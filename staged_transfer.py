"""Staged transfer pipeline: S1 base block -> S2 growth -> S3 warmup -> S4 full.

Implements STAGED_DESIGN.md (locked 2026-11-18): copy the first n_base teacher
layers, re-initialize the new-architecture components, train the base (S1),
duplicate the trained blocks to the target depth replacing each copy's FFN with
the shared experts of a distinct teacher layer (S2, function-preserving via
zero gates), warm the whole volume with a block-output orthogonality penalty
(S3), then optional full training on the real dataset (S4).

Model classes are a staged variant of the reviewed experiment.py (same math,
plus per-block residual gates and block-output pooling for the penalty);
experiment.py itself stays untouched as the transfer diagnostic.

Env interface (all optional, defaults for a bounded mechanics test):
  K3_STAGED_RUN  smoke|real   smoke: synthetic data, no teacher download, FFN
                              copies duplicate the base (pure stacking ablation).
  K3_S1_STEPS / K3_S3_STEPS / K3_S4_STEPS   optimizer steps per stage (S4 off at 0)
  K3_LAMBDA_ORTH (1e-2), K3_ORTH_SCOPE all|adjacent|sublayer
  K3_N_BASE (8), K3_TARGET (16), K3_SEQ, K3_BATCH (0 = device count)
  K3_ARTIFACTS, K3_CACHE_DIR, K3_ENWIK8_PATH, K3_REVISION, HF_TOKEN
Scope: candidate student; no teacher parity, speed or quality claims.
"""
import os, sys, time, json, math, gc, re, struct, traceback, hashlib
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import jax, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import ml_dtypes, requests
import importlib.metadata

RUN = os.environ.get('K3_STAGED_RUN', 'smoke')
S1_STEPS = int(os.environ.get('K3_S1_STEPS', 8 if RUN == 'smoke' else 300))
S3_STEPS = int(os.environ.get('K3_S3_STEPS', 8 if RUN == 'smoke' else 600))
S4_STEPS = int(os.environ.get('K3_S4_STEPS', 0))
LAMBDA_ORTH = float(os.environ.get('K3_LAMBDA_ORTH', 1e-2))
ORTH_SCOPE = os.environ.get('K3_ORTH_SCOPE', 'all')
N_BASE = int(os.environ.get('K3_N_BASE', 8))
TARGET = int(os.environ.get('K3_TARGET', 16))
SEQ = int(os.environ.get('K3_SEQ', 64 if RUN == 'smoke' else 128))
BATCH_REQUESTED = int(os.environ.get('K3_BATCH', 0))
ARTIFACTS = os.environ.get('K3_ARTIFACTS', 'artifacts')
os.makedirs(ARTIFACTS, exist_ok=True)
REPORT = {'run': RUN, 'status': 'RUNNING', 'results': [],
          'scope': 'staged transfer candidate; no teacher parity claim'}
RPT_PATH = os.path.join(ARTIFACTS, 'report.json')


def save_rpt():
    p = Path(RPT_PATH); tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(REPORT, indent=2, default=str), encoding='utf-8'); os.replace(tmp, p)


def chk(cat, tid, ok, detail=''):
    REPORT['results'].append(dict(cat=cat, id=tid, status='PASS' if ok else 'FAIL', detail=str(detail)))
    print(f"[{'PASS' if ok else 'FAIL'}] {cat}.{tid}: {detail}", flush=True); save_rpt()
    if not ok: raise AssertionError(f'{cat}.{tid}: {detail}')


def exception_hook(typ, value, tb):
    REPORT['status'] = 'FAILED'; REPORT['exception_type'] = typ.__name__; save_rpt()
    sys.__excepthook__(typ, value, tb)
sys.excepthook = exception_hook
sys.path.insert(0, str(Path(__file__).resolve().parent))
import staged_core
from staged_core import orthogonality_penalty, build_growth_plan

HIGHEST = jax.lax.Precision.HIGHEST
devices = jax.devices(); N_DEV = len(devices)
if RUN == 'real':
    if jax.default_backend() != 'tpu':
        raise RuntimeError('Staged real run requires a TPU backend; detected ' + jax.default_backend())
BATCH = N_DEV if BATCH_REQUESTED == 0 else max(BATCH_REQUESTED, 1)
mesh = Mesh(np.array(devices).reshape(-1), ('fsdp',))


def fsdp_spec(shape, n_dev=N_DEV):
    if n_dev <= 1: return P()
    c = [(s, i) for i, s in enumerate(shape) if s >= n_dev and s % n_dev == 0]
    if not c: return P()
    _, i = max(c); sp = [None] * len(shape); sp[i] = 'fsdp'; return P(*sp)


def sharded_like(abstract):
    return jax.tree_util.tree_map(lambda a: NamedSharding(mesh, fsdp_spec(a.shape)), abstract)


REPORT['env'] = {'python': sys.version, 'jax': jax.__version__, 'backend': jax.default_backend(),
                 'devices': [str(x) for x in devices], 'run': RUN, 'stages': {'S1': S1_STEPS, 'S3': S3_STEPS, 'S4': S4_STEPS},
                 'lambda_orth': LAMBDA_ORTH, 'orth_scope': ORTH_SCOPE, 'n_base': N_BASE, 'target': TARGET, 'seq': SEQ, 'batch': BATCH}
REPORT['packages'] = {k: importlib.metadata.version(k) for k in ['jax', 'jaxlib', 'flax', 'optax', 'numpy', 'ml_dtypes', 'requests']}
save_rpt()
HF_TOKEN = os.environ.get('HF_TOKEN')
K3_SHA = None

# ----------------------------- model (staged variant) -----------------------------
# Copied from the reviewed experiment.py; deltas: per-block residual gate
# (block_gate; base init 1, copies 0) and optional block-output pooling.
import dataclasses
from typing import Tuple
import flax.linen as nn
BF = dict(dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)


def situ(x, beta=4.0): return beta * jnp.tanh(x / beta) * jax.nn.sigmoid(x)
def situ_glu(gate, up, bg=4.0, bu=25.0): return situ(gate.astype(jnp.float32), bg) * (bu * jnp.tanh(up.astype(jnp.float32) / bu))


@dataclasses.dataclass(frozen=True)
class StudentConfig:
    d_model: int = 7168; num_blocks: int = 4; layers_per_block: int = 4; n_layers: int = 16; vocab: int = 163840
    tie_embeddings: bool = False; n_heads: int = 32; d_head: int = 128; mla_positions: Tuple[int, ...] = (3, 7, 11, 15)
    q_lora_rank: int = 1536; kv_lora_rank: int = 512; qk_rope_head_dim: int = 64; rope_theta: float = 10000.0
    d_ff: int = 6144; d_conv: int = 4; dar_latent: int = 256
    @property
    def d_attn_proj(self): return self.n_heads * self.d_head


class RoPE(nn.Module):
    dim: int; theta: float = 10000.0
    @nn.compact
    def __call__(self, L):
        half = self.dim // 2; fr = 1.0 / (self.theta ** (jnp.arange(0, half) / half)); ang = jnp.outer(jnp.arange(L), fr)
        return jnp.cos(ang), jnp.sin(ang)


class DeltaAttentionResidual(nn.Module):
    d_model: int; d_latent: int
    @nn.compact
    def __call__(self, cur, sources):
        a = self.param("alpha_scale", nn.initializers.zeros, ())
        w = self.param("w_l", nn.initializers.zeros, (self.d_model,))
        norm = nn.RMSNorm(epsilon=1e-6, param_dtype=jnp.float32, name="src_norm")
        stack = jnp.stack(sources, 0) if sources else jnp.zeros((1,) + cur.shape, cur.dtype)
        sn = norm(stack.astype(jnp.float32))
        weights = jax.nn.softmax(jnp.einsum("d,nbld->nbl", w, sn), axis=0)
        retrieved = jnp.einsum("nbl,nbld->bld", weights, stack.astype(jnp.float32))
        return (a * retrieved).astype(cur.dtype)


class GDN2Mixer(nn.Module):
    cfg: StudentConfig
    @nn.compact
    def __call__(self, x):
        cfg = self.cfg; B, L, _ = x.shape; H, Dh, Pp = cfg.n_heads, cfg.d_head, cfg.d_attn_proj; eps = 1e-6
        def short_conv(name, u):
            w = self.param(f"{name}_conv_w", nn.initializers.normal(0.02), (u.shape[-1], cfg.d_conv))
            b = self.param(f"{name}_conv_b", nn.initializers.zeros, (u.shape[-1],))
            y = jax.lax.conv_general_dilated(u.astype(jnp.float32), w.T[:, None, :].astype(jnp.float32), window_strides=(1,),
                padding=[(cfg.d_conv - 1, 0)], feature_group_count=u.shape[-1], dimension_numbers=("NHC", "HIO", "NHC"))
            return (y + b[None, None, :]).astype(jnp.float32)
        D = lambda n, o, bias=False: nn.Dense(o, use_bias=bias, name=n, **BF)
        q = jax.nn.silu(short_conv("q", D("q_proj", Pp)(x))).reshape(B, L, H, Dh)
        k = jax.nn.silu(short_conv("k", D("k_proj", Pp)(x))).reshape(B, L, H, Dh)
        v = jax.nn.silu(short_conv("v", D("v_proj", Pp)(x))).reshape(B, L, H, Dh)
        l2n = lambda t: t * jax.lax.rsqrt(jnp.sum(t * t, -1, keepdims=True) + eps ** 2)
        q, k = l2n(q), l2n(k)
        erase = jax.nn.sigmoid(D("erase_gate", Pp, True)(x)).reshape(B, L, H, Dh)
        write = jax.nn.sigmoid(D("write_gate", Pp, True)(x)).reshape(B, L, H, Dh)
        decay_a = self.param("decay_a", nn.initializers.zeros, (H,)).astype(jnp.float32)
        f_b = D("f_b_proj", Pp)(D("f_a_proj", 128)(x))
        dt_bias = self.param("dt_bias", nn.initializers.zeros, (Pp,))
        raw_gate = f_b.reshape(B, L, H, Dh).astype(jnp.float32) + dt_bias.reshape(H, Dh)
        log_decay = -5.0 * jax.nn.sigmoid(jnp.exp(decay_a)[None, None, :, None] * raw_gate)
        self.sow('diagnostics', 'g_min', jnp.min(log_decay))
        self.sow('diagnostics', 'g_max', jnp.max(log_decay))
        out_gate = D("out_gate", Pp)(x)
        S0 = jnp.zeros((B, H, Dh, Dh), jnp.float32)
        def step(S, xs):
            q_t, k_t, v_t, e_t, w_t, g_t = xs
            S = S * jnp.exp(g_t)[..., None, :]
            Sk = jnp.einsum("bhij,bhj->bhi", S, e_t * k_t)
            S = S + jnp.einsum("bhi,bhj->bhij", w_t * v_t - Sk, k_t)
            return S, jnp.einsum("bhij,bhj->bhi", S, q_t)
        xs = jax.tree.map(lambda t: jnp.swapaxes(t.astype(jnp.float32), 1, 0), (q, k, v, erase, write, log_decay))
        _, o = jax.lax.scan(jax.checkpoint(step), S0, xs); o = jnp.swapaxes(o, 0, 1).reshape(B, L, Pp)
        o = o / math.sqrt(Dh)
        o = nn.RMSNorm(epsilon=1e-5, param_dtype=jnp.bfloat16, name="o_norm")(o.reshape(B, L, H, Dh)).reshape(B, L, Pp) * jax.nn.sigmoid(out_gate)
        return nn.Dense(cfg.d_model, use_bias=False, name="o_proj", **BF)(o)


class MLAMixer(nn.Module):
    cfg: StudentConfig
    @nn.compact
    def __call__(self, x, cos, sin):
        cfg = self.cfg; B, L, _ = x.shape; H, D = cfg.n_heads, cfg.d_head; R = cfg.qk_rope_head_dim
        dense = lambda n, o: nn.Dense(o, use_bias=False, name=n, **BF)
        qa = nn.RMSNorm(epsilon=1e-6, param_dtype=jnp.bfloat16, name="q_a_layernorm")(dense("q_a_proj", cfg.q_lora_rank)(x))
        q = dense("q_b_proj", H * (D + R))(qa).reshape(B, L, H, D + R)
        kc, kr = jnp.split(dense("kv_a_proj_with_mqa", cfg.kv_lora_rank + R)(x), [cfg.kv_lora_rank], -1)
        kc = nn.RMSNorm(epsilon=1e-6, param_dtype=jnp.bfloat16, name="kv_a_layernorm")(kc)
        kv = dense("kv_b_proj", H * 2 * D)(kc).reshape(B, L, H, 2 * D)
        k, v = jnp.split(kv, 2, -1)
        k = jnp.concatenate([k, jnp.broadcast_to(kr[:, :, None, :], (B, L, H, R))], -1)
        scores = jnp.einsum("blhd,bmhd->bhlm", q.astype(jnp.float32), k.astype(jnp.float32)) / math.sqrt(D + R)
        att = jax.nn.softmax(jnp.where(jnp.tril(jnp.ones((L, L), bool))[None, None], scores, -1e30), -1)
        o = jnp.einsum("bhlm,bmhd->blhd", att, v.astype(jnp.float32)).reshape(B, L, H * D)
        o = o * jax.nn.sigmoid(dense("g_proj", H * D)(x))
        return dense("o_proj", cfg.d_model)(o)


class FFNSublayer(nn.Module):
    cfg: StudentConfig
    @nn.compact
    def __call__(self, x):
        g = nn.Dense(self.cfg.d_ff, use_bias=False, name="gate_proj", **BF)(x); u = nn.Dense(self.cfg.d_ff, use_bias=False, name="up_proj", **BF)(x)
        return nn.Dense(self.cfg.d_model, use_bias=False, name="down_proj", **BF)(situ_glu(g, u))


class DARLayer(nn.Module):
    """Returns (normed input, dar delta, mixer delta) WITHOUT adding them; the
    block applies the residual gate, so a copied block with gate 0 is an exact
    identity including its DAR contribution."""
    cfg: StudentConfig; layer_idx: int
    @nn.compact
    def __call__(self, x, local_deltas, history, cos, sin):
        src = [history[j] for j in range(history.shape[0])] + list(local_deltas)
        dar = DeltaAttentionResidual(self.cfg.d_model, self.cfg.dar_latent, name="dar")(x, src)
        x1 = x + dar
        nx = nn.RMSNorm(epsilon=1e-6, param_dtype=jnp.bfloat16, name="pre_sublayer_norm")(x1)
        d = MLAMixer(cfg=self.cfg, name="mixer")(nx, cos, sin) if self.layer_idx in self.cfg.mla_positions else GDN2Mixer(cfg=self.cfg, name="mixer")(nx)
        return x1, dar, d


class DARBlock(nn.Module):
    cfg: StudentConfig; block_idx: int; layer_idx_start: int
    @nn.compact
    def __call__(self, x, history, cos, sin):
        g = self.param("block_gate", nn.initializers.ones, ())
        ld = []; pooled = None
        for i in range(self.cfg.layers_per_block):
            idx = self.layer_idx_start + i
            x1, dar_d, mixer_d = DARLayer(cfg=self.cfg, layer_idx=idx, name=f"layer_{idx}")(x, ld, history, cos, sin)
            x = x1 + g * dar_d
            d = g * mixer_d
            ld.append(d); x = x + d
        nh = jnp.concatenate([history, sum(ld)[None]], 0)
        nx = nn.RMSNorm(epsilon=1e-6, param_dtype=jnp.bfloat16, name="pre_ffn_norm")(x)
        x = x + g * FFNSublayer(cfg=self.cfg, name="ffn")(nx)
        pooled = jnp.mean(x.astype(jnp.float32), axis=1)          # [B, d] for the orthogonality gram
        return x, nh, pooled


class StudentLM(nn.Module):
    cfg: StudentConfig
    @nn.compact
    def __call__(self, ids, collect_blocks=False):
        B, L = ids.shape
        embed = nn.Embed(self.cfg.vocab, self.cfg.d_model, name="embed", **BF)
        x = embed(ids); cos, sin = RoPE(self.cfg.qk_rope_head_dim, self.cfg.rope_theta)(L)
        hist = jnp.zeros((0, B, L, self.cfg.d_model), dtype=x.dtype); RB = nn.remat(DARBlock); pooled = []
        for bi in range(self.cfg.num_blocks):
            x, hist, pb = RB(cfg=self.cfg, block_idx=bi, layer_idx_start=bi * self.cfg.layers_per_block, name=f"block_{bi}")(x, hist, cos, sin)
            pooled.append(pb)
        x = nn.RMSNorm(epsilon=1e-6, param_dtype=jnp.bfloat16, name="final_norm")(x)
        logits = embed.attend(x) if self.cfg.tie_embeddings else nn.Dense(self.cfg.vocab, use_bias=False, name="lm_head", **BF)(x)
        return (logits, jnp.stack(pooled, 0)) if collect_blocks else logits


# ----------------------------- report/optimizer helpers -----------------------------
import optax
from flax.traverse_util import flatten_dict, unflatten_dict
import jax.tree_util as jtu

NEW_PAT = re.compile(r'/dar/|/erase_gate/|/write_gate/|/dt_bias$|/pre_ffn_norm/scale$|/block_gate$')


def trainable_mask(flat, policy, copy_blocks):
    out = {}
    for k in flat:
        new = bool(NEW_PAT.search(k))
        in_copy = any(k.startswith(f'block_{b}/') for b in copy_blocks)
        out[k] = new or (policy == 'all') or (policy == 'blocks' and in_copy)
    return out


def merge(a, b):
    fa, fb = flatten_dict(a, sep='/'), flatten_dict(b, sep='/')
    return unflatten_dict({**fb, **fa}, sep='/')


def sample(ids, rng, bs, seq):
    starts = jax.random.randint(rng, (bs,), 0, len(ids) - seq - 1)
    return np.stack([np.asarray(ids[s:s + seq + 1]) for s in np.asarray(starts)])


def run_stage(tag, model, tr, fr, tx_state, steps, ids, report_key, collect_gram):
    """Shared training loop; returns (tr, final_optimizer_state, report_dict)."""
    rep = {'steps_planned': steps}; t0 = time.monotonic(); last = {}; op = tx_state[1]
    if steps:
        @jax.jit
        def train_step(a, op, b, batch):
            def loss_of(params):
                full = merge(params, b)
                out = model.apply({'params': full}, batch[:, :-1], collect_blocks=True)
                logits, pooled = out if collect_gram else (out, None)
                ce = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), batch[:, 1:]).mean()
                if not collect_gram:
                    return ce, None, None
                pen, gram = orthogonality_penalty(pooled, LAMBDA_ORTH, ORTH_SCOPE)
                return ce + pen, ce, gram
            (loss, ce, gram), grads = jax.value_and_grad(loss_of, has_aux=True)(a)
            gn = optax.global_norm(grads); ok = jnp.isfinite(loss) & jnp.isfinite(gn)
            updates, nop = tx_state[0].update(grads, op, a)
            ntr = jax.tree.map(lambda new, old: jnp.where(ok, new, old), updates, a)
            na = jax.tree.map(lambda new, old: jnp.where(ok, new, old), nop, op)
            return ntr, na, loss, ce, gram, gn, ok
        for step in range(1, steps + 1):
            tr, op, lv, ce, gram, gn, ok = train_step(tr, op, fr, sample(ids, jax.random.PRNGKey(step), BATCH, SEQ))
            row = {'step': step, 'loss': float(lv), 'ce': float(ce), 'grad_norm': float(gn), 'accepted': bool(ok)}
            if gram is not None:
                g = np.asarray(gram); n = g.shape[0]
                row['orth_mean_abs_cos'] = float(np.abs(g - np.eye(n, dtype=bool) * g).sum() / max(n * (n - 1), 1))
                row['gram_min'] = float(g.min()); row['gram_max'] = float(g.max())
            rep.setdefault('metrics', []).append(row); REPORT['last_step'] = row; save_rpt()
            print(f'[{tag}]', {k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            if not row['accepted']:
                rep['status'] = 'FAILED_NONFINITE'; raise FloatingPointError(f'{tag}: nonfinite training step; update skipped')
            last = row
    rep['status'] = 'COMPLETED'; rep['elapsed_seconds'] = time.monotonic() - t0
    if last: rep['last'] = last
    REPORT['stages'][report_key] = rep; save_rpt()
    return tr, op, rep


def save_trainable(tag, tr):
    flat = {k: np.asarray(v) for k, v in flatten_dict(jax.device_get(tr), sep='/').items()}
    path = Path(ARTIFACTS, f'{tag}_trainable.npz'); temp = path.with_suffix('.tmp')
    with open(temp, 'wb') as f: np.savez(f, **flat)
    os.replace(temp, path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    REPORT['stages'][tag]['checkpoint'] = {'file': path.name, 'sha256': sha, 'kind': 'trainable-only', 'exact_resume': False}
    save_rpt(); print(f'[checkpoint] {path.name} sha256={sha[:16]}', flush=True)


# ----------------------------- S1: base -----------------------------
REPORT['stages'] = {}
if ORTH_SCOPE not in ('all', 'adjacent'):
    raise ValueError("ORTH_SCOPE 'sublayer' is designed but not wired yet (STAGED_DESIGN.md); use all|adjacent")

"""enwik8 identity: https://mattmahoney.net/dc/textdata.html (size + SHA-1); loader copied from reviewed experiment.py."""
import gzip, io, zipfile as _zipfile
ENWIK8_SIZE = 100_000_000
ENWIK8_SHA1 = '57b8363b814821dc9d47aa4d41f58733519076b2'
ENWIK8_URLS = ['https://raw.githubusercontent.com/lucidrains/x-transformers/main/data/enwik8.gz',
               'https://www.mattmahoney.net/dc/enwik8.zip', 'https://mattmahoney.net/dc/enwik8.zip']


def decode_enwik8(blob, size=ENWIK8_SIZE, sha1=ENWIK8_SHA1):
    if blob.startswith(b'PK\x03\x04'):
        with _zipfile.ZipFile(io.BytesIO(blob)) as z:
            names = [n for n in z.namelist() if n.rsplit('/', 1)[-1] == 'enwik8']
            if len(names) != 1 or z.getinfo(names[0]).file_size != size: raise ValueError('ZIP has no unique enwik8 of expected size')
            raw = z.read(names[0])
    elif blob.startswith(b'\x1f\x8b'):
        with gzip.GzipFile(fileobj=io.BytesIO(blob)) as f: raw = f.read(size + 1)
    elif len(blob) == size: raw = blob
    else: raise ValueError(f'Response is neither ZIP, gzip nor expected raw file; bytes={len(blob)}, magic_hex={blob[:8].hex()}')
    if len(raw) != size or hashlib.sha1(raw).hexdigest() != sha1: raise ValueError('enwik8 size/SHA-1 mismatch')
    return raw


def load_enwik8(cache_dir, local_path='', report=None, save_report=lambda: None):
    cache = Path(cache_dir); cache.mkdir(parents=True, exist_ok=True)
    target = cache / 'enwik8'; attempts = []
    if report is not None: report['data_download_attempts'] = attempts
    raw = None; source = ''
    if local_path:
        raw = decode_enwik8(Path(local_path).read_bytes()); source = 'user file'
    elif target.is_file():
        try:
            raw = decode_enwik8(target.read_bytes()); source = 'verified local cache'
        except (ValueError, OSError, _zipfile.BadZipFile): raw = None
    if raw is None:
        for url in ENWIK8_URLS:
            record = {'source': url.split('?', 1)[0]}; attempts.append(record)
            try:
                with requests.get(url, stream=True, timeout=(30, 180), headers={'User-Agent': 'K3-transfer-dataset/4'}) as r:
                    record.update(status=r.status_code); r.raise_for_status(); body = bytearray()
                    for chunk in r.iter_content(1024 * 1024):
                        body.extend(chunk)
                        if len(body) > ENWIK8_SIZE + 1024 * 1024: raise ValueError('Dataset response exceeds download bound')
                raw = decode_enwik8(bytes(body)); source = record['source']; record['verified'] = True
                save_report(); break
            except Exception as exc:
                record.update(verified=False, error_type=type(exc).__name__); save_report(); print('[data source rejected]', record, flush=True)
        if raw is None: raise RuntimeError('No verified enwik8 source succeeded; set K3_ENWIK8_PATH.')
    temp = target.with_suffix('.tmp'); temp.write_bytes(raw); os.replace(temp, target)
    if report is not None:
        report['enwik8_identity'] = {'bytes': len(raw), 'sha1': hashlib.sha1(raw).hexdigest(), 'source': source, 'verified': True}
    save_report(); print('[data] verified enwik8', len(raw), 'bytes; source:', source, flush=True)
    return raw

if RUN == 'smoke':
    base_cfg = StudentConfig(d_model=64, num_blocks=N_BASE // 4, n_layers=N_BASE, vocab=256, n_heads=2, d_head=32,
                             q_lora_rank=32, kv_lora_rank=16, d_ff=128, mla_positions=tuple(p for p in (3, 7, 11, 15) if p < N_BASE), dar_latent=16)
    train_ids = np.tile(np.arange(256, dtype=np.int32), 400); val_ids = np.tile(np.arange(256, dtype=np.int32), 40)
    REPORT['data'] = {'synthetic': True, 'note': 'smoke mechanics only'}
else:
    base_cfg = StudentConfig(num_blocks=N_BASE // 4, n_layers=N_BASE, mla_positions=tuple(p for p in (3, 7, 11, 15) if p < N_BASE))
    from transformers import AutoTokenizer
    from huggingface_hub import HfApi
    K3_SHA = HfApi(token=HF_TOKEN).model_info('moonshotai/Kimi-K3', revision=os.environ.get('K3_REVISION', 'main')).sha
    REPORT['k3_sha'] = K3_SHA
    tok = AutoTokenizer.from_pretrained('moonshotai/Kimi-K3', revision=K3_SHA, token=HF_TOKEN, trust_remote_code=True)
    raw = load_enwik8(Path(os.environ.get('K3_CACHE_DIR', '/content/k3_cache')) / 'data',
                      local_path=os.environ.get('K3_ENWIK8_PATH', ''), report=REPORT, save_report=save_rpt)
    text = raw[:2_000_000].decode('utf-8', errors='replace')
    train_ids = np.asarray(tok.encode(text, add_special_tokens=False), np.int32)
    val_ids = train_ids[-20000:]
    REPORT['data'] = {'train_tokens': len(train_ids), 'bounded_subset': True, 'full_enwik8': False}
    save_rpt()

base_model = StudentLM(cfg=base_cfg)
ids_small = np.zeros((1, 8), np.int32)
base_abstract = jax.eval_shape(lambda r: base_model.init(r, ids_small)['params'], jax.random.PRNGKey(0))
base_shard = sharded_like(base_abstract)
rng_init = jax.random.PRNGKey(0)
params_host = {}

if RUN == 'smoke':
    params_host = jax.device_get(base_model.init(rng_init, ids_small)['params'])
    chk('S1', 'smoke_init', len(flatten_dict(params_host, sep='/')) > 0)
else:
    # ---- teacher fetch (copied from reviewed experiment.py; provenance identical) ----
    REPO = 'moonshotai/Kimi-K3'; REV = K3_SHA
    HF_HEADERS = {'Authorization': f'Bearer {HF_TOKEN}'} if HF_TOKEN else {}
    def hf_url(path): return f'https://huggingface.co/{REPO}/resolve/{REV}/{path}'
    def http_get(url, byte_range=None, retries=4):
        h = dict(HF_HEADERS); last = None
        if byte_range: h['Range'] = f'bytes={byte_range[0]}-{byte_range[1]}'
        for a in range(retries):
            try: r = requests.get(url, headers=h, allow_redirects=True, timeout=180, stream=True)
            except requests.RequestException as e: last = e; time.sleep(2 ** a); continue
            if byte_range and r.status_code != 206:
                r.close(); raise RuntimeError(f'Range not honored: HTTP {r.status_code}')
            if r.status_code in (200, 206):
                data = r.content; r.close()
                if byte_range and len(data) != byte_range[1] - byte_range[0] + 1: raise ValueError('Truncated range')
                return data
            if r.status_code in (429, 500, 502, 503, 504): last = RuntimeError(f'HTTP {r.status_code}'); time.sleep(2 ** a); continue
            r.raise_for_status()
        raise RuntimeError(f'GET failed: {last}')
    NP_DTYPES = {'BF16': ml_dtypes.bfloat16, 'F16': np.float16, 'F32': np.float32, 'U8': np.uint8, 'I8': np.int8, 'I32': np.int32, 'I64': np.int64,
                 'F8_E4M3': ml_dtypes.float8_e4m3fn, 'F8_E5M2': ml_dtypes.float8_e5m2, 'F8_E8M0': getattr(ml_dtypes, 'float8_e8m0fnu', None)}
    def load_index(): return json.loads(http_get(hf_url('model.safetensors.index.json')))['weight_map']
    _hc = {}
    def read_shard_header(shard):
        key = shard
        if key in _hc: return _hc[key]
        url = hf_url(shard); (n,) = struct.unpack('<Q', http_get(url, (0, 7)))
        hd = json.loads(http_get(url, (8, 7 + n))); hd.pop('__metadata__', None); _hc[key] = (hd, 8 + n); return _hc[key]
    def cached_tensor_bytes(p):
        key = hashlib.sha256(json.dumps({k: p[k] for k in ['name', 'url', 'start', 'end', 'dtype', 'shape']}, sort_keys=True).encode()).hexdigest()
        root = Path(os.environ.get('K3_CACHE_DIR', '/content/k3_cache')) / 'tensor_ranges' / REV
        root.mkdir(parents=True, exist_ok=True)
        data_path = root / (key + '.bin'); meta_path = root / (key + '.json'); expected = p['end'] - p['start'] + 1
        if data_path.is_file() and meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text()); buf = data_path.read_bytes()
                if len(buf) == expected and hashlib.sha256(buf).hexdigest() == meta['sha256']: return buf
            except (OSError, ValueError, KeyError): pass
        buf = http_get(p['url'], (p['start'], p['end']))
        if len(buf) != expected: raise ValueError('Tensor byte-range length mismatch')
        try:
            tmp = data_path.with_suffix('.tmp'); tmp.write_bytes(buf); os.replace(tmp, data_path)
            tmp_meta = meta_path.with_suffix('.tmp'); tmp_meta.write_text(json.dumps({'name': p['name'], 'sha256': hashlib.sha256(buf).hexdigest(), 'bytes': len(buf)})); os.replace(tmp_meta, meta_path)
        except OSError as exc: print('[tensor cache write skipped]', type(exc).__name__, flush=True)
        return buf
    def plan_fetch(names, wm):
        bs = defaultdict(list)
        for n in names: bs[wm[n]].append(n)
        plan = []
        for shard, ns in bs.items():
            hd, base = read_shard_header(shard); url = hf_url(shard)
            for n in ns:
                i = hd[n]; s, e = i['data_offsets']
                plan.append(dict(name=n, url=url, start=base + s, end=base + e - 1, dtype=i['dtype'], shape=i['shape']))
        return plan
    def fetch_tensors(names, wm, workers=8):
        plan = plan_fetch(names, wm); tot = sum(p['end'] - p['start'] + 1 for p in plan)
        print(f'{len(plan)} tensors, {tot / 1e9:.2f} GB', flush=True)
        def get(p): return p['name'], np.frombuffer(bytearray(cached_tensor_bytes(p)), dtype=NP_DTYPES[p['dtype']]).reshape(p['shape']).copy()
        with ThreadPoolExecutor(workers) as ex: return dict(ex.map(get, plan))
    f32 = lambda a: np.asarray(a)

    wm = load_index()
    KDA_SUF = ['self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight', 'self_attn.o_proj.weight', 'self_attn.g_proj.weight',
               'self_attn.q_conv1d.weight', 'self_attn.k_conv1d.weight', 'self_attn.v_conv1d.weight', 'self_attn.b_proj.weight',
               'self_attn.f_a_proj.weight', 'self_attn.f_b_proj.weight', 'self_attn.A_log', 'self_attn.dt_bias', 'self_attn.o_norm.weight', 'input_layernorm.weight']
    MLA_SUF = ['self_attn.q_a_proj.weight', 'self_attn.q_b_proj.weight', 'self_attn.kv_a_proj_with_mqa.weight', 'self_attn.kv_b_proj.weight',
               'self_attn.q_a_layernorm.weight', 'self_attn.kv_a_layernorm.weight', 'self_attn.o_proj.weight', 'self_attn.g_proj.weight', 'input_layernorm.weight']
    BASE_FFN_LAYER = 1
    need = [f'language_model.model.layers.{L}.{s}' for L in range(base_cfg.n_layers) for s in (MLA_SUF if L in base_cfg.mla_positions else KDA_SUF)]
    pre_sh = f'language_model.model.layers.{BASE_FFN_LAYER}.block_sparse_moe.shared_experts'
    need += [f'{pre_sh}.{w}.weight' for w in ('gate_proj', 'up_proj', 'down_proj')]
    need += ['language_model.model.embed_tokens.weight', 'language_model.model.norm.weight', 'language_model.lm_head.weight']
    missing = [n for n in need if n not in wm]
    if missing: raise KeyError(f'Missing required pretrained tensors: {missing}')
    REPORT['fetch_S1'] = {'tensors': len(need)}
    save_rpt()
    tensors = fetch_tensors(need, wm); print(f'[fetch] loaded {len(tensors)}', flush=True)

    def _sl_rows(W, n): return np.asarray(W)[:n, ...]
    def _sl_cols(W, n): return np.asarray(W)[..., :n]
    def _gate_dense(b_k3, H_s, Dh):
        b = f32(b_k3); bh = b[:H_s]
        return np.repeat(bh[:, None, :], Dh, axis=1).reshape(H_s * Dh, -1), np.zeros(H_s * Dh, np.float32)
    def _kda(layer, H_s, Dh):
        p = f'language_model.model.layers.{layer}.self_attn'; Ps = H_s * Dh; T = tensors; o = {}
        o['mixer/q_proj/kernel'] = _sl_rows(T[f'{p}.q_proj.weight'], Ps).T; o['mixer/k_proj/kernel'] = _sl_rows(T[f'{p}.k_proj.weight'], Ps).T
        o['mixer/v_proj/kernel'] = _sl_rows(T[f'{p}.v_proj.weight'], Ps).T; o['mixer/o_proj/kernel'] = _sl_cols(T[f'{p}.o_proj.weight'], Ps)
        o['mixer/out_gate/kernel'] = _sl_rows(T[f'{p}.g_proj.weight'], Ps).T
        for n in 'qkv': o[f'mixer/{n}_conv_w'] = _sl_rows(T[f'{p}.{n}_conv1d.weight'].reshape(-1, 4), Ps)
        eW, eb = _gate_dense(T[f'{p}.b_proj.weight'], H_s, Dh)
        o['mixer/erase_gate/kernel'] = eW.T; o['mixer/write_gate/kernel'] = eW.T; o['mixer/erase_gate/bias'] = eb; o['mixer/write_gate/bias'] = eb
        o['mixer/f_a_proj/kernel'] = f32(T[f'{p}.f_a_proj.weight']).T; o['mixer/f_b_proj/kernel'] = _sl_rows(T[f'{p}.f_b_proj.weight'], Ps)
        A = f32(T[f'{p}.A_log']); o['mixer/decay_a'] = A[:H_s]
        o['mixer/dt_bias'] = f32(T[f'{p}.dt_bias'])[:Ps]
        o['mixer/o_norm/scale'] = f32(T[f'{p}.o_norm.weight'])
        o['pre_sublayer_norm/scale'] = f32(T[f'language_model.model.layers.{layer}.input_layernorm.weight']); return o
    def _mla(layer, H_s, Dh):
        p = f'language_model.model.layers.{layer}.self_attn'; Ps = H_s * Dh; T = tensors; o = {}
        o['mixer/q_a_proj/kernel'] = f32(T[f'{p}.q_a_proj.weight']).T; o['mixer/q_a_layernorm/scale'] = f32(T[f'{p}.q_a_layernorm.weight'])
        o['mixer/q_b_proj/kernel'] = _sl_rows(T[f'{p}.q_b_proj.weight'], H_s * (Dh + base_cfg.qk_rope_head_dim)).T
        o['mixer/g_proj/kernel'] = _sl_rows(T[f'{p}.g_proj.weight'], Ps).T
        o['mixer/kv_a_proj_with_mqa/kernel'] = f32(T[f'{p}.kv_a_proj_with_mqa.weight']).T; o['mixer/kv_a_layernorm/scale'] = f32(T[f'{p}.kv_a_layernorm.weight'])
        kb = f32(T[f'{p}.kv_b_proj.weight']).reshape(-1, 2 * Dh, base_cfg.kv_lora_rank)[:H_s].reshape(2 * Ps, base_cfg.kv_lora_rank)
        o['mixer/kv_b_proj/kernel'] = kb.T; o['mixer/o_proj/kernel'] = _sl_cols(T[f'{p}.o_proj.weight'], Ps)
        o['pre_sublayer_norm/scale'] = f32(T[f'language_model.model.layers.{layer}.input_layernorm.weight']); return o

    H_s, Dh = base_cfg.n_heads, base_cfg.d_head; patch = {}
    for L in range(base_cfg.n_layers):
        for k, v in (_mla if L in base_cfg.mla_positions else _kda)(L, H_s, Dh).items():
            patch[f'block_{L // base_cfg.layers_per_block}/layer_{L}/{k}'] = v
    G, U, Dn = (f32(tensors[f'{pre_sh}.{w}.weight']) for w in ('gate_proj', 'up_proj', 'down_proj'))
    for bi in range(base_cfg.num_blocks):
        patch[f'block_{bi}/ffn/gate_proj/kernel'] = G.T; patch[f'block_{bi}/ffn/up_proj/kernel'] = U.T; patch[f'block_{bi}/ffn/down_proj/kernel'] = Dn.T
    patch['embed/embedding'] = tensors['language_model.model.embed_tokens.weight']
    patch['final_norm/scale'] = tensors['language_model.model.norm.weight']
    patch['lm_head/kernel'] = tensors['language_model.lm_head.weight'].T
    del tensors; gc.collect()
    # apply patches against the initialized tree; DAR/gates/dt_bias/norms = new components
    init_host = jax.device_get(base_model.init(rng_init, ids_small)['params'])
    iflat = flatten_dict(init_host, sep='/'); unused = set(patch); applied = fb = 0
    rng_new = np.random.default_rng(42)
    for key, val in iflat.items():
        dt = np.dtype(jax.dtypes.canonicalize_dtype(jnp.asarray(val).dtype)); shape = tuple(np.asarray(val).shape)
        if key in patch:
            a = np.asarray(patch[key])
            if a.shape != shape:
                if a.T.shape == shape: a = a.T
                elif a.ndim == 1 and len(shape) == 1 and shape[0] % a.shape[0] == 0: a = np.tile(a, shape[0] // a.shape[0])
                else: raise ValueError(f'{key}: patch {a.shape} != model {shape}')
            iflat[key] = a.astype(dt, copy=False); applied += 1; unused.discard(key)
        elif NEW_PAT.search(key) and key.endswith('block_gate'):
            iflat[key] = np.ones(shape, dt); fb += 1
        elif '/dar/' in key:
            iflat[key] = (np.ones(shape, dt) if key.endswith('/scale') else np.zeros(shape, dt)); fb += 1
        elif key.endswith('pre_ffn_norm/scale'):
            iflat[key] = np.ones(shape, dt); fb += 1
        elif key.endswith('_conv_b'):
            iflat[key] = np.zeros(shape, dt); fb += 1
        elif '/erase_gate/' in key or '/write_gate/' in key or key.endswith('/dt_bias'):
            iflat[key] = np.zeros(shape, dt); fb += 1
        else:
            raise KeyError(f'unexpected unpatched teacher leaf: {key}')
    if unused: raise KeyError(f'unused patch entries (ledger must close): {sorted(unused)[:8]}')
    params_host = unflatten_dict(iflat, sep='/')
    REPORT['S1_patch'] = {'applied': applied, 'new_components': fb, 'unused': 0}; save_rpt()

params_base = jax.device_put(params_host, base_shard)
pflat = flatten_dict(params_base, sep='/')
mask1 = trainable_mask(pflat, 'new', [])
tr1 = unflatten_dict({k: pflat[k] for k in pflat if mask1[k]}, sep='/')
fr1 = unflatten_dict({k: pflat[k] for k in pflat if not mask1[k]}, sep='/')
tx1 = optax.chain(optax.clip_by_global_norm(1.0), optax.masked(optax.adamw(1e-4, weight_decay=.01), jax.tree.map(lambda v: True, tr1)))
state1 = tx1.init(tr1)

def eval_ce(model, tr, fr, ids, tag):
    @jax.jit
    def ce(a, b, batch):
        full = merge(a, b)
        logits = model.apply({'params': full}, batch[:, :-1])
        return optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), batch[:, 1:]).mean()
    return float(ce(tr, fr, sample(ids, jax.random.PRNGKey(777), BATCH, SEQ)))

ce0 = eval_ce(base_model, tr1, fr1, train_ids, 'S1')
REPORT['stages']['S1_base'] = {'ce0': ce0}
chk('S1', 'finite_ce0', np.isfinite(ce0), ce0)
if RUN == 'real' and ce0 >= math.log(base_cfg.vocab):
    raise RuntimeError('S1 CE(0) >= ln(vocab); staged run stopped (heuristic, not teacher parity)')
tr1, op1, rep1 = run_stage('S1', base_model, tr1, fr1, (tx1, state1), S1_STEPS, train_ids, 'S1_base', collect_gram=False)
if S1_STEPS: save_trainable('S1_base', tr1)
chk('S1', 'completed', rep1['status'] == 'COMPLETED', rep1.get('last', {}).get('ce'))

# ----------------------------- S2: growth -----------------------------
pool = list(range(N_BASE, 93)) if RUN == 'real' else []
plan = build_growth_plan(N_BASE, TARGET, pool if pool else list(range(N_BASE, TARGET)) if TARGET > N_BASE else [N_BASE],
                         strategy='spread')
if RUN == 'smoke':
    for entry in plan:
        if entry['role'] == 'copy': entry['ffn_source'] = None   # no teacher: duplicate base FFN
REPORT['growth_plan'] = plan; save_rpt()
copy_sources = {e['ffn_source'] for e in plan if e['role'] == 'copy' and e['ffn_source'] is not None}
chk('S2', 'distinct_ffn_sources', len(copy_sources) == len([e for e in plan if e['role'] == 'copy']) or RUN == 'smoke', str(sorted(copy_sources)))
chk('S2', 'sources_outside_base', all((s is None) or s >= N_BASE for s in copy_sources))

grown_cfg = StudentConfig(d_model=base_cfg.d_model, num_blocks=TARGET // 4, layers_per_block=base_cfg.layers_per_block,
                          n_layers=TARGET, vocab=base_cfg.vocab, tie_embeddings=base_cfg.tie_embeddings, n_heads=base_cfg.n_heads,
                          d_head=base_cfg.d_head, mla_positions=tuple(p for p in (3, 7, 11, 15) if p < TARGET),
                          q_lora_rank=base_cfg.q_lora_rank, kv_lora_rank=base_cfg.kv_lora_rank,
                          qk_rope_head_dim=base_cfg.qk_rope_head_dim, d_ff=base_cfg.d_ff, dar_latent=base_cfg.dar_latent)
grown_model = StudentLM(cfg=grown_cfg)
grown_abstract = jax.eval_shape(lambda r: grown_model.init(r, ids_small)['params'], jax.random.PRNGKey(0))
gflat = flatten_dict(jax.device_get(grown_abstract), sep='/')
h1 = flatten_dict(jax.device_get(params_base), sep='/')

copy_ffn = {}
if RUN == 'real':
    ffn_layers = sorted({e['ffn_source'] for e in plan if e['role'] == 'copy' and e['ffn_source'] is not None})
    need2 = []
    for t in ffn_layers:
        pre = f'language_model.model.layers.{t}.block_sparse_moe.shared_experts'
        need2 += [f'{pre}.{w}.weight' for w in ('gate_proj', 'up_proj', 'down_proj')]
    missing2 = [n for n in need2 if n not in wm]
    if missing2: raise KeyError(f'Missing copy FFN tensors: {missing2}')
    t2 = fetch_tensors(need2, wm)
    for t in ffn_layers:
        pre = f'language_model.model.layers.{t}.block_sparse_moe.shared_experts'
        copy_ffn[t] = {w: f32(t2[f'{pre}.{w}.weight']).T for w in ('gate_proj', 'up_proj', 'down_proj')}
    del t2; gc.collect()

new_flat = {}
for key, aval in gflat.items():
    shape = tuple(aval.shape); dt = np.dtype(jax.dtypes.canonicalize_dtype(jnp.asarray(aval).dtype))
    parts = key.split('/')
    if key in h1 and parts[0] in [f'block_{b}' for b in range(N_BASE // 4)] or key in h1 and not key.startswith('block_'):
        new_flat[key] = np.asarray(h1[key]); continue
    blk = int(parts[0].split('_')[1])
    entry = next(e for e in plan if e['block'] == blk)
    if entry['role'] == 'base':
        if key in h1: new_flat[key] = np.asarray(h1[key]); continue
        raise KeyError(f'base block leaf missing from S1 params: {key}')
    # role == copy: duplicate mixer_source layer wholesale
    src_layer = entry['mixer_source']
    src_key = f'block_{src_layer // 4}/layer_{src_layer}/' + '/'.join(parts[2:])
    if parts[1] == 'layer':
        if src_key not in h1: raise KeyError(f'mixer_source leaf missing: {src_key}')
        new_flat[key] = np.asarray(h1[src_key])
    elif key == f'block_{blk}/block_gate':
        new_flat[key] = np.zeros(shape, dt)
    elif key == f'block_{blk}/pre_ffn_norm/scale':
        new_flat[key] = np.ones(shape, dt)
    elif parts[1] == 'ffn':
        if entry['ffn_source'] is None:
            new_flat[key] = np.asarray(h1[f'block_{entry["mixer_source"] // 4}/ffn/' + '/'.join(parts[2:])])
        else:
            name = parts[2] if len(parts) == 3 else '/'.join(parts[2:])
            if name not in copy_ffn[entry['ffn_source']]: raise KeyError(f'no teacher FFN tensor for {key}')
            a = np.asarray(copy_ffn[entry['ffn_source']][name])
            if a.T.shape == shape: a = a.T
            if a.shape != shape: raise ValueError(f'{key}: FFN patch {a.shape} != {shape}')
            new_flat[key] = a.astype(dt, copy=False)
    else:
        raise KeyError(f'unexpected grown leaf: {key}')
missing_grown = [k for k in gflat if k not in new_flat]
if missing_grown: raise KeyError(f'grown params incomplete: {missing_grown[:8]}')
params_grown = jax.device_put(unflatten_dict(new_flat, sep='/'), sharded_like(grown_abstract))
REPORT['S2_transplant'] = {'leaves': len(new_flat), 'copies': [e['block'] for e in plan if e['role'] == 'copy']}; save_rpt()

# function-preservation gate: zero gates must reproduce the base function exactly
@jax.jit
def _logits_base(a, b, batch): return base_model.apply({'params': merge(a, b)}, batch)
@jax.jit
def _logits_grown(a, b, batch): return grown_model.apply({'params': merge(a, b)}, batch)
probe = sample(train_ids, jax.random.PRNGKey(1234), BATCH, SEQ)
lg_base = np.asarray(_logits_base(tr1, fr1, probe))
lg_grown = np.asarray(_logits_grown(params_grown, {}, probe))
identical = np.array_equal(lg_base.astype(np.float32), lg_grown.astype(np.float32))
maxdiff = float(np.abs(lg_base.astype(np.float32) - lg_grown.astype(np.float32)).max())
chk('S2', 'function_preservation_zero_gates', identical, f'max_abs_diff={maxdiff}')
_, pooled_probe = grown_model.apply({'params': params_grown}, probe, collect_blocks=True)
_, gram0 = orthogonality_penalty(jax.device_get(pooled_probe), 1.0, 'all')
REPORT['S2_growth'] = {'gram_init_mean_abs_offdiag': float(np.abs(np.asarray(gram0) - np.eye(len(gram0), dtype=bool) * np.asarray(gram0)).sum() / max(len(gram0) * (len(gram0) - 1), 1)),
                       'plan': plan}; save_rpt()

# ----------------------------- S3: warmup with orthogonality -----------------------------
pg = flatten_dict(params_grown, sep='/')
copy_blocks = [e['block'] // base_cfg.layers_per_block for e in plan if e['role'] == 'copy']
mask3 = trainable_mask(pg, 'blocks', copy_blocks)
tr3 = unflatten_dict({k: pg[k] for k in pg if mask3[k]}, sep='/')
fr3 = unflatten_dict({k: pg[k] for k in pg if not mask3[k]}, sep='/')
tx3 = optax.chain(optax.clip_by_global_norm(1.0), optax.masked(optax.adamw(1e-4, weight_decay=.01), jax.tree.map(lambda v: True, tr3)))
state3 = tx3.init(tr3)
tr3, op3, rep3 = run_stage('S3', grown_model, tr3, fr3, (tx3, state3), S3_STEPS, train_ids, 'S3_warmup', collect_gram=True)
if S3_STEPS: save_trainable('S3_warmup', tr3)
# final gate values (block_gate per block)
gates = {k: float(v) for k, v in flatten_dict(jax.device_get(merge(tr3, fr3)), sep='/').items() if k.endswith('block_gate')}
REPORT['S3_warmup']['block_gates'] = gates; save_rpt()

# ----------------------------- S4: optional full training -----------------------------
if S4_STEPS:
    tr4, op4, rep4 = run_stage('S4', grown_model, tr3, fr3, (tx3, op3), S4_STEPS, train_ids, 'S4_full', collect_gram=True)
    if S4_STEPS: save_trainable('S4_full', tr4)
else:
    REPORT['stages']['S4_full'] = {'status': 'SKIPPED', 'steps_planned': 0,
                                   'note': 'full training on the real dataset is a separate authorized run (STAGED_DESIGN.md S4)'}
save_rpt()
REPORT['status'] = 'COMPLETED'; save_rpt()
print(json.dumps({'status': REPORT['status'], 'stages': {k: v.get('status') for k, v in REPORT['stages'].items()}}, indent=1), flush=True)
