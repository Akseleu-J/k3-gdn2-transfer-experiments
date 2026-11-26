"""Staged transfer pipeline: S1 base block -> S2 growth -> S3 warmup -> S4 full.

Revision 2 (2026-11-19, after AUDIT_RESPONSE_2026-11-19): fixes C1-C6 and
M1/M3-M6/M8-M10, and adopts F1+F2:
  - growth copies use LLaMA-Pro-style zero-init output matrices (mixer o_proj,
    FFN down_proj) and DAR alpha_scale=0 instead of scalar gates: the block is
    an exact identity at growth time and its functional tangent space is
    full-rank from step one;
  - the anti-redundancy loss is centered linear CKA between PRE-PROJECTION
    contribution representations of blocks (mixer before o_proj, FFN hidden
    before down_proj), restricted to (copy, source) and (copy, copy) pairs.
The residual-stream orthogonality penalty of revision 1 is deprecated.

Model classes are a staged variant of the reviewed experiment.py (same math
plus pre-projection representation outputs); experiment.py stays untouched.

Env interface (defaults for a bounded mechanics test):
  K3_STAGED_RUN  smoke|real
  K3_S1_STEPS / K3_S3_STEPS / K3_S4_STEPS   optimizer steps per stage (S4 off at 0)
  K3_LAMBDA_ORTH (1e-2), K3_CKA_TOKENS (512)
  K3_N_BASE (8), K3_TARGET (16), K3_SEQ, K3_BATCH (0 = device count)
  K3_S1_TRAINABLE new|new_lm_head, K3_ALLOW_HIGH_CE0 1|0 (preregistered diagnostic bypass)
  K3_ARTIFACTS, K3_CACHE_DIR, K3_ENWIK8_PATH, K3_REVISION, HF_TOKEN, K3_SEED
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
CKA_TOKENS = int(os.environ.get('K3_CKA_TOKENS', 512))
N_BASE = int(os.environ.get('K3_N_BASE', 8))
TARGET = int(os.environ.get('K3_TARGET', 16))
SEQ = int(os.environ.get('K3_SEQ', 64 if RUN == 'smoke' else 128))
BATCH_REQUESTED = int(os.environ.get('K3_BATCH', 0))
SEED = int(os.environ.get('K3_SEED', 0))
S1_TRAINABLE = os.environ.get('K3_S1_TRAINABLE', 'new')  # 'new' or 'new_lm_head' (diagnostic arm A+B)
ALLOW_HIGH_CE0 = os.environ.get('K3_ALLOW_HIGH_CE0', '0') == '1'  # preregistered gate bypass
COPY_SOURCE = os.environ.get('K3_COPY_SOURCE', 'base')   # 'base': duplicate trained base blocks; 'teacher': next teacher layers (bert2BERT AKI)
EVAL_EVERY = int(os.environ.get('K3_EVAL_EVERY', 0))       # held-out CE every N steps (0 = off)
STAGE_MAX_SEC = float(os.environ.get('K3_STAGE_MAX_SEC', 0))   # graceful per-stage wall-clock budget (0 = off)
CKPT_MAX_LEAF_MB = float(os.environ.get('K3_CKPT_MAX_LEAF_MB', 0))   # skip checkpoint leaves above this size (0 = keep all)
ARM_DEADLINE_SEC = float(os.environ.get('K3_ARM_DEADLINE_SEC', 0))    # whole-process soft deadline set by the driver
HBM_DIRTY_FRAC = float(os.environ.get('K3_HBM_DIRTY_FRAC', 0.10))
PROC_T0 = time.monotonic()
assert COPY_SOURCE in ('base', 'teacher'), COPY_SOURCE
ARTIFACTS = os.environ.get('K3_ARTIFACTS', 'artifacts')
os.makedirs(ARTIFACTS, exist_ok=True)
REPORT = {'run': RUN, 'revision': '2-audit-2026-11-19', 'status': 'RUNNING', 'results': [],
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
import mem_guard
from staged_core import cka_penalty, build_growth_plan

HIGHEST = jax.lax.Precision.HIGHEST
devices = jax.devices(); N_DEV = len(devices)
if RUN == 'real':
    if jax.default_backend() != 'tpu':
        raise RuntimeError('Staged real run requires a TPU backend; detected ' + jax.default_backend())
BATCH = N_DEV if BATCH_REQUESTED == 0 else max(BATCH_REQUESTED, 1)
mesh = Mesh(np.array(devices).reshape(-1), ('fsdp',))
batch_sharding = NamedSharding(mesh, P('fsdp', None) if N_DEV > 1 else P())


def fsdp_spec(shape, n_dev=N_DEV):
    if n_dev <= 1: return P()
    c = [(s, i) for i, s in enumerate(shape) if s >= n_dev and s % n_dev == 0]
    if not c: return P()
    _, i = max(c); sp = [None] * len(shape); sp[i] = 'fsdp'; return P(*sp)


def sharded_like(abstract):
    return jax.tree_util.tree_map(lambda a: NamedSharding(mesh, fsdp_spec(a.shape)), abstract)


def reshard_host(tree):
    """Explicit sharded placement for host/device trees (audit run_082324): without
    it, eager fp32 masters and Adam state for lm_head (~14 GB in arm A+B) land on
    one chip and HBM OOMs at tx.init (4.38 G single buffer)."""
    return jax.device_put(tree, jax.tree_util.tree_map(lambda v: NamedSharding(mesh, fsdp_spec(np.shape(v))), tree))


REPORT['env'] = {'python': sys.version, 'jax': jax.__version__, 'backend': jax.default_backend(),
                 'devices': [str(x) for x in devices], 'run': RUN, 'stages': {'S1': S1_STEPS, 'S3': S3_STEPS, 'S4': S4_STEPS},
                 'lambda_orth': LAMBDA_ORTH, 'cka_tokens': CKA_TOKENS, 'n_base': N_BASE, 'target': TARGET,
                 'seq': SEQ, 'batch': BATCH, 'seed': SEED}
REPORT['packages'] = {k: importlib.metadata.version(k) for k in ['jax', 'jaxlib', 'flax', 'optax', 'numpy', 'ml_dtypes', 'requests']}
save_rpt()
HF_TOKEN = os.environ.get('HF_TOKEN')
K3_SHA = None
# Memory guard: refuse to start on an occupied accelerator (the driver resets and retries),
# then sample allocator statistics passively for the whole process.
REPORT['hbm_start'] = mem_guard.hbm_stats(jax.local_devices())
_dirty = mem_guard.hbm_dirty(REPORT['hbm_start'], HBM_DIRTY_FRAC)
_fake = os.environ.get('K3_TEST_DIRTY_ONCE', '') if RUN == 'smoke' else ''   # test hook, smoke only
if _fake and not os.path.exists(_fake):
    Path(_fake).write_text('x'); _dirty = [0]
if _dirty:
    REPORT['status'] = 'HBM_DIRTY'; REPORT['hbm_dirty_devices'] = _dirty; save_rpt()
    print('[mem-guard] accelerator memory occupied at start on devices', _dirty, flush=True)
    sys.exit(mem_guard.HBM_DIRTY_EXIT)
mem_guard.start_hbm_monitor(jax.local_devices(), os.path.join(ARTIFACTS, 'hbm_monitor.jsonl'))

# ----------------------------- model (staged variant, rev 2) -----------------------------
# Math identical to the reviewed experiment.py; deltas: mixers/FFN additionally
# return their PRE-PROJECTION representation (for the CKA penalty), the dead
# o/sqrt(Dh) scale and unused sow diagnostics are removed.
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


def _amax(t):
    return jnp.max(jnp.abs(jnp.asarray(t).astype(jnp.float32)))      # NaN-propagating


DIAG_FIELDS = ('x_in', 'dar', 'normed_in', 'rep', 'delta', 'inner', 'nonfinite_rep', 'nonfinite_delta')


def _diag_row(x, dar, nx, rep, delta, inner):
    """Per-sublayer magnitudes for localizing non-finite values (S2 diagnostics):
    inner = max|S| of the GDN state / max|scores| of MLA; FFN row has dar=inner=0."""
    nf = lambda t: jnp.sum(~jnp.isfinite(jnp.asarray(t).astype(jnp.float32))).astype(jnp.float32)
    return jnp.stack([_amax(x), _amax(dar), _amax(nx), _amax(rep), _amax(delta), jnp.asarray(inner, jnp.float32), nf(rep), nf(delta)])


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
        out_gate = D("out_gate", Pp)(x)
        S0 = jnp.zeros((B, H, Dh, Dh), jnp.float32)
        def step(carry, xs):
            S, smax = carry
            q_t, k_t, v_t, e_t, w_t, g_t = xs
            S = S * jnp.exp(g_t)[..., None, :]
            Sk = jnp.einsum("bhij,bhj->bhi", S, e_t * k_t)
            S = S + jnp.einsum("bhi,bhj->bhij", w_t * v_t - Sk, k_t)
            smax = jnp.maximum(smax, jnp.max(jnp.abs(S)))   # NaN-propagating state magnitude (diagnostics)
            return (S, smax), jnp.einsum("bhij,bhj->bhi", S, q_t)
        xs = jax.tree.map(lambda t: jnp.swapaxes(t.astype(jnp.float32), 1, 0), (q, k, v, erase, write, log_decay))
        (_, smax), o = jax.lax.scan(jax.checkpoint(step), (S0, jnp.zeros((), jnp.float32)), xs); o = jnp.swapaxes(o, 0, 1).reshape(B, L, Pp)
        rep = nn.RMSNorm(epsilon=1e-5, param_dtype=jnp.bfloat16, name="o_norm")(o.reshape(B, L, H, Dh)).reshape(B, L, Pp) * jax.nn.sigmoid(out_gate)
        return nn.Dense(cfg.d_model, use_bias=False, name="o_proj", **BF)(rep), rep, jnp.maximum(smax, _amax(o))


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
        rep = o * jax.nn.sigmoid(dense("g_proj", H * D)(x))
        return dense("o_proj", cfg.d_model)(rep), rep, _amax(scores)


class FFNSublayer(nn.Module):
    cfg: StudentConfig
    @nn.compact
    def __call__(self, x):
        g = nn.Dense(self.cfg.d_ff, use_bias=False, name="gate_proj", **BF)(x); u = nn.Dense(self.cfg.d_ff, use_bias=False, name="up_proj", **BF)(x)
        rep = situ_glu(g, u)
        return nn.Dense(self.cfg.d_model, use_bias=False, name="down_proj", **BF)(rep), rep


class DARLayer(nn.Module):
    """Baseline math (experiment.py): DAR is added inside the layer; returns
    (x, mixer delta, pre-projection mixer rep)."""
    cfg: StudentConfig; layer_idx: int
    @nn.compact
    def __call__(self, x, local_deltas, history, cos, sin):
        src = [history[j] for j in range(history.shape[0])] + list(local_deltas)
        dar = DeltaAttentionResidual(self.cfg.d_model, self.cfg.dar_latent, name="dar")(x, src)
        x1 = x + dar
        nx = nn.RMSNorm(epsilon=1e-6, param_dtype=jnp.bfloat16, name="pre_sublayer_norm")(x1)
        if self.layer_idx in self.cfg.mla_positions:
            d, mrep, inner = MLAMixer(cfg=self.cfg, name="mixer")(nx, cos, sin)
        else:
            d, mrep, inner = GDN2Mixer(cfg=self.cfg, name="mixer")(nx)
        diag = _diag_row(x, dar, nx, mrep, d, inner)
        return x1, d, mrep, diag


class DARBlock(nn.Module):
    cfg: StudentConfig; block_idx: int; layer_idx_start: int
    @nn.compact
    def __call__(self, x, history, cos, sin):
        ld = []; mreps = []; diags = []
        for i in range(self.cfg.layers_per_block):
            idx = self.layer_idx_start + i
            x, d, mrep, dg = DARLayer(cfg=self.cfg, layer_idx=idx, name=f"layer_{idx}")(x, ld, history, cos, sin)
            ld.append(d); mreps.append(mrep); diags.append(dg)
            x = x + d
        nh = jnp.concatenate([history, sum(ld)[None]], 0)
        nx = nn.RMSNorm(epsilon=1e-6, param_dtype=jnp.bfloat16, name="pre_ffn_norm")(x)
        f, frep = FFNSublayer(cfg=self.cfg, name="ffn")(nx)
        diags.append(_diag_row(x, jnp.zeros((), jnp.float32), nx, frep, f, jnp.zeros((), jnp.float32)))
        x = x + f
        mixer_rep = jnp.sum(jnp.stack(mreps, 0), 0)      # F_b^{mix} = sum of layer contributions
        return x, nh, mixer_rep, frep, jnp.stack(diags, 0)


class StudentLM(nn.Module):
    cfg: StudentConfig
    @nn.compact
    def __call__(self, ids, collect_blocks=False):
        B, L = ids.shape
        embed = nn.Embed(self.cfg.vocab, self.cfg.d_model, name="embed", **BF)
        x = embed(ids); cos, sin = RoPE(self.cfg.qk_rope_head_dim, self.cfg.rope_theta)(L)
        hist = jnp.zeros((0, B, L, self.cfg.d_model), dtype=x.dtype); RB = nn.remat(DARBlock); mrs = []; frs = []; dgs = []
        for bi in range(self.cfg.num_blocks):
            x, hist, mr, fr, dg = RB(cfg=self.cfg, block_idx=bi, layer_idx_start=bi * self.cfg.layers_per_block, name=f"block_{bi}")(x, hist, cos, sin)
            mrs.append(mr); frs.append(fr); dgs.append(dg)
        x = nn.RMSNorm(epsilon=1e-6, param_dtype=jnp.bfloat16, name="final_norm")(x)
        logits = embed.attend(x) if self.cfg.tie_embeddings else nn.Dense(self.cfg.vocab, use_bias=False, name="lm_head", **BF)(x)
        if not collect_blocks:
            return logits
        return logits, {'mixer': jnp.stack(mrs, 0), 'ffn': jnp.stack(frs, 0), 'diag': jnp.stack(dgs, 0),
                        'final_absmax': _amax(x)}


# ----------------------------- report/optimizer helpers -----------------------------
import optax
from flax.traverse_util import flatten_dict, unflatten_dict
import jax.tree_util as jtu

NEW_PAT = re.compile(r'/dar/|/erase_gate/|/write_gate/|/dt_bias$|/pre_ffn_norm/scale$')


def trainable_mask(flat, policy, copy_blocks, extra_prefixes=()):
    out = {}
    for k in flat:
        new = bool(NEW_PAT.search(k))
        in_copy = any(k.startswith(f'block_{b}/') for b in copy_blocks)
        out[k] = new or (policy == 'all') or (policy == 'blocks' and in_copy)             or any(k.startswith(p) for p in extra_prefixes)
    return out


def merge(a, b):
    fa, fb = flatten_dict(a, sep='/'), flatten_dict(b, sep='/')
    return unflatten_dict({**fb, **fa}, sep='/')


def to_master(tr):
    """fp32 master copy of the trainable subtree: a bf16 param cannot absorb an
    Adam step of ~lr=1e-4 once |w| > ~0.03 (bf16 ulp), and norm scales (~1,
    ulp ~8e-3) would never move. Flax Dense/RMSNorm cast to the compute dtype."""
    return jax.tree.map(lambda v: jnp.asarray(v, jnp.float32) if jnp.issubdtype(jnp.asarray(v).dtype, jnp.floating) else v, tr)


def make_tx(tr):
    """clip + AdamW with decoupled weight decay only on >=2D matrices (audit M6):
    gates, norm scales, DAR scalars and biases are never decayed."""
    mask2d = jax.tree.map(lambda v: bool(np.ndim(v) >= 2), tr)
    return optax.chain(optax.clip_by_global_norm(1.0),
                       optax.adamw(1e-4, weight_decay=0.01, mask=mask2d))


def sample(ids, key, bs, seq):
    starts = jax.random.randint(key, (bs,), 0, len(ids) - seq - 1)
    return np.stack([np.asarray(ids[s:s + seq + 1]) for s in np.asarray(starts)])


def data_key(stage_tag, step):
    # stage-folded keys: S1 and S3 never see the same batch sequence (audit M4)
    return jax.random.fold_in(jax.random.PRNGKey(SEED), stage_tag * 10 ** 6 + step)


def run_stage(tag, model, tr, fr, tx, op, steps, ids, report_key, stage_tag, collect_gram, extra=None, eval_fn=None):
    """Shared training loop; returns (tr, final_op, report_dict). Optional periodic
    held-out CE (eval_fn, every EVAL_EVERY steps) and a graceful wall-clock budget."""
    rep = {'steps_planned': steps}; t0 = time.monotonic(); last = {}; stopped = False
    mpath = Path(ARTIFACTS, f'{report_key}_metrics.jsonl')
    if extra: rep.update(extra)
    if steps:
        @jax.jit
        def train_step(a, op, b, batch, step):
            def loss_of(params):
                full = merge(params, b)
                logits, reps = model.apply({'params': full}, batch[:, :-1], collect_blocks=True)
                ce = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), batch[:, 1:]).mean()
                if not collect_gram:
                    return ce, (ce, None)
                pen, gram = cka_penalty(reps['mixer'], LAMBDA_ORTH, PAIRS, CKA_TOKENS, stage_tag * 10 ** 6 + step)
                pen_f, gram_f = cka_penalty(reps['ffn'], LAMBDA_ORTH, PAIRS, CKA_TOKENS, stage_tag * 10 ** 6 + step)
                return ce + pen + pen_f, (ce, (gram, gram_f))
            (loss, (ce, gram)), grads = jax.value_and_grad(loss_of, has_aux=True)(a)
            gn = optax.global_norm(grads); ok = jnp.isfinite(loss) & jnp.isfinite(gn)
            updates, nop = tx.update(grads, op, a)
            na = optax.apply_updates(a, updates)                      # audit C3: apply, not raw deltas
            ntr = jax.tree.map(lambda new, old: jnp.where(ok, new, old), na, a)
            na = jax.tree.map(lambda new, old: jnp.where(ok, new, old), nop, op)
            return ntr, na, loss, ce, gram, gn, ok
        for step in range(1, steps + 1):
            batch = jax.device_put(sample(ids, data_key(stage_tag, step), BATCH, SEQ), batch_sharding)   # audit M1
            tr, op, lv, ce, gram, gn, ok = train_step(tr, op, fr, batch, step)
            row = {'step': step, 'loss': float(lv), 'ce': float(ce), 'grad_norm': float(gn), 'accepted': bool(ok)}
            if gram is not None:
                for name, g in (('mixer', gram[0]), ('ffn', gram[1])):
                    gm = np.asarray(g); n = gm.shape[0]
                    off = gm[~np.eye(n, dtype=bool)]
                    row[f'{name}_cka_mean'] = float(off.mean()); row[f'{name}_cka_max'] = float(off.max())
                    if step % 100 == 1 or step == steps: row[f'{name}_gram'] = gm.round(4).tolist()
            if eval_fn is not None and EVAL_EVERY and (step % EVAL_EVERY == 0 or step == steps):
                row['ce_val'] = eval_fn(tr)
            slim = {k: v for k, v in row.items() if not k.endswith('_gram')}
            rep.setdefault('metrics', []).append(slim)
            with mpath.open('a', encoding='utf-8') as mf: mf.write(json.dumps(row) + chr(10))
            REPORT['last_step'] = slim
            if step % 10 == 0 or step == steps or 'ce_val' in row or not row['accepted']: save_rpt()
            print(f'[{tag}]', {k: (round(v, 6) if isinstance(v, float) else v) for k, v in slim.items()}, flush=True)
            if not row['accepted']:
                rep['status'] = 'FAILED_NONFINITE'; REPORT['stages'][report_key] = rep; save_rpt()
                raise FloatingPointError(f'{tag}: nonfinite training step; update skipped')
            last = row
            over_stage = STAGE_MAX_SEC and time.monotonic() - t0 > STAGE_MAX_SEC
            over_arm = ARM_DEADLINE_SEC and time.monotonic() - PROC_T0 > ARM_DEADLINE_SEC
            if (over_stage or over_arm) and step < steps:
                stopped = True; rep['stopped_at_step'] = step
                if eval_fn is not None and 'ce_val' not in row: last['ce_val'] = eval_fn(tr)
                rep['stop_reason'] = 'arm_deadline' if over_arm else 'stage_budget'
                print(f"[{tag}] wall-clock {rep['stop_reason']} reached at step {step}; stopping stage gracefully", flush=True)
                break
    rep['status'] = 'STOPPED_TIME_BUDGET' if stopped else 'COMPLETED'; rep['elapsed_seconds'] = time.monotonic() - t0
    rep['hbm'] = mem_guard.hbm_stats(jax.local_devices())
    if len(rep.get('metrics', [])) > 2000:   # keep report.json bounded; full trace is in *_metrics.jsonl
        m = rep['metrics']; rep['metrics'] = m[:500] + m[-1500:]; rep['metrics_truncated'] = {'kept': 2000, 'total': len(m), 'full': mpath.name}
    if last: rep['last'] = last
    REPORT['stages'][report_key] = rep; save_rpt()
    return tr, op, rep


def save_trainable(tag, tr):
    flat = {k: np.asarray(v) for k, v in flatten_dict(jax.device_get(tr), sep='/').items()}
    skipped = []
    if CKPT_MAX_LEAF_MB:
        skipped = sorted(k for k, v in flat.items() if v.nbytes > CKPT_MAX_LEAF_MB * 2 ** 20)
        flat = {k: v for k, v in flat.items() if k not in skipped}
    path = Path(ARTIFACTS, f'{tag}_trainable.npz'); temp = path.with_suffix('.tmp')
    with open(temp, 'wb') as f: np.savez(f, **flat)
    os.replace(temp, path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    REPORT['stages'][tag]['checkpoint'] = {'file': path.name, 'sha256': sha, 'kind': 'trainable-only', 'exact_resume': False,
                                           'skipped_large_leaves': skipped}
    save_rpt(); print(f'[checkpoint] {path.name} sha256={sha[:16]}', flush=True)


def s2_diag_report(db, dg, copy_blocks, base_logits, grown_logits, grown_logits_fp32, final_absmax):
    """Readable S2 localization: rows = sublayers (layers + 'ffn') per block."""
    def rows(d):
        out = []
        for b in range(d.shape[0]):
            for r in range(d.shape[1]):
                out.append(dict(block=b, sub=('ffn' if r == d.shape[1] - 1 else f'layer_{b * (d.shape[1] - 1) + r}'),
                                **{f: (float(d[b, r, i]) if np.isfinite(d[b, r, i]) else str(d[b, r, i])) for i, f in enumerate(DIAG_FIELDS)}))
        return out
    def first_bad(d):
        for b in range(d.shape[0]):
            for r in range(d.shape[1]):
                for i, f in enumerate(DIAG_FIELDS):
                    v = d[b, r, i]
                    if not np.isfinite(v) or (f.startswith('nonfinite') and v > 0):
                        return dict(block=b, row=r, field=f, value=str(v))
        return None
    nf = lambda a: int(np.sum(~np.isfinite(a)))
    summary = dict(base_first_nonfinite=first_bad(db), grown_first_nonfinite=first_bad(dg),
                   nonfinite_logits=dict(base=nf(base_logits), grown_bf16=nf(grown_logits), grown_fp32=nf(grown_logits_fp32)),
                   fp32_grown_vs_base_maxdiff=float(np.nanmax(np.abs(grown_logits_fp32 - base_logits))) if nf(grown_logits_fp32) == 0 else None,
                   final_absmax=dict(base=final_absmax[0], grown=final_absmax[1]),
                   max_inner_by_block_grown=[float(np.nanmax(dg[b, :, DIAG_FIELDS.index('inner')])) if np.isfinite(dg[b, :, DIAG_FIELDS.index('inner')]).all() else 'nonfinite' for b in range(dg.shape[0])],
                   copies=list(copy_blocks))
    return dict(summary=summary, fields=list(DIAG_FIELDS), base=rows(db), grown=rows(dg))


def transplant_params(h1, gflat, plan, n_base, layers_per_block, copy_ffn):
    """Block-wise transplant (audit C5 fix). Base blocks copy the S1 state;
    copies duplicate their mixer_source block wholesale and zero-init the
    output matrices (mixer o_proj, FFN down_proj) and DAR alpha_scale — an
    exact identity with a full-rank functional tangent (LLaMA Pro style)."""
    nbb = n_base // layers_per_block
    out = {}; block_first = {}
    for e in plan:
        if e['block'] % layers_per_block == 0: block_first[e['block'] // layers_per_block] = e
    for key, aval in gflat.items():
        parts = key.split('/')
        if not parts[0].startswith('block_'):           # embed / final_norm / lm_head: S1 state
            if key not in h1: raise KeyError(f'non-block leaf missing from S1 params: {key}')
            out[key] = np.asarray(h1[key]); continue
        blk = int(parts[0].split('_')[1])
        shape = tuple(aval.shape); dt = np.dtype(jax.dtypes.canonicalize_dtype(aval.dtype))
        e = block_first[blk]
        if e['role'] == 'base':
            if key not in h1: raise KeyError(f'base block leaf missing from S1 params: {key}')
            out[key] = np.asarray(h1[key]); continue
        if parts[1].startswith('layer_'):
            li = int(parts[1].split('_')[1])
            src_block = e['mixer_source'] // layers_per_block
            src_key = f'block_{src_block}/layer_{src_block * layers_per_block + li % layers_per_block}/' + '/'.join(parts[2:])
            if src_key not in h1: raise KeyError(f'mixer_source leaf missing: {src_key}')
            out[key] = np.asarray(h1[src_key])
            if key.endswith('/mixer/o_proj/kernel') or key.endswith('/dar/alpha_scale'):
                out[key] = np.zeros(shape, dt)
            continue
        if key == f'block_{blk}/pre_ffn_norm/scale':
            out[key] = np.ones(shape, dt); continue
        if parts[1] != 'ffn': raise KeyError(f'unexpected grown leaf: {key}')
        name = parts[2]
        if name == 'down_proj':
            out[key] = np.zeros(shape, dt); continue
        if e['ffn_source'] is None:                    # smoke: duplicate source FFN (pure stacking ablation)
            src_key = f'block_{e["mixer_source"] // layers_per_block}/ffn/' + '/'.join(parts[2:])
            out[key] = np.asarray(h1[src_key]); continue
        if name not in copy_ffn.get(e['ffn_source'], {}): raise KeyError(f'no teacher FFN tensor for {key}')
        a = np.asarray(copy_ffn[e['ffn_source']][name])
        if a.T.shape == shape: a = a.T
        if a.shape != shape: raise ValueError(f'{key}: FFN patch {a.shape} != {shape}')
        out[key] = a.astype(dt, copy=False)
    missing = [k for k in gflat if k not in out]
    if missing: raise KeyError(f'grown params incomplete: {missing[:8]}')
    return out


# ----------------------------- enwik8 loader (copied from reviewed experiment.py) -----------------------------
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


def run_pipeline():
    global PAIRS
    REPORT['stages'] = {}

    if RUN == 'smoke':
        base_cfg = StudentConfig(d_model=16, num_blocks=N_BASE // 4, n_layers=N_BASE, vocab=64, n_heads=1, d_head=8,
                                 q_lora_rank=16, kv_lora_rank=8, d_ff=32, mla_positions=tuple(4 * b + 3 for b in range(N_BASE // 4)), dar_latent=8)
        train_ids = np.tile(np.arange(64, dtype=np.int32), 800)
        val_ids = np.tile((np.arange(64) + 17) % 64, 80)          # disjoint phase; audit M3
        REPORT['data'] = {'synthetic': True, 'note': 'smoke mechanics only (micro sizes, audit 1.4.1)'}
    else:
        base_cfg = StudentConfig(num_blocks=N_BASE // 4, n_layers=N_BASE, mla_positions=tuple(4 * b + 3 for b in range(N_BASE // 4)))
        from transformers import AutoTokenizer
        from huggingface_hub import HfApi
        K3_SHA = HfApi(token=HF_TOKEN).model_info('moonshotai/Kimi-K3', revision=os.environ.get('K3_REVISION', 'main')).sha
        REPORT['k3_sha'] = K3_SHA
        tok = AutoTokenizer.from_pretrained('moonshotai/Kimi-K3', revision=K3_SHA, token=HF_TOKEN, trust_remote_code=True)
        raw = load_enwik8(Path(os.environ.get('K3_CACHE_DIR', '/content/k3_cache')) / 'data',
                          local_path=os.environ.get('K3_ENWIK8_PATH', ''), report=REPORT, save_report=save_rpt)
        text = raw[:2_000_000].decode('utf-8', errors='replace')             # train slice (baseline convention)
        val_text = raw[95_000_000:95_200_000].decode('utf-8', errors='replace')   # disjoint held-out; audit M3
        train_ids = np.asarray(tok.encode(text, add_special_tokens=False), np.int32)
        val_ids = np.asarray(tok.encode(val_text, add_special_tokens=False), np.int32)
        REPORT['data'] = {'train_tokens': len(train_ids), 'val_tokens': len(val_ids), 'disjoint_slices': True, 'full_enwik8': False}
        save_rpt()

    base_model = StudentLM(cfg=base_cfg)
    ids_small = np.zeros((1, 8), np.int32)
    base_abstract = jax.eval_shape(lambda r: base_model.init(r, ids_small)['params'], jax.random.PRNGKey(0))
    rng_init = jax.random.PRNGKey(0)
    wm = None
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
        # Geometry drift gate (audit C1 fix: defined after hf_url/http_get).
        source_config = json.loads(http_get(hf_url('config.json')))
        Path(ARTIFACTS, 'k3_config.json').write_text(json.dumps(source_config, indent=2), encoding='utf-8')
        tc = source_config['text_config']
        expected = {'hidden_size': 7168, 'num_attention_heads': 96, 'qk_nope_head_dim': 128,
                    'qk_rope_head_dim': 64, 'v_head_dim': 128, 'tie_word_embeddings': False, 'mla_use_output_gate': True}
        for k, v in expected.items(): chk('CONFIG', k, tc.get(k) == v, str(tc.get(k)))
        chk('CONFIG', 'safe_gate', tc['linear_attn_config'].get('gate_lower_bound') == -5.0)
        actual_mla = tuple(i - 1 for i in tc['linear_attn_config']['full_attn_layers'] if i <= TARGET)
        chk('CONFIG', 'layer_mapping', actual_mla == tuple(p for p in (3, 7, 11, 15) if p < TARGET), actual_mla)
        REPORT['k3_geometry'] = {'hidden_size': tc['hidden_size'], 'mla_positions': actual_mla}; save_rpt()
        NP_DTYPES = {'BF16': ml_dtypes.bfloat16, 'F16': np.float16, 'F32': np.float32, 'U8': np.uint8, 'I8': np.int8, 'I32': np.int32, 'I64': np.int64,
                     'F8_E4M3': ml_dtypes.float8_e4m3fn, 'F8_E5M2': ml_dtypes.float8_e5m2, 'F8_E8M0': getattr(ml_dtypes, 'float8_e8m0fnu', None)}
        def load_index(): return json.loads(http_get(hf_url('model.safetensors.index.json')))['weight_map']
        _hc = {}
        def read_shard_header(shard):
            if shard in _hc: return _hc[shard]
            url = hf_url(shard); (n,) = struct.unpack('<Q', http_get(url, (0, 7)))
            hd = json.loads(http_get(url, (8, 7 + n))); hd.pop('__metadata__', None); _hc[shard] = (hd, 8 + n); return _hc[shard]
        def cached_tensor_bytes(p):
            key = hashlib.sha256(json.dumps({k: p[k] for k in ['name', 'url', 'start', 'end', 'dtype', 'shape']}, sort_keys=True).encode()).hexdigest()
            root = Path(os.environ.get('K3_CACHE_DIR', '/content/k3_cache')) / 'tensor_ranges' / REV
            root.mkdir(parents=True, exist_ok=True)
            data_path = root / (key + '.bin'); meta_path = root / (key + '.json'); expected_sz = p['end'] - p['start'] + 1
            if data_path.is_file() and meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text()); buf = data_path.read_bytes()
                    if len(buf) == expected_sz and hashlib.sha256(buf).hexdigest() == meta['sha256']: return buf
                except (OSError, ValueError, KeyError): pass
            buf = http_get(p['url'], (p['start'], p['end']))
            if len(buf) != expected_sz: raise ValueError('Tensor byte-range length mismatch')
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
        max_teacher_layer = max(int(m.split('.')[3]) for m in wm if m.startswith('language_model.model.layers.'))
        REPORT['teacher_depth'] = max_teacher_layer + 1
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
        REPORT['fetch_S1'] = {'tensors': len(need)}; save_rpt()
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
        init_host = jax.device_get(base_model.init(rng_init, ids_small)['params'])
        iflat = flatten_dict(init_host, sep='/'); unused = set(patch); applied = fb = 0
        for key, val in iflat.items():
            dt = np.dtype(jax.dtypes.canonicalize_dtype(jnp.asarray(val).dtype)); shape = tuple(np.asarray(val).shape)
            if key in patch:
                a = np.asarray(patch[key])
                if a.shape != shape:
                    if a.T.shape == shape: a = a.T
                    elif a.ndim == 1 and len(shape) == 1 and shape[0] % a.shape[0] == 0: a = np.tile(a, shape[0] // a.shape[0])
                    else: raise ValueError(f'{key}: patch {a.shape} != model {shape}')
                iflat[key] = a.astype(dt, copy=False); applied += 1; unused.discard(key)
            elif key.endswith('pre_ffn_norm/scale'):
                iflat[key] = np.ones(shape, dt); fb += 1
            elif '/dar/' in key:
                iflat[key] = (np.ones(shape, dt) if key.endswith('/scale') else np.zeros(shape, dt)); fb += 1
            elif key.endswith('_conv_b'):
                iflat[key] = np.zeros(shape, dt); fb += 1
            elif '/erase_gate/' in key or '/write_gate/' in key or key.endswith('/dt_bias'):
                iflat[key] = np.zeros(shape, dt); fb += 1
            else:
                raise KeyError(f'unexpected unpatched teacher leaf: {key}')
        if unused: raise KeyError(f'unused patch entries (ledger must close): {sorted(unused)[:8]}')
        params_host = unflatten_dict(iflat, sep='/')
        REPORT['S1_patch'] = {'applied': applied, 'new_components': fb, 'unused': 0}; save_rpt()

    params_base = jax.device_put(params_host, sharded_like(base_abstract))
    pflat = flatten_dict(params_base, sep='/')
    mask1 = trainable_mask(pflat, 'new', [], extra_prefixes=('lm_head/',) if S1_TRAINABLE == 'new_lm_head' else ())
    tr1 = reshard_host(to_master(unflatten_dict({k: pflat[k] for k in pflat if mask1[k]}, sep='/')))
    fr1 = reshard_host(unflatten_dict({k: pflat[k] for k in pflat if not mask1[k]}, sep='/'))
    del params_base, pflat, params_host
    gc.collect()
    tx1 = make_tx(tr1); op1 = tx1.init(tr1)

    def eval_ce(model, tr, fr, ids, n_batches=2):
        @jax.jit
        def ce(a, b, batch):
            full = merge(a, b)
            logits = model.apply({'params': full}, batch[:, :-1])
            return optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), batch[:, 1:]).mean()
        vals = [float(ce(tr, fr, jax.device_put(sample(ids, data_key(99, i), BATCH, SEQ), batch_sharding))) for i in range(n_batches)]
        return float(np.mean(vals))

    ce0 = eval_ce(base_model, tr1, fr1, val_ids)
    counts = np.bincount(train_ids, minlength=base_cfg.vocab).astype(np.float64) + 1.0
    uni_ce = float(np.mean(-np.log((counts[train_ids[:200000]] / counts.sum()))))
    threshold = min(math.log(base_cfg.vocab), uni_ce)
    REPORT['S1_reference'] = {'ce0_val': ce0, 'unigram_ce_train': uni_ce, 'stop_threshold': threshold,
                              's1_trainable': S1_TRAINABLE, 'gate_bypassed': False}; save_rpt()
    chk('S1', 'finite_ce0', np.isfinite(ce0), ce0)
    if RUN == 'real' and ce0 >= threshold:
        if not ALLOW_HIGH_CE0:
            raise RuntimeError(f'S1 CE(0) {ce0:.3f} >= threshold {threshold:.3f} (ln vocab or unigram); staged run stopped (heuristic)')
        REPORT['S1_reference']['gate_bypassed'] = True
        REPORT['S1_reference']['gate_bypass_note'] = 'preregistered diagnostic arm A+B (AUDIT_RUNS_2026-11-21); no quality claim from this run'
        save_rpt(); print('[gate] CE(0) stop bypassed by preregistered diagnostic flag; no quality claim', flush=True)
    tr1, op1, rep1 = run_stage('S1', base_model, tr1, fr1, tx1, op1, S1_STEPS, train_ids, 'S1_base', 1, collect_gram=False,
                               extra={'ce0_val': ce0}, eval_fn=lambda t: eval_ce(base_model, t, fr1, val_ids))
    if S1_STEPS: save_trainable('S1_base', tr1)
    chk('S1', 'completed', rep1['status'] in ('COMPLETED', 'STOPPED_TIME_BUDGET'), f"{rep1['status']} ce={rep1.get('last', {}).get('ce')}")
    REPORT['stages']['S1_base']['ce_val_end'] = eval_ce(base_model, tr1, fr1, val_ids); save_rpt()

    if TARGET == N_BASE:   # compute-matched control arm: no growth, S1 only
        for k in ('S3_warmup', 'S4_full'):
            REPORT['stages'][k] = {'status': 'SKIPPED_CONTROL', 'note': 'K3_TARGET == K3_N_BASE: no-growth control arm'}
        REPORT['status'] = 'COMPLETED'; save_rpt()
        print(json.dumps({'status': 'COMPLETED', 'control_no_growth': True, 'ce_val_end': REPORT['stages']['S1_base']['ce_val_end']}), flush=True)
        return

    # ----------------------------- S2: growth -----------------------------
    if RUN == 'real':
        ffn_key_pat = 'block_sparse_moe.shared_experts.gate_proj.weight'
        candidates = [t for t in range(3, max_teacher_layer - 1) if all(
            f'language_model.model.layers.{t}.block_sparse_moe.shared_experts.{w}.weight' in wm for w in ('gate_proj', 'up_proj', 'down_proj'))]
        plan = build_growth_plan(N_BASE, TARGET, candidates, strategy='depth_proportional')
        for entry in plan:
            if entry['role'] == 'base': entry['ffn_source'] = BASE_FFN_LAYER   # report what S1 actually loaded
    else:
        plan = build_growth_plan(N_BASE, TARGET, list(range(N_BASE, TARGET)), strategy='spread')
        for entry in plan:
            if entry['role'] == 'copy': entry['ffn_source'] = None   # no teacher: duplicate base FFN
    REPORT['growth_plan'] = plan; save_rpt()
    copy_pairs_src = [(e['block'] // 4, e['mixer_source'] // 4) for e in plan if e['role'] == 'copy' and e['block'] % 4 == 0]
    copy_block_ids = sorted({b for b, _ in copy_pairs_src})
    pair_set = set()
    for c, s in copy_pairs_src:
        pair_set.add((c, s))
    if len(copy_block_ids) > 1: pair_set.add(tuple(copy_block_ids[:2]))
    PAIRS = sorted(pair_set)
    REPORT['penalty_pairs'] = PAIRS; save_rpt()

    grown_cfg = StudentConfig(d_model=base_cfg.d_model, num_blocks=TARGET // 4, layers_per_block=base_cfg.layers_per_block,
                              n_layers=TARGET, vocab=base_cfg.vocab, tie_embeddings=base_cfg.tie_embeddings, n_heads=base_cfg.n_heads,
                              d_head=base_cfg.d_head, mla_positions=tuple(4 * b + 3 for b in range(TARGET // 4)),
                              q_lora_rank=base_cfg.q_lora_rank, kv_lora_rank=base_cfg.kv_lora_rank,
                              qk_rope_head_dim=base_cfg.qk_rope_head_dim, d_ff=base_cfg.d_ff, dar_latent=base_cfg.dar_latent)
    grown_model = StudentLM(cfg=grown_cfg)
    grown_abstract = jax.eval_shape(lambda r: grown_model.init(r, ids_small)['params'], jax.random.PRNGKey(0))
    gflat = flatten_dict(jax.device_get(grown_abstract), sep='/')
    h1 = flatten_dict(jax.device_get(merge(tr1, fr1)), sep='/')   # S1-TRAINED state (not the pre-S1 init)

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

    new_flat = transplant_params(h1, gflat, plan, N_BASE, base_cfg.layers_per_block, copy_ffn)
    REPORT['copy_source'] = COPY_SOURCE
    if COPY_SOURCE == 'teacher' and RUN == 'real':
        # bert2BERT-AKI arm: copy blocks take the mixers of the NEXT teacher layers
        # (N_BASE..TARGET-1) instead of duplicating base blocks; DAR/norm scales still
        # come from the S1 source block, outputs (o_proj, down_proj, DAR scale) stay 0.
        aki_layers = list(range(N_BASE, TARGET))
        need3 = [f'language_model.model.layers.{L}.{s}' for L in aki_layers for s in (MLA_SUF if L in grown_cfg.mla_positions else KDA_SUF)]
        miss3 = [n for n in need3 if n not in wm]
        if miss3: raise KeyError(f'Missing AKI teacher tensors: {miss3[:4]}')
        tensors = fetch_tensors(need3, wm)
        replaced = 0
        for L in aki_layers:
            for k, v in (_mla if L in grown_cfg.mla_positions else _kda)(L, H_s, Dh).items():
                key = f'block_{L // base_cfg.layers_per_block}/layer_{L}/{k}'
                if key.endswith('/mixer/o_proj/kernel'): continue          # stays exactly zero (identity at growth)
                if key not in new_flat: raise KeyError(f'AKI leaf not in grown tree: {key}')
                tgt = new_flat[key]; a = np.asarray(v)
                if a.shape != tgt.shape:
                    if a.T.shape == tgt.shape: a = a.T
                    elif a.ndim == 1 and tgt.ndim == 1 and tgt.shape[0] % a.shape[0] == 0: a = np.tile(a, tgt.shape[0] // a.shape[0])
                    else: raise ValueError(f'{key}: AKI patch {a.shape} != {tgt.shape}')
                new_flat[key] = a.astype(tgt.dtype, copy=False); replaced += 1
        del tensors; gc.collect()
        REPORT['S2_aki'] = {'teacher_layers': aki_layers, 'replaced_leaves': replaced}; save_rpt()
    elif COPY_SOURCE == 'teacher':
        REPORT['S2_aki'] = {'note': 'smoke: no teacher; COPY_SOURCE=teacher falls back to base duplication'}
    params_grown = jax.device_put(unflatten_dict(new_flat, sep='/'), sharded_like(grown_abstract))
    REPORT['S2_transplant'] = {'leaves': len(new_flat), 'copies': copy_block_ids,
                               'zero_init': ['mixer/o_proj', 'ffn/down_proj', 'dar/alpha_scale (copies)']}; save_rpt()

    # function-preservation gate (audit M7): tolerance + KL; bitwise is informational
    @jax.jit
    def _logits_base(a, b, batch): return base_model.apply({'params': merge(a, b)}, batch)
    @jax.jit
    def _logits_grown(a, b, batch): return grown_model.apply({'params': merge(a, b)}, batch)
    probe = jax.device_put(sample(val_ids, data_key(2, 0), BATCH, SEQ), batch_sharding)
    lb = _logits_base(tr1, fr1, probe).astype(jnp.float32)
    lg = _logits_grown(params_grown, {}, probe).astype(jnp.float32)
    maxdiff = float(np.abs(np.asarray(lb) - np.asarray(lg)).max())
    pb = jax.nn.log_softmax(np.asarray(lb), axis=-1); pg = jax.nn.log_softmax(np.asarray(lg), axis=-1)
    kl = float(np.mean(np.sum(np.exp(pb) * (pb - pg), axis=-1)))
    bitwise = bool(np.array_equal(np.asarray(lb), np.asarray(lg)))
    # Exactness is structural: every copy output path is multiplied by an exactly-zero
    # leaf, so the grown function equals the base function in exact arithmetic (eager
    # base-vs-grown diff is 0.0 in local diagnostics). Two separately compiled bf16
    # programs differ by a few bf16 ulps of the logits (local: same model eager vs jit
    # 4.1e-2 > base-vs-grown 1.6e-2), so the numeric gate is ulp-relative.
    zero_leaves = [k for k in new_flat if any(k.startswith(f'block_{b}/') for b in copy_block_ids)
                   and (k.endswith('/mixer/o_proj/kernel') or k.endswith('/ffn/down_proj/kernel') or k.endswith('/dar/alpha_scale'))]
    chk('S2', 'copy_outputs_exactly_zero', len(zero_leaves) > 0 and all(not np.any(np.asarray(new_flat[k])) for k in zero_leaves),
        f'{len(zero_leaves)} zero-init leaves')
    # Localization diagnostics (run 2026-11-26 failed here with NaN grown logits): per
    # block/sublayer magnitudes for base and grown, first non-finite site, and a fp32
    # re-run of the grown forward (separates bf16 overflow from genuine instability).
    @jax.jit
    def _diag_base(a, b, batch): return base_model.apply({'params': merge(a, b)}, batch, collect_blocks=True)[1]
    @jax.jit
    def _diag_grown(a, b, batch): return grown_model.apply({'params': merge(a, b)}, batch, collect_blocks=True)[1]
    db = jax.device_get(_diag_base(tr1, fr1, probe)); dg = jax.device_get(_diag_grown(params_grown, {}, probe))
    g32 = jax.tree.map(lambda v: v.astype(jnp.float32) if jnp.issubdtype(v.dtype, jnp.floating) else v, params_grown)
    lg32 = np.asarray(_logits_grown(g32, {}, probe).astype(jnp.float32)); del g32
    REPORT['S2_diag'] = s2_diag_report(np.asarray(db['diag']), np.asarray(dg['diag']), copy_block_ids,
                                       base_logits=np.asarray(lb), grown_logits=np.asarray(lg), grown_logits_fp32=lg32,
                                       final_absmax=(float(db['final_absmax']), float(dg['final_absmax'])))
    save_rpt(); print('[S2 diag]', json.dumps(REPORT['S2_diag']['summary']), flush=True)
    lmax = float(np.nanmax(np.abs(np.asarray(lb)))) if np.isfinite(np.asarray(lb)).any() else 1.0
    tol = 4.0 * 2.0 ** (math.floor(math.log2(max(lmax, 1e-30))) - 7)   # 4 bf16 ulps at max|logit|
    chk('S2', 'function_preservation_numeric', maxdiff <= tol and kl <= 1e-4,
        f'max_abs_diff={maxdiff:.2e} tol={tol:.2e} kl={kl:.2e} bitwise={bitwise}')
    REPORT['S2_growth'] = {'max_abs_diff': maxdiff, 'tol_4ulp_bf16': tol, 'kl': kl, 'bitwise': bitwise, 'penalty_pairs': PAIRS}; save_rpt()

    # ----------------------------- S3: warmup with CKA penalty on contributions -----------------------------
    pg = flatten_dict(params_grown, sep='/')
    mask3 = trainable_mask(pg, 'blocks', copy_block_ids)
    tr3 = reshard_host(to_master(unflatten_dict({k: pg[k] for k in pg if mask3[k]}, sep='/')))
    fr3 = reshard_host(unflatten_dict({k: pg[k] for k in pg if not mask3[k]}, sep='/'))
    del params_grown, pg, new_flat
    gc.collect()
    tx3 = make_tx(tr3); op3 = tx3.init(tr3)
    tr3, op3, rep3 = run_stage('S3', grown_model, tr3, fr3, tx3, op3, S3_STEPS, train_ids, 'S3_warmup', 3, collect_gram=True,
                               eval_fn=lambda t: eval_ce(grown_model, t, fr3, val_ids))
    if S3_STEPS: save_trainable('S3_warmup', tr3)
    zg = flatten_dict(jax.device_get(merge(tr3, fr3)), sep='/')
    zero_norms = {k: float(np.linalg.norm(np.asarray(v))) for k, v in zg.items()
                  if any(k.startswith(f'block_{b}/') and (k.endswith('/mixer/o_proj/kernel') or k.endswith('/ffn/down_proj/kernel')) for b in copy_block_ids)}
    REPORT['stages']['S3_warmup']['copy_output_norms'] = zero_norms   # audit C6 fix
    REPORT['stages']['S3_warmup']['ce_val_end'] = eval_ce(grown_model, tr3, fr3, val_ids); save_rpt()

    # ----------------------------- S4: optional full training -----------------------------
    if S4_STEPS:
        tr4, op4, rep4 = run_stage('S4', grown_model, tr3, fr3, tx3, op3, S4_STEPS, train_ids, 'S4_full', 4, collect_gram=True)
        save_trainable('S4_full', tr4)
    else:
        REPORT['stages']['S4_full'] = {'status': 'SKIPPED', 'steps_planned': 0,
                                       'note': 'full training on the real dataset is a separate authorized run (STAGED_DESIGN.md S4)'}
    save_rpt()
    REPORT['status'] = 'COMPLETED'; save_rpt()
    print(json.dumps({'status': REPORT['status'], 'stages': {k: v.get('status') for k, v in REPORT['stages'].items()}}, indent=1), flush=True)


if __name__ == '__main__':
    run_pipeline()
