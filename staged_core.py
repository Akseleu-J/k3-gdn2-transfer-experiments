"""Math core of the staged transfer pipeline (S1 base block -> S2 growth -> S3 warmup).

Pure differentiable operators, no model code: composable with the student model
in experiment.py after narrowing. References for each operator are in
STAGED_DESIGN.md; the growth plan follows the depthwise-stacking line
(Du et al., NeurIPS 2024 "Stacking Your Transformers"; bert2BERT, ACL 2021)
and the orthogonality penalty follows MoE expert-specialization losses
(Guo et al., arXiv:2505.22323) applied here to block outputs of duplicated
layers - the explicitly-unexplored variant per the 2026-11-18 literature scan.
"""
import numpy as np
import jax
import jax.numpy as jnp

STAGED_DEFAULTS = dict(
    n_base=8,            # teacher layers copied into the base block (S1)
    target_layers=16,    # total student depth after growth (S2)
    lambda_orth=1e-2,    # weight of the block-output orthogonality penalty
    orth_scope='all',    # 'all' pairs or 'adjacent' neighbors only
    ffn_strategy='spread',  # 'spread': each copy takes a distinct teacher FFN layer
)


def cka_penalty(reps, lam, pairs, n_tokens=512, key=0, eps=1e-6):
    """Anti-redundancy on UNGATED block contributions via centered linear CKA.

    reps: [n_blocks, ..., d] pre-projection contribution representations
    (e.g. mixer output before o_proj, FFN hidden before down_proj); all axes
    between the first and the last are flattened into the token/sample axis.
    Linear CKA (Kornblith et al., ICML 2019) with token-centered features X_i:
        CKA(i, j) = <G_i, G_j>_F / (||G_i||_F ||G_j||_F),  G_i = X_i X_i^T,
    which equals ||X_i^T X_j||_F^2 / (||X_i^T X_i||_F ||X_j^T X_j||_F); it is
    1 for identical (or isotropically scaled / rotated) representations and
    in [0, 1] otherwise. Only the requested pairs contribute (copy<->source,
    copy<->copy), per the ICLR 2025 warning against pushing ALL layer pairs.
    Returns (lam * mean CKA over pairs, full CKA matrix over all blocks).
    """
    R = jnp.asarray(reps).astype(jnp.float32)
    nb, d = R.shape[0], R.shape[-1]
    R = R.reshape(nb, -1, d)
    n = R.shape[1]
    if n_tokens and n_tokens < n:
        k = jnp.asarray(key, jnp.uint32)
        take = jax.random.permutation(jax.random.PRNGKey(k), n)[:n_tokens]
        R = R[:, take, :]
    R = R - R.mean(axis=1, keepdims=True)
    G = jnp.einsum('bnd,bmd->bnm', R, R)               # [nb, n, n] token Grams
    hsic = jnp.einsum('inm,jnm->ij', G, G)             # <G_i, G_j>_F
    fg = jnp.sqrt(jnp.clip(jnp.diagonal(hsic), 0.0))   # ||G_i||_F
    gram = hsic / (fg[:, None] * fg[None, :] + eps)
    pm = np.zeros((nb, nb), np.float32)
    for i, j in pairs:
        pm[i, j] = 1.0
    count = max(len(pairs), 1)
    return lam * jnp.sum(gram * pm) / count, gram


def orthogonality_penalty(block_outputs, lam, scope='all', eps=1e-6):
    """DEPRECATED (audit 2026-11-19): penalizes accumulated residual-stream
    states, not block contributions; gradient w.r.t. zero gates is first-order
    zero (AUDIT_RESPONSE 2026-11-19 §2.1). Kept only for A/B reference; use
    cka_penalty on pre-projection contributions instead.

    Mean off-diagonal |cosine| between pooled block outputs (MoE-style L_O).

    block_outputs: [n_blocks, T, d] pooled per block over the token axis; the
    penalty is differentiable and reported together with the full similarity
    matrix for the run report. Identical blocks give the maximum; orthogonal
    outputs give zero.
    """
    v = block_outputs.mean(axis=1)                       # [n_blocks, d]
    v = v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + eps)
    gram = v @ v.T                                       # [n_blocks, n_blocks] cosine
    n = gram.shape[0]
    if scope == 'adjacent':
        off = jnp.abs(gram.diagonal(1))          # symmetric gram: each pair once
        count = max(n - 1, 1)
    else:
        iu = jnp.triu_indices(n, k=1)
        off = jnp.abs(gram[iu])
        count = n * (n - 1) // 2
    mean_off = off.sum() / max(count, 1)
    return lam * mean_off, gram


def build_growth_plan(n_base, target_layers, teacher_ffn_layers, strategy='spread', layers_per_block=4):
    """Deterministic plan: which teacher FFN initializes each student block.

    Base blocks 0..n_base-1 keep their own teacher layers and S1-trained state
    (gate_init None = loaded from the S1 checkpoint). Each copy duplicates a
    trained base block (mixer_source) and takes a *distinct* teacher FFN source
    (bert2BERT found duplicating *different* layers beats identical
    duplication); 'spread' picks sources evenly across the unused teacher
    depth, 'cycle' repeats when teacher depth is exhausted.
    """
    assert n_base >= 1 and target_layers >= n_base, 'growth must keep or add depth'
    assert len(teacher_ffn_layers) >= n_base, 'teacher must provide the base FFN sources'
    assert n_base % layers_per_block == 0 and target_layers % layers_per_block == 0, 'block-aligned depths required'
    n_base_blocks = n_base // layers_per_block
    plan = []
    for b in range(0, target_layers, layers_per_block):
        if b < n_base:
            for i in range(layers_per_block):
                plan.append(dict(block=b + i, role='base', mixer_source=b + i,
                                 ffn_source=teacher_ffn_layers[b + i], gate_init=None))
            continue
        pool = [t for t in teacher_ffn_layers if t >= n_base] or list(teacher_ffn_layers)
        if strategy == 'spread':
            copies = range(n_base, target_layers, layers_per_block)
            need = len(copies)
            idx = [round(i * (len(pool) - 1) / max(need - 1, 1)) for i in range(need)]
            src = pool[idx[(b - n_base) // layers_per_block]]
        elif strategy == 'depth_proportional':
            # copy block at relative depth r := (b+3)/target gets the teacher FFN
            # layer at the same relative position (uniform/skip mapping used in
            # layer-matching distillation); collisions resolved to the nearest
            # unused candidate (audit M8/Q13).
            src = None
            r = (b + layers_per_block - 1) / max(target_layers - 1, 1)
            order = sorted(range(len(pool)), key=lambda i: (abs(i / max(len(pool) - 1, 1) - r), abs(i - (b // layers_per_block))))
            used = {p['ffn_source'] for p in plan}
            for i in order:
                if pool[i] not in used:
                    src = pool[i]; break
            if src is None: src = pool[int(r * (len(pool) - 1))]
        elif strategy == 'cycle':
            src = pool[((b - n_base) // layers_per_block) % len(pool)]
        else:
            raise ValueError('unknown ffn strategy: ' + str(strategy))
        for i in range(layers_per_block):
            plan.append(dict(block=b + i, role='copy', mixer_source=(b % n_base) + i,
                             ffn_source=src, gate_init=0.0))
    return plan


def residual_gate(gate_value):
    """ReZero-style scalar gate (zero init = function-preserving stack)."""
    return jnp.asarray(gate_value)


def apply_residual(x, gate, f_out):
    """x + gate * F(x); gate==0 makes an appended block an exact identity."""
    return x + gate * f_out


def stage_report_template(plan, cfg_updates):
    """Report contract shared by all staged runs (gates per stage, no quality claims)."""
    return dict(
        scope='staged transfer candidate; no teacher parity claim',
        stages={'S1_base': 'PENDING', 'S2_growth': 'PENDING', 'S3_warmup': 'PENDING', 'S4_full': 'PENDING'},
        plan=plan, config=cfg_updates,
    )
