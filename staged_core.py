"""Math core of the staged transfer pipeline (S1 base block -> S2 growth -> S3 warmup).

Pure differentiable operators, no model code: composable with the student model
in experiment.py after narrowing. References for each operator are in
STAGED_DESIGN.md; the growth plan follows the depthwise-stacking line
(Du et al., NeurIPS 2024 "Stacking Your Transformers"; bert2BERT, ACL 2021)
and the orthogonality penalty follows MoE expert-specialization losses
(Guo et al., arXiv:2505.22323) applied here to block outputs of duplicated
layers - the explicitly-unexplored variant per the 2026-11-18 literature scan.
"""
import jax.numpy as jnp

STAGED_DEFAULTS = dict(
    n_base=8,            # teacher layers copied into the base block (S1)
    target_layers=16,    # total student depth after growth (S2)
    lambda_orth=1e-2,    # weight of the block-output orthogonality penalty
    orth_scope='all',    # 'all' pairs or 'adjacent' neighbors only
    ffn_strategy='spread',  # 'spread': each copy takes a distinct teacher FFN layer
)


def orthogonality_penalty(block_outputs, lam, scope='all', eps=1e-6):
    """Mean off-diagonal |cosine| between pooled block outputs (MoE-style L_O).

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


def build_growth_plan(n_base, target_layers, teacher_ffn_layers, strategy='spread'):
    """Deterministic plan: which teacher FFN initializes each student block.

    Base blocks 0..n_base-1 keep their own teacher layers. Each copy k takes a
    distinct teacher FFN source (bert2BERT found duplicating *different*
    layers beats identical duplication); 'spread' picks them evenly across the
    unused teacher depth, 'cycle' repeats when teacher depth is exhausted.
    """
    assert n_base >= 1 and target_layers >= n_base, 'growth must keep or add depth'
    assert len(teacher_ffn_layers) >= n_base, 'teacher must provide the base FFN sources'
    plan = []
    for b in range(target_layers):
        if b < n_base:
            # base blocks load their S1-trained state; gate_init None = from checkpoint
            plan.append(dict(block=b, role='base', ffn_source=teacher_ffn_layers[b], gate_init=None))
            continue
        pool = [t for t in teacher_ffn_layers if t >= n_base] or list(teacher_ffn_layers)
        if strategy == 'spread':
            need = target_layers - n_base
            idx = [round(i * (len(pool) - 1) / max(need - 1, 1)) for i in range(need)]
            src = pool[idx[b - n_base]]
        elif strategy == 'cycle':
            src = pool[(b - n_base) % len(pool)]
        else:
            raise ValueError('unknown ffn strategy: ' + str(strategy))
        plan.append(dict(block=b, role='copy', ffn_source=src, gate_init=0.0))
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
