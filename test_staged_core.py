"""Local unit checks for staged_core (CPU, tiny arrays; not a model/TPU test)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jax
import jax.numpy as jnp
import staged_core as sc

results = []
def check(name, ok, detail=''):
    assert ok, (name, detail)
    results.append((name, detail)); print(f'[PASS] {name} {detail}')

# 1) penalty bounds: identical blocks -> ~max, orthogonal -> ~0
base = jax.random.normal(jax.random.PRNGKey(0), (32, 8))
same = jnp.repeat(base[None], 4, axis=0)          # 4 identical blocks
p_same, g_same = sc.orthogonality_penalty(same, 1.0)
check('penalty_identical_is_max', abs(float(p_same) - 1.0) < 1e-5, f'{float(p_same):.6f}')
eye = jnp.eye(8); outs = jnp.stack([eye[i].reshape(1, 8).repeat(32, 0) for i in range(4)])
p_orth, _ = sc.orthogonality_penalty(outs, 1.0)
check('penalty_orthogonal_is_zero', float(p_orth) < 1e-5, f'{float(p_orth):.2e}')
# lam scales linearly
p_half, _ = sc.orthogonality_penalty(same, 0.5)
check('penalty_scales_with_lambda', abs(float(p_half) - 0.5 * float(p_same)) < 1e-6)

# 2) differentiability: gradient flows to the blocks
grad = jax.grad(lambda o: sc.orthogonality_penalty(o, 1.0)[0])(same)
check('penalty_differentiable', bool(jnp.isfinite(grad).all()) and float(jnp.abs(grad).sum()) > 0)
# perturbing one of two identical blocks strictly lowers the penalty
b2 = jax.random.normal(jax.random.PRNGKey(1), (16, 8))
x = jnp.repeat(b2[None], 2, axis=0)
x = x.at[1].add(1e-2 * jax.random.normal(jax.random.PRNGKey(2), x.shape[1:]))
p_pert, _ = sc.orthogonality_penalty(x, 1.0)
check('perturbation_reduces_penalty', float(p_pert) < float(sc.orthogonality_penalty(jnp.repeat(x[0][None], 2, axis=0), 1.0)[0]))

# 3) adjacent scope counts n-1 pairs
p_adj, _ = sc.orthogonality_penalty(same, 1.0, scope='adjacent')
check('adjacent_scope_mean_is_one_for_identical', abs(float(p_adj) - 1.0) < 1e-5)

# 4) growth plan: distinct FFN sources for copies, deterministic
plan = sc.build_growth_plan(8, 16, list(range(8, 48)))
check('plan_length', len(plan) == 16)
check('base_roles', [p['role'] for p in plan[:8]] == ['base'] * 8)
check('copy_roles', [p['role'] for p in plan[8:]] == ['copy'] * 8)
check('copy_sources_distinct', len({p['ffn_source'] for p in plan[8:]}) == 8, str([p['ffn_source'] for p in plan[8:]]))
check('sources_outside_base', all(p['ffn_source'] >= 8 for p in plan[8:]))
check('copy_gates_zero', all(p['gate_init'] == 0.0 for p in plan[8:]))
check('base_gates_from_checkpoint', all(p['gate_init'] is None for p in plan[:8]))
plan2 = sc.build_growth_plan(8, 16, list(range(8, 48)))
check('plan_deterministic', plan == plan2)
plan_cycle = sc.build_growth_plan(8, 40, list(range(0, 12)), strategy='cycle')
check('cycle_covers_exhausted_pool', len({p['ffn_source'] for p in plan_cycle[8:]}) <= 4)

# 5) zero gate is an exact identity (function-preserving stack)
x = jax.random.normal(jax.random.PRNGKey(3), (2, 16, 8))
y = sc.apply_residual(x, sc.residual_gate(0.0), jnp.ones_like(x) * 100.0)
check('zero_gate_identity', bool(jnp.array_equal(x, y)))
y7 = sc.apply_residual(x, sc.residual_gate(0.7), jnp.ones_like(x))
check('gate_linear_response', bool(jnp.allclose(y7, x + 0.7, atol=1e-6)))

# 6) report contract
rep = sc.stage_report_template(plan, {'lambda_orth': 1e-2})
check('stage_report_contract', all(s in rep['stages'] for s in ('S1_base', 'S2_growth', 'S3_warmup', 'S4_full'))
      and rep['scope'].startswith('staged'))

print(f'ALL {len(results)} STAGED-CORE CHECKS PASS')
