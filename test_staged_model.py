"""Model-level unit checks for the staged pipeline (audit 1.4.3, CPU, micro sizes).

Covers the failure classes the 19 core tests could not see:
  (i)  grown-model identity with NONZERO DAR alpha (would catch C4),
  (ii) one optimizer step moves params by ~lr, not ~params (would catch C3),
  (iii) transplant covers every grown leaf with zero-init outputs (would catch C5).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import jax, jax.numpy as jnp
import optax
from flax.traverse_util import flatten_dict, unflatten_dict
import staged_core
from staged_core import build_growth_plan, cka_penalty
from staged_transfer import (StudentConfig, StudentLM, transplant_params, make_tx, to_master,
                             trainable_mask, merge, sample)

results = []
def check(name, ok, detail=''):
    assert ok, (name, detail)
    results.append(name); print(f'[PASS] {name} {detail}')

BASE = StudentConfig(d_model=16, num_blocks=2, n_layers=8, vocab=64, n_heads=1, d_head=8,
                     q_lora_rank=16, kv_lora_rank=8, d_ff=32, mla_positions=(3, 7), dar_latent=8)
GROWN = StudentConfig(d_model=16, num_blocks=4, n_layers=16, vocab=64, n_heads=1, d_head=8,
                      q_lora_rank=16, kv_lora_rank=8, d_ff=32, mla_positions=(3, 7, 11, 15), dar_latent=8)
base_model = StudentLM(cfg=BASE); grown_model = StudentLM(cfg=GROWN)
ids_small = np.zeros((1, 8), np.int32)
params_base = jax.device_get(base_model.init(jax.random.PRNGKey(0), ids_small)['params'])

# force NONZERO DAR alpha so an un-gated/un-transplanted DAR contribution breaks identity
flat = flatten_dict(params_base, sep='/')
alpha_leaves = [k for k in flat if k.endswith('/dar/alpha_scale')]
assert len(alpha_leaves) == BASE.n_layers
for k in alpha_leaves: flat[k] = np.asarray(flat[k]) + 0.5
params_base = unflatten_dict(flat, sep='/')

plan = build_growth_plan(8, 16, list(range(8, 16)), strategy='spread')
for e in plan:
    if e['role'] == 'copy': e['ffn_source'] = None
grown_abstract = jax.eval_shape(lambda r: grown_model.init(jax.random.PRNGKey(0), ids_small)['params'], jax.random.PRNGKey(0))
gflat = flatten_dict(jax.device_get(grown_abstract), sep='/')
h1 = flatten_dict(params_base, sep='/')
new_flat = transplant_params(h1, gflat, plan, 8, 4, {})

# (iii) coverage + zero-init of copies (audit C5)
check('transplant_covers_all_leaves', set(new_flat) == set(gflat), f'{len(new_flat)} leaves')
check('copy_o_proj_zero_init', float(np.abs(np.asarray(new_flat['block_2/layer_8/mixer/o_proj/kernel'])).sum()) == 0.0)
check('copy_down_proj_zero_init', float(np.abs(np.asarray(new_flat['block_2/ffn/down_proj/kernel'])).sum()) == 0.0)
check('copy_dar_alpha_zero_init', float(np.abs(np.asarray(new_flat['block_2/layer_8/dar/alpha_scale'])).sum()) == 0.0)
check('copy_mixer_duplicated_not_zero', float(np.abs(np.asarray(new_flat['block_2/layer_8/mixer/q_proj/kernel'])).sum()) > 0,
      'dup of base mixer')
check('copy_ffn_gate_duplicated', float(np.abs(np.asarray(new_flat['block_2/ffn/gate_proj/kernel'])).sum()) > 0)
check('base_block_kept', float(np.abs(np.asarray(new_flat['block_0/layer_0/dar/alpha_scale']) - 0.5).sum()) < 1e-9,
      'base alpha stays 0.5')
check('base_ffn_kept', np.array_equal(np.asarray(new_flat['block_1/ffn/down_proj/kernel']),
                                      np.asarray(h1['block_1/ffn/down_proj/kernel'])))

params_grown = unflatten_dict(new_flat, sep='/')
probe_ids = jnp.asarray(np.random.default_rng(5).integers(0, 64, size=(1, 16)), jnp.int32)

# (i) grown identity with NONZERO base DAR alpha (audit C4 class)
lb = base_model.apply({'params': params_base}, probe_ids).astype(jnp.float32)
lg = grown_model.apply({'params': params_grown}, probe_ids).astype(jnp.float32)
d = float(np.abs(np.asarray(lb) - np.asarray(lg)).max())
check('grown_identity_nonzero_alpha', d < 1e-4, f'max_abs_diff={d:.2e}')

# representation shapes for the penalty (audit F1: pre-projection contributions)
_, reps = grown_model.apply({'params': params_grown}, probe_ids, collect_blocks=True)
check('reps_shapes', reps['mixer'].shape == (4, 1, 16, 8) and reps['ffn'].shape == (4, 1, 16, 32),
      f"{reps['mixer'].shape} {reps['ffn'].shape}")
_, cka_g = cka_penalty(reps['mixer'], 1.0, [(2, 0)], 0, 0)
check('copy_source_cka_below_one', float(cka_g[2, 0]) < 0.999, f'CKA(copy2, src0)={float(cka_g[2, 0]):.4f} (different inputs)')
# copy layer_8 must be a duplicate of base layer_0 (index fix: src layer = src_block*4 + li%4)
check('copy_layer_maps_to_source_layer', np.array_equal(np.asarray(new_flat['block_3/layer_13/mixer/q_proj/kernel']),
                                                        np.asarray(h1['block_1/layer_5/mixer/q_proj/kernel'])))
flat_grown = flatten_dict(params_grown, sep='/')
copy_blocks = [2, 3]
mask = trainable_mask(flat_grown, 'blocks', copy_blocks)
check('mask_covers_copies', all(mask[k] for k in flat_grown if k.startswith(('block_2/', 'block_3/'))))
tr3 = to_master(unflatten_dict({k: flat_grown[k] for k in flat_grown if mask[k]}, sep='/'))
fr3 = unflatten_dict({k: flat_grown[k] for k in flat_grown if not mask[k]}, sep='/')

# (ii) optimizer step: params move by ~lr, magnitudes preserved (audit C3 class)
tx = make_tx(tr3); op = tx.init(tr3)
batch = jnp.asarray(np.random.default_rng(6).integers(0, 64, size=(1, 17)), jnp.int32)
def loss_of(params):
    full = merge(params, fr3)
    logits, reps = grown_model.apply({'params': full}, batch[:, :-1], collect_blocks=True)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), batch[:, 1:]).mean()
    pen, _ = cka_penalty(reps['mixer'], 0.01, [(2, 0), (3, 1), (2, 3)], 16, 7)
    pen_f, _ = cka_penalty(reps['ffn'], 0.01, [(2, 0), (3, 1), (2, 3)], 16, 7)
    return ce + pen + pen_f, (ce, None)
(loss, (ce, _)), grads = jax.value_and_grad(loss_of, has_aux=True)(tr3)
updates, nop = tx.update(grads, op, tr3)
na = optax.apply_updates(tr3, updates)
w_old = np.asarray(tr3['block_2']['layer_8']['mixer']['q_proj']['kernel'])
w_new = np.asarray(na['block_2']['layer_8']['mixer']['q_proj']['kernel'])
step_size = float(np.abs(w_new - w_old).max())
check('step_moves_by_lr_bound', step_size <= 2.6e-4, f'max|delta|={step_size:.2e} (lr=1e-4)')
g_old = float(np.asarray(tr3['block_2']['pre_ffn_norm']['scale']).astype(np.float32).mean())
g_new = float(np.asarray(na['block_2']['pre_ffn_norm']['scale']).astype(np.float32).mean())
check('no_decay_or_blowup_on_1d', abs(g_new - g_old) <= 2.6e-4, f'norm scale {g_old:.5f}->{g_new:.5f}')
check('step_preserves_magnitude', float(np.abs(w_old).max()) > 1e-3 and float(np.abs(w_new).max()) > float(np.abs(w_old).max()) * 0.9,
      f'|old|max={float(np.abs(w_old).max()):.4f} |new|max={float(np.abs(w_new).max()):.4f}')
check('gradient_flows_to_copy_params', float(np.abs(np.asarray(grads['block_2']['layer_8']['mixer']['q_proj']['kernel'])).max()) > 0,
      'CKA on contributions gives grad at zero-init (audit F1)')
check('ce_finite_step', np.isfinite(float(ce)), f'ce={float(ce):.4f}')

# (iv) A+B arm: lm_head-trainable mask (audit run 2 follow-up)
mask_default = trainable_mask(flat_grown, 'new', [])
mask_lm = trainable_mask(flat_grown, 'new', [], extra_prefixes=('lm_head/',))
lm_keys = [k for k in flat_grown if k.startswith('lm_head/')]
check('mask_default_excludes_lm_head', all(not mask_default[k] for k in lm_keys))
check('mask_new_lm_head_includes_lm_head', all(mask_lm[k] for k in lm_keys) and len(lm_keys) == 1)
check('mask_new_lm_head_superset', all(mask_lm[k] == (mask_default[k] or k.startswith('lm_head/')) for k in flat_grown))

print(f'ALL {len(results)} STAGED-MODEL CHECKS PASS')
