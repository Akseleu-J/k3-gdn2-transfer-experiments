# STAGED_DESIGN.md — staged transfer pipeline (S1→S2→S3→S4)

Status: design locked 2026-11-18 after literature scan and user narrowing.
Core operators implemented and unit-tested in `staged_core.py` (18 checks PASS);
the full pipeline lives in `staged_transfer.py`. Scope: candidate student, no
teacher parity claim anywhere.

## Revision 2 (2026-11-20, after AUDIT_RESPONSE_2026-11-19)

Supersedes the gate/penalty parts of the sections below where they conflict.

- **Growth identity**: copies duplicate their source block wholesale, with
  mixer `o_proj`, FFN `down_proj` and DAR `alpha_scale` zero-initialized
  (LLaMA Pro style) instead of a scalar `block_gate`. Exact identity at growth,
  full-rank functional tangent from step one. DAR math is identical to
  experiment.py (no double-added DAR).
- **Anti-redundancy loss**: centered linear CKA (`staged_core.cka_penalty`)
  between PRE-PROJECTION contributions (mixer before `o_proj`, FFN hidden
  before `down_proj`), only on (copy, source) and (copy, copy) pairs. Gradient
  reaches copy parameters while outputs are still exactly zero. The
  residual-stream `orthogonality_penalty` is deprecated (kept for A/B only).
- **S2 gate**: structural (all copy output leaves exactly zero) + numeric
  (max|Δlogits| ≤ 4 bf16 ulps at max|logit|, KL ≤ 1e-4). Bitwise equality is
  informational: two compiled bf16 programs differ by a few ulps even for the
  same model (local: eager vs jit 4.1e-2).
- **S2 source state**: growth uses the S1-TRAINED parameters.
- **Optimizer**: fp32 master copy of the trainable subtree; AdamW with weight
  decay only on ≥2D matrices.
- **Data/eval**: disjoint held-out slice (enwik8 bytes 95.0M–95.2M), stage-folded
  data keys, `ce_val_end` after S1 and S3, stop threshold min(ln V, unigram CE).
- **Copy FFN sources (real)**: `depth_proportional` over teacher layers with
  shared experts present, excluding the last two layers.

## Locked decisions (user, 2026-11-18)

| Decision | Value |
|---|---|
| FFN source for growth copies | **shared experts of distinct teacher layers** (one teacher layer per copy, no repeats). Base blocks keep the reviewed baseline convention (teacher layer 1 shared experts). bert2BERT evidence: duplicating *different* layers converges faster. Naive MoE→dense averaging is rejected per moe-to-dense (KRAFTON 2026). |
| Depth | base **8** layers = 2 DARBlocks (mla_positions (3,7)) → target **16** layers = 4 blocks (mla_positions (3,7,11,15)); 2 copies. |
| Orthogonality penalty | **pooled block outputs, all pairs** first; per-sublayer (attention/FFN grams) available behind `orth_scope='sublayer'` if needed. Weight `lambda_orth`, default 1e-2. |
| Function preservation at growth | ReZero-style scalar gates on copied blocks (mixer delta and FFN output), init 0 → grown model reproduces the base function exactly at growth time (asserted as a stage gate). G_stack (NeurIPS 2024 Spotlight) shows cross-stage convergence within a few hundred steps for such stacks. |

## Stages and gates

**S1_base** — student with first 8 teacher layers. Weights: mixer/FFN/embed/norms
patched from teacher (same `_kda`/`_mla`/shared-experts mapping as experiment.py);
new-architecture components re-initialized (DAR zeros, erase/write gates from
teacher `_gate_dense`, dt_bias zeros, pre_ffn_norm ones) — «сбросим то что лишнее».
Train the base for `s1_steps` on the S1 dataset. Gate: finite CE(0); every
optimizer step accepted (finite loss + grad norm); report gate trajectories.

**S2_growth** — build the 16-layer student: blocks 0–1 keep the S1-trained state;
blocks 2–3 are **duplicates of the trained base blocks** (`mixer_source`:
block 2 ← block 0, block 3 ← block 1 — DAR, mixer, norms, gates copied) whose
FFN sublayer is replaced by the shared experts of the plan-assigned distinct
teacher layer; `pre_ffn_norm` starts at ones; copied-block output gates
(`gate_mixer`, `gate_ffn`) = 0.
Gates: (a) **function preservation** — logits of the grown model on a fixed
probe batch equal the base model logits bitwise (all added contributions are
zero); (b) patch ledger closes (`unused` empty, no unexpected fallback);
(c) growth plan matches `staged_core.build_growth_plan` output for the locked
config (deterministic, distinct FFN sources ≥ n_base).

**S3_warmup** — continued training of the whole 16-layer volume:
`loss = CE + lambda_orth * mean_offdiag_cos(block_outputs)`.
Dynamics note: with gates at 0 the gradient reaches copy parameters only through
the gates, so the orthogonality term first steers *gate* trajectories toward
decorrelated blocks; log per-block gate values and the full similarity gram
every report interval. Gate: finite steps; gram recorded; stop criteria per
run request.

**S4_full** — full training mode on the planned real dataset (enwik8 splits as
in experiment.py; tokenizer = K3's). Orth penalty stays configurable
(`orth_in_s4`). Gate: CE(0) < ln(vocab) heuristic stop; checkpoint cadence;
final report marks scope `candidate student; no teacher parity claim`.

## Model/patch integration notes

- FFN is **per DARBlock** in the student; the duplication unit is therefore the
  DARBlock (4 DAR layers + one FFN). Teacher FFN tensors =
  `language_model.model.layers.{t}.block_sparse_moe.shared_experts.{gate,up,down}_proj.weight`.
- Copied blocks duplicate the assigned trained base block wholesale (DAR, mixer,
  norms, per-layer gates); only the FFN comes from the assigned teacher layer
  and `pre_ffn_norm` starts at ones. Function preservation is carried by the
  per-copy `gate_mixer`/`gate_ffn` scalars (init 0, trainable).
- Teacher layer pool for copies: `t ≥ n_base` (do not reuse base-layer FFNs);
  `spread` strategy = evenly spaced distinct layers (staged_core plan).
- `orth_scope='sublayer'` switch: two grams (per-block mixer deltas, per-block
  FFN outputs) averaged; implemented as a separate penalty call, same lambda.

## References (2026-11-18 scan)

- Du et al., *Stacking Your Transformers: A Closer Look at Model Growth for
  Efficient LLM Pre-Training*, NeurIPS 2024 Spotlight, arXiv:2405.15319 — G_stack,
  function-preserving identity init, few-hundred-step cross-stage convergence.
- Chen et al., *bert2BERT*, ACL 2021, arXiv:2110.07143 — duplicating different
  layers beats identical duplication; function-preserving growth init.
- Chen et al., *Net2Net*, ICLR 2016, arXiv:1511.05641 — identity/depth growth.
- Grafting (DiT, NeurIPS 2025 Oral, arXiv:2506.05340; ViT arXiv:2210.15943) —
  replacing attention components while reusing pretrained weights.
- Guo et al., *Advancing Expert Specialization for Better MoE*, arXiv:2505.22323 —
  expert orthogonality/variance losses (adapted here to block outputs).
- *Pruning and Distilling MoE into Dense* (KRAFTON), arXiv:2605.28207 —
  diverse-expert selection beats naive expert averaging.
- Csordás et al., arXiv:2406.11784 — depth as loop-like computation (motivation
  for stacking); Gromov et al., arXiv:2403.17887 — deep-layer redundancy
  (motivation for the orthogonality penalty).
- SOLAR 10.7B DUS, arXiv:2312.15166; LiGO ICLR 2023 arXiv:2303.00980;
  LEMON arXiv:2406.08870; HyperCloning arXiv:2404.09720 — growth operators.

## Explicit non-claims

Tiny/smoke runs verify mechanics only. No speed, quality, teacher-parity or
100M-token claims. Cross-stage convergence cost is measured, not assumed.
Every stage writes its own report section, checkpoint and gate outcomes into
the shared run report; stage failures stop later stages.
