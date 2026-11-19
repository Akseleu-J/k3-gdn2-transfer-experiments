# K3 → GDN-2/MLA transfer experiments

Readable sources for the K3 → GDN-2/MLA transfer diagnostic, packaged for the
protocol-compliant launcher flow (see `KAGGLE_PROTOCOL.md` in the local project
folder: no embedded `PAYLOAD`, no `exec`/`eval` blobs, sources are cloned from
this repository at a pinned full commit).

## Contents

| File | Role |
|---|---|
| `experiment.py` | The diagnostic itself: tiny (mechanism/backward gates, no weights) and real (loads pinned Kimi-K3 revision, checks transfer, measures CE(0), shuffled CE, unigram baseline, optional bounded training). Configured purely via environment variables: `K3_MODE`, `K3_STEPS`, `K3_SEQ`, `K3_BATCH`, `K3_ARTIFACTS`, `K3_REVISION`, `HF_TOKEN`. |
| `colab_run.py` | Launcher driver: creates an isolated venv (`--without-pip` + parent-pip bootstrap), installs pinned dependencies, runs the selected script per phase (`--script`, default `experiment.py`), writes `status.json`, keeps one atomic results ZIP (with `manifest.json` and SHA-256 sidecar) refreshed every 60 s, redacts secret-looking strings from logs and archives. Stdlib only. |
| `staged_core.py` | Math core of the staged pipeline: block-output orthogonality penalty, deterministic growth plan (block-aligned copies with `mixer_source` + distinct teacher FFN sources), ReZero-style function-preserving gates, stage report contract. 19 local unit checks in `test_staged_core.py`. |
| `staged_transfer.py` | The staged pipeline itself (smoke and real paths): S1 base-block training on copied teacher layers, S2 growth by duplicating trained blocks with per-copy FFN from distinct teacher shared-experts layers and zero gates + function-preservation gate, S3 whole-volume warmup with the orthogonality penalty, S4 optional full training. See `STAGED_DESIGN.md` for the locked design and references. |

## How to run (Colab)

Use the companion launcher notebook `K3_GDN2_Launcher_Colab.ipynb` (built from
this repository; it contains only settings, `git clone` of this repo at a
pinned commit, revision check and the documented CLI call):

1. The repository is currently **public**: the notebooks clone it without any
   credentials. If it is switched back to private, add a fine-grained GitHub
   read-only token (Contents: Read, access limited to this repository) as a
   Colab Secret named `GITHUB_READ_TOKEN`; the launcher uses it only for the
   clone subprocess — never in URLs, git config, logs or source.
2. `real` mode additionally accepts an optional `HF_TOKEN` Colab Secret for
   Kimi-K3 Hub access; public access is attempted without it.
3. Run all. Results: `/content/k3_runs/<profile>-<RUN_ID>.zip` (+ `.sha256`),
   mirrored to `MyDrive/K3_GDN2_results/<run>/` when Drive is mounted.

Manual equivalent without the notebook:

```bash
git clone https://github.com/Akseleu-J/k3-gdn2-transfer-experiments.git
cd k3-gdn2-transfer-experiments
git checkout <PINNED_FULL_COMMIT_SHA>
python colab_run.py --profile tiny --tiny-steps 30 --seq 128          # diagnostic
python colab_run.py --profile staged --script staged_transfer.py      # staged pipeline (env knobs: K3_STAGED_RUN, K3_S1_STEPS, ...)
```

## Staged pipeline summary

Copy the first `K3_N_BASE` (default 8) teacher layers, re-initialize the new
architecture components, train the base (S1); duplicate the trained blocks to
`K3_TARGET` (default 16) with each copy's FFN replaced by the shared experts of
a **distinct** teacher layer and zero residual gates so the grown model exactly
reproduces the base function (S2, asserted); warm the whole volume with a
MoE-style orthogonality penalty on pooled block outputs (S3); optional full
training on the real dataset (S4, off by default). Motivation and prior work:
depthwise stacking converges cross-stage within a few hundred steps
(NeurIPS 2024 Spotlight), duplicating *different* layers beats identical
duplication (ACL 2021), deep-layer redundancy motivates the diversity penalty
(arXiv:2403.17887) — an explicit orthogonality loss between grown blocks is the
under-explored piece this pipeline tests. Full contract: `STAGED_DESIGN.md`.

## Dependencies (installed into the isolated venv by `colab_run.py`)

`jax[tpu]==0.11.2`, `flax==0.12.9`, `optax==0.2.6`, `requests`,
`huggingface_hub`, `transformers>=4.56,<5`, `tiktoken`, `sentencepiece`,
`safetensors`. The Kimi-K3 modeling source revision is pinned separately via
`--k3-revision`; `real` mode refuses geometry drift from the pinned config.

## What this does NOT establish

Tiny mode checks mechanisms and backward, not K3 quality. CE(0) ≥ ln(vocab)
stops real training as a heuristic — not a proof of teacher equivalence. The
candidate student (93 → 16 layers, 96 → 32 heads, no routed experts, changed
residual topology) intentionally changes the K3 function even at zero DAR;
teacher parity, full-transfer quality, speed and TPU stability are not claimed
until measured. `report.json` marks scope as `candidate student; no teacher
parity claim`.

## Provenance contract

Every archive contains `manifest.json` (per-file size + SHA-256), `run_plan.json`
(profile, phases, repo SHA, K3 revision, source hashes) and byte-copies of
`experiment.py` / `colab_run.py` that actually ran. Integrity of a ZIP does not
establish model correctness or quality.

## License / status

Research code of the project team, temporarily public (may return to private;
no redistribution of derived training artifacts implied). Historical notebooks
with embedded source payloads are archived artifacts and must not be
re-submitted (KAGGLE_PROTOCOL.md §9).
