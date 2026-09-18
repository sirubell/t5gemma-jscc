# Studies: preview, export, train and evaluate

A study is an explicit list of task recipes, seeds and model variations. It describes experiments; it does not allocate hardware. The same design variation is applied to COCO and HellaSwag.

## Study format

```yaml
name: example
task_configs: [../tasks/coco.yaml, ../tasks/hellaswag.yaml]
seeds: [0, 1, 2]
experiments:
  - name: shared
    model_overrides: {}
  - name: narrow
    model_overrides:
      codec: {bottleneck_dim: 256}
```

This expands to 2 tasks × 2 variations × 3 seeds = 12 training configurations. Model overrides support model/split/codec/channel fields; task training and evaluation settings come from the selected task recipe. A split override supplies a complete stack/where/index definition, while other model sections update named fields. Unknown fields and duplicate names/seeds are rejected to avoid silent unintended experiments.

Expansion order is task, variation, then seed. --task coco or --task hellaswag filters the plan and assigns a new contiguous index range. A seed currently affects COCO data/demo selection as well as training randomness: comparisons at the same seed share that selection, but a multi-seed study is not purely an initialization study with one fixed report panel.

## Preview and export

```bash
uv run --locked python study.py --config configs/studies/baseline.yaml
uv run --locked python study.py --config configs/studies/splits.yaml --task coco
uv run --locked python study.py --config configs/studies/smoke_h200.yaml --output runs/studies/smoke-h200
```

Without --output, the CLI only prints a plan. Export creates a NEW directory and refuses to overwrite an existing one:

```text
prepared/
  manifest.json          # Task/variation/seed/index and paths
  configs/               # Complete resolved single-run YAMLs
  links/                 # Filled after each successful training process
  runs/                  # Created by actual training, not by export
```

The original task/model files are not needed to read an exported config. Its output directory is relative to the prepared folder, so the folder can be transferred before execution. The manifest records the source revision and dirty state when available, but it does not copy or pin code: run from the intended source checkout. After execution, run links contain host-specific paths to the actual training output.

The pre-H200 `configs/studies/hellaswag_diagnostic.yaml` plan is intentionally
HellaSwag-only and expands to five entries: an `enc_fn` reference, old/new
`enc_l9`, and old/new `dec_l8`. Its task recipe stops at 4,000 optimizer
updates while retaining a 20,000-update schedule horizon, uses the full
512-row selection holdout, and evaluates a fixed short `no_noise`/`-6`/`18`
panel plus vanilla. Decoder entries keep `codec.memory.layernorm` fixed while
varying the main hidden codec. It is a WS/RTX 5090 diagnostic gate, not a
formal H200 sweep.

## Corrected baseline after the diagnostic

The corrected baseline is a separate protocol. Keep its results separate from
the historical task recipe and the five-row diagnostic; the protocol changes
power/mask semantics, trainable precision and loss aggregation. The plan is
`configs/studies/hellaswag_corrected.yaml` and its task recipe is
`configs/tasks/hellaswag_corrected.yaml`. It expands to three HellaSwag
entries:

| Entry | Split | Main codec boundary norm | Memory codec boundary norm |
|---|---|---|---|
| `corrected_enc_fn` | encoder final norm | `both` | `both` (inherited control) |
| `corrected_enc_l9` | encoder layer 9 | `none` | `both` (inherited control) |
| `corrected_dec_l8` | decoder layer 8 | `none` | `both` |

All entries use T5Gemma 2, residual codec B512/H1152, FiLM off, seed 0,
training batch `16 x 2`, 4,000 optimizer updates, a 20,000-step schedule
horizon, the fixed 512-row validation holdout, and evaluation at `no_noise`,
`-6`, `18`, plus vanilla. The plan is a reusable definition; individual
submission IDs and completion evidence belong to the execution record.
The resolved task config carries `protocol: corrected-baseline-v1`, so result
indexers can keep this cohort separate from legacy evidence.

Prepare the portable H200 plan without submitting it:

```bash
uv run --locked python study.py \
  --config configs/studies/hellaswag_corrected.yaml \
  --task hellaswag \
  --output runs/studies/hellaswag-corrected-h200
```

After a separate execution approval and a target-host preflight, the expected
Slurm shape is one training array with the three entries in parallel, followed
by a matching evaluation array. The commands below are a template only; fill
in the site's partition/account/QOS and the returned training job ID:

```bash
sbatch --partition=YOUR_PARTITION --account=YOUR_ACCOUNT --qos=YOUR_QOS \
  --array=0-2%3 --job-name=hs-corrected-train \
  --output=runs/slurm/%x-%A_%a.out \
  --error=runs/slurm/%x-%A_%a.err \
  scripts/slurm_study.sh train \
  runs/studies/hellaswag-corrected-h200/manifest.json

sbatch --partition=YOUR_PARTITION --account=YOUR_ACCOUNT --qos=YOUR_QOS \
  --array=0-2%3 --dependency=aftercorr:TRAIN_ARRAY_ID \
  --kill-on-invalid-dep=yes --job-name=hs-corrected-eval \
  --output=runs/slurm/%x-%A_%a.out \
  --error=runs/slurm/%x-%A_%a.err \
  scripts/slurm_study.sh evaluate \
  runs/studies/hellaswag-corrected-h200/manifest.json
```

Each array element requests one GPU. `aftercorr` keeps evaluation index `i`
paired with the successful training index `i`; it does not start evaluation
for a failed or timed-out training route. The `study.py --output` step and the
commands above do not submit anything by themselves. The H200 execution
decision remains a separate gate after code review, local tests and a target
GPU smoke.

Run outputs and completed links remain on the execution host. Transferring a plan before execution does not automatically sync later results back to the development machine.

Use a fresh prepared folder for a new training attempt. Each entry is independent; rerunning evaluation creates a new evaluation directory for the same recorded checkpoint. A completed training link is not overwritten by another training invocation.

## Separate Slurm jobs

For the two-entry smoke plan, submit two arrays with matching indices. Replace site placeholders using ignored local environment notes; use --time=00:15:00 for the smoke instead of the script's longer default.

```bash
mkdir -p runs/slurm
sbatch --partition=YOUR_PARTITION --account=YOUR_ACCOUNT --qos=YOUR_QOS --array=0-1%2 --time=00:15:00 --job-name=smoke-train --output=runs/slurm/%x-%A_%a.out --error=runs/slurm/%x-%A_%a.err scripts/slurm_study.sh train runs/studies/smoke-h200/manifest.json
# Replace TRAIN_ARRAY_ID with the ID returned above.
sbatch --partition=YOUR_PARTITION --account=YOUR_ACCOUNT --qos=YOUR_QOS --array=0-1%2 --dependency=aftercorr:TRAIN_ARRAY_ID --kill-on-invalid-dep=yes --time=00:15:00 --job-name=smoke-eval --output=runs/slurm/%x-%A_%a.out --error=runs/slurm/%x-%A_%a.err scripts/slurm_study.sh evaluate runs/studies/smoke-h200/manifest.json
```

Each array element requests one GPU. aftercorr lets evaluation index i run only after training index i succeeds. The manifest and run link select the exact matching run; evaluation does not guess from directory timestamps. A failed or timed-out train process does not create a success link, and dependent evaluation cannot count as completed. The invalid-dependency option lets Slurm cancel such evaluations instead of keeping impossible dependencies pending.

Use the manifest entry count for the index range (for N entries: 0 through N−1). The %2 concurrency cap is per array, not global across all arrays; account/QOS limits still apply. Budget training and evaluation separately. The two-array submission commands are examples for explicit execution, not actions taken by study.py.

The same entry runner can be invoked on a standalone GPU workstation (without Slurm), or inside an appropriate allocation without arrays. Run train and evaluation sequentially for each desired index on a single GPU:

```bash
uv run --locked python -m jscc.study_task train --manifest runs/studies/smoke-h200/manifest.json --index 0
uv run --locked python -m jscc.study_task evaluate --manifest runs/studies/smoke-h200/manifest.json --index 0
```

## Scope and results

All supplied split/bottleneck entries have small-model CPU forward/backward/generation tests. Full-weight GPU smoke has covered the shared baseline, not every sweep combination. A full-duration sweep is only needed to collect research results; for software/GPU readiness, first run short checks on the target GPU, especially early encoder splits and larger-memory variants. See [validation levels](validation.md#what-a-sweep-test-does-and-does-not-prove).

Training retains its existing validation/checkpoint rules. Each evaluation uses best.pt by default and the saved configuration; the module CLI can select --checkpoint last.pt. All SNRs for one checkpoint remain in one evaluation process. Vanilla results are currently computed per entry rather than deduplicated across a study.

This workflow provides planning and train/eval pairing. It does not automatically resume timeouts, tune resource requests, aggregate thesis figures, or reproduce historical protocols. Existing results remain separate from newly prepared plans.
