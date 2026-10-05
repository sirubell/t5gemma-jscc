# Recorded experiment operations

Run commands from the project root after `uv sync --locked`. Training, evaluation and live submission require their applicable scope and budget; these examples are not launch receipts. [Running](running.md) covers local/Colab/CUDA setup. Local research starts at `docs/local/research/HANDOFF.md`, then the session guide and assigned ticket; those ignored files are optional for a fresh clone.

Acquire any missing pinned model/tokenizer/data bytes in a separately scoped setup step using [the existing cache/setup guide](running.md). Then keep these normal operations cache-only:

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

The existing ordinary loaders honor this environment; missing cached inputs fail instead of starting an implicit acquisition. Recipe commands also require their acquired registration receipts. `--check` resolves settings without loading model/data, and neither setup nor these examples grants training/submission authority.

## Ordinary training, evaluation and continuation

Use explicit fresh roots for the supported recorded COCO and HellaSwag route. The complete [resumable examples](../configs/operations/README.md) use the shared model design, with `data.num_workers: 0` set **before** binding. The named COCO example adds a nonterminal save at100 while retaining selection at500 and the500-update budget; this extra save is an explicit example policy. COCO retains supplied scientific settings. The named HellaSwag variant additionally sets `training.presentation_stream: null`, replacing the default fixed-presentation stream with the ordinary map loader; this changes sampling/presentation policy and requires a fresh scientific identity. Prompts/objectives/update budgets and paired randomness are preserved. Supplied `configs/tasks/coco.yaml` remains valid fresh training with four workers, but cannot exactly continue. The standard HellaSwag default also remains supported fresh training, but its fixed-presentation loader cannot exactly continue. A later YAML edit cannot add a cursor to an existing run.

```bash
uv run --locked python train.py --config configs/operations/coco-resumable.yaml --check
uv run --locked python train.py --config configs/operations/hellaswag-resumable.yaml --check
uv run --locked python train.py --config configs/operations/coco-resumable.yaml --output runs/coco-case
uv run --locked python evaluate.py --run runs/coco-case
uv run --locked python train.py --config configs/operations/hellaswag-resumable.yaml --output runs/hellaswag-case
uv run --locked python evaluate.py --run runs/hellaswag-case
```

`--check` prints composed settings without weights/data/output creation. Actual training requires `--output`, which selects the exact logical root; existing paths cannot silently restart or overwrite. Bound `config.yaml` freezes complete effective settings even if authored YAML changes. Training artifacts live in `attempts/training-0001/`; `training.json` identifies the current attempt, while `checkpoint.json` identifies the eligible default and its own ancestry. COCO retains caption/CIDEr evaluation and batch16/accumulation4/500 updates. HellaSwag retains five-shot prompts, accuracy scoring and batch64/accumulation1/10,000 updates. These settings share effective batch64 but have different data and evaluation policies.

Continuation is a separate operation after the parent is finalized, using a checkpoint actually published by that parent:

```bash
uv run --locked python train.py --config configs/operations/coco-resumable.yaml --output runs/coco-case \
  --resume runs/coco-case/attempts/training-0001/step_000100.pt
uv run --locked python train.py --config configs/operations/hellaswag-resumable.yaml --output runs/hellaswag-case \
  --resume runs/hellaswag-case/attempts/training-0001/step_001000.pt
```

These paths are examples of scheduled saves, not claims they exist or remain nonterminal. A checkpoint at the completed maximum/early-stop boundary cannot add updates. Continuation receives an exclusive deterministic attempt name `resume-<parent-id>-<saved-step>`; repeat invocation rejects reuse. Restore includes optimizer/LR/scaler/RNG, counters, ordinary batch cursor and checkpoint-time best selection. Saving1000 then interrupting after unsaved1500 restores1000 and next executes1001. Source/runtime/config/input identity must match. A crashed `running` parent needs explicit stopped-worker/outcome reconciliation before continuation. See [ordinary records and exact support limits](ordinary-runs.md#exact-ordinary-continuation), including unsupported multi-worker/fixed-presentation loaders and metadata-only pending-publication recovery.

Default evaluation requires the recorded eligible best target, never a newest search or partial fallback. A previous eligible default can survive a partial child with its original ancestry. Explicit current-attempt checkpoints/overrides are supplemental, with fresh independent evaluation attempts:

```bash
uv run --locked python evaluate.py --run runs/coco-case --checkpoint step_000500.pt --expected-step 500 --attempt step500
uv run --locked python evaluate.py --run runs/hellaswag-case --checkpoint step_001000.pt --expected-step 1000 --attempt step1000
```

Historical timestamped engine directories retain their saved layout and iterator restart semantics. Current actual training requires an explicit output root; use the recorded route above for supported exact continuation. Old state cannot be silently upgraded.

## Recipe preparation, cases and recorded evaluation

[Retained original13](../configs/recipes/original13.yaml) declares thirteen locations, quota10,000 and nine independent width/ratio cases. [Changed](../configs/recipes/changed.yaml) declares ordered `[dec_l3, enc_l2, enc_emb]`, quota80 and one `width-32-ratio-50` case. Editing locations, rows/selection, supported optimizer/schedule, objective coefficients and saves in these existing YAMLs changes science without Python edits. Keep their complete observation/evaluation sections and declaring-file-relative task references. Functional KL remains additive to reconstruction; quota spans existing phases/reset semantics rather than applying anew to each phase. Native16/accumulation1 and supported HellaSwag semantics remain required; this does not introduce COCO shared-codec or arbitrary recipe channels/batching.

First acquire/register pinned bytes explicitly using [existing acquisition commands](experiment-recipes.md#acquire-and-prepare). Given a retained prepared receipt, registration reuses its existing assets without downloads or training:

```bash
uv run --locked python scripts/experiment_recipe.py --register-from-prepared /existing/retained/prepared.json \
  --model-name google/t5gemma-2-1b-1b --output assets/registrations/pinned.json
uv run --locked python train.py --config configs/recipes/original13.yaml --registrations assets/registrations --check
uv run --locked python train.py --config configs/recipes/changed.yaml --registrations assets/registrations --check
uv run --locked python train.py --config configs/recipes/original13.yaml --registrations assets/registrations --output runs/retained-comparison --prepare-only
uv run --locked python train.py --config configs/recipes/original13.yaml --output runs/retained-comparison --case width-512-ratio-75
uv run --locked python train.py --config configs/recipes/changed.yaml --registrations assets/registrations --output runs/changed-comparison --prepare-only
uv run --locked python train.py --config configs/recipes/changed.yaml --output runs/changed-comparison --case width-32-ratio-50
uv run --locked python evaluate.py --run runs/retained-comparison --case width-512-ratio-75
uv run --locked python evaluate.py --run runs/changed-comparison --case width-32-ratio-50
```

`--prepare-only` binds fresh small records/initializers/templates and complete effective configs from exact acquired registrations; it executes no training. Missing/conflicting identity fails without guessing latest. Existing bound roots require an explicit case for training. Independent local cases run sequentially; separately allocated jobs can run their own selected cases. These examples do not repeat or take ownership of the active frozen original13 campaign.

| Count | Retained original13 | Changed |
|---|---:|---:|
| Selected locations N / quota Q | 13 / 10,000 | 3 / 80 |
| Updates per case N×Q | 130,000 | 240 |
| Base-stream presentations Q×16 | 160,000 | 1,280 |
| Declared processed presentations N×Q×16 | 2,080,000 | 3,840 |
| Declared cases / comparison updates | 9 / 1,170,000 | 1 / 240 |

Selected eligible rows are separately observed during preparation: changed requests12; the retained measured workload had39,058 eligible rows. Neither number is an update/presentation budget. Three locations at quota10,000 declare30,000 updates versus130,000 for thirteen; totals sum independent cases, and actual processed exposures/updates remain engine records. Recipe `--check` requires existing registrations for pinned model topology (or a bound `--output` root); it reports resolved settings/derived declared counts without new acquisition or execution, not observed eligible rows.

Recipe continuation selects a path from the case's `checkpoints.json` and a fresh attempt, preserving pre/post-reset role:

```bash
uv run --locked python train.py --config configs/recipes/changed.yaml --output runs/changed-comparison \
  --case width-32-ratio-50 --resume runs/changed-comparison/cases/width-32-ratio-50/attempts/training-0001/engine/step_000120_pre_reset.pt --attempt 0002
```

After confirming a crashed previous worker stopped, `--reconcile-interrupted 0001` explicitly reconciles its running reservation. `--finalize-terminal` can publish a valid durable full-budget terminal with zero updates; nonterminal state cannot use it. Default recipe evaluation requires the own completed frozen terminal. Explicit `--scope development` needs a YAML-declared observation checkpoint/expected step; `--scope supplemental` uses frozen YAML supplemental conditions and a fresh `--attempt`. [Detailed recipe commands](experiment-recipes.md#public-recorded-evaluation) document exact syntax and obligations, including pending prior observations and retained failed/missing denominators.

## Finite reviewed graph and mock recovery

Existing `recipe-case-submission-v1` job YAML selects finite `cases` and `evaluation_cases`, resources, source SHA/path, registrations, fresh run/submission roots and attempt. `cases: null` selects all declared cases; `evaluation_cases: []` adds none. Preview displays exact commands, selected tasks, source/runtime/effective settings, dependencies, per-task resources, array caps and separate science totals. This public graph is recipe-specific; it makes no ordinary-task Slurm array promise. Use [the existing job/limits example](source-lifetime.md#reviewed-public-recipe-case-arrays), adjusted to real approved absolute paths and finite resources.

```bash
uv run --locked python scripts/submit_recipe.py action --job reviewed-cases.yaml --limits approved-limits.yaml --output reviews/new-comparison.json
uv run --locked python scripts/submit_recipe.py preview --bundle reviews/new-comparison.json
uv run --locked python scripts/submit_recipe.py check --bundle reviews/new-comparison.json
```

`action` without scheduler flags only prepares/previews. Source binding requires a clean full-SHA checkout and fixed locked environment held for every queued/running consumer; different concurrent revisions use separate pinned worktrees. Separate arrays use stable indices and `aftercorr` own-training gates; compatible combined allocations still record separate train/evaluation outcomes. Array tasks, allocated GPUs and optimizer updates are distinct.

For a reviewed one-group **local fixture** bundle, explicitly save fake scheduler responses using the existing payload shape. Mock submits no scheduler command; it still creates durable reservations/receipts, so use an isolated fresh fixture output:

```bash
python - <<'PY'
import json
from pathlib import Path
Path('fake-responses.json').write_text(json.dumps([
    {'returncode': 0, 'stdout': '900001\n', 'stderr': ''}
]))
PY
uv run --locked python scripts/submit_recipe.py submit --bundle reviews/local-fixture.json --mock-results fake-responses.json
```

Provide one response per selected group (two for separate train/evaluation arrays). The mock/live payload is the same; mock acceptance does not authorize live submission or establish real IDs. Intent and partial/unknown responses persist before/after adapter invocation. Unknown acceptance blocks duplicate retry. Recovery consumes retained exact evidence without querying/submitting:

```bash
uv run --locked python scripts/submit_recipe.py intents --bundle submission/bundle.json
uv run --locked python scripts/submit_recipe.py reconcile --bundle submission/bundle.json --evidence scheduler-evidence.json
uv run --locked python scripts/submit_recipe.py prepare-retry --bundle submission/bundle.json --output retry-bundle.json --retry-output /absolute/approved/root/retry-submission --attempt 0002
uv run --locked python scripts/submit_recipe.py authorize-retry --bundle submission/bundle.json --new-bundle retry-bundle.json --authorization retry-approval.json
uv run --locked python scripts/submit_recipe.py terminal --bundle submission/bundle.json
uv run --locked python scripts/submit_recipe.py release --bundle submission/bundle.json
```

Only proven unaccepted/stopped unattempted groups qualify for a separately approved recovery child. Accepted work never resubmits; every task needs durable terminal evidence before source release. [Recovery evidence/approval fields](source-lifetime.md#explicit-scheduler-reconciliation-and-recovery) retain authoritative-match/nonacceptance/conflict rules and ancestry. Preparation and retry authorization execute no live submission.

## Collect and record one scientific question

```bash
uv run --locked python scripts/collect_results.py --run runs/coco-case --snapshot collection-snapshots/coco-01 --output collected/coco-01
uv run --locked python scripts/collect_results.py --run runs/hellaswag-case --snapshot collection-snapshots/hellaswag-01 --output collected/hellaswag-01
uv run --locked python scripts/collect_results.py --run runs/changed-comparison --snapshot collection-snapshots/changed-01 --output collected/changed-01
```

Snapshots/destinations stay outside scientific roots. Repeat the same command for the same snapshot retry; new results use fresh names. Default collection retains final/intermediate tables, every recorded scalar row and small identity/status companions, with selected/copied/verified/reused/missing/failed accounting. Large raw/checkpoint/input bytes remain retrievable at exact references. [Collection details](result-collection.md) cover explicit selections, snapshot limits and separate incomplete exit status; this local action performs no network transfer.

Use one canonical local experiment ticket per scientific question, whether one job or several cases. For a singleton, record purpose, unchanged task YAML/source identity, exact root, budget/authority, training attempt, recorded checkpoint, evaluation attempt and collection receipt. For a comparison, add stable case IDs, each effective config/source/budget and retained operational attempts; a source/science/budget change is an explicitly linked revision/case, not an exact retry. A new question gets a new linked ticket. Link the existing `experiment.json`, `case.json`, `training.json`, `checkpoint.json`, `attempt.json`, recipe records and scheduler/collection receipts instead of duplicating a tracker/schema.

Update training, evaluation and collection outcomes separately. Keep failed/partial cases and pending observations visible with denominators. Reports/charts are optional; closure records the interpretation and explicit owner acceptance of any partial/deferred scope, without converting fixture checks or collection success into scientific success. Follow the local session guide for status/claim/map changes and the coordinator-owned HANDOFF for navigation. Offline/mock checks, future live scheduler acceptance, target/GPU fit and scientific conclusions are separate evidence.
