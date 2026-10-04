# Shared codec recipe contracts

The YAML resolver and source-bound commands connect cached native HellaSwag rows,
ordered training, recoverable optimizer state, fixed observations and terminal
reporting. Preparation creates a small receipt; execution requires an explicit
source pin and fresh output. Legacy original13 packages and COCO commands remain
callable with their established contracts.

```bash
uv run --locked python scripts/experiment_recipe.py \
  --config configs/recipes/original13.yaml \
  --model-config /existing/pinned/backbone/config.json \
  --output /fresh/output/resolved.json
```

Use the `config.json` from the selected locally cached backbone. Resolution reads
its encoder/decoder layer counts and hidden dimensions (including T5Gemma2
`encoder.text_config`) without downloading or
loading weights. A revision recorded in metadata must match the task model pin.
The runtime loader must validate locations against the loaded backbone again;
a JSON file alone does not establish loaded runtime identity. Output is exclusive
creation: choose a fresh file for each resolution. References in recipe YAML are
relative to that YAML; references in its task YAML retain the existing resolver's
semantics. Supported recipe channels are AWGN and identity with empty constructor kwargs;
SNR endpoints must be finite and ordered. The resolved output contains the composed model, task, data, optimizer,
channel, schedule, save states, scoring requests and all derived counts.

`updates_per_location` is authoritative. For N ordered locations and Q updates
per location, native16/accumulation1 produces N×Q optimizer updates, Q×16 reused
base-stream presentations and N×Q×16 site presentations. Dataset selection counts
are independent. Every complete view traverses locations in order before the
stream advances. Repeated data content is allowed. Unsupported batch semantics,
independent declarations of derived budgets, invalid/duplicate boundaries, empty
selections and fractional phase/save quotas fail explicitly.

The retained example reuses `sharing_native16_long_b512.yaml` and its existing
`sharing_direct_no_norm_b512.yaml` design: one direct-affine body, no nested
memory codec, FiLM off. It resolves nine independent trajectories with 13 locations,
Q=10,000, widths 2304/1152/512 and reconstruction percentages 50/75/90. Each has
130,000 updates, 160,000 base-stream presentations and 2,080,000 site presentations.
Quarter-phase states plus initialization, post-reset and the functional 10%
observation give 11 checkpoint states. Pre/post reset remain distinct at one
update. Two 256-question observations preserve the chosen four sites and NN/−6
conditions. Encoder hidden nMSE is .1; decoder hidden/memory are .05 each;
functional training adds KL1.0. These values are example settings, not validator
literals. The integrated learner honors the resolved coefficients.

## Public training command and independent cases

Explicitly register acquired pinned model/cache/fixed-panel assets once using
the registration commands below. Routine training then needs only YAML, the
registration directory and an exact output root; preparation is automatic:

```bash
uv run --locked python train.py --config configs/recipes/changed.yaml --registrations assets/registrations --output runs/changed
uv run --locked python train.py --config configs/recipes/original13.yaml --registrations assets/registrations --output runs/comparison --case width-512-ratio-75
uv run --locked python train.py --config configs/recipes/changed.yaml --registrations assets/registrations --check
```

The first fresh invocation binds every declared width/ratio case, even when
selecting one. Case IDs are `width-<width>-ratio-<reconstruction-percent>`;
reordering YAML does not rename a case. A fresh invocation without `--case`
executes the declared cases sequentially on the one local device. Case selection
is an execution choice and does not approve a sweep or retry. Other cases remain
bound and unexecuted after a selected run or failure.

To prepare once for independently resource-assigned case jobs, bind without
model loading, then select a pending case in each job. Different case attempts
can run concurrently; each case excludes concurrent training of itself:

```bash
uv run --locked python train.py --config configs/recipes/original13.yaml --registrations assets/registrations --output runs/comparison --prepare-only
uv run --locked python train.py --config configs/recipes/original13.yaml --output runs/comparison --case width-512-ratio-75
uv run --locked python train.py --config configs/recipes/original13.yaml --output runs/comparison --case width-512-ratio-50
```

The bound root uses its saved prepared receipt and complete effective configs;
subsequent edits to authored YAML have no effect. A changed experiment needs a
fresh root. Existing roots require one explicit case and cannot repeat all cases
implicitly. Preparation/attempt failures preserve partial files, other cases and
prior attempts. A process killed without cleanup can retain a `running` attempt and its
`.training-active` reservation. After confirming the prior worker has stopped,
add `--reconcile-interrupted 0001` to the explicit continuation command below.
That invocation asserts the prior worker will make no further writes, retains its
checkpoint inventory and reconciliation receipt, marks it interrupted and releases
its reservation. It never infers liveness or steals an active worker automatically.

Exact continuation uses a checkpoint path published in that case's attempt
`checkpoints.json`, with its recorded hash and size. No newest checkpoint is
selected. Use a new explicit attempt name; a completed case is evaluated rather
than trained again:

```bash
uv run --locked python train.py --config configs/recipes/changed.yaml --output runs/changed --case width-32-ratio-50 --resume runs/changed/cases/width-32-ratio-50/attempts/training-0001/engine/step_000120_pre_reset.pt --attempt 0002
```

An interruption after a durable terminal save can be finalized with the same
explicit checkpoint/attempt syntax plus `--finalize-terminal`. It restores and
validates the complete full-budget state, publishes a new terminal/freeze record
with **zero optimizer updates**, and retains prior observation obligations
separately. Nonterminal checkpoints cannot be finalized; terminal checkpoints
cannot start additional training. Prior attempts and checkpoint bytes are retained.

This restores the existing recipe engine's optimizer/LR/scaler/RNG and indexed
row/location/phase state. A pre-reset checkpoint resets before the next update;
a post-reset checkpoint does not reset twice. Earlier observation obligations
remain separately recorded. `--timeout-seconds` sets a per-case execution deadline
(default 86,400 seconds). Portable standalone CUDA/Colab runs require one visible
BF16-capable GPU; scheduler callers can additionally pass `--device-uuid` to
verify their exact `CUDA_VISIBLE_DEVICES` assignment. CPU execution is restricted
to explicitly registered synthetic fixtures. This contract grants no scheduler
or experiment execution authority.

Root record `recipe-run.json` (`recipe-run-v1`) publishes stable case paths,
width/ratio, preparation reference/hash, per-case counts and comparison totals.
Each `cases/<id>` contains ordinary `experiment-run-v1` case records and bound
`config.yaml`, plus `recipe.json`, `counts.json` and `data_selection.json`.
`attempts/training-<name>/attempt.json` retains status/request/source/runtime;
its fresh `engine/` contains the existing campaign, metrics, full checkpoint and
freeze records. `training.json` records both attempt and engine paths.
`checkpoints.json` lists durable checkpoint path/hash/size/step/role, including
pre/post reset separately. `checkpoint.json` is published only after completed
training and identifies the exact `recipe-terminal` policy target and engine.
Evaluation consumers use that engine and the prepared fixed terminal template;
recipe evaluation through `evaluate.py` is a separate integration release.

Counts distinguish actual selected rows, `Q × 16` base presentations and
`N × Q × 16` processed/site presentations, with `N × Q` optimizer updates.
Comparison selected-row totals sum per-case row counts, including reused rows;
they are not distinct dataset rows. Actual completed/segment updates and exposures
are recorded by the engine independently of declared budgets. Three versus
thirteen locations at Q=10,000 declare 30,000 versus 130,000 updates per case;
the retained nine cases declare 1,170,000 updates and 18,720,000 processed
presentations. Supported native16/accumulation1, additive objectives, phase/reset,
family exclusions and pinned noise/initializer policies remain with the existing
recipe engine. Unsupported COCO shared-codec, custom recipe channels and batching
fail explicitly; ordinary COCO/HellaSwag and their custom channels remain supported.

## Acquire and prepare

Acquire the existing model once, after obtaining any required target authority:

```bash
uv run --locked python scripts/experiment_recipe.py --acquire-model \
  --model-path /existing/pinned/backbone --model-revision EXACT_MODEL_REVISION \
  --output /fresh/model-binding.json
```

Model acquisition hashes the consumed files once. Explicitly acquire existing
Arrow/tokenizer bytes with `bind_recipe_data_cache`, then register the model and
cache receipts and acquired fixed scoring panels. Registration is an explicit
local operation; it never downloads or trains. The panel acquisition defines
ordered validation IDs, source families, prompt, reference corpus, noise,
precision/backend and native layout. Input checksums are computed automatically:

```bash
mkdir /existing/recipe-registrations
uv run --locked python scripts/experiment_recipe.py --register-assets \
  --model-name google/t5gemma-2-1b-1b \
  --model-binding /existing/model-binding.json --cache /existing/cache.json \
  --observation-template /existing/fixed-observation.json \
  --terminal-template /existing/fixed-terminal.json \
  --output /existing/recipe-registrations/pinned.json
```

`--model-name` must equal the task YAML's exact model name; the example uses the retained
recipe model name. Model and dataset revisions come from the
checksum-bound acquisition receipts. For a retained `recipe-prepared-v1` receipt,
registration can instead reuse its existing assets:

```bash
uv run --locked python scripts/experiment_recipe.py \
  --register-from-prepared /existing/retained/prepared.json \
  --model-name EXACT_TASK_MODEL_NAME \
  --output /existing/recipe-registrations/pinned.json
```

Thereafter, edit only the supported task/recipe YAML and prepare a fresh owned
directory. No handwritten assets JSON, initializer file, tensor package per
update or new model acquisition is required between recipes:

```bash
uv run --locked python scripts/experiment_recipe.py \
  --config configs/recipes/changed.yaml \
  --registrations /existing/recipe-registrations --output /fresh/preparation
uv run --locked python scripts/original13_training.py check \
  --prepared /fresh/preparation/prepared.json --prepared-sha256 EXACT_SHA256
```

Registration discovery requires one exact model-name/revision and dataset-name/
revision match. Duplicate identical registrations are equivalent; conflicting
matches fail and require a directory containing the intended registration. No
latest revision is selected. A missing acquisition fails with an explicit
registration instruction. Normal preparation and check make no Hub calls.

Preparation creates complete effective plans, fixed scoring templates,
`prepared.json` and a small timing receipt. Width/seed initializers are generated
once under the registration directory's `.initializers/`, keyed by codec design,
seed, codec implementation and Torch version. Later preparations reuse these
immutable tensors using checksum-bound acquisition receipts and stat checks. Keep this directory with its
registrations and prepared consumers. A concurrent first-use caller waits up to ten seconds for an atomically
published acquisition receipt; an incomplete reservation then fails explicitly.
Retain it and reconcile before retry. Output
directories are exclusive and a failed operation may retain partial evidence.

Smaller YAML panels use the acquired ordered prefix while preserving the full
cache's family exclusions and retained training pool. A larger panel or different
few-shot/scoring policy requires explicit matching acquisition; noise, prompt,
reference and layout are preserved. Ordered row selection and presentation
budgets still use the existing cached-data loader; no per-update package is made.

Production model paths use pinned HF snapshots. Links to files directly in the
same HF repository's `blobs` directory are equivalent to snapshot files; links
outside it, traversal and unexpected indirection fail. The approved `/work` to
`/raid/work` alias is canonicalized at the asset seam. New model/cache receipts
bind target path, device/inode, size and nanosecond modification/change times.
Changed content or link targets fail cheap checks and require reacquisition.
An explicit model audit uses `verify_model_assets(..., verify_bytes=True)`;
reacquire a cache with `bind_recipe_data_cache` after replacement/transfer.
Legacy receipts without stat bindings retain their prior presence/size policy;
reacquire them to gain the stronger checks. Preserve frozen source and receipts.

The callable consumer contract is
`register_recipe_assets(...)` or `register_prepared_assets(...)`, followed by
`prepare_recipe_assets(recipe_path, registrations=..., output=...,
study_pairing_id=...)`, returning the existing `recipe-prepared-v1` manifest.
`discover_recipe_assets` performs exact selection. Consumers use the resulting
`prepared.json` checksum and existing loader; they must preserve registration,
initializer and cache lifetimes. Automatic preparation does not wire public
train/evaluate dispatch or scheduler preflight; those consumers integrate this
interface separately. Existing `--assets` and low-level binding APIs remain
available for already prepared callers.

## Train, resume and evaluate

```bash
uv run --locked python scripts/original13_training.py train \
  --prepared /fresh/prepared.json --prepared-sha256 EXACT_SHA256 \
  --recipe configs/recipes/changed.yaml --width 32 --ratio 50 \
  --approved-source FULL_ACCEPTED_SOURCE_SHA --timeout-seconds 7200 \
  --device-uuid GPU-ASSIGNED_UUID --output /fresh/training
uv run --locked python scripts/original13_training.py resume \
  --prepared /fresh/prepared.json --prepared-sha256 EXACT_SHA256 \
  --recipe configs/recipes/changed.yaml --width 32 --ratio 50 \
  --approved-source FULL_ACCEPTED_SOURCE_SHA --timeout-seconds 7200 \
  --device-uuid GPU-ASSIGNED_UUID --output /fresh/continued \
  --resume-checkpoint /existing/state.pt --resume-sha256 EXACT_SHA256 --resume-size EXACT_BYTES
uv run --locked python scripts/original13_evaluation.py request \
  --training-run /fresh/training --template /existing/fixed-terminal.json \
  --template-sha256 EXACT_SHA256 --scope initial --output /fresh/request.json
uv run --locked python scripts/original13_evaluation.py run \
  --request /fresh/request.json --request-sha256 EXACT_SHA256 \
  --prepared /fresh/prepared.json --prepared-sha256 EXACT_SHA256 \
  --approved-source FULL_ACCEPTED_SOURCE_SHA --timeout-seconds 7200 \
  --device-uuid GPU-ASSIGNED_UUID --output /fresh/evaluation
uv run --locked python scripts/original13_evaluation.py report \
  --training-run /fresh/training --initial /fresh/evaluation \
  --observation /fresh/training/EXPLICIT_OBSERVATION_DIRECTORY --output /fresh/report.json
```

The optional `--recipe` resolves changed settings against the same acquired assets;
without it, execution uses the prepared resolution. Routine subset/seed/quota,
location/order, objective, optimizer/save/channel changes require YAML only.
Resume requires identical source/config/data/initializer/fixed observations and
full Adam/LR/scaler/RNG/runtime state; a changed recipe is a new run. Both phases,
partial location cycles and pre/post-reset checkpoints retain their exact cursor.
The runner binds each fresh cached batch once and reuses it across locations.
Identity and AWGN idle state adapters are supported. Observation callbacks restore
training state in finally; only typed scoring failures can continue after verified
restoration. Save, nonfinite, checkpoint or restoration failures stop the segment.

Terminal requests select this run's one durable terminal, including an explicit
supplemental request when desired. The generalized report validates
`recipe-training-event-v1`, actual YAML coefficients/LR/counts and supplied
observation receipts. Ancestor obligations remain pending until explicitly
completed; supplied receipts do not silently rewrite them. Legacy events retain
their separate reader and frozen package validation.

## One submission command

```bash
uv run --locked python scripts/submit_recipe.py prepare \
  --job /existing/job.yaml --limits /approved/limits.yaml --output /fresh/bundle.json
uv run --locked python scripts/submit_recipe.py check --bundle /fresh/bundle.json
uv run --locked python scripts/submit_recipe.py render \
  --bundle /fresh/bundle.json --output /fresh/job.sbatch
uv run --locked python scripts/submit_recipe.py submit \
  --bundle /fresh/bundle.json --mock-results /existing/mock-results.json
```

The job schema is `shared-codec-submission-v1`: recipe path, root, full source
commit/path, model path/config digest, prepared digest/candidate paths, Python,
cache, fresh output, resources and array groups. Limits separately pin root,
partition/account/qos/node and bound GPU/CPU/memory/walltime/concurrency/task count.
Every trajectory occurs once; dependencies refer to earlier groups. `/work`
canonicalizes to `/raid/work`; lexical escapes and changed host symlinks fail.
Preparation accepts generalized small receipts and retains legacy exact-compatible
packages. Launch uses the persisted resolved bundle through `--resolved`, so a
later YAML edit cannot alter allocated work. Mock and live modes share payload,
reservation, arguments and durable attempted-command receipts. Uncertain outcomes
retain the reservation and never retry automatically. Live submit additionally
requires exact human authorization for bundle/source/target/operation; these
commands and resource limits provide no such authorization.

## Cost and acceptance limits

Acquisition hashes bytes once. Startup checks checksums for small receipts and
required asset paths/sizes/stat signatures; replacing an immutable cache or
model/data file requires reacquisition or explicit byte audit. New target copies need target
acquisition evidence. Training tokenizes selected rows once, builds Q actual
BatchViews once and assembles one fresh padded batch per view/location cycle.
Q view binding remains a setup cost; no full per-update tensor package is built.
Evaluation currently repeats selected-row loading/view binding for its source/data
check. Initializer acquisition and checkpoint integrity remain full-byte trust boundaries.

Offline tiny CPU and mocked scheduler evidence establish mechanics, not pretrained
model quality, target memory fit, GPU/backend equivalence or the ten-minute live
preparation/submission target. Queue wait and training are outside that target.
COCO remains on its existing loader and commands. No live deployment, scheduler
or scientific execution is authorized by preparation or offline acceptance.

Measured offline acceptance at runtime source `5bbd1db7e466a4d8eb8402f960450e861bd54af1`
uses a saved 26-layer tiny CPU backbone, a 40-row synthetic Arrow fixture and a
fast tokenizer. These single warm local measurements compare different workloads,
not a before/after training speedup:

| Operation | Workload | Observed wall time |
|---|---|---|
| Initial model acquisition | One tiny weight file, hashed once | 1.58–2.16 ms |
| Fresh prepared receipt | Reused model binding; no weight hashes | 78.4–87.1 ms |
| Cached rows and views | 12/20 selected rows, Q8 native16 views | 22.6–25.0 ms |
| Retained training | 520 real updates, 11 states; two fixed256 scorer fixtures | 15.77 s |
| Changed training | 24 real updates plus native fixed2 observations | 2.28–2.54 s |
| Continuation | Pre/post reset or functional checkpoint | 0.65–1.87 s |
| Native terminal evaluation | 3 questions × 3 sites × NN | 1.26–1.29 s |
| Mock submission | One resource-bound array group | 10.8 ms |

The dedicated rebind case acquires once, prepares twice and records zero subsequent
weight-file hashes. Real large-model acquisition/loading, full-pool tokenization,
server filesystem costs and live submission remain unmeasured. The retained256
fixture proves lifecycle/count parity; native tiny scoring and the retained
legacy real-scorer regression provide separate mechanical evidence.
