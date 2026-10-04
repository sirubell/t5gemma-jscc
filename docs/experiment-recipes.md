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

## Acquire and prepare

Acquire the existing model once, after obtaining any required target authority:

```bash
uv run --locked python scripts/experiment_recipe.py --acquire-model \
  --model-path /existing/pinned/backbone --model-revision EXACT_MODEL_REVISION \
  --output /fresh/model-binding.json
```

This is the full-byte acquisition boundary. Reuse its checksum-bound JSON for
new source/recipe prepared receipts. Replacement, transfer or explicit byte audit
requires new acquisition evidence; a source receipt never repeats weight hashing.
Acquire immutable dataset/tokenizer assets once with
`bind_recipe_data_cache`; bind existing initializer tensor references and fixed
scoring templates. These are Python acquisition APIs for new assets, not edits
required between recipes. A small `assets.json` supplies:

```json
{
  "model_path": "/existing/pinned/backbone",
  "model_binding": {"path": "/existing/model-binding.json", "sha256": "EXACT_SHA256"},
  "cache_manifest": "/existing/cache.json",
  "cache_sha256": "EXACT_SHA256",
  "initializations": {
    "512": {"path": "/existing/initial-B512.pt", "sha256": "EXACT_SHA256",
            "state_identity": "EXACT_STATE_IDENTITY", "seed": 0}
  },
  "observation_template": {"path": "/existing/fixed-observation.json", "sha256": "EXACT_SHA256"},
  "terminal_template": {"path": "/existing/fixed-terminal.json", "sha256": "EXACT_SHA256"},
  "study_pairing_id": "EXPLICIT_PAIRED_STUDY"
}
```

Every requested width requires its own matching initializer. Production assets
use the pinned HF snapshot path. Templates contain task, ordered validation IDs,
source families, settings, noise, data settings, scorer, precision, backend,
reference corpus, prompt policy and layout. Their membership must exactly match
cache `development` and `final_validation`; changed panel sizes need separately
acquired templates and cache receipts. No implicit resize occurs.

```bash
uv run --locked python scripts/experiment_recipe.py \
  --config configs/recipes/changed.yaml \
  --model-config /existing/pinned/backbone/config.json \
  --assets /existing/assets.json --output /fresh/prepared.json
uv run --locked python scripts/original13_training.py check \
  --prepared /fresh/prepared.json --prepared-sha256 EXACT_SHA256
```

`recipe-prepared-v1` binds the integrated source inventory, model acquisition,
cache checksum, initializers, templates and complete effective YAML resolution.
Operational model paths and acquired asset bindings are saved in the effective
config and included in its identity. A new source needs a fresh small prepared
receipt; it reuses unchanged model/cache/initializer bytes. Frozen source and
historical receipts are never relabeled. Check validates small receipts, required asset presence/sizes and initializer
bytes. It does not load backbone weights or assert target runtime acceptance.

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
required asset paths/sizes; replacing an immutable cache or same-size model/data
file requires rebind or explicit byte audit. New target copies need target
acquisition evidence. Training tokenizes selected rows once, builds Q actual
BatchViews once and assembles one fresh padded batch per view/location cycle.
Q view binding remains a setup cost; no full per-update tensor package is built.
Evaluation currently repeats selected-row loading/view binding for its source/data
check. Initializer and checkpoint integrity remain full-byte trust boundaries.

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
