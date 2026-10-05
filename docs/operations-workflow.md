# One-operation YAML workflow

Invoke `uv run --locked python experiment.py --config FILE`. Every new public request has `schema: jscc-experiment-v1`, `workflow: ordinary|shared`, exactly one `operation`, and an `execution` mapping. Authored scientific settings go in `experiment`; requests for an existing bound root may omit science. If supplied, science must match the binding. All relative paths resolve from their declaring file. A root, case, attempt or checkpoint is always explicit; there is no latest lookup or implicit retry.

| Operation | Effects and owned fields |
|---|---|
| check | Compose/describe only; optional `resolution.output` exclusively writes resolved JSON. Shared `resolution.model_config` reads pinned local topology, not tensors. A bound root supplies immutable science. |
| acquire | `acquisition: {model_path, model_revision, output, synthetic_cpu}` attests complete existing local model bytes to fresh JSON. Shared `kind: coco-cache` binds existing dataset/images/processor/demo/reference/scorer assets as described below. No download or branch resolution. |
| register | `registration` owns exact `model_name`, fresh `output`, and either checksum-bound `references` or `from_prepared`. It preserves existing acquisition identity. |
| prepare | Authored science plus `preparation`; reserve/bind a fresh root or standalone prepared destination. No training. Shared chooses `registrations` or `assets` plus `model_config`; ordinary uses an empty preparation and explicit root. |
| train | Explicit root and new attempt. Fresh ordinary science is composed once; exact ordinary preparation can be consumed once. Shared uses an already bound root or exact train-only `prepared: {path, sha256}`. No implicit registration/acquisition. |
| resume | Bound root, new attempt and `continuation` exact durable checkpoint/role/step/parent/size/hash. Shared requires one case and may explicitly reconcile a stopped parent or finalize terminal. No unsaved counters are borrowed. |
| evaluate | Bound root, new attempt and optional exact `evaluation` checkpoint/expected_step/overrides/scope. Ordinary defaults to recorded eligible best; shared to own frozen terminal. Development shared evaluation needs its declared checkpoint. |
| collect | Explicit root and `collection` destination/snapshot/selection. Copies verified small artifacts; raw inputs and checkpoints stay referenced. Incomplete collection exits nonzero with receipts. |
| preview | `plan.kind: study` expands/optionally exports; `kind: submission` binds/renders a CASE plan. Neither invokes a scheduler. Study is a plan kind, not an operation. |
| submit | Existing CASE plan, limits, fresh exact receipt directory and source-bound authorization. Uses existing scheduler/dependency/resource/source-consumer guards. A request file is not live launch authorization. |

Unknown, inactive, conflicting and unsupported fields reject before their effects. Acquire/register/collect/preview/submit reject authored science. Check rejects preparation/acquisition/registration. Prepare selects exactly one destination and one assets path. No command selects a scientific recipe, resource budget or report deadline for you.

A fresh ordinary request copies the scientific `experiment` from the supplied COCO or HellaSwag check example and changes only its operation/execution:

```yaml
schema: jscc-experiment-v1
workflow: ordinary
operation: train
execution:
  root: ../runs/NEW-ordinary
  attempt: training-NEW
  success_link: ../links/NEW-ordinary.txt
experiment:
  # Full task/model/data/training/evaluation authoring from the task example.
```

The comment above is a shape illustration, not a runnable partial experiment. Use complete science. `success_link` is optional and must be fresh and disjoint from the root in both ancestry directions. It publishes only after the operation returns; receipt publication failure does not rerun completed training.

Bound ordinary evaluation does not reread mutable training YAML:

```yaml
schema: jscc-experiment-v1
workflow: ordinary
operation: evaluate
execution:
  root: ../runs/RECORDED-ordinary
  attempt: evaluation-NEW
  evaluation: {}
```

Set `evaluation.checkpoint` to a recorded filename and `expected_step` for an exact supplemental target. An `overrides` file contains evaluation controls only. COCO generation batch/max tokens differ from HellaSwag candidate batch/fewshot/scoring policy; they are not interchangeable. Ordinary configured custom channels still reach the model/channel consumers.

For shared COCO cache acquisition, `execution.acquisition` declares `kind: coco-cache`, `dataset_path`, `assets`, `dataset_name`, `dataset_revision`, `model_revision`, `karpathy_path`, `karpathy_name`, `karpathy_revision`, `fixed_data_ids` and fresh `output`. Paths identify existing local Arrow/image and processor assets plus the exact Karpathy membership; `fixed_data_ids` requires unique disjoint `train_ids`, `demo_ids`, `selection_ids` and `report_ids`; optional `development` and `final_validation` each carry matching `input_ids` and `source_family_ids`. Selection/report IDs belong to Karpathy validation; demos belong to restval; train/demo IDs exclude validation/test. The acquired receipt binds complete image/reference/demo inventories and local PTB/CIDEr scorer assets. It is separate from model acquisition and does not train or download. Registration/preparation consume its exact checksum-bound cache reference. COCO caption policy/seed and image-aware targets stay in effective science; no HellaSwag token-only fixture stands in for multimodal acceptance.

A shared request uses science from the shared example. Check requires local `resolution.model_config`; prepare selects one explicit acquisition path:

```yaml
schema: jscc-experiment-v1
workflow: shared
operation: prepare
execution:
  root: ../runs/NEW-shared
  preparation:
    registrations: ../acquired/EXACT-registrations
experiment:
  # Full shared task_config, locations, quotas, widths, phases, objectives and panels.
```

Standalone preparation uses `preparation.output` instead of root. Explicit assets use `preparation.assets` and `model_config`; the assets JSON owns its reference paths, cache hashes, initializers and fixed templates. Registration accepts `references` with model_binding/cache/observation_template/terminal_template `{path, sha256}` records and explicit `synthetic_cpu`, or an exact `from_prepared` path without new references.

After preparation, shared training and terminal evaluation are separate calls:

```yaml
schema: jscc-experiment-v1
workflow: shared
operation: train
execution:
  root: ../runs/RECORDED-shared
  case: width-RECORDED-ratio-RECORDED
  attempt: training-NEW
```

Use `cases` for an explicit unique set, or omit selection to execute declared cases. Resume requires one case. Shared evaluation uses `operation: evaluate`, a new attempt, and `evaluation: {scope: initial}`. Completed/failed/missing/pending denominators remain visible; incomplete evaluation cannot pass final eligibility. Shared COCO uses acquired image/reference/demo/scorer identities. A single named shared COCO or HellaSwag case can distribute whole site/condition panels to independent processes through `evaluation.parallel`.

```yaml
schema: jscc-experiment-v1
workflow: shared
operation: resume
execution:
  root: ../runs/RECORDED-shared
  case: RECORDED_CASE
  attempt: resume-NEW
  continuation:
    path: ../runs/RECORDED-shared/cases/RECORDED_CASE/attempts/training-PARENT/engine/checkpoint.pt
    parent_attempt: ../runs/RECORDED-shared/cases/RECORDED_CASE/attempts/training-PARENT
    role: RECORDED_ROLE
    step: 1 # Replace with recorded positive step.
    size_bytes: 1 # Replace with recorded byte size.
    sha256: RECORDED_SHA256
```

The checkpoint may be inside the engine subdirectory; parent_attempt identifies its owning durable training attempt. Training compatibility still verifies source/runtime/config and complete optimizer/LR/scaler/RNG/view/phase state. New source cannot masquerade as an old source-compatible resume.

Study preview/export is nonexecuting:

```yaml
schema: jscc-experiment-v1
workflow: ordinary
operation: preview
execution:
  plan:
    kind: study
    path: ../configs/studies/baseline.yaml
    output: ../prepared/NEW-study
```

Exports contain complete one-operation train requests and a manifest with task/run links. Optional Slurm study transport generates fresh evaluation requests from the exact recorded training link. It keeps fixed-step and distinct codec/vanilla attempts. `scripts/slurm_job.sh` accepts explicit requests; it does not infer a train/evaluate sequence.

Submission preview uses `plan: {kind: submission, path: CASE_PLAN}` and `submission: {limits: LIMITS, bundle_output: FRESH_JSON}`. Submit uses `submission: {plan: CASE_PLAN, limits: LIMITS, bundle_output: FRESH_RECEIPT_DIR/bundle.json, receipt_dir: FRESH_RECEIPT_DIR, authorization: EXACT_APPROVAL_MAPPING}`. The existing CASE source/runtime/resource/array/dependency guards remain authoritative. Generated train/evaluate requests preserve root/case/attempt and live in the bundle's owned requests directory. Old queued bundles keep their original source and argv identity.

Collection remains separate from training and evaluation:

```yaml
schema: jscc-experiment-v1
workflow: shared
operation: collect
execution:
  root: ../runs/RECORDED
  collection:
    snapshot: ../snapshots/NEW
    destination: ../results/RECORDED
    select: []
```

An exact retry reuses the same snapshot/root/selection/retrieval identity. Missing or failed copies remain incomplete; scientific completion is retained independently.

## Parallel panels of one shared case

```yaml
schema: jscc-experiment-v1
workflow: shared
operation: evaluate
execution:
  root: ../runs/RECORDED-shared
  case: RECORDED_CASE
  attempt: evaluation-NEW
  evaluation:
    scope: initial
    parallel:
      backend: local
      workers: 2
      devices: [cpu, cpu]
```

Repeated CPU entries identify separate processes and require registered synthetic CPU assets. CUDA entries use distinct visible device indices, for example `cuda:0`; each child retains one assigned device and loads one runtime/model and one whole-panel data view. The current CASE resource admission allows one GPU, so its parallel CUDA request admits only `cuda:0`. Worker count is an execution choice, separate from scientific item counts and scheduler array concurrency. Parallel execution requires `case`, rejects `cases`, overrides and a parent `device_uuid`, and keeps the recorded checkpoint, source, runtime, input, scorer and noise policy. Both tasks score their entire ordered panel inventory; COCO never averages CIDEr image shards.

A parent process alone publishes live `aggregate.json` and immutable snapshots below the evaluation attempt's `panels/worker-attempts/ATTEMPT`. Worker records preserve started, completed, failed, missing, pending and interrupted outcomes. The standard panel receipt retains full task metrics and artifact identities. Final eligibility requires every panel complete and the required training observations verified complete. A complete panel set cannot complete pending training observations. Recorded noise policy identity is distinct from sampled noise audit evidence; CPU tests verify actual paired draws on tiny task bodies. CUDA timing and speedup require separate execution evidence.

For a stopped partial/failed/interrupted evaluation, use a fresh public request with a new `execution.attempt` and `evaluation.parallel.parent_attempt` naming the exact prior `panels/worker-attempts/ATTEMPT` directory. Its registered terminal snapshot and evidence inventory must still match. Only complete panels are inherited; the parent failures and bytes remain retained, and unfinished panels get new worker outputs. A different case, checkpoint, input or scoring policy needs its own evaluation plan.

A CASE job may supply `evaluation: {scope: initial, parallel: ...}`. It retains one evaluation task for each selected original case index and the existing `aftercorr` training dependency. Panel indices never become scheduler task indices. A retry parent in a CASE job requires `combined: false` and exactly one evaluation case. Scheduler recovery still applies only to explicitly eligible unaccepted/unattempted groups; retrying accepted failed scientific execution uses a separate fresh evaluation request. Source holds remain subject to the existing terminal/release authority.
