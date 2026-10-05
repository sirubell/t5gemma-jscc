# T5Gemma-2 JSCC research

A frozen T5Gemma-2 backbone sends transmitter representations through a trainable codec and channel. The project supports independent ordinary COCO captioning and HellaSwag selection, and a shared codec trained across ordered encoder/decoder sites. Decoder receiver memory crosses the communication boundary and counts toward traffic.

Use one public command with one YAML-selected operation per invocation:

```bash
uv run --locked python experiment.py --config configs/experiments/ordinary/coco.yaml
uv run --locked python experiment.py --config configs/experiments/ordinary/hellaswag.yaml
```

The supplied ordinary examples select `operation: check`. These commands compose and validate configuration without loading model/data, creating a run or downloading assets. To train, author a separate request selecting `operation: train` with an explicit fresh `execution.root` and `execution.attempt`. See [operations](docs/operations-workflow.md) for effect boundaries and examples of all operations.

| Workflow/task | Current consumer | Authored example |
|---|---|---|
| Ordinary COCO | `ordinary_runs.train_task` → `training.train`; caption/CIDEr evaluation | [COCO](configs/experiments/ordinary/coco.yaml) |
| Ordinary HellaSwag | Same ordinary engine with candidate scoring and training holdout | [HellaSwag](configs/experiments/ordinary/hellaswag.yaml) |
| Shared HellaSwag | `shared_runs` → `shared_training`, recorded terminal/panel evaluation | [Reference](configs/experiments/shared/shared_hellaswag_reference.yaml), [changed example](configs/experiments/shared/shared_hellaswag_example.yaml) |
| Shared COCO | Image-aware indexed views, clean targets and whole-panel caption/CIDEr scoring | [COCO check example](configs/experiments/shared/shared_coco_example.yaml) |

Shared checks require `execution.resolution.model_config` pointing to a local pinned backbone `config.json`. Shared acquisition attests existing complete local assets; registration binds exact cache/model/scorer receipts. Preparation and training are separate invocations. Missing assets fail with explicit requirements. No command implicitly downloads, resumes, retries, selects the latest checkpoint or trains an exported study.

Named model designs live in [configs/models](configs/README.md). Ordinary runtime and custom channels retain their existing consumers. Width-only authoring families share one design with an explicit `experiment.codec.bottleneck_dim`; q200/effective64, q400/local and native16-long remain separate protocols. A complete effective configuration is saved in each run. Declaring-file-relative paths make requests independent of shell cwd.

Ordinary runs preserve their own configured boundary, optional distinct receiver-memory codec, heldout selection and eligible best checkpoint. Shared runs preserve one canonical codec owner, deterministic paired views, ordered sites, durable state and their own frozen terminal. Shared support remains native batch16/accumulation1, direct-affine/native arithmetic, AWGN or identity, reconstruction followed by additive reconstruction+KL, and one optimizer/LR reset. Unsupported or inactive authored controls reject early. HellaSwag native source 2048/target 512 remain unchanged.

| Responsibility | Modules |
|---|---|
| Public request composition/operation dispatch | `experiment.py`, `jscc/experiment.py`, `config.py`, `shared_config.py` |
| Frozen backbone, codec/channel precision, task losses and numerical runtime | `models/split_model.py`, `models/codec.py`, `models/channel.py`, `models/precision.py`, `losses.py`, `runtime.py` |
| Ordinary training, continuation and scoring | `ordinary_runs.py`, `training.py`, `ordinary_continuation.py`, `data/`, `evaluation.py` |
| Shared lifecycle, clock, state and registration | `shared_runs.py`, `shared_training.py`, `shared_plan.py`, `shared_state.py`, `shared_assets.py`, `shared_preparation.py` |
| Indexed paired inputs, image/cache/processor/scorer binding, routing and clean targets | `shared_inputs.py`, `site_routing.py`, `reconstruction.py`, `data/coco.py`, `data/hellaswag.py` |
| Recorded checkpoints, whole-panel task scoring and observations | `shared_evaluation.py`, `panel_evaluation.py`, `observation_policy.py` |
| Isolated panel processes, device/output ownership, partial aggregates and linked retries | `panel_workers.py` |
| Root/attempt/source/submission/result records | `run_records.py`, `experiment_records.py`, `submission.py`, `source_lifetime.py`, `result_collection.py` |
| Immutable checkpoint eligibility, asset trust and scientific identity | `checkpoint_policy.py`, `asset_identity.py`, `asset_paths.py`, `identity_compatibility.py` |
| Explicit historical reading and useful diagnostics | `checkpoint_readers.py`, `sharing_admission.py` (saved admission references only), `diagnostic_scoring.py`, `gradient_diagnostics.py`, `qualification.py` |

Shared COCO binds indexed image/caption/demo/reference inventories, processor assets and scorer bytes; it retains image-aware clean targets and receiver memory. CIDEr scores each complete image/reference panel. Isolated parallel evaluation distributes whole site/condition panels for one named shared case, with one model load per worker and one aggregate writer. A fresh linked retry inherits verified completed panels while retaining failed/missing/pending/interrupted evidence. See [parallel controls](docs/operations-workflow.md#parallel-panels-of-one-shared-case). Tiny CPU/mock checks establish mechanics; full-weight GPU execution and scientific quality remain separate.

Study preview/export uses `operation: preview`, `execution.plan.kind: study`. Submission preview uses `kind: submission`; only an explicitly authorized `operation: submit` may invoke a scheduler. `scripts/submit_recipe.py` is CASE payload transport and reconciliation, not a second scientific authoring frontend. Slurm is optional. Collection is a separate operation and reports copied/verified/missing/failed status independently of science.

See [architecture](docs/architecture.md), [transmission vocabulary](CONTEXT.md), [running and Colab](docs/running.md), [ordinary roots](docs/ordinary-runs.md), [shared authoring](docs/experiment-recipes.md), [studies](docs/studies.md), [channels](docs/channel-integration.md), [source lifetime](docs/source-lifetime.md), [collection](docs/result-collection.md), and [validation](docs/validation.md). Historical protocol documents refer to their frozen source and immutable records. Refactor CPU/mock acceptance does not establish full-weight GPU fit, speed or scientific quality.
