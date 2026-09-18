# T5Gemma-2 JSCC

A research project for studying how compression and wireless channel noise affect T5Gemma-2 representations and downstream task performance. It supports COCO image captioning and HellaSwag multiple-choice evaluation with a frozen backbone and a trainable residual codec.

Both tasks share the training loop, model design and codec implementation. A shared model YAML controls the split, bottleneck size, LayerNorm, SNR conditioning and channel; task YAMLs define data, training and evaluation. Training uses KL distillation plus normalized reconstruction MSE.

```text
hidden representation → codec encoder → power normalization → channel
                      → codec decoder → downstream model computation
```

For a decoder split, receiver layers obtain encoder memory through a second codec/channel transmission. Transmitter layers retain original memory. See [architecture](docs/architecture.md) for signal paths and [terminology](CONTEXT.md) for the system definition.

## Setup

Use Python 3.13+ and uv. The research configurations use CUDA / bfloat16. COCO CIDEr scoring also requires Java. Model access and Hugging Face caches must be available on the machine that runs the experiment.

The model code is ordinary PyTorch/CUDA, not an H200-specific implementation. A standalone RTX 5090 can use the same Python entry points without Slurm; its smaller memory budget may require smaller training/evaluation batches. See [the workstation guide](docs/running.md#standalone-gpu-workstations-including-rtx-5090) for compatibility checks and a conservative starting setup.

From the project root:

```bash
uv sync --locked

uv run --locked python train.py --config configs/tasks/coco.yaml --check
uv run --locked python train.py --config configs/tasks/hellaswag.yaml --check
```

The check commands parse and print the configuration without loading weights or data. Model weights and datasets are downloaded on first use when online, then reused from the local cache. See [running on other machines](docs/running.md) for environment setup and Slurm.

## Train an experiment

The configuration files have distinct responsibilities:

| File | Contents |
|---|---|
| `configs/model.yaml` | Shared backbone, split, codec and channel design; FiLM off |
| `configs/tasks/coco.yaml` | COCO data, training, evaluation and a reference to model.yaml |
| `configs/tasks/hellaswag.yaml` | HellaSwag data, training, evaluation and the same reference |
| `configs/tasks/hellaswag_diagnostic.yaml` | Five-setting short HellaSwag diagnostic recipe |
| `configs/smoke/` | Small CPU/H200 task budgets; the same shared model design |
| `configs/evaluation/` | Evaluation-only overrides for small checks or custom channels |
| `configs/studies/` | Multi-experiment plans shared by both tasks |

Both task files contain `model_config: ../model.yaml`. The reference resolves relative to the task YAML, so copying a task into another directory requires adjusting that reference. There is one model/task composition level, not a recursive inheritance chain. Run either task with the same entry point:

```bash
uv run --locked python train.py --config configs/tasks/coco.yaml
uv run --locked python train.py --config configs/tasks/hellaswag.yaml
```

Edit model.yaml to change the common design for both tasks. For a separate model experiment, copy it to a named model file and point that experiment's task YAML at the copy. Keep model/split/codec/channel sections in the model file rather than overriding them inside a task file.

CPU smoke files use `runtime: {device: cpu, dtype: float32}` to override only execution settings. After loading, the run receives a complete resolved configuration with no dependency on the source YAML files. Legacy flat configurations remain readable.

## Plan multiple experiments

Studies list task recipes, seeds and named model overrides. Both tasks use the same variation list. Previewing is the default and does not load a model or submit jobs:

```bash
uv run --locked python study.py --config configs/studies/baseline.yaml
uv run --locked python study.py --config configs/studies/splits.yaml --task coco
uv run --locked python study.py --config configs/studies/bottleneck.yaml
```

Use `--output runs/studies/NEW_NAME` to write complete YAMLs and a train/eval index manifest. Export uses a new directory and never starts training. Prepared plans can be moved to the execution host before use. See [config layout](configs/README.md) and [study execution](docs/studies.md) for counts, paths and separate Slurm train/eval arrays.

| Setting | Meaning |
|---|---|
| `run.name`, `run.output_dir` | Experiment name and output directory; relative output paths resolve from the YAML directory |
| `split.stack`, `split.where`, `split.index` | Encoder/decoder insertion point; layer indices are zero-based |
| `codec.bottleneck_dim` | Dimension of the transmitted representation |
| `codec.layernorm` | `none`, `pre`, `post`, or `both` |
| `codec.snr_film` | Receiver-side SNR conditioning |
| `channel.type`, `channel.kwargs` | Built-in AWGN/identity or a custom Python module |
| `training.batch_size`, `gradient_accumulation` | Microbatch size and number of accumulated microbatches |
| `training.max_steps`, `schedule_steps`, `eval_every` | Stop budget, optional LR schedule horizon and validation interval |
| `training.monitor`, `patience` | Checkpoint metric and early stopping; null patience disables early stopping |
| `training.save_steps` | Additional checkpoint steps |
| `training.max_minutes` | Optional training time budget, checked between updates; reserve time for final validation and evaluation |

Batch size 16 with accumulation 2 gives an effective batch size of 32. A 6,000-step run performs 12,000 microbatches.

External LayerNorm `pre` is before codec encoding, and `post` is after decoding; residual blocks retain their internal LayerNorm. Both tasks now start from the same encoder-final-norm split and residual codec with external LayerNorm `both`. A decoder experiment may add `codec.memory` overrides to hold the receiver-memory codec's settings fixed while varying the main hidden codec. SNR-FiLM is disabled in all default and smoke recipes. Task data, training budgets and evaluation methods remain task-specific. See [the shared baseline](docs/architecture.md#shared-baseline) for the design and compatibility limits.

W&B is optional: set `run.wandb_project` and add `--extra wandb` to `uv run --locked`.

## Read results and evaluate

Each training invocation prints its new run directory:

```text
runs/<time>-<name>-<id>/
  config.yaml          # Resolved experiment configuration
  data_ids.json        # Selected dataset rows/IDs
  run.json                 # Source revision/dirty state, runtime and budget metadata
  metrics.jsonl        # Training/validation metrics and logged timing/memory
  best.pt              # Best checkpoint under the configured improvement rule
  last.pt              # Most recent validation checkpoint
  step_002000.pt       # Optional extra checkpoint
  completion.json      # Written when training returns normally
  evaluations/         # Separate settings and results for each evaluation
```

Checkpoints contain codec/channel and optimizer state, configuration and data IDs. The backbone is reloaded from its model revision.

```bash
uv run --locked python evaluate.py --run runs/<coco-run>
uv run --locked python evaluate.py --run runs/<hellaswag-run>
```

The checkpoint identifies the task. Both commands default to `best.pt`; use `--checkpoint last.pt` to select another saved checkpoint.

COCO writes captions, CIDEr and EOS/empty/truncation rates. HellaSwag uses lm-eval to report accuracy and normalized accuracy. Default evaluations cover no-noise, nine SNR values and vanilla. No-noise retains the codec; vanilla bypasses both codec and channel.

Pass `--config` an evaluation-only YAML to adjust sample count, batch size or SNRs. Fields are top-level, for example:

```yaml
snrs: [no_noise, 0, 18]
num_samples: 32
batch_size: 4
vanilla: true
```

To resume a saved experiment when needed:

```bash
uv run --locked python train.py --config configs/tasks/coco.yaml --resume runs/<coco-run>/last.pt
uv run --locked python train.py --config configs/tasks/hellaswag.yaml --resume runs/<hellaswag-run>/last.pt
```

Resume creates a new run using the saved training recipe; the supplied YAML selects the new run name, destination and device. The data iterator restarts, so batch order is not guaranteed to reproduce an uninterrupted run. A forced Slurm timeout does not save the current step: only the most recently written checkpoint is recoverable.

## Development and tests

| Area | Files |
|---|---|
| Shared model and task recipes | `configs/model.yaml`, `configs/tasks/coco.yaml`, `configs/tasks/hellaswag.yaml` |
| Codec, channel and split hooks | `jscc/models/` |
| Training and objectives | `jscc/training.py`, `jscc/losses.py` |
| Task data | `jscc/data/coco.py`, `jscc/data/hellaswag.py` |
| Task evaluation | `jscc/evaluation.py` |
| COCO and HellaSwag coverage | `tests/test_coco.py`, `tests/test_hellaswag.py` |
| Shared training/codec coverage | `tests/test_core.py`, `tests/test_training.py` |
| All supplied sweep routes | `tests/test_sweep_routes.py` |

```bash
uv sync --locked --extra dev
uv run --locked --extra dev pyright
uv run --locked --extra dev ruff check .
uv run --locked --extra dev python -m pytest -q
uv run --locked --extra dev python -m pytest -q tests/test_coco.py
uv run --locked --extra dev python -m pytest -q tests/test_hellaswag.py
uv run --locked --extra dev python -m pytest -q tests/test_sweep_routes.py
```

These tests use CPU tensors and tiny real Transformers modules, without downloading pretrained weights. Full-weight smoke recipes are available for both tasks in `configs/smoke/coco_cpu.yaml` and `configs/smoke/hellaswag_cpu.yaml`; use a machine with enough RAM.

Pyright and Ruff are project dev dependencies with locked versions, installed in .venv by uv. Pyright's Node runtime is also included through its nodejs extra. The commands above do not depend on an editor's Mason/global installations. Editors may use their own server binaries, but should select this project's .venv for Python imports; the uv commands are the reproducible project checks. Pyright configuration uses Python 3.13 and the project-local .venv. Optional W&B is loaded only when enabled; core development does not require that extra.

The sweep-route tests exercise all supplied split and bottleneck variants on small CPU Transformers models. They check actual forward/backward/generation paths, not only YAML expansion. They do not measure full-model GPU memory or prove convergence; a full research sweep is separate from software validation.

For a quick H200 check, both `configs/smoke/*_h200.yaml` recipes use four optimizer updates, accumulation two, validation every two updates, and two evaluation samples across no-noise, 0 dB and vanilla. `configs/studies/smoke_h200.yaml` groups them into one plan. These checks cover the main train/checkpoint/evaluation paths, not every configuration or convergence.

Earlier full-weight checks completed HellaSwag training/evaluation and COCO small-scale generation/CIDEr; the older long COCO run timed out. The subsequent receiver-memory and HellaSwag holdout corrections have CPU regression coverage and still need full-weight validation before a new research sweep. See [validation scope](docs/validation.md) for the exact versions and limits.

## Documentation and collaboration

- [Architecture, data, objectives and checkpoint semantics](docs/architecture.md)
- [Configuration directory guide](configs/README.md)
- [Study preview, export and train/eval jobs](docs/studies.md)
- [Research synthesis, evidence gaps and future ablations](docs/research-roadmap.md)
- [Execution environments, uv and Slurm](docs/running.md)
- [Validation coverage](docs/validation.md)
- [Migration decisions and compatibility](docs/migration.md)
- [Custom wireless channel integration](docs/channel-integration.md)
- [Environment template](docs/environment.example.md)

Git tracks source, YAML, uv.lock and portable documentation. Personal machine/account settings and experiment paths belong in ignored `docs/local/`; credentials stay in their existing SSH or authentication tools. Model weights, datasets, run outputs and virtual environments are also ignored.

A collaborator can clone the repository and use the standard commands above. Machine-specific setup is optional until remote execution is needed; copy the environment template locally and fill in your own details. Channel collaboration has its own guide and does not change the project's normal research workflow.

Record the source revision before an experiment. Currently run.json does not automatically record the Git commit. Use branches and pull requests for shared changes; publishing a GitHub remote is a separate step from local version control.
