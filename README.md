# T5Gemma-2 JSCC research

A frozen T5Gemma-2 encoder–decoder is split at one of its layers. The hidden state at the split is encoded by a trainable codec to width b, sent through a channel (AWGN by default, or a plugged-in physical-layer simulator), decoded and handed to the rest of the frozen model. At decoder splits the encoder memory used by the receiver layers is transmitted too. Only the codec (and any trainable channel) is trained, by distillation from the unsplit model. Tasks: HellaSwag five-shot `acc_norm` and COCO captioning CIDEr.

## Run

```bash
uv sync --locked --extra dev
python experiment.py train configs/examples/shared-13-hellaswag.yaml       # one codec for all 13 sites
python experiment.py train configs/examples/per-site-enc_l9-hellaswag.yaml # one codec for one site
python experiment.py train --resume runs/<run>
python experiment.py evaluate runs/<run> [--checkpoint best|last|PATH]
python scripts/results.py                                                  # recent runs and their status
```

On a Slurm cluster: `sbatch scripts/job.sh train CONFIG`. Put machine-specific settings in an untracked `.env` (for example `HF_HOME`, `JSCC_DATA` for the COCO image file, `JSCC_PYTHON`).

Each run directory holds `config.yaml`, `metrics.jsonl` (training and validation curves), `best_*.pt`, `last.pt`, `eval/<checkpoint>/results.json`, and a human-readable `SUMMARY.md`. Every start, finish, failure or time-limit interruption adds a line to `runs/INDEX.md`.

## Configuration

`configs/hellaswag.yaml` and `configs/coco.yaml` hold the base settings; experiment files name one of them with `base:` and override keys. The main choices:

| Key | Meaning |
|---|---|
| `training.sites` | Split sites trained by this codec: one entry = per-site codec, all 13 = shared codec |
| `codec.architecture` | `linear`, `mlp` (`depth`, `hidden_dim`), `residual` |
| `codec.bottleneck_dim` | Transmitted width b (D = 1152; b > D is expansion) |
| `codec.film` | Give the receiver the SNR |
| `channel.type` | `awgn`, `identity` or `package.module:ClassName` ([guide](docs/channel-integration.md)) |
| `training.early_stopping` | Validate every N updates; stop after `patience` validations without `min_delta` relative improvement |
| `training.reconstruction.enabled` | Feature-reconstruction pretraining phase before distillation |
| `evaluation.conditions` | `vanilla`, `no_noise`, SNR values in dB |

## Training rule

Each update trains one site; sites rotate, so every site gets the same number of updates and the k-th update of every site uses the same batch k in per-site and shared runs. Validation runs on a held-out part of the training data (never the reported evaluation set) for every site under no noise and −6 dB. Training stops when the mean validation loss stops improving, and the best-validation checkpoint is evaluated. The LR is constant after warmup, so a plateau reflects the model rather than a decayed schedule.

See [design](docs/design-v2.md), [vocabulary](CONTEXT.md), [running](docs/running.md) and [channel integration](docs/channel-integration.md). Code before the 2026-10 rewrite is at tag `archive/codex-v1`.
