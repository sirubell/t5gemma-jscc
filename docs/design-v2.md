# Design: research/v2

Status: draft, 2026-10-15. The code before this rewrite is preserved at tag `archive/codex-v1`.

## Goals

1. Keep only the science: frozen T5Gemma-2 split, codec, channel, losses, data, training loop, task scoring.
2. One training engine for both comparison arms. A per-site codec is a run whose `sites` list has one entry; the shared codec is the same run with 13 entries. Data, loss, optimizer, stopping rule and evaluation are identical by construction.
3. Training ends by a stated criterion (validation-based early stopping), not a fixed step count, and keeps the best-validation checkpoint.
4. Any channel with the documented contract plugs into every run, including trainable physical-layer simulators.
5. A finished job leaves a human-readable summary next to its outputs and one line in a results index.

Non-goals: reading old receipts, manifests or run records; reproducing old workflows. Old results stay readable from `archive/codex-v1`.

## What is kept from the safety machinery

| Kept | Why |
|---|---|
| Stop on nonfinite loss or gradient, in every run | Protects result validity |
| Atomic checkpoint save (write temp, rename) | A crash never leaves a half-written checkpoint |
| `git rev-parse HEAD` + dirty flag + resolved config written once per run | Enough to know which code produced a result |
| Pinned Hugging Face model revision in YAML | The backbone cannot silently change |

Everything else (sha256 bindings, cache/batch identities, manifests, receipts, admission and eligibility gates, reconciliation) is removed.

## Modules

| Module | Responsibility |
|---|---|
| `experiment.py` | CLI: `python experiment.py {train,evaluate} CONFIG.yaml [--resume RUN_DIR]` |
| `jscc/config.py` | Load task YAML, merge its `model_config` design file, fill defaults, save resolved config |
| `jscc/codec.py` | Codec designs: `linear`, `mlp` (depth, hidden width, activation), `residual`; optional boundary norm and SNR-FiLM |
| `jscc/channel.py` | Power normalization, AWGN, identity, plugin loader and the channel contract |
| `jscc/model.py` | Load backbone/processor; `SplitModel` hooks for all 13 sites, route switching, decoder receiver-memory stream |
| `jscc/losses.py` | KL distillation and nMSE reconstruction statistics |
| `jscc/data/hellaswag.py`, `jscc/data/coco.py` | Train stream, selection (validation) batches, report set, collation |
| `jscc/train.py` | The single trainer: site schedule, optional reconstruction phase, AdamW, LR schedule, periodic validation, early stopping, checkpoints, metrics |
| `jscc/evaluate.py` | HellaSwag (lm-eval, five-shot `acc_norm`) and COCO (greedy captions, PTB + CIDEr) per site × condition |
| `jscc/report.py` | `summary.json` / `SUMMARY.md` per run and the shared results index |
| `scripts/slurm/` | Job templates: train then evaluate, each writing its summary |
| `scripts/results.py` | Print the results index (newest first) |

## Training engine

- `sites`: ordered list. Each update trains one site; sites rotate round-robin, so every site gets the same number of updates.
- Phases: optional reconstruction phase (nMSE only) of a given fraction, then task distillation (KL + weighted nMSE). Ratio 0 means distillation only.
- Optimizer: AdamW, gradient clipping, warmup then constant LR (optionally reduce-on-plateau). No cosine-to-zero, so a plateau reflects the model, not the schedule.
- Validation every `val_every` updates on the held-out selection split, for every site, under `no_noise` and one noisy SNR. Logged per site.
- Early stopping: stop when the mean validation loss over sites has not improved by `min_delta` (relative) for `patience` validations, or at `max_updates`. Keep `best.pt` (best validation) and `last.pt`.
- Metrics: one JSON line per logged update and per validation, in `metrics.jsonl`.
- Resume: from `last.pt` (codec, channel, optimizer, scheduler, step, RNG, early-stopping state). Data order is a deterministic function of seed and step, so resuming continues the same stream.

## Channel contract

```python
class MyChannel(torch.nn.Module):
    normalizes_power: bool = False   # True: the caller skips power normalization
    snr_mode: str = "per_sample"     # or "per_batch": receives one float per call

    def forward(self, z, snr_db):    # z: [batch, tokens, b] real; snr_db: None, float or [batch,1,1]
        ...                          # return same shape; may also set self.aux_loss (scalar tensor)
```

- Trainable parameters are optimized with the codec and saved in checkpoints.
- `train()` / `eval()` follow the codec (BatchNorm-style layers behave correctly).
- `aux_loss`, when set, is added with weight `channel.aux_weight`.
- `snr_db=None` means no noise.
- Config: `channel: {type: "package.module:ClassName", kwargs: {...}}`.

## Run directory

```
runs/<name>/
  config.yaml      resolved config
  source.txt       git commit and dirty flag
  metrics.jsonl    training and validation curves
  best.pt, last.pt
  eval/<checkpoint>/results.json
  summary.json, SUMMARY.md
```

## Validation of the rewrite

1. CPU unit tests on a tiny random T5Gemma-2 config: hooks at every site, power normalization, channel plugin, loss values, one train/validate/early-stop cycle, resume equality.
2. Score parity: evaluate an existing original13 terminal checkpoint with v2 on a few panels and compare with the recorded scores.
