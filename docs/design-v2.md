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
| `jscc/tasks.py` | HellaSwag and COCO data: train stream, selection (validation) batches, collation, model inputs |
| `jscc/train.py` | The single trainer: site schedule, optional reconstruction phase, AdamW, LR schedule, periodic validation, early stopping, checkpoints, metrics, SIGTERM stop |
| `jscc/evaluate.py` | HellaSwag (lm-eval, five-shot `acc_norm`) and COCO (greedy captions, PTB + CIDEr) per site × condition. COCO report images are preprocessed once for all panels; generation stops at the first newline (only the text before it is scored), except in noisy decoder-site panels, where it would shift later batches' noise |
| `jscc/report.py` | `summary.json` / `SUMMARY.md` per run and the shared results index |
| `scripts/job.sh` | Slurm job: train (then evaluate) or evaluate, each writing its summary |
| `scripts/results.py` | Print the results index (newest first) |

## Training engine

- `sites`: ordered list. Each update trains one site; sites rotate round-robin, so every site gets the same number of updates.
- Phases: optional reconstruction phase (nMSE only) of a given fraction, then task distillation (KL + weighted nMSE). Ratio 0 means distillation only.
- Optimizer: AdamW, gradient clipping, warmup then constant LR (optionally reduce-on-plateau). No cosine-to-zero, so a plateau reflects the model, not the schedule.
- Validation every `val_every` updates on the held-out selection split, for every site, under `no_noise` and one noisy SNR. Logged per site.
- Validation metric: in distillation, the KL(teacher ‖ student) per valid target token of each selection batch (the training loss's nMSE terms are not included), averaged over batches for each site × condition cell, then over all cells; in reconstruction, the hidden-state nMSE.
- Early stopping: stop when that mean has not improved by `min_delta` (relative to the last value that did) for `patience` validations, or at `max_updates`. `best_<phase>.pt` is saved at every new minimum, however small; `min_delta` only drives the patience counter. `last.pt` is saved at every validation and holds the final weights when training ends. `patience: null` disables early stopping (fixed budget of `max_updates`). `stop: false` trains on to `max_updates` but copies the best checkpoint at the rule's first stop to `best_<phase>_early_stop.pt` (`evaluate --checkpoint early_stop`), so one run gives both the early-stopped and the fixed-budget result. With `selection.reference` (per site|condition reference KL) each validation also logs the normalized mean (KL ÷ reference, averaged over cells) and `best_<phase>_normalized.pt` is kept at every new normalized minimum.
- Metrics: one JSON line per logged update and per validation, in `metrics.jsonl`.
- Speed without changing the arithmetic: the sites of one batch k share its model inputs and teacher pass (built once, freed when k changes), and batch k + 1 is built in a background thread. The teacher pass also keeps the frozen-prefix hidden states the run's sites need (the input of encoder layer 0, the outputs of the split encoder layers and of the last one, the encoder memory); each student pass, in training and validation, starts at its split from them instead of recomputing the vision tower and the layers before the split. This is exact because the backbone is frozen, runs without dropout under the same autocast and masks, and the teacher draws no noise. `tests/test_model.py` guards it for every site kind, with images, against the full forward.
- Resume: from `last.pt` (codec, channel, optimizer, scheduler, step, RNG, early-stopping state). Data order is a deterministic function of seed and step, so resuming continues the same stream. On SIGTERM the trainer saves `last.pt` at the next update boundary; a validation in progress is aborted (not logged) and repeated on resume. Records in `metrics.jsonl` written after the checkpoint (a hard kill) are dropped on resume, so each update and validation appears once. A run can also be resumed from a copy of its `best_<phase>.pt` renamed to `last.pt` (e.g. to continue an early-stopped run with a changed config); the validation it was saved at counts as done.

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
