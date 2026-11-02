# Running on a workstation or cluster

The code needs a CUDA GPU with a CUDA-enabled PyTorch build (the lock file pins PyTorch 2.10.0; RTX 5090 needs the cu128 build). Slurm is optional.

## Setup

```bash
uv sync --locked            # add --extra dev for tests and linters
```

Create an untracked `.env` in the repo root for machine-specific settings; `scripts/job.sh` sources it:

```bash
export HF_HOME=/path/to/huggingface          # model and HellaSwag dataset cache
export JSCC_DATA=/path/to/coco               # directory with coco-historical-12516.arrow
export JSCC_PYTHON=.venv/bin/python
```

Jobs run offline (`HF_HUB_OFFLINE=1`), so the model `google/t5gemma-2-1b-1b@dd0a268` and `Rowan/hellaswag@218ec52e` must already be in `HF_HOME`.

## Memory

At b = 2304 with batch 16, HellaSwag training used about 25 GB on the GPU and COCO training about 74 GB (H200 measurements). HellaSwag fits a 32 GB RTX 5090 if the GPU is otherwise free; COCO needs the H200.

## Workflow

1. Update the checkout on the server to the intended commit (`git pull`), keep it clean.
2. Submit: `sbatch scripts/job.sh train configs/examples/<file>.yaml` on Slurm, or queue `scripts/job.sh ...` with pueue on a workstation (below).
3. Check: `python scripts/results.py` (index and status of recent runs), then open `runs/<run>/SUMMARY.md`.
4. A run that hit the time limit shows `interrupted` (`scripts/job.sh` asks Slurm for SIGTERM two minutes early; training then saves `last.pt`); continue with `train --resume runs/<run>`.

## Extra panels into another directory

`python experiment.py evaluate runs/<run> --checkpoint best --conditions -10 0 6 --out DIR` scores a checkpoint into `DIR` (`config.yaml`, `source.json`, `results.json`, plus `items.npz` for HellaSwag or the captions files for COCO); the run directory and `runs/INDEX.md` are not written. `--checkpoint` defaults to `best`.

Running it again with the same run and checkpoint resumes: panels already in `DIR/results.json` are kept and only the missing ones are scored. A different run or checkpoint name is refused with an error, because `source.json` records the pair, so give each checkpoint its own directory (a failed attempt, such as `early_stop` when the rule never fired, also leaves its `source.json`). Only the names are compared: if the file behind a name changed, for example `last` after more training, use a new directory.

## Job queue on lab workstations (pueue)

The lab workstations without Slurm (5090B, Spark) queue jobs with [pueue](https://github.com/Nukesor/pueue): a user-level daemon whose queue survives restarts. Install the static `pueue` and `pueued` release binaries for the machine's architecture into `~/.local/bin` (no sudo), and start the daemon at boot with a user crontab line:

```
@reboot $HOME/.local/bin/pueued -d
```

The `default` group runs one task at a time, so GPU jobs run one after another. Add jobs from the repo root (tasks run in the directory they were added from):

```bash
pueue add -l <label> -- scripts/job.sh train configs/examples/<file>.yaml
pueue add -l <label> -- scripts/job.sh train --resume runs/<run>
pueue status                 # one-command check; pueue log <id> shows a task's output
```

Non-login SSH commands do not have `~/.local/bin` on `PATH`; use `ssh <host> ~/.local/bin/pueue status`. When starting the daemon over SSH, redirect its output (`pueued -d </dev/null >/dev/null 2>&1`) or the SSH session hangs.

Nothing is retried automatically. A task that fails or is killed shows `Failed`/`Killed`; after a power loss or daemon crash the interrupted task shows `Killed` and its group comes back paused, so queued tasks wait. To continue: check `nvidia-smi` (a daemon crash alone can leave the training process running), add `train --resume runs/<run>` for the interrupted run, then `pueue start`. pueue does not check whether another user is using a shared GPU; look at `nvidia-smi` before adding GPU jobs.

Personal access details (SSH aliases, VPN, scheduler account) belong in `docs/local/environment.md`.
