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
2. Submit: `sbatch scripts/job.sh train configs/examples/<file>.yaml`, or run `python experiment.py ...` directly on a workstation (inside `tmux`).
3. Check: `python scripts/results.py` (index and status of recent runs), then open `runs/<run>/SUMMARY.md`.
4. A run that hit the time limit shows `interrupted`; continue with `train --resume runs/<run>`.

Personal access details (SSH aliases, VPN, scheduler account) belong in `docs/local/environment.md`.
