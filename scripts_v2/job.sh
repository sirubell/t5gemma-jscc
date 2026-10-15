#!/bin/bash
# Usage (from the repo root on the cluster):
#   sbatch scripts/job.sh train configs/examples/shared-13-hellaswag.yaml
#   sbatch scripts/job.sh evaluate runs/<run>
#   sbatch --time=48:00:00 scripts/job.sh train --resume runs/<run>
# The job writes runs/<run>/SUMMARY.md and a line in runs/INDEX.md when it starts,
# finishes, fails or hits the time limit (Slurm sends SIGTERM two minutes before).
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --signal=TERM@120
#SBATCH --output=runs/slurm-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
export HF_HOME="${HF_HOME:-/work/ShaoYuLien/siru/.cache/huggingface}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
exec "${JSCC_PYTHON:-.venv/bin/python}" experiment.py "$@"
