#!/usr/bin/env bash
# Single-GPU execution only; sbatch resources are supplied by reviewed dry-run commands.
set -euo pipefail
phase="${1:?train, evaluate, or vanilla}"
plan="${2:?prepared plan directory}"
frozen_hash="${3:?approved PREPARED-FILES.json SHA256}"
case "$phase" in
  train|evaluate) index="${SLURM_ARRAY_TASK_ID:?Submit as a study array}" ;;
  vanilla) index=5 ;;
  *) echo 'Unknown COCO phase' >&2; exit 2 ;;
esac
: "${SLURM_JOB_ID:?Run through Slurm}"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
uv run --locked --no-sync python -m scripts.coco_native_study execute "$phase" \
  --plan "$plan" --index "$index" --prepared-sha256 "$frozen_hash"
