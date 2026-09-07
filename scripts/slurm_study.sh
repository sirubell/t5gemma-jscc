#!/bin/bash
# Pass site-specific resources/log paths and --array to sbatch.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=18:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
uv run --locked --no-sync python -m jscc.study_task "${1:?train or evaluate}" \
  --manifest "${2:?path to manifest.json}" --index "${SLURM_ARRAY_TASK_ID:?Use --array}"
