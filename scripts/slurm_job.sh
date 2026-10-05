#!/bin/bash
# Pass site-specific partition, account, QOS and log paths to sbatch.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=18:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
if (( $# == 0 )); then
    echo "At least one explicit YAML request is required" >&2; exit 2
fi
for request in "$@"; do
    uv run --locked --no-sync python experiment.py --config "$request"
done
