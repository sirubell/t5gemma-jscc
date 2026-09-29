#!/bin/bash
set -euo pipefail
release=${1:?release}
root=${2:?root}
phase=${3:?phase}
item=${4:?split or family}
cd "$release"
export PATH="$HOME/.local/bin:$PATH"
: "${UV_PROJECT_ENVIRONMENT:?prepared environment}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 NVIDIA_TF32_OVERRIDE=0
nvidia-smi --id="${CUDA_VISIBLE_DEVICES:?GPU allocation}" --query-gpu=timestamp,uuid,name,utilization.gpu,memory.used,power.draw --format=csv -l 30 > "$root/logs/hardware-${SLURM_JOB_ID}.csv" &
telemetry=$!
trap 'kill "$telemetry" 2>/dev/null || true' EXIT
if [[ "$phase" == preflight ]]; then
 uv run --locked --no-sync python -m scripts.exact64_group --root "$root" --family "$item"
else
 uv run --locked --no-sync python -m scripts.prompt_policy_job --root "$root" --split "$item" --phase "$phase"
fi
