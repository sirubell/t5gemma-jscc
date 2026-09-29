#!/bin/bash
set -euo pipefail
release=${1:?release}
root=${2:?root}
family=${3:?family}
cd "$release"
export PATH="$HOME/.local/bin:$PATH"
: "${UV_PROJECT_ENVIRONMENT:?prepared uv environment}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 NVIDIA_TF32_OVERRIDE=0
nvidia-smi --id="${CUDA_VISIBLE_DEVICES:?allocated GPU}" --query-gpu=timestamp,uuid,name,utilization.gpu,memory.used,power.draw --format=csv -l 10 > "$root/logs/hardware-${SLURM_JOB_ID}.csv" &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true' EXIT
uv run --locked --no-sync python -m scripts.five_shot_preflight_group --root "$root" --family "$family"
