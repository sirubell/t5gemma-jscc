#!/bin/bash
set -euo pipefail
release=${1:?release}
root=${2:?root}
phase=${3:?phase}
item=${4:?family}
if [[ "$phase" != preflight ]]; then
 echo "Prompt-policy phase retired or unsupported: $phase; only preflight is supported" >&2
 exit 2
fi
cd "$release"
export PATH="$HOME/.local/bin:$PATH"
: "${UV_PROJECT_ENVIRONMENT:?prepared environment}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 NVIDIA_TF32_OVERRIDE=0
nvidia-smi --id="${CUDA_VISIBLE_DEVICES:?GPU allocation}" --query-gpu=timestamp,uuid,name,utilization.gpu,memory.used,power.draw --format=csv -l 30 > "$root/logs/hardware-${SLURM_JOB_ID}.csv" &
telemetry=$!
trap 'kill "$telemetry" 2>/dev/null || true' EXIT
uv run --locked --no-sync python -m scripts.exact64_group --root "$root" --family "$item"
