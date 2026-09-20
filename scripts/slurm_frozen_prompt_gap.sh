#!/bin/bash
# One authorized inference-only allocation; resources supplied explicitly at submission.
set -euo pipefail
root=${1:?new diagnostic root}
release=${2:?immutable numerical release}
cd "$release"
export GIT_CEILING_DIRECTORIES="$(dirname "$release")"
export PATH="$HOME/.local/bin:$PATH"
: "${UV_PROJECT_ENVIRONMENT:?set shared project environment at submission}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8 NVIDIA_TF32_OVERRIDE=0 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
nvidia-smi --id="${CUDA_VISIBLE_DEVICES:?allocated GPU}" --query-gpu=timestamp,uuid,name,utilization.gpu,memory.used,power.draw --format=csv -l 30 > "$root/hardware.csv" &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true' EXIT
uv run --locked --no-sync python "$root/code/frozen_prompt_gap.py" run --input "$root/input" --source-release "$release" --prepared "$root/prepared" --output "$root/results"
