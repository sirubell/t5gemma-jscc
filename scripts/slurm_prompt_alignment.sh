#!/bin/bash
set -euo pipefail
release=${1:?release}
root=${2:?study root}
phase=${3:?benchmark or pair}
split=${4:?split}
cd "$release"
export GIT_CEILING_DIRECTORIES="$(dirname "$release")"
export PATH="$HOME/.local/bin:$PATH"
: "${UV_PROJECT_ENVIRONMENT:?shared prepared uv environment}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 NVIDIA_TF32_OVERRIDE=0
nvidia-smi --id="${CUDA_VISIBLE_DEVICES:?allocated GPU}" --query-gpu=timestamp,uuid,name,utilization.gpu,memory.used,power.draw --format=csv -l 30 > "$root/logs/hardware-${SLURM_JOB_ID}.csv" &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true' EXIT
case "$phase" in
 benchmark)
   uv run --locked --no-sync python -m scripts.prompt_alignment_benchmark benchmark --configs "$root/input/configs/$split-raw.yaml" "$root/input/configs/$split-five_shot.yaml" --output "$root/benchmarks/$split" --microbatch 32
   ;;
 pair)
   uv run --locked --no-sync python -m scripts.prompt_alignment_pair --plan "$root/input/LOCKED.json" --split "$split" --prepared "$root/input/fixture" --output "$root/pairs/$split"
   ;;
 *) exit 2 ;;
esac
