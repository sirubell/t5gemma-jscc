#!/bin/bash
# Site resources are supplied explicitly at submission; one GPU per allocation.
set -euo pipefail
release=${1:?release}
prepared=${2:?prepared study}
phase=${3:?capacity, scoring, train, evaluate, or vanilla}
output=${4:?artifact directory}
cd "$release"
# A source-only release must not inherit the parent checkout's unrelated commit.
export GIT_CEILING_DIRECTORIES="$(dirname "$release")"
export SLURM_SUBMIT_DIR="$release"
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 NVIDIA_TF32_OVERRIDE=0
mkdir -p "$output"
nvidia-smi --query-gpu=timestamp,uuid,name,utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem --format=csv -l 30 > "$output/hardware-${SLURM_JOB_ID}.csv" &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true' EXIT
case "$phase" in
  capacity)
    uv run --locked --no-sync python -m scripts.outer_ln_capacity --manifest "$prepared/study-manifest.proposed.json" --indices "${5:?indices}" --output "$output/capacity-${SLURM_ARRAY_TASK_ID:-0}"
    ;;
  scoring)
    # Exact archived preflight checkpoints are evaluation fixtures only.
    for route in enc_l9 dec_l8; do
      run=$(python3 - "$release/outer-input/scoring-runs.json" "$route" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))[sys.argv[2]])
PY
)
      fixture_args=()
      if [[ "$route" == enc_l9 ]]; then fixture_args=(--tensor-fixtures "$release/outer-input/anomaly-fixtures.json"); fi
      uv run --locked --no-sync python -m scripts.outer_ln_scoring_check --run "$run" --checkpoint step_000080.pt --samples "$release/outer-input/scoring-documents.jsonl" --output "$output/scoring-$route" "${fixture_args[@]}"
    done
    ;;
  train)
    bash scripts/slurm_study.sh train "$prepared/study-manifest.proposed.json"
    ;;
  evaluate)
    bash scripts/slurm_study.sh evaluate "$prepared/study-manifest.proposed.json" --fixed-step --checkpoint step_005000.pt --expected-step 5000
    ;;
  vanilla)
    export SLURM_ARRAY_TASK_ID=0
    bash scripts/slurm_study.sh evaluate "$prepared/vanilla-manifest.proposed.json" --fixed-step --checkpoint step_005000.pt --expected-step 5000 --evaluation-config "$prepared/vanilla.yaml"
    uv run --locked --no-sync python -m jscc.outer_ln_gate --manifest "$prepared/study-manifest.proposed.json" --output "$output/first-wave-gate.json"
    ;;
  *) echo "Unknown phase: $phase" >&2; exit 2 ;;
esac
