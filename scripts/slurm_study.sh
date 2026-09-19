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
phase="${1:?train or evaluate}"
manifest="${2:?path to manifest.json}"
shift 2
fixed_step=false
checkpoint=""
expected_step=""
evaluation_config=""
while (( $# )); do
  case "$1" in
    --fixed-step) fixed_step=true; shift ;;
    --checkpoint|--expected-step|--evaluation-config)
      if (( $# < 2 )) || [[ -z "$2" || "$2" == --* ]]; then
        echo "$1 requires a value" >&2; exit 2
      fi
      if [[ "$1" == --checkpoint ]]; then checkpoint="$2"; elif [[ "$1" == --expected-step ]]; then expected_step="$2"; else evaluation_config="$2"; fi
      shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "$fixed_step" == true || -n "$expected_step" ]]; then
  if [[ "$phase" != evaluate || -z "$checkpoint" || ! "$expected_step" =~ ^[1-9][0-9]*$ ]]; then
    echo "Fixed-step evaluation requires evaluate, --checkpoint and positive --expected-step together" >&2
    exit 2
  fi
fi
if [[ -n "$evaluation_config" && "$phase" != evaluate ]]; then
  echo "--evaluation-config requires evaluate" >&2; exit 2
fi
args=("$phase" --manifest "$manifest" --index "${SLURM_ARRAY_TASK_ID:?Use --array}")
if [[ -n "$checkpoint" ]]; then args+=(--checkpoint "$checkpoint"); fi
if [[ -n "$expected_step" ]]; then args+=(--expected-step "$expected_step"); fi
if [[ -n "$evaluation_config" ]]; then args+=(--evaluation-config "$evaluation_config"); fi
uv run --locked --no-sync python -m jscc.study_task "${args[@]}"
