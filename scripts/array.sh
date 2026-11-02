#!/bin/bash
# Slurm job array over a list of experiment.py commands, one per line (from the repo root on the cluster):
#   sbatch --array=1-13 scripts/array.sh runs/lists/w1.txt
# Task i runs line i through scripts/job.sh, e.g. "train configs/x.yaml" or
# "evaluate runs/r --checkpoint best --conditions -10 0 --out runs/v1/r". The line is word-split and
# globbed at run time, so "runs/name-*" finds a run dir created by an earlier job.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=runs/slurm-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
line=$(sed -n "${SLURM_ARRAY_TASK_ID:?submit with --array}p" "${1:?list file}")
[[ -n $line ]] || { echo "no line $SLURM_ARRAY_TASK_ID in $1" >&2; exit 1; }
echo "task $SLURM_ARRAY_TASK_ID: $line"
# shellcheck disable=SC2086  # word splitting and globbing are intended
exec scripts/job.sh $line
