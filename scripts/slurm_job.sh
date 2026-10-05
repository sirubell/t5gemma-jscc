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
job_id="${SLURM_JOB_ID:?A plain job identity is required}"
if [[ ! "$job_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || (( $# == 0 )); then
    echo "A plain job identity and at least one config are required" >&2; exit 2
fi
job_directory="runs/jobs/$job_id"
if [[ -L "$job_directory" || -L "$job_directory/links" ]]; then
    echo "Job infrastructure may not be symlink aliases" >&2; exit 2
fi
ordinal=0
for config in "$@"; do
    ((ordinal+=1))
    printf -v number '%04d' "$ordinal"
    output="$job_directory/training-$number"
    path_file="$job_directory/links/$number.run.txt"
    if [[ -e "$output" || -L "$output" || -e "$path_file" || -L "$path_file" ]]; then
        echo "Used job root or link; explicit continuation or a new authorized job is required" >&2; exit 2
    fi
done
mkdir -p "$job_directory/links"
hostname
nvidia-smi
uv run --locked --no-sync python -c 'import torch; print({"torch":str(torch.__version__),"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(),"bf16":torch.cuda.is_bf16_supported(),"capability":torch.cuda.get_device_capability()})'
java -version
ordinal=0
for config in "$@"; do
    ((ordinal+=1))
    printf -v number '%04d' "$ordinal"
    output="$job_directory/training-$number"
    path_file="$job_directory/links/$number.run.txt"
    uv run --locked --no-sync python train.py --config "$config" --output "$output" --run-path-file "$path_file"
    uv run --locked --no-sync python evaluate.py --run "$(cat "$path_file")"
done
