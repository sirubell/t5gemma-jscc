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
mkdir -p "runs/jobs/$SLURM_JOB_ID"
hostname
nvidia-smi
uv run --locked --no-sync python -c 'import torch; print({"torch":str(torch.__version__),"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(),"bf16":torch.cuda.is_bf16_supported(),"capability":torch.cuda.get_device_capability()})'
java -version
for config in "$@"; do
    name=$(basename "$config" .yaml)
    path_file="runs/jobs/$SLURM_JOB_ID/$name.run.txt"
    uv run --locked --no-sync python train.py --config "$config" --run-path-file "$path_file"
    uv run --locked --no-sync python evaluate.py --run "$(cat "$path_file")"
done
