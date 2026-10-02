# Execution environments

Run the same project and YAML entry points on a workstation or a scheduled GPU server. Connection details belong to each user's environment.

## Personal configuration

Copy the tracked template once:

```bash
mkdir -p docs/local
cp -n docs/environment.example.md docs/local/environment.md
```

Fill in your checkout locations, SSH aliases, VPN procedure, scheduler account and cache locations. docs/local/ is ignored by Git, so local agents can read it without including personal settings in a clone. Keep actual credentials in SSH configuration, an agent/keychain or the appropriate authentication tool.

An existing local environment file takes precedence over the blank template. The template is documentation, not automatically loaded configuration.

## uv and caches

Prepare a fresh environment from the project root:

```bash
uv sync --locked --extra dev
uv run --locked --extra dev pyright
uv run --locked --extra dev ruff check .
uv run --locked --extra dev python -m pytest -q
uv run --locked python train.py --config configs/tasks/coco.yaml --check
uv run --locked python train.py --config configs/tasks/hellaswag.yaml --check
```

The project pins Python dependencies in uv.lock. Each machine creates its own .venv; do not copy an environment between machines. Prepare model access and Hugging Face datasets before offline jobs. Use a login shell if a remote non-interactive command lacks uv or Slurm in PATH.

Pyright reads the relative .venv and Python version from pyproject.toml. Editors may also have interpreter selection settings: prefer the current project's .venv, and avoid a cached interpreter from another checkout. Editor-specific configuration belongs to the user's editor setup rather than the repository.

Pyright, Ruff and Pyright's Node runtime are included in the dev extra and locked with the project. They are optional for training-only installations. Install the environment while network access is available before using it offline. No global npm/Mason installation is required for the project CLI checks.

## Standalone GPU workstations, including RTX 5090

The Python model/training/evaluation code does not require H200 or Slurm. RTX 5090 is a Blackwell GPU with 32 GB GDDR7. PyTorch introduced Blackwell support with CUDA 12.8 builds; the project's Linux installations have used PyTorch 2.10.0+cu128. Use a compatible NVIDIA driver and CUDA-enabled PyTorch build, not a CPU-only wheel. Sources: [NVIDIA specification](https://marketplace.nvidia.com/en-us/consumer/graphics-cards/nvidia-geforce-rtx-5090/), [PyTorch Blackwell support](https://pytorch.org/blog/pytorch-2-7/).

On the target workstation, inspect the installed runtime before loading the model:

```bash
nvidia-smi
uv run --locked python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_arch_list())"
```

When CUDA is available, inspect the selected GPU and BF16 support:

```bash
uv run --locked python -c "import torch; print(torch.cuda.get_device_name(), torch.cuda.is_bf16_supported())"
```

The *_h200.yaml smoke filenames record where those CUDA/bfloat16 recipes were validated. Their contents do not test for an H200 or request Slurm resources. They can be used for an initial small check on another compatible CUDA GPU:

```bash
uv run --locked python train.py --config configs/smoke/coco_h200.yaml
# Evaluate the run directory printed above.
uv run --locked python evaluate.py --run runs/<coco-smoke-run>
uv run --locked python train.py --config configs/smoke/hellaswag_h200.yaml
uv run --locked python evaluate.py --run runs/<hellaswag-smoke-run>
```

Run sequentially on one GPU. These recipes already use batch size 1 for training/evaluation, but memory fit still depends on available VRAM, the model build and the selected split. The historical enc_fn / external LayerNorm-both recipe passed full-weight training/checkpoint/evaluation smoke on a Linux RTX 5090 using PyTorch 2.10.0+cu128 and BF16. That observation does not establish acceptance of the current enc_l9 defaults or the new two-stage paths on a rebuilt workstation. At batch 1, recorded training peak allocated memory was about 4.80 GiB for COCO and 4.34 GiB for HellaSwag. These are PyTorch training peaks, not total driver-reserved memory or a guarantee for larger batches/other splits.

For an initial fit check, use a separate smoke recipe and record its batch policy. The current research defaults both have effective batch 64: COCO uses microbatch 16 with accumulation 4; HellaSwag uses microbatch 64 with accumulation 1. A microbatch of 1 with accumulation 64 preserves the nominal effective batch, but does not establish numerical equivalence: batch composition can affect BF16 computation and paired AWGN shapes. Changing evaluation batch size can also affect candidate scores. Keep the declared microbatch, accumulation and evaluation batch for matched research comparisons; investigate an OOM before approving a separate execution policy. See [current defaults](current-defaults.md) for recipe lineage. The smoke recipe intentionally has a smaller effective batch. Early encoder splits can require more activation memory than late splits, so a successful baseline smoke does not prove every sweep configuration fits.

If CPU memory/data loading is constrained, start with data.num_workers=0 or a small value. Keep the same model.yaml, task definitions and metrics when comparing model quality. The shell examples and Slurm wrappers target Unix-like environments; Windows/WSL installations need their own environment check.

## Slurm

[scripts/slurm_job.sh](../scripts/slurm_job.sh) requests one GPU, 16 CPUs, 128 GiB RAM and 18 hours by default. Partition, account and QOS are supplied at submission time because they vary by site. A cluster may also adjust the requested CPU/RAM values.

Replace the placeholders using your local environment notes:

```bash
mkdir -p runs/slurm
sbatch --partition=YOUR_PARTITION --account=YOUR_ACCOUNT --qos=YOUR_QOS --job-name=coco --output=runs/slurm/%x-%j.out --error=runs/slurm/%x-%j.err scripts/slurm_job.sh configs/tasks/coco.yaml
sbatch --partition=YOUR_PARTITION --account=YOUR_ACCOUNT --qos=YOUR_QOS --job-name=hellaswag --output=runs/slurm/%x-%j.out --error=runs/slurm/%x-%j.err scripts/slurm_job.sh configs/tasks/hellaswag.yaml
```

The script uses uv run --locked --no-sync. Sync dependencies before submitting concurrent jobs. It expects NVIDIA/CUDA and Java, prints environment information, trains, and evaluates the returned run only when training succeeds. Its offline flags require existing model/data caches.

The same wrapper accepts either H200 smoke YAML in configs/smoke/. Override the Slurm time limit for small checks; a smoke YAML's training time budget does not include the subsequent evaluation. For multi-experiment plans with separate training and evaluation allocations, use the [study workflow](studies.md) and scripts/slurm_study.sh.

## Read results

```bash
squeue --me
sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS,AllocTRES,ExitCode
scontrol show job JOBID
```

Inspect the configured stdout/stderr paths. The Run line identifies the artifact directory. runs/jobs/JOBID/*.run.txt is written only after training returns successfully. MaxRSS measures host RAM, not GPU VRAM; GPU allocation is reported by the training metric peak_memory_gib or nvidia-smi inside an allocation.

## Time limits and recovery

Slurm time limits and YAML training.max_minutes are separate. The main task YAMLs do not set max_minutes. When supplied, it is checked after an optimizer update; final validation and saving still take time. Budget separately for task evaluation.

Forced scheduler termination does not save the current step or proceed to evaluation. last.pt records the last completed save, and completion.json is only written when the Python training loop returns normally.

Changing a script affects future submissions, not running allocations. Whether users can extend a running job is site-dependent. If a future experiment needs recovery, inspect the actual checkpoint step before using the README's resume commands. Do not infer that a timeout requires an automatic continuation.

## Performance work

Measure training, data preparation and evaluation separately before changing settings. Compare microbatch/accumulation combinations at a fixed effective batch size. Benchmark worker count and attention kernels on actual shapes. BF16 has been validated on H200; FP8, compile and shared teacher/student computation have not been benchmarked for this project.

See [validation levels](validation.md#what-a-sweep-test-does-and-does-not-prove) before deciding between CPU route tests, a target-GPU smoke and a full research run.

## Common setup issues

- Missing Python imports: run uv sync with the required extras and verify the editor selected this project's .venv.
- CUDA unavailable or unsupported GPU architecture: inspect the NVIDIA driver and the installed PyTorch CUDA build before changing model code.
- Ruff reports invalid UTF-8 in ._*.py after copying from macOS: these may be AppleDouble metadata sidecars. Confirm their format and move them out of the source tree; use a Git checkout/archive for future source transfers. The project explicitly excludes docs/local/ backups and ._* metadata even in deployments without a .git directory. Real Python source encoding errors should still be investigated normally.
- lm-eval prints a Git lookup failure in a source-only archive deployment: inspect the process exit code and results.json. The metadata lookup can fail while evaluation succeeds; a normal Git checkout provides the missing repository metadata.

## Colab runbook（2026-10-02）

沿用 [簡短研究流程](research-roadmap.md#研究工作流程2026-10-02)。以下是現有 owner 回報的已測連線與執行方式；不要重新建立帳號流程或重跑別人的 GPU probe。

1. 使用現有官方 Colab CLI 與既有登入狀態。從當前 Colab owner 的 handoff 取得實際 CLI 路徑、runtime ID 與既有 launcher；不要猜 subcommand、重印 token 或把登入資料放進文件。先確認當前 assignment，重用已指定的 runtime。
2. 明確 stage 固定版本 source 與 launcher 所需 modules；launcher 的 script directory 必須在 `sys.path`。用 project lockfile 執行 `uv sync --locked`，後續固定使用 project Python。Notebook kernel 啟動的 child process 已確認可使用 CUDA。
3. SSH 路徑需要在啟動 Python **之前**設定 process-local driver library path。既有 launcher 的命令前綴為：

   ```bash
   LD_LIBRARY_PATH=/usr/lib64-nvidia .venv/bin/python <existing-launcher.py> <approved-arguments>
   ```

   這是命令樣式，placeholder 必須換成 owner 已驗證的實際路徑／參數；不要改全系統 library 設定。
4. 必要 weights 先壓縮上傳、解壓，再由 model loader 讀取；確認 input/checkpoint/source 身分與輸出位置。`/content` 是暫存，終止前匯出必要結果。私人永久 Drive 儲存尚未設定或授權，不把 mount/upload 當成既定下一步。
5. 保留有界的 remote job、結果 export、stop 與 **zero assignments** 驗證。CLI/client timeout 不是遠端 GPU cutoff。背景 runtime 的存活不能視為 AI Pro 保證；斷線後先查既有 job 狀態，避免重複啟動。

已測範圍：L4 上16-document、BF16、batch16 通過；model load 約7.5秒、evaluation15.09秒，peak allocated6.03GiB／reserved7.16GiB。當次觀察 L4 為1.54 CU/hour、A100 為5.30 CU/hour；這些是當次數值，不保證未來費率。完整 panels／training 尚未由這個小測試驗證。256-document probe 由現有 owner 處理，先讀其結果，不另開重複工作。

本機 evidence workspace：`/Users/tim_c_wang/Documents/Codex/2026-10-01/colab-readiness-cost-probe/`；資產身分來源為 [COLAB-ASSET-HANDOFF.json](local/research/short-trials-20261001/COLAB-ASSET-HANDOFF.json)。外部 session 不會自動取得這些本機／ignored 檔案；由 owner 提供需要的非機密 handoff。此 runbook 不代表已完成永久儲存、完整訓練 qualification 或新的 deployment。
