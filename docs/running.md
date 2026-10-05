# Execution environments

Current public calls use one YAML-selected operation through `experiment.py --config FILE`; see [request fields and effect boundaries](operations-workflow.md).
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
uv run --locked python experiment.py --config configs/experiments/ordinary/coco.yaml
uv run --locked python experiment.py --config configs/experiments/ordinary/hellaswag.yaml
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

The *_h200.yaml smoke filenames retain the bounded CUDA/BF16 science used for historical checks. Their current requests select `operation: check`; they do not request Slurm resources or launch training:

```bash
uv run --locked python experiment.py --config configs/smoke/coco_h200.yaml
uv run --locked python experiment.py --config configs/smoke/hellaswag_h200.yaml
```

For an authorized target smoke, copy the complete request, select `operation: train`, and set a fresh explicit root and attempt. Evaluate that exact recorded root with a separate `operation: evaluate` request and new attempt. Check examples and historical smoke observations do not qualify the current runtime.

Run sequentially on one GPU. These recipes already use batch size 1 for training/evaluation, but memory fit still depends on available VRAM, the model build and the selected split. The historical enc_fn / external LayerNorm-both recipe passed full-weight training/checkpoint/evaluation smoke on a Linux RTX 5090 using PyTorch 2.10.0+cu128 and BF16. That observation does not establish acceptance of the current enc_l9 defaults or the new two-stage paths on a rebuilt workstation. At batch 1, recorded training peak allocated memory was about 4.80 GiB for COCO and 4.34 GiB for HellaSwag. These are PyTorch training peaks, not total driver-reserved memory or a guarantee for larger batches/other splits.

For an initial fit check, use a separate smoke recipe and record its batch policy. The current research defaults both have effective batch 64: COCO uses microbatch 16 with accumulation 4; HellaSwag uses microbatch 64 with accumulation 1. A microbatch of 1 with accumulation 64 preserves the nominal effective batch, but does not establish numerical equivalence: batch composition can affect BF16 computation and paired AWGN shapes. Changing evaluation batch size can also affect candidate scores. Keep the declared microbatch, accumulation and evaluation batch for matched research comparisons; investigate an OOM before approving a separate execution policy. See [current defaults](current-defaults.md) for recipe lineage. The smoke recipe intentionally has a smaller effective batch. Early encoder splits can require more activation memory than late splits, so a successful baseline smoke does not prove every sweep configuration fits.

If CPU memory/data loading is constrained, start with data.num_workers=0 or a small value. Keep the same model.yaml, task definitions and metrics when comparing model quality. The shell examples and Slurm wrappers target Unix-like environments; Windows/WSL installations need their own environment check.

## Slurm

[scripts/slurm_job.sh](../scripts/slurm_job.sh) requests one GPU, 16 CPUs, 128 GiB RAM and 18 hours by default. Partition, account and QOS are supplied at submission time because they vary by site. A cluster may also adjust the requested CPU/RAM values.

Replace the placeholders using your local environment notes:

```bash
mkdir -p runs/slurm
sbatch --partition=YOUR_PARTITION --account=YOUR_ACCOUNT --qos=YOUR_QOS --job-name=coco --output=runs/slurm/%x-%j.out --error=runs/slurm/%x-%j.err scripts/slurm_job.sh /absolute/approved/coco-request.yaml
sbatch --partition=YOUR_PARTITION --account=YOUR_ACCOUNT --qos=YOUR_QOS --job-name=hellaswag --output=runs/slurm/%x-%j.out --error=runs/slurm/%x-%j.err scripts/slurm_job.sh /absolute/approved/hellaswag-request.yaml
```

The script uses `uv run --locked --no-sync python experiment.py --config REQUEST` for each supplied request in order and stops on a nonzero exit. Prepare the environment before submitting concurrent jobs. It sets offline cache flags and requires existing model/data assets. It does not infer evaluation after training; supply a separate evaluation request for the exact bound root and a new attempt when the approved plan requires it.

The same wrapper accepts copied smoke requests with explicit operations. The supplied check examples perform checks only; training and evaluation have separate declared operations and budgets. For multi-experiment plans with separate training and evaluation allocations, use the [study workflow](studies.md) and scripts/slurm_study.sh.

## Read results

```bash
squeue --me
sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS,AllocTRES,ExitCode
scontrol show job JOBID
```

Inspect the configured stdout/stderr paths and the request's exact root/attempt. The wrapper does not invent job-number roots or success links. A supplied `execution.success_link` is published only after that operation returns and cannot substitute for scientific completion. Optional study transport records its training link and generates separate evaluation attempts; see [studies](studies.md). Retained failed roots and occupied attempt/link paths require explicit fresh identities, not replay over existing outputs. MaxRSS measures host RAM, not GPU VRAM; GPU allocation is reported by training telemetry or nvidia-smi inside an allocation.

## Time limits and recovery

Slurm time limits and YAML training.max_minutes are separate. The main task YAMLs do not set max_minutes. When supplied, it is checked after an optimizer update; final validation and saving still take time. Budget separately for task evaluation.

Forced scheduler termination does not save the current step or proceed to evaluation. last.pt records the last completed save, and completion.json is only written when the Python training loop returns normally.

Changing a script affects future submissions, not running allocations. Whether users can extend a running job is site-dependent. If a future experiment needs recovery, inspect the actual checkpoint step before authoring a separate `operation: resume` request with the exact durable checkpoint/role/step/parent/size/hash in [operations](operations-workflow.md). Do not infer that a timeout requires an automatic continuation.

## Performance work

Measure training, data preparation and evaluation separately before changing settings. Compare microbatch/accumulation combinations at a fixed effective batch size. Benchmark worker count and attention kernels on actual shapes. BF16 has been validated on H200; FP8, compile and shared teacher/student computation have not been benchmarked for this project.

See [validation levels](validation.md#what-a-sweep-test-does-and-does-not-prove) before deciding between CPU route tests, a target-GPU smoke and a full research run.

## Common setup issues

- Missing Python imports: run uv sync with the required extras and verify the editor selected this project's .venv.
- CUDA unavailable or unsupported GPU architecture: inspect the NVIDIA driver and the installed PyTorch CUDA build before changing model code.
- Ruff reports invalid UTF-8 in ._*.py after copying from macOS: these may be AppleDouble metadata sidecars. Confirm their format and move them out of the source tree; use a Git checkout/archive for future source transfers. The project explicitly excludes docs/local/ backups and ._* metadata even in deployments without a .git directory. Real Python source encoding errors should still be investigated normally.
- lm-eval prints a Git lookup failure in a source-only archive deployment: inspect the process exit code and results.json. The metadata lookup can fail while evaluation succeeds; a normal Git checkout provides the missing repository metadata.

## Colab runbook (2026-10-02)

Follow the [concise research workflow](research-roadmap.md#research-workflow-2026-10-02). These connection and execution observations come from the existing owner. Reuse that work rather than rebuilding account setup or duplicating GPU probes.

1. Use the existing official Colab CLI and login. Obtain the actual CLI path, runtime ID and existing launcher from the current Colab owner's handoff; do not guess subcommands, print tokens or put credentials in documents. Check the current assignment and reuse the designated runtime.
2. Explicitly stage the pinned source and modules required by the launcher. The launcher's script directory must be on `sys.path`. Run `uv sync --locked` with the project lockfile, then use the fixed project Python. A child process launched from the notebook kernel has been verified to use CUDA.
3. Through SSH, set the process-local driver library path **before** starting Python. Prefix the existing launcher command as follows:

   ```bash
   LD_LIBRARY_PATH=/usr/lib64-nvidia .venv/bin/python <existing-launcher.py> <approved-arguments>
   ```

   This is a command pattern: replace placeholders with the owner's verified paths and arguments. Keep system-wide library settings unchanged.
4. For the verified private Drive path, connect the runtime in the browser, use Files → Mount Drive and complete any requested consent, then attach the existing worker to that same runtime. Copy only the approved model/checkpoint files into `/content/colab-probe` before loading; verify sizes, streaming hashes and independent destination hashes. Fixed source, inputs, model configuration and tokenizer assets are still required. The [private handoff](local/research/COLAB-HANDOFF.md) identifies the approved folder, exact revisions/hashes and existing copy helper. `/content` remains ephemeral; export results before termination, while private Drive originals persist. Direct compressed upload/decompression remains the earlier alternative.
5. Retain a bounded remote job, result export, stop and **zero assignments** verification. A CLI/client timeout is not a remote GPU cutoff. AI Pro does not establish guaranteed background runtime survival; after disconnecting, inspect the existing job before launching a duplicate.

Verified scope: an L4 passed a 16-document BF16 evaluation at batch16; model loading took approximately7.5 seconds and evaluation15.09 seconds, with peak6.03GiB allocated/7.16GiB reserved. Observed rates were1.54 CU/hour for L4 and5.30 CU/hour for A100; these observations do not guarantee future rates. This small test does not qualify full panels or training. The existing owner handles the256-document probe; read its results before starting overlapping work.

Local evidence workspace: `/Users/tim_c_wang/Documents/Codex/2026-10-01/colab-readiness-cost-probe/`; asset identities are recorded in [COLAB-ASSET-HANDOFF.json](local/research/short-trials-20261001/COLAB-ASSET-HANDOFF.json). External sessions do not automatically receive these local/ignored files; the owner supplies the necessary non-secret handoff. The approved private Drive assets are now stored and transfer-verified; this does not establish full training qualification or authorize a new deployment.

Private Drive transfer was verified on CPU: model copy plus streaming hash took91.4 seconds (46.3 MB/s), and both files plus independent readback took114 seconds. The whole probe used approximately0.01983 CU including consent waiting; the observed CPU rate was0.08 CU/hour, not free. Release at18:00:45 Taipei on October2 verified zero assignments. This establishes transfer only, not model loading or GPU performance.

The CLI deep-link did not reliably attach to the allocated endpoint: browser Connect created a replacement CPU runtime. The previous runtime was released and the continuation used the browser runtime without a further allocation. Reconcile assignments before attaching a worker. Upload authorization (`drive.file`) and native Colab mount consent are separate; the latter is broader and may recur. Read-only filesystem mounting does not narrow OAuth consent. Exact private identifiers and immutable receipts stay in the ignored handoff, not this versioned guide.
