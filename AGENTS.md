# Project guide

## Read the relevant documents

- Start with README.md for both task workflows.
- Before changing model/data/loss behavior, read docs/architecture.md; for historical compatibility, read docs/migration.md.
- For test coverage and known limits, read docs/validation.md.
- For config layout, read configs/README.md; for study expansion and paired jobs, read docs/studies.md.
- Before remote access or Slurm work, read docs/running.md and the local docs/local/environment.md if present.
- For actual run IDs, paths and decisions, consult docs/local/experiments.md if present.
- For custom channels, read docs/channel-integration.md.
- For retrospective paper organization and future ablations, read docs/research-roadmap.md.

## Working conventions

This project primarily supports the owner's research workflow. Keep the README centered on features and experiments. Put collaboration details in the channel guide and personal infrastructure details in docs/local/.

Use uv. Both COCO and HellaSwag must remain visible in command documentation and task tests. Task YAMLs reference one shared model_config file; the loader saves a complete resolved configuration in every run. Put architectural variations in a named model design file rather than duplicating them inside task YAMLs.

Keep the supplied task and smoke model designs aligned with the shared baseline in docs/architecture.md. SNR-FiLM is off by default; enable it only for an explicitly requested FiLM experiment, not as part of routine channel integration.

Study preview/export prepares configurations only. Do not treat a listed study or an exported manifest as authorization to run a full sweep. Use a fresh prepared directory for each new execution; evaluate an entry through its recorded training-run link rather than searching for the newest checkpoint.

Git tracks code, configuration, uv.lock and portable docs. docs/local/ is intentionally ignored but remains readable by local agents. When it is absent, use docs/environment.example.md and request the missing machine/account details before remote work. Do not guess another user's SSH aliases, VPN commands or filesystem paths.

Keep passwords, access tokens and private keys out of both portable and local Markdown; refer to the installed credential mechanism. Update only current commits to remove personal notes; preserve existing Git history unless the user explicitly requests a rewrite.

The real-environment executability check is complete. A historical COCO timeout is not a pending request to resume training. New compute work must serve the user's current research request.

## Checks

```bash
uv sync --locked --extra dev
uv run --locked --extra dev pyright
uv run --locked --extra dev ruff check .
uv run --locked --extra dev python -m pytest -q
uv run --locked python train.py --config configs/tasks/coco.yaml --check
uv run --locked python train.py --config configs/tasks/hellaswag.yaml --check
```

Use focused CPU tests for behavior changes. For documentation-only edits, check content, links and commands rather than starting model runs. Preserve unrelated user edits.

Use the project's uv-managed Pyright/Ruff rather than assuming the owner's editor tools exist. Model code is CUDA-generic; Slurm is optional. Do not claim that a full configuration fits an RTX 5090 based on an H200 run, or that a full sweep has been GPU-tested because its small-model tests pass.

When recording a real experiment, include the recipe, source revision, job/run identifiers, evaluation checkpoint and completion state in local experiment notes; distinguish observed results from estimates.
