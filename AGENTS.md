# Project guide

## Communication and delivery

- Explain research work to the user in Traditional Chinese unless requested otherwise. Write documentation, agent instructions, technical artifacts and worker prompts in English.
- PRs: implement, validate and independently review; explain the proposal in its task and obtain publication approval. Then handle CI and review comments autonomously. Tim merges manually.
- Reviews: rely on repository-triggered initial reviews and retain automatic reviewer assignments. Request re-review only when changes warrant it; report meaningful developments rather than repeatedly polling unchanged status.
- Commits: include a concise body covering motivation, key changes and actual validation.
- Tools: prefer `rg` and `rg --files` for searches and structured parsers for JSON/YAML. Use available modern CLI tools when useful; do not install tools solely to satisfy this preference.

## Lean engineering

- Implement the simplest solution that meets the current task and its operational requirements. Add frameworks, wrappers or fallback paths only for a concrete need.
- Reuse existing functions and tools. Introduce abstractions for actual repetition or a materially simpler flow.
- Validate stable invariants at setup or trust boundaries, and revalidate when relevant state changes. Avoid redundant hot-path checks; retain validation of untrusted inputs, changing runtime state and security-sensitive operations.
- Preserve checks that prevent silent incorrect results, data loss or security failures. Surface exceptions and make configuration changes explicit; never silently swallow failures or substitute settings.
- Run the smallest sufficient relevant tests. Reuse valid evidence when code, data and environment are unchanged; broaden checks for changed risks, failures or unresolved concerns, and satisfy required project checks.
- Keep small changes lightweight. Update existing documentation when behavior or usage changes; add process documents or repeat reviews only for a concrete need or applicable requirement. Report the change, actual validation and remaining limits concisely.

These principles preserve project-specific safety, security, correctness and production requirements.

## Research session quick start

For research, experiment or Colab sessions, start with the concise [research workflow](docs/research-roadmap.md#research-workflow-2026-10-02), then the current local HANDOFF and assigned ticket. Colab setup is in [the runbook](docs/running.md#colab-runbook-2026-10-02).

## Read the relevant documents

- Start with README.md for both task workflows.
- Before changing model/data/loss behavior, read docs/architecture.md; for historical compatibility, read docs/migration.md. For local research changes, also follow HANDOFF.md to the frozen baseline record; preserve earlier baselines and create a new record when adopting a changed design.
- For test coverage and known limits, read docs/validation.md.
- For config layout, read configs/README.md; for study expansion and paired jobs, read docs/studies.md.
- Before remote access or Slurm work, read docs/running.md and the local docs/local/environment.md if present.
- For actual run IDs, paths and decisions, consult docs/local/experiments.md if present.
- When resuming research or scheduling experiments, read docs/local/research/HANDOFF.md if present for current state, owners and resume pointers. For backup work, also read backup-status.md if present. Current owner decisions supersede earlier proposals.
- Research planning uses one local Markdown Wayfinder tracker; read docs/local/research/wayfinder/codec-research/SESSION-GUIDE.md and the assigned ticket before starting a research task. Verify dependencies and claim exclusively before work. Keep decisions in tickets, execution state in HANDOFF.md and scientific evidence in result reports. Local research notes remain outside portable Git documentation.
- For custom channels, read docs/channel-integration.md.
- For retrospective research, start with docs/local/research/reported-experiments.md when present: prioritize presented experiments and map each slide to raw evidence. Use docs/research-roadmap.md for the portable research scope.
- For transmitter/receiver semantics, read CONTEXT.md and docs/adr/0001-transmission-boundary.md. For storage cleanup, read docs/local/research/cleanup-plan.md before proposing deletion paths.

## Working conventions

This project primarily supports the owner's research workflow. Keep the README centered on features and experiments. Put collaboration details in the channel guide and personal infrastructure details in docs/local/.

Use uv. Both COCO and HellaSwag must remain visible in command documentation and task tests. Task YAMLs reference one shared model_config file; the loader saves a complete resolved configuration in every run. Put architectural variations in a named model design file rather than duplicating them inside task YAMLs.

Keep the supplied task and smoke model designs aligned with the shared baseline in docs/architecture.md. SNR-FiLM is off by default; enable it only for an explicitly requested FiLM experiment, not as part of routine channel integration.

Study preview/export prepares configurations only. Do not treat a listed study or an exported manifest as authorization to run a full sweep. Use a fresh prepared directory for each new execution; evaluate an entry through its recorded training-run link rather than searching for the newest checkpoint.

Git tracks code, configuration, uv.lock and portable docs. docs/local/ is intentionally ignored but remains readable by local agents. When it is absent, use docs/environment.example.md and request the missing machine/account details before remote work. Do not guess another user's SSH aliases, VPN commands or filesystem paths.

Keep passwords, access tokens and private keys out of both portable and local Markdown; refer to the installed credential mechanism. Update only current commits to remove personal notes; preserve existing Git history unless the user explicitly requests a rewrite.

Earlier real-environment checks predate the receiver-only decoder-memory correction and HellaSwag selection holdout. CPU regression tests do not establish full-weight performance of those changes. New GPU work requires a concrete experiment scope and budget; a historical COCO timeout is not a pending request to resume training.

## Research validation

Validate fixed input/model/checkpoint identities when loading the experiment. During training, retain GPU-assignment, nonfinite-value and save-success checks. Record training, evaluation and collection status separately so reporting or upload failures do not erase verified scientific results.

## Checks

```bash
uv sync --locked --extra dev
uv run --locked --extra dev pyright
uv run --locked --extra dev ruff check .
uv run --locked --extra dev python -m pytest -q
uv run --locked python train.py --config configs/tasks/coco.yaml --check
uv run --locked python train.py --config configs/tasks/hellaswag.yaml --check
```

For documentation-only edits, check content, links and commands rather than starting model runs. Preserve unrelated user edits.

Use the project's uv-managed Pyright/Ruff rather than assuming the owner's editor tools exist. Model code is CUDA-generic; Slurm is optional. Do not claim that a full configuration fits an RTX 5090 based on an H200 run, or that a full sweep has been GPU-tested because its small-model tests pass.

When recording a real experiment, include the recipe, source revision, job/run identifiers, evaluation checkpoint and completion state in local experiment notes; distinguish observed results from estimates.
