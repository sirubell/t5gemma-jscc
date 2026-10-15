# Project guide

Read README.md first, then docs/design-v2.md before changing model, data, loss or training behavior. Research state, plans and the professor's requirements are in the git-ignored `docs/local/research/` (start at `HANDOFF.md`, `STATUS-*.md`, `PLAN-*.md`, `PROFESSOR-FEEDBACK.md`; list files with `rg --files --no-ignore docs/local/research`). Cluster and workstation access details are in `docs/local/environment.md`.

## Principles

- Keep the code small. Do not add identity hashes, receipts, manifests, admission gates or review layers. The code identity of a run is the git commit recorded in its run directory; run experiments from a clean checkout of the intended commit.
- What protects results stays: stop on nonfinite loss/gradient, atomic checkpoint saves, best-validation checkpoint selection, pinned model revision.
- Results must be converged and comparable: use the early-stopping rule, change one factor per comparison, and use the same data, loss and evaluation for per-site and shared codecs (they share one trainer).
- Collaborators' channels plug in through the contract in docs/channel-integration.md; keep it working for every run type.
- Research choices belong in YAML, not in Python.

## Working conventions

- Explain results to the owner in Traditional Chinese; write code, docs and agent prompts in English.
- GPU jobs need an agreed scope; do not start extra jobs, retries or monitoring loops on your own.
- Report observed results separately from estimates; a passing CPU test is not a GPU or scientific result.
- Keep passwords, tokens and private keys out of all files. Machine paths go in an untracked `.env` or `docs/local/`.

## Checks

```bash
uv sync --locked --extra dev
uv run --locked --extra dev ruff check .
uv run --locked --extra dev pyright
uv run --locked --extra dev python -m pytest -q
```
