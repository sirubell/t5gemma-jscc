# Local environment template

Copy this file to docs/local/environment.md. Replace placeholders with your own non-secret settings. Keep credentials in SSH/authentication tools, not in Markdown.

## Locations

| Purpose | Local value |
|---|---|
| Development checkout | <path> |
| GPU workstation checkout, if used | <path> |
| Scheduled server checkout, if used | <path> |
| Existing model/data caches | <paths or environment variable names> |
| Experiment artifacts | <path> |

## Connections

- SSH aliases: <aliases configured in your own ~/.ssh/config>
- Jump hosts: <route, if needed>
- VPN activation: <existing command, if needed>
- Login-shell requirement: <how PATH/modules are loaded>
- Credentials: <name of the existing credential mechanism; no key/token contents>

## Hardware and development tools

- OS / shell: <environment>
- GPU model, available VRAM and driver: <values>
- PyTorch version / CUDA build / BF16 support: <verified values>
- Available host RAM and worker count: <values>
- Project tools: install with uv sync --locked --extra dev; use uv-managed Pyright/Ruff.
- Editor Python interpreter: <project .venv path; editor configuration stays outside Git>
- Tested batch/accumulation/evaluation batch: <values and tested split>

## Scheduler, if used

- Partition: <partition>
- Account: <account>
- QOS: <qos, if required>
- GPU request syntax and limits: <site-specific settings>
- CPU/RAM rules and maximum wall time: <limits>
- Can users extend running jobs? <verified result or unknown>

## Current work

- Source revision / remote: <revision and repository location>
- Current request: <scope>
- Experiment notes: docs/local/experiments.md
- Known issues or differences between checkouts: <notes>
- Last verified: <date>

Store historical artifact paths separately in docs/local/legacy.md when useful. A result marked incomplete does not itself authorize another run.
