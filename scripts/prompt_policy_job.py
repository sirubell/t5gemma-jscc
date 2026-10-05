"""Strict readers for historical fixed-budget prompt-policy engine runs.

The prompt-policy training/evaluation executor is retired. Historical readers
retain exact configuration, source, budget and checkpoint bindings.
"""

import hashlib
import json
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def validate_run(run, expected, config_hash):
    c = json.loads((run / "completion.json").read_text())
    m = json.loads((run / "run.json").read_text())
    from jscc.presentation import training_source_digest

    if not (
        c["status"] == "FULL_BUDGET_COMPLETED"
        and c["step"] == c["optimizer_updates"] == expected
        and c["presentations"] == 640000
        and c["presentation_budget_verified"]
        and c["source_verified"]
    ):
        raise ValueError("Incomplete fixed study budget")
    if (
        sha(run / "config.yaml") != m["resolved_config_sha256"]
        or m["training_source_sha256"] != training_source_digest()
    ):
        raise ValueError("Run config/source binding mismatch")
    # The resolved run path differs from the prepared recipe; check the original binding saved at launch.
    if (
        json.loads((run / "launch-binding.json").read_text())["prepared_config_sha256"]
        != config_hash
    ):
        raise ValueError("Prepared config mismatch")
    checkpoint = run / f"step_{expected:06d}.pt"
    if (
        c["final_checkpoint"]["file"] != checkpoint.name
        or sha(checkpoint) != c["final_checkpoint"]["sha256"]
    ):
        raise ValueError("Checkpoint binding mismatch")
    return checkpoint


if __name__ == "__main__":
    raise SystemExit("Prompt-policy execution is retired; use explicit ordinary training/evaluation commands.")
