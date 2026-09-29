"""Execute an approved fixed-config train or eval task without checkpoint fallback."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--split", required=True)
    p.add_argument(
        "--phase",
        choices=["train", "eval5", "eval0", "vanilla5", "vanilla0"],
        required=True,
    )
    a = p.parse_args()
    root = a.root
    manifest = json.loads((root / "APPROVAL-CARD.json").read_text())
    config = root / "configs" / f"{a.split}.yaml"
    expected = manifest["config_sha256"][config.name]
    if sha(config) != expected:
        raise ValueError("Frozen config changed")
    link = root / "links" / f"{a.split}-train.txt"
    if a.phase == "train":
        subprocess.run(
            [
                sys.executable,
                "train.py",
                "--config",
                str(config),
                "--run-path-file",
                str(link),
            ],
            check=True,
        )
        run = Path(link.read_text().strip())
        (run / "launch-binding.json").write_text(
            json.dumps({"prepared_config_sha256": expected}) + "\n"
        )
        validate_run(run, 10000, expected)
    else:
        evaluation_config = root / "evaluation" / f"{a.phase}.yaml"
        if (
            sha(evaluation_config)
            != manifest["evaluation_sha256"][evaluation_config.name]
        ):
            raise ValueError("Evaluation config changed")
        run = Path(link.read_text().strip())
        checkpoint = validate_run(run, 10000, expected)
        out = root / "links" / f"{a.split}-{a.phase}.txt"
        subprocess.run(
            [
                sys.executable,
                "evaluate.py",
                "--run",
                str(run),
                "--checkpoint",
                checkpoint.name,
                "--expected-step",
                "10000",
                "--config",
                str(root / "evaluation" / f"{a.phase}.yaml"),
                "--output-path-file",
                str(out),
            ],
            check=True,
        )
        result = json.loads(
            (Path(out.read_text().strip()) / "results.json").read_text()
        )
        expected_conditions = (
            ["no_noise", -6, 6, 18]
            if a.phase == "eval5"
            else (["vanilla"] if a.phase.startswith("vanilla") else ["no_noise"])
        )
        count = len(expected_conditions)
        if result["optimizer_step"] != 10000 or len(result["conditions"]) != count:
            raise ValueError("Incomplete formal evaluation")
        if [c["condition"] for c in result["conditions"]] != expected_conditions:
            raise ValueError("Wrong condition scope")
        for c in result["conditions"]:
            if not (
                c["compact_num_samples"] == 10042
                and c["finite_scores"]
                and c["num_fewshot"] == (0 if a.phase.endswith("0") else 5)
                and c["candidate_forward_requests"] == 40168
            ):
                raise ValueError("Incomplete sample/request/scoring scope")


if __name__ == "__main__":
    main()
