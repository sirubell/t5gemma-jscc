"""Execute exactly one prepared study entry in a train or evaluation job."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def run_task(manifest_path, index, phase, checkpoint=None, expected_step=None, evaluation_config=None):
    if evaluation_config and phase != "evaluate":
        raise ValueError("evaluation config is only supported for evaluate")
    if expected_step is not None:
        if phase != "evaluate":
            raise ValueError("expected step is only supported for evaluate")
        if not checkpoint:
            raise ValueError("expected step requires an explicit checkpoint")
        if type(expected_step) is not int or expected_step < 1:
            raise ValueError("expected step must be a positive integer")
    checkpoint = checkpoint or "best.pt"
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest["version"] != 1:
        raise ValueError("Unsupported study manifest version")
    if index < 0 or index >= len(manifest["runs"]):
        raise ValueError("Study index is out of range")
    entry = manifest["runs"][index]
    directory = manifest_path.parent
    link = directory / entry["run_link"]
    root = Path(__file__).resolve().parents[1]
    if phase == "train":
        if link.exists():
            raise FileExistsError("This study entry already has a completed training run; export a new plan for a new run")
        command = [sys.executable, str(root / "train.py"), "--config", str(directory / entry["config"]),
                   "--run-path-file", str(link)]
    elif phase == "evaluate":
        # The train command writes this only after it returns successfully.
        run = link.read_text().strip()
        if not run:
            raise ValueError("Training run link is empty")
        command = [sys.executable, str(root / "evaluate.py"), "--run", run, "--checkpoint", checkpoint]
        if expected_step is not None:
            command += ["--expected-step", str(expected_step)]
        if evaluation_config:
            command += ["--config", str(Path(evaluation_config).resolve())]
        if entry.get("evaluation_link"):
            result_link = directory / entry["evaluation_link"]
            if result_link.exists():
                raise FileExistsError("This study entry already has a completed evaluation")
            command += ["--output-path-file", str(result_link)]
    else:
        raise ValueError("phase must be train or evaluate")
    print(f"{phase}: index={index} task={entry['task']} experiment={entry['experiment']} seed={entry['seed']}", flush=True)
    subprocess.run(command, cwd=root, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "evaluate"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--checkpoint", help="checkpoint filename (default: best.pt)")
    parser.add_argument("--expected-step", type=int, help="require this exact saved optimizer step")
    parser.add_argument("--evaluation-config", help="evaluation-only overrides, e.g. a shared vanilla panel")
    args = parser.parse_args()
    if args.evaluation_config:
        run_task(args.manifest, args.index, args.phase, args.checkpoint, args.expected_step, args.evaluation_config)
    else:
        run_task(args.manifest, args.index, args.phase, args.checkpoint, args.expected_step)


if __name__ == "__main__":
    main()
