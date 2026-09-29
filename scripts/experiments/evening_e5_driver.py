"""One bounded three-family E5 allocation; separate processes, no retries."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    root = a.root
    auth = json.loads((root / "AUTHORIZATION.json").read_text())
    deadline = auth["gpu_work_stop_at_unix"]
    if not auth["execution_authorized"] or time.time() + 2100 > deadline:
        raise ValueError("Three-arm allocation cannot fit remaining authorized window")
    release = Path(__file__).resolve().parents[2]
    for name, h in json.loads(
        (root / "source/execution-source.json").read_text()
    ).items():
        if sha(release / name) != h:
            raise ValueError("Source differs: " + name)
    for name, h in json.loads(
        (root / "source/input-manifest.json").read_text()
    ).items():
        if sha(root / name) != h:
            raise ValueError("Input differs: " + name)
    (root / "e5.started").open("x").write(str(time.time()))
    (root / "logs").mkdir(exist_ok=True)
    (root / "results").mkdir(exist_ok=True)
    started = time.time()
    records = []

    def execute(name, cmd, seconds):
        with (root / "commands.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    {
                        "name": name,
                        "command": cmd,
                        "timeout_seconds": seconds,
                        "start_unix": time.time(),
                    }
                )
                + "\n"
            )
        t = time.time()
        remaining = deadline - t
        if remaining <= 0:
            raise TimeoutError("Global GPU deadline reached before child dispatch")
        row: dict[str, Any] = {"name": name, "exit_code": None, "timed_out": False}
        try:
            with (root / "logs" / f"{name}.log").open("w") as f:
                proc = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=min(seconds, remaining),
                    check=False,
                )
            row["exit_code"] = proc.returncode
        except subprocess.TimeoutExpired:
            row["timed_out"] = True
            raise
        finally:
            row["seconds"] = time.time() - t
            records.append(row)
            (root / "e5-progress.json").write_text(json.dumps(records, indent=2))
        if row["exit_code"]:
            raise RuntimeError(f"{name} failed; no automatic retry")

    result: dict[str, Any] = {"status": "NOT_COMPLETED"}
    try:
        for family in ["affine_core", "shallow_nonlinear", "current_residual"]:
            out = root / "pilots" / family
            execute(
                "train-" + family,
                [
                    sys.executable,
                    "-m",
                    "scripts.experiments.evening_pilot",
                    "--config",
                    str(root / "configs/enc_l9.yaml"),
                    "--family",
                    family,
                    "--output",
                    str(out),
                    "--seconds",
                    "600",
                ],
                650,
            )
            report = json.loads((out / "report.json").read_text())
            run = Path(report["run"])
            c = json.loads((run / "completion.json").read_text())
            assert (
                c["status"] == "FULL_BUDGET_COMPLETED"
                and c["source_verified"]
                and c["presentation_budget_verified"]
            )
            assert (
                c["step"] == c["optimizer_updates"] == 500
                and c["presentations"] == 32000
            )
            ckpt = run / "step_000500.pt"
            h = sha(ckpt)
            assert c["final_checkpoint"]["sha256"] == h
            execute(
                "eval-" + family,
                [
                    sys.executable,
                    "-m",
                    "scripts.experiments.evening_eval",
                    "--root",
                    str(root),
                    "--route",
                    "enc_l9",
                    "--checkpoint",
                    str(ckpt),
                    "--checkpoint-sha256",
                    h,
                    "--step",
                    "500",
                    "--mode",
                    "pilot",
                ],
                50,
            )
        result = {
            "status": "COMPLETE",
            "optimizer_updates": 1500,
            "training_presentations": 96000,
        }
    except BaseException as e:
        result = {"status": "FAILED", "error": repr(e)}
        raise
    finally:
        result.update(records=records, elapsed_seconds=time.time() - started)
        (root / "e5-completion.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
