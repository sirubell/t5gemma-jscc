"""Run the authorized zero-update suite sequentially on one WS GPU.

No training or automatic retries. Parent may dispatch pilots separately only
once the core outputs and remaining shared budgets have been inspected.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root
    auth = json.loads((root / "AUTHORIZATION.json").read_text())
    if not auth["execution_authorized"] or auth["plan_id"] != "EVENING_SMALL_SUITE_V1":
        raise ValueError("Missing authorization")
    records = json.loads((root / "checkpoints/ws-receipts.json").read_text())
    lookup = {(r["route"], r["step"]): r for r in records}

    def sha(path):
        h = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    release = Path(__file__).resolve().parents[2]
    for name, expected in json.loads(
        (root / "source/execution-source.json").read_text()
    ).items():
        if sha(release / name) != expected:
            raise ValueError(f"Executed source changed: {name}")
    for name, expected in json.loads(
        (root / "source/input-manifest.json").read_text()
    ).items():
        if sha(root / name) != expected:
            raise ValueError(f"Frozen suite input changed: {name}")
    for record in records:
        if sha(record["path"]) != record["sha256"]:
            raise ValueError(
                f"Checkpoint bytes differ: {record['route']} step{record['step']}"
            )
    (root / "dispatch-identity-verification.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "checked_unix": time.time(),
                "checkpoint_hashes": {r["path"]: r["sha256"] for r in records},
                "source_manifest_sha256": sha(root / "source/execution-source.json"),
                "input_manifest_sha256": sha(root / "source/input-manifest.json"),
            },
            indent=2,
        )
        + "\n"
    )
    (root / "logs").mkdir(exist_ok=True)
    (root / "results").mkdir(exist_ok=True)
    deadline = auth["gpu_work_stop_at_unix"]
    commands = []
    # Keep all three E3 routes independent from evaluator controls/OOMs.
    for route in ["enc_l9", "dec_l0", "dec_l20"]:
        r = lookup[route, 10000]
        commands.append(
            (
                f"E3-{route}",
                900,
                [
                    sys.executable,
                    "-m",
                    "scripts.experiments.evening_gradients",
                    "--run",
                    r["run"],
                    "--checkpoint",
                    "step_010000.pt",
                    "--route",
                    route,
                    "--output",
                    str(root / "results" / f"E3-{route}"),
                    "--seconds",
                    "850",
                    "--vjp-cap",
                    "32" if route == "enc_l9" else "48",
                    "--deadline-epoch",
                    str(deadline),
                ],
            )
        )
    for route in ["dec_l0", "dec_l20"]:
        r = lookup[route, 10000]
        cmd = [
            sys.executable,
            "-m",
            "scripts.experiments.evening_eval",
            "--root",
            str(root),
            "--route",
            route,
            "--checkpoint",
            r["path"],
            "--checkpoint-sha256",
            r["sha256"],
            "--step",
            "10000",
            "--mode",
            "decoder-final",
        ]
        if route == "dec_l0":
            cmd += ["--vanilla"]
        commands.append((f"E1-E2-{route}", 900, cmd))
    for route, steps in [("enc_l9", [1000, 5000, 10000]), ("dec_l20", [1000, 5000])]:
        for step in steps:
            r = lookup[route, step]
            commands.append(
                (
                    f"E4-{route}-{step}",
                    300,
                    [
                        sys.executable,
                        "-m",
                        "scripts.experiments.evening_eval",
                        "--root",
                        str(root),
                        "--route",
                        route,
                        "--checkpoint",
                        r["path"],
                        "--checkpoint-sha256",
                        r["sha256"],
                        "--step",
                        str(step),
                        "--mode",
                        "curve",
                    ],
                )
            )
    # A single invocation owns all WS reservations, including load/failure time.
    # No retry is implicit; attempts below reserve 100 GPU-min in total.
    reservation = sum(seconds for _, seconds, _ in commands)
    if reservation != 6000 or reservation > auth["max_gpu_minutes"] * 60:
        raise ValueError("Incorrect reserved GPU budget")
    (root / "driver.started").open("x").write(str(time.time()))
    log = root / "commands.jsonl"
    outcomes = []
    for name, seconds, cmd in commands:
        if time.time() + seconds > deadline:
            outcomes.append(
                {
                    "name": name,
                    "status": "NOT_RUN",
                    "reason": "insufficient remaining bounded window",
                }
            )
            continue
        start = time.time()
        with log.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "name": name,
                        "command": cmd,
                        "start_unix": start,
                        "timeout_seconds": seconds,
                    }
                )
                + "\n"
            )
        try:
            with (root / "logs" / f"{name}.log").open("w") as f:
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=seconds,
                    check=False,
                )
            row = {
                "name": name,
                "exit_code": result.returncode,
                "status": "COMPLETED" if result.returncode == 0 else "FAILED",
            }
        except subprocess.TimeoutExpired:
            row = {"name": name, "status": "TIMEOUT", "exit_code": None}
        row.update(
            start_unix=start, end_unix=time.time(), gpu_seconds=time.time() - start
        )
        outcomes.append(row)
        (root / "driver-progress.json").write_text(
            json.dumps(outcomes, indent=2) + "\n"
        )
        print(json.dumps(row), flush=True)
    (root / "driver-completion.json").write_text(
        json.dumps(
            {
                "outcomes": outcomes,
                "reserved_gpu_seconds": reservation,
                "actual_gpu_seconds": sum(x.get("gpu_seconds", 0) for x in outcomes),
                "optimizer_updates": 0,
                "source_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
