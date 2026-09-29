"""Bounded approved preflight family; no retries or formal submission."""

import argparse
import json
import hashlib
from pathlib import Path
import subprocess
import sys

TIMED = {"enc_emb", "enc_l9", "enc_l14", "enc_fn", "dec_l0", "dec_l24"}
ROUTES = {
    "enc": {"enc_emb", "enc_l4", "enc_l9", "enc_l14", "enc_l19", "enc_fn"},
    "dec": {"dec_l0", "dec_l4", "dec_l8", "dec_l12", "dec_l16", "dec_l20", "dec_l24"},
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--family", choices=["enc", "dec"], required=True)
    a = p.parse_args()
    root = a.root
    out = root / "preflight" / a.family
    out.mkdir(parents=True, exist_ok=False)
    configs = sorted((root / "configs").glob(a.family + "_*.yaml"))
    if {c.stem for c in configs} != ROUTES[a.family]:
        raise ValueError("Incomplete approved roster")
    bindings = json.loads((root / "INPUT-MANIFEST.json").read_text())
    for relative, digest in bindings.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
            raise ValueError("Prepared input hash mismatch")
    records = []
    updates = 0
    requests = 0
    for config in configs:
        name = config.stem
        measured = 20 if name in TIMED else 0
        trial = out / name
        updates += 3 + measured
        (out / "budget.json").write_text(
            json.dumps(
                {"optimizer_updates_reserved": updates, "requests_reserved": requests}
            )
            + "\n"
        )
        command = [
            sys.executable,
            "-m",
            "scripts.exact64_preflight",
            "--config",
            str(config),
            "--output",
            str(trial),
            "--measured",
            str(measured),
        ]
        with (out / f"{name}.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        c = json.loads((trial / "completion.json").read_text())
        records.append(c)
        if name in {"enc_fn", "dec_l0"}:
            for mode in ["smoke5", "smoke0"]:
                requests += 256 if mode == "smoke5" else 128
                (out / "budget.json").write_text(
                    json.dumps(
                        {
                            "optimizer_updates_reserved": updates,
                            "requests_reserved": requests,
                        }
                    )
                    + "\n"
                )
                link = trial / f"{mode}-eval.txt"
                cmd = [
                    sys.executable,
                    "evaluate.py",
                    "--run",
                    c["run_path"],
                    "--checkpoint",
                    c["checkpoint"],
                    "--expected-step",
                    str(c["expected_step"]),
                    "--config",
                    str(root / "evaluation" / f"{mode}.yaml"),
                    "--output-path-file",
                    str(link),
                ]
                with (out / f"{name}-{mode}.log").open("w") as log:
                    subprocess.run(
                        cmd, check=True, stdout=log, stderr=subprocess.STDOUT
                    )
                result = json.loads(
                    (Path(link.read_text().strip()) / "results.json").read_text()
                )
                assert result["optimizer_step"] == c["expected_step"] and len(
                    result["conditions"]
                ) == (2 if mode == "smoke5" else 1)
                expected_conditions = (
                    ["no_noise", -6] if mode == "smoke5" else ["no_noise"]
                )
                assert [
                    x["condition"] for x in result["conditions"]
                ] == expected_conditions
                for row in result["conditions"]:
                    assert row["compact_num_samples"] == 32 and row["finite_scores"]
                    assert row["num_fewshot"] == (5 if mode == "smoke5" else 0)
                    assert row["candidate_forward_requests"] == 128
        (out / "progress.json").write_text(json.dumps(records, indent=2) + "\n")
    (out / "completion.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "family": a.family,
                "routes": len(records),
                "updates_reserved": updates,
                "requests_reserved": requests,
                "results": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
