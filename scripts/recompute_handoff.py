#!/usr/bin/env python3
"""Recompute summaries from a compact semcomm-speed evidence directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


def read_jsonl(path: Path):
    if not path.exists():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def summarize(root: Path) -> dict:
    repeats = read_jsonl(root / "results" / "benchmark-repeats.jsonl")
    evaluations = read_jsonl(root / "results" / "evaluation-repeats.jsonl")
    grouped = {}
    for row in repeats:
        if row.get("status") != "completed":
            continue
        grouped.setdefault(row["variant_id"], []).append(row)
    speed = {}
    for variant, rows in grouped.items():
        values = [row["updates_per_second"] for row in rows if row.get("updates_per_second") is not None]
        memories = [row["peak_allocated_gib"] for row in rows if row.get("peak_allocated_gib") is not None]
        speed[variant] = {
            "repeats": len(rows),
            "updates_per_second_median": median(values) if values else None,
            "updates_per_second_range": [min(values), max(values)] if values else None,
            "peak_allocated_gib_median": median(memories) if memories else None,
        }
    evaluation_summary = {}
    for row in evaluations:
        if row.get("status") != "completed":
            continue
        key = f"batch{row['batch_size']}/{row['condition']}"
        evaluation_summary.setdefault(key, []).append(row)
    evaluation = {}
    for key, rows in evaluation_summary.items():
        times = [row["scoring_seconds"] for row in rows]
        scores = [row.get("acc_norm") for row in rows if row.get("acc_norm") is not None]
        evaluation[key] = {
            "repeats": len(rows),
            "scoring_seconds_median": median(times),
            "scoring_seconds_range": [min(times), max(times)],
            "acc_norm_values": scores,
        }
    samples = []
    for path in sorted((root / "samples").glob("compact_*.jsonl")):
        samples.extend(read_jsonl(path))
    sample_checks = {"records": len(samples), "conditions": sorted({row.get("condition") for row in samples})}
    return {"training": speed, "evaluation": evaluation, "sample_checks": sample_checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.root.resolve())
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
