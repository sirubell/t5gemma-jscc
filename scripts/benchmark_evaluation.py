#!/usr/bin/env python3
"""Benchmark HellaSwag task scoring at several evaluation batch sizes.

The script uses one immutable corrected step-4000 checkpoint and the pinned
five-shot 512-row panel.  It does not write lm-eval samples; the existing
remote evaluation artifacts remain the provenance source for per-example
analysis.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import torch

from jscc.evaluation import evaluate_hellaswag
from jscc.models.split_model import build_model
from jscc.runtime import isolated_rng


def run_batch(run_path: Path, checkpoint_name: str, batch_size: int,
              repeats: int, output: Path):
    checkpoint_path = run_path / checkpoint_name
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = copy.deepcopy(state["config"])
    config["model"] = dict(config["model"])
    config["model"]["device"] = "cuda"
    settings = dict(config["evaluation"])
    settings.update({"batch_size": batch_size, "num_samples": 512,
                     "num_fewshot": 5})
    processor, model = build_model(config)
    model.load_communication_state(state)
    model.eval()
    results = []
    conditions = ["no_noise", "vanilla"]
    for repeat in range(repeats):
        for condition in conditions:
            snr = None if condition in ("no_noise", "vanilla") else float(condition)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with isolated_rng(config["seed"]), model.transmission(
                snr, bypass=condition == "vanilla"
            ):
                metrics = evaluate_hellaswag(
                    model, processor, settings, output=None,
                    condition=condition, data_settings=config["data"],
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            results.append({
                "status": "completed",
                "batch_size": batch_size,
                "repeat_id": repeat,
                "condition": condition,
                "num_samples": metrics.get("num_samples"),
                "num_fewshot": settings["num_fewshot"],
                "acc": metrics.get("acc"),
                "acc_norm": metrics.get("acc_norm"),
                "scoring_seconds": elapsed,
                "peak_allocated_gib": (
                    torch.cuda.max_memory_allocated() / 2**30
                    if torch.cuda.is_available() else None
                ),
                "peak_reserved_gib": (
                    torch.cuda.max_memory_reserved() / 2**30
                    if torch.cuda.is_available() else None
                ),
                "cache_policy": "fresh_harness_per_condition",
                "checkpoint_step": state["step"],
            })
        # Keep the model/checkpoint fixed across repeats.  The task harness
        # owns its own request cache policy; this script records that policy.
    output.mkdir(parents=True, exist_ok=True)
    with (output / "evaluation-repeats.jsonl").open("a") as stream:
        for row in results:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    print(json.dumps({"batch_size": batch_size, "results": results}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--checkpoint", default="step_004000.pt")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    try:
        run_batch(args.run.resolve(), args.checkpoint, args.batch_size,
                  args.repeats, args.output.resolve())
    except torch.cuda.OutOfMemoryError as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        row = {"status": "oom", "batch_size": args.batch_size,
               "repeat_id": None, "condition": None, "error": str(exc)}
        with (args.output / "evaluation-repeats.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row))
        raise SystemExit(0)


if __name__ == "__main__":
    main()
