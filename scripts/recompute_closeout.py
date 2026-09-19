"""Offline evidence audit. Uses only the Python standard library and exported relative paths."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def bf16_round(value):
    bits = struct.unpack("I", struct.pack("f", value))[0]
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("f", struct.pack("I", rounded))[0]


def recompute(root):
    result = {
        "evaluation": {},
        "paired": {},
        "historical_timing": {},
        "historical_training": {},
        "cuda": {},
        "research": {},
    }
    panels = {}
    documents = {r["doc_id"]: r for r in rows(root / "samples/documents.jsonl")}
    for path in sorted((root / "results/evaluator").glob("*/*.json")):
        if path.name == "request-manifest.json":
            continue
        data = json.loads(path.read_text())
        request_manifest = json.loads(
            (path.parent / "request-manifest.json").read_text()
        )
        for req in request_manifest:
            arguments = documents[req["doc_id"]]["exact_candidate_arguments"][
                req["choice"]
            ]
            expected = hashlib.sha256(
                json.dumps(arguments[0], separators=(",", ":"), sort_keys=True).encode()
            ).hexdigest()
            assert expected == req["prompt_hash"]
        compact = data["compact"]
        ids = [r["doc_id"] for r in compact]
        assert len(ids) == len(set(ids)) == 512 and set(ids) == set(range(512)), path
        for row in compact:
            norm = [x / d for x, d in zip(row["raw_scores"], row["denominators"])]
            assert norm == row["normalized_scores"]
            assert max(range(4), key=lambda i: norm[i]) == row["prediction_norm"]
            assert row["gold_margin"] == norm[row["gold"]] - max(
                v for i, v in enumerate(norm) if i != row["gold"]
            )
        for capture in data["captures"]:
            assert (
                abs(bf16_round(sum(capture["old_log_probs"])) - capture["old_sum"])
                <= 1e-6
            )
            assert abs(
                sum(capture["fp32_log_probs"]) - capture["fp32_sum"]
            ) <= 1e-5 + 1e-5 * abs(capture["fp32_sum"])
        key = f"b{data['batch']}-r{data['repeat']}/{data['condition']}"
        result["evaluation"][key] = {
            "n": len(compact),
            "acc": sum(r["prediction"] == r["gold"] for r in compact) / len(compact),
            "acc_norm": sum(r["prediction_norm"] == r["gold"] for r in compact)
            / len(compact),
            "candidate_requests": data["request_count"],
            "model_forwards": data["forward_count"],
        }
        order = [
            [r["doc_id"], r["choice"]] for b in data["batches"] for r in b["companions"]
        ]
        h = hashlib.sha256(
            json.dumps(order, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
        assert h == data["request_order_hash"]
        panels[key] = data
    for condition in ["no_noise", "vanilla"]:
        a = panels["b8-r0/" + condition]
        for batch, repeat in [(8, 1), (16, 0), (32, 0)]:
            b = panels[f"b{batch}-r{repeat}/{condition}"]
            aa = {(r["doc_id"], r["choice"]): r for r in a["captures"]}
            bb = {(r["doc_id"], r["choice"]): r for r in b["captures"]}
            result["paired"][f"{condition}:8r0-vs-{batch}r{repeat}"] = {
                "normalized_prediction_flips": sum(
                    r["prediction_norm"] != s["prediction_norm"]
                    for r, s in zip(a["compact"], b["compact"])
                ),
                "max_raw_score_difference": max(
                    abs(x - y)
                    for r, s in zip(a["compact"], b["compact"])
                    for x, y in zip(r["raw_scores"], s["raw_scores"])
                ),
                "captured_valid_tokens_match": all(
                    r["context_ids"] == bb[k]["context_ids"]
                    and r["target_ids"] == bb[k]["target_ids"]
                    for k, r in aa.items()
                ),
                "captured_fp32_score_max_difference": max(
                    abs(r["fp32_sum"] - bb[k]["fp32_sum"]) for k, r in aa.items()
                ),
            }
    for path in sorted((root / "results/historical-evaluation").glob("*.jsonl")):
        groups = {}
        for row in rows(path):
            groups.setdefault(row["condition"], []).append(row["scoring_seconds"])
        result["historical_timing"][path.stem] = {
            k: statistics.median(v) for k, v in groups.items()
        }
    old = root / "results/historical-benchmark-repeats.jsonl"
    if old.exists():
        groups = {}
        for row in rows(old):
            groups.setdefault(row["matrix_id"], []).append(row)
        for key, group in groups.items():
            result["historical_training"][key] = {
                "updates_per_second": statistics.median(
                    r["measured_updates"] / r["wall_seconds"] for r in group
                ),
                "peak_allocated_gib": max(r["peak_allocated_gib"] for r in group),
            }
        b = result["historical_training"]
        b0, b2, b4 = (b[k] for k in ["B0", "B2", "B4_batch32"])
        result["historical_speedup"] = {
            "B2_vs_B0": b2["updates_per_second"] / b0["updates_per_second"] - 1,
            "B4_vs_B2": b4["updates_per_second"] / b2["updates_per_second"] - 1,
            "B4_time_reduction_vs_B2": 1
            - b2["updates_per_second"] / b4["updates_per_second"],
            "B2_allocated_reduction_vs_B0": 1
            - b2["peak_allocated_gib"] / b0["peak_allocated_gib"],
        }
    for path in sorted((root / "results").rglob("per-parameter*.jsonl")):
        entries = rows(path)
        tolerances = json.loads(
            (root / "configs/equivalence-tolerances.json").read_text()
        )
        for row in entries:
            near = row["reference_norm"] <= tolerances["near_zero_norm"]
            quantity = row["quantity"]
            absolute = (
                tolerances["near_zero_delta_absolute"]
                if quantity == "delta"
                else tolerances["near_zero_absolute"]
            )
            passed = (
                row["max_abs"] <= absolute
                if near
                else row["relative_l2"] <= tolerances[quantity + "_relative_l2"]
                and (
                    quantity in ("exp_avg", "exp_avg_sq")
                    or row["cosine"] is not None
                    and row["cosine"] >= tolerances["cosine_min"]
                )
            )
            assert passed == row["pass"], (path, row["parameter"])
            if row["reference_norm"] > 1e-8:
                rel = row["difference_norm"] / row["reference_norm"]
                assert math.isclose(rel, row["relative_l2"], rel_tol=1e-6, abs_tol=1e-8)
                if row["candidate_norm"] > 0:
                    cos = row["dot_product"] / (
                        row["reference_norm"] * row["candidate_norm"]
                    )
                    assert math.isclose(cos, row["cosine"], rel_tol=1e-6, abs_tol=1e-8)
        result["cuda"][str(path.relative_to(root))] = {
            "tensor_comparisons": len(entries),
            "failed_thresholds": sum(not r["pass"] for r in entries),
        }
    production = rows(root / "results/production-repeats.jsonl")
    measured = [r for r in production if r["measured_updates"] == 60]
    for row in production:
        expected = hashlib.sha256(
            json.dumps(row["actual_train_row_ids"]).encode()
        ).hexdigest()
        assert expected == row["data_order_hash"]
        if "snr_vectors" in row:
            expected = hashlib.sha256(
                json.dumps(row["snr_vectors"]).encode()
            ).hexdigest()
            assert expected == row["snr_hash"]
    by_variant = {}
    for variant in ["reference", "candidate"]:
        group = [r for r in measured if r["variant"] == variant]
        by_variant[variant] = {
            "median_updates_per_second": statistics.median(
                r["measured_updates"] / r["measured"]["wall_seconds"] for r in group
            ),
            "peak_allocated_gib": max(
                r["measured"]["training_peak_allocated_gib"] for r in group
            ),
        }
    ref, cand = by_variant["reference"], by_variant["candidate"]
    result["production"] = by_variant
    result["production_ratios"] = {
        "throughput_gain": cand["median_updates_per_second"]
        / ref["median_updates_per_second"]
        - 1,
        "time_reduction": 1
        - ref["median_updates_per_second"] / cand["median_updates_per_second"],
        "allocated_reduction": 1
        - cand["peak_allocated_gib"] / ref["peak_allocated_gib"],
    }
    probe = json.loads((root / "results/backbone-probe.json").read_text())
    result["backbone_probe"] = {}
    for precision in ["bfloat16", "float32"]:
        group = [r for r in probe if r["precision"] == precision]
        reference = next(r["score_fp32"] for r in group if r["batch_size"] == 8)
        result["backbone_probe"][precision] = {
            "max_score_difference_vs_batch8": max(
                abs(r["score_fp32"] - reference) for r in group
            )
        }
    ledger = root / "research/experiment-ledger.json"
    if ledger.exists():
        data = json.loads(ledger.read_text())
        if isinstance(data, dict):
            data = data.get("rows", data.get("experiments", []))
        result["research"] = {
            "ledger_rows": len(data),
            "cohorts": len({r["cohort_id"] for r in data}),
        }
    budget = json.loads((root / "budget-ledger.json").read_text())
    numerical_paths = [
        "results/cuda-enc_l9.json",
        "results/cuda-dec_l8.json",
        "results/cuda-deterministic/cuda-enc_l9.json",
        "results/cuda-deterministic-paired/cuda-enc_l9.json",
        "results/cuda-deterministic-paired/cuda-dec_l8.json",
    ]
    numerical = [
        json.loads((root / name).read_text())["budget"] for name in numerical_paths
    ]
    numerical_examples = sum(r["completed_backward_examples"] for r in numerical)
    numerical_steps = sum(r["optimizer_steps"] for r in numerical)
    production_all = [
        json.loads(p.read_text())
        for p in (root / "results/production").rglob("production_check.json")
    ]
    production_steps = sum(r["counters"]["optimizer_steps"] for r in production_all)
    smoke = json.loads((root / "results/split-smoke-budget.json").read_text())
    backward_examples = (
        numerical_examples
        + production_steps * 32
        + smoke["completed_backward_examples"]
    )
    optimizer_steps = numerical_steps + production_steps + smoke["optimizer_steps"]
    assert backward_examples == budget["backward"]["total_backward_examples"]
    assert optimizer_steps == budget["backward"]["actual_optimizer_steps"]
    assert backward_examples / 32 == budget["backward"]["total_update_equivalents"]
    request_count = sum(r["candidate_requests"] for r in result["evaluation"].values())
    probe_count = sum(r["forward_requests"] for r in probe)
    assert request_count == budget["task_inference"]["completed_candidate_requests"]
    assert probe_count == budget["task_inference"]["backbone_probe_candidate_requests"]
    upper = (request_count + probe_count) / 2048 + budget["task_inference"][
        "failed_instrumentation_panel_upper_bound"
    ]
    assert upper == budget["task_inference"]["total_panel_upper_bound"]
    assert upper <= budget["limits"]["task_panel_equivalents"]
    assert (
        backward_examples / 32
        <= budget["limits"]["effective_batch32_update_equivalents"]
    )
    result["budget_recomputed"] = {
        "backward_examples": backward_examples,
        "update_equivalents": backward_examples / 32,
        "optimizer_steps": optimizer_steps,
        "task_panel_upper_bound": upper,
    }
    required = {
        "gate_id",
        "status",
        "scope",
        "execution_source_hash",
        "config_hash",
        "evidence_paths",
        "observed",
        "thresholds",
        "limitations",
        "blocker",
        "next_action",
    }
    gates = json.loads((root / "gates.json").read_text())
    if isinstance(gates, dict):
        gates = gates["gates"]
    for gate in gates:
        assert required <= gate.keys(), gate["gate_id"]
        for path in gate["evidence_paths"]:
            assert (root / path).exists(), path
    result["gate_statuses"] = {g["gate_id"]: g["status"] for g in gates}
    for path in root.rglob("*.json"):
        json.loads(
            path.read_text(),
            parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)),
        )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1]
    )
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = recompute(args.root)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
