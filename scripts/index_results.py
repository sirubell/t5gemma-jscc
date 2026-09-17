"""Archive small historical result files and index metrics without loading weights.

Usage: uv run python scripts/index_results.py --legacy-root /path/to/mtk --output docs/local/research
The output is private: it records original paths. It never deletes source files.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil


def metric_rows(data, family, metadata=None):
    """Keep reported scales/labels; absent sample counts remain unknown."""
    flags = ["historical_protocol", "source_not_bound_to_execution"]
    if family == "march":
        metadata = metadata or {}
        if metadata.get("history_run", {}).get("run_id") != metadata.get("eval_run", {}).get("run_id"):
            flags.append("mixed_run_provenance")
        rows = []
        for key, value in data.items():
            match = re.fullmatch(r"eval/(.+)_(acc_norm|acc)", key)
            if match and isinstance(value, (int, float)):
                rows.append({"task": "hellaswag", "split": "", "condition": match[1],
                             "metric": match[2], "value": value, "sample_count": None,
                             "flags": list(flags)})
        return rows
    result = data.get("result", data.get("results", {}))
    split = data.get("split", "")
    if "dec_" in split:
        flags.append("clean_memory_bypass_documented")
    rows = []
    for key, metric, task in (("acc,none", "acc", "hellaswag"),
                              ("acc_norm,none", "acc_norm", "hellaswag"),
                              ("coco_CIDEr,none", "cider", "coco")):
        if key in result:
            rows.append({"task": task, "split": split,
                         "condition": "vanilla" if split == "vanilla" else
                                      "no_noise" if data.get("snr") is None else str(data["snr"]),
                         "metric": metric, "value": result[key],
                         "sample_count": result.get("sample_len"), "flags": list(flags),
                         "seed": data.get("seed")})
    return rows


def candidates(root):
    for path in sorted((root / "june-experiment/results/eval_logs").rglob("*.json")):
        yield path, "june", "aggregate_result"
    for path in sorted((root / "march-experiment/experiments").rglob("*")):
        if path.is_file() and path.suffix in {".json", ".csv"}:
            yield path, "march", "aggregate_result" if path.name == "eval_summary.json" else "supporting_artifact"
    for path in sorted((root / "2025-experiments/results").rglob("*.csv")):
        yield path, "llama_2025", "unparsed_csv"
    path = root / "july/analysis/h200-coco-metrics.tsv"
    if path.is_file():
        yield path, "july_original", "downloaded_summary"
    documents = ["july/analysis/h200-source-and-next-session-handoff-2026-09-06.md",
                 "july/analysis/findings.md", "july/analysis/protocol_comparison.csv",
                 "july/snapshots/redesign-current/docs/experiment_ledger.md",
                 "july/snapshots/redesign-current/docs/arch_issue_decoder_crossattn.md",
                 "june-experiment/docs/may_vs_june_fulldata.md",
                 "t5gemma-jscc/docs/local/experiments.md", "t5gemma-jscc/docs/local/ws-validation.md"]
    for name in documents:
        path = root / name
        if path.is_file():
            yield path, "historical_documents", "documentary_evidence"


def build_index(root, output):
    root, output = root.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    artifacts, rows = [], []
    for source, family, kind in candidates(root):
        relative = source.relative_to(root)
        archived = output / "archive" / relative
        archived.parent.mkdir(parents=True, exist_ok=True)
        content = source.read_bytes()
        sha = hashlib.sha256(content).hexdigest()
        shutil.copy2(source, archived)
        if hashlib.sha256(archived.read_bytes()).hexdigest() != sha:
            raise RuntimeError(f"Archive verification failed: {source}")
        artifact = {"family": family, "kind": kind, "original": str(source),
                    "archive": str(archived.relative_to(output)), "sha256": sha,
                    "bytes": len(content), "metric_rows": 0}
        parsed = []
        if kind == "aggregate_result":
            try:
                data = json.loads(content)
                meta_path = source.with_name("run_meta.json")
                metadata = json.loads(meta_path.read_text()) if family == "march" and meta_path.exists() else None
                parsed = metric_rows(data, family, metadata)
                artifact["parse_status"] = "recognized" if parsed else "unrecognized_schema"
            except (ValueError, TypeError, AttributeError) as exc:
                artifact["parse_status"] = "error"
                artifact["parse_error"] = str(exc)
        elif kind == "downloaded_summary":
            for line in csv.reader(content.decode().splitlines(), delimiter="\t"):
                if len(line) >= 3:
                    parsed.append({"task": "coco", "split": line[0], "condition": line[1],
                                   "metric": "cider", "value": float(line[2]), "sample_count": None,
                                   "flags": ["downloaded_summary", "eval_train_overlap_documented",
                                             "historical_protocol", "not_new_baseline_evidence"]})
        artifact["metric_rows"] = len(parsed)
        artifacts.append(artifact)
        for row in parsed:
            rows.append({"family": family, "artifact": str(relative), "sha256": sha, **row})
    (output / "artifacts.json").write_text(json.dumps(artifacts, indent=2) + "\n")
    (output / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    fields = ["family", "artifact", "sha256", "task", "split", "condition", "metric",
              "value", "sample_count", "seed", "flags"]
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows({**row, "flags": ";".join(row["flags"])} for row in rows)
    summary = {"artifacts": len(artifacts), "metric_rows": len(rows),
               "archived_bytes": sum(a["bytes"] for a in artifacts),
               "unrecognized_or_error": [a["original"] for a in artifacts
                                         if a.get("parse_status") in {"error", "unrecognized_schema"}]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_index(args.legacy_root, args.output)
