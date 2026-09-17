"""Index already archived remote JSONs. No network, training or deletion.

Run: uv run python -m scripts.index_remote_results docs/local/research
"""
import argparse
import csv
import json
from pathlib import Path

from scripts.index_results import metric_rows


def observations(data):
    if not isinstance(data, dict):
        return []
    if isinstance(data.get("conditions"), list):
        rows = []
        for condition in data["conditions"]:
            for metric in ("acc", "acc_norm", "cider"):
                if metric in condition:
                    rows.append({"task": data.get("task", "coco" if metric == "cider" else "hellaswag"), "split": data.get("split"),
                                 "condition": condition["condition"], "metric": metric,
                                 "value": condition[metric], "sample_count": condition.get("num_samples", data.get("eval_samples")),
                                 "flags": ["historical_configuration", "source_not_bound_to_execution"]})
        return rows
    if "cider" in data:
        flags = ["historical_protocol", "reported_provenance_not_independently_reexecuted"]
        if "dec_" in data.get("split", ""):
            flags.append("clean_memory_bypass_documented")
        if "snapshot" in data:
            flags.append("selection_panel_not_final_report")
        return [{"task": "coco", "split": data.get("split"),
                 "condition": data.get("condition_id", data.get("snr", "unspecified")),
                 "metric": "cider", "value": data["cider"],
                 "sample_count": data.get("provenance", {}).get("eval_id_count"), "flags": flags}]
    if "acc_norm_codec" in data or "cider_codec" in data:
        metric = "acc_norm" if "acc_norm_codec" in data else "cider"
        rows = []
        flags = ["historical_protocol", "source_not_bound_to_execution"]
        if "dec_" in data.get("split", ""):
            flags.append("global_memory_coding_documented" if data.get("codec_encoder_memory")
                         else "clean_memory_bypass")
        for role in ("codec", "vanilla"):
            key = f"{metric}_{role}"
            if key in data:
                rows.append({"task": "hellaswag" if metric == "acc_norm" else "coco",
                             "split": data.get("split"), "condition": role,
                             "metric": metric, "value": data[key],
                             "sample_count": data.get("eval_samples", data.get("num_eval")),
                             "flags": flags})
        return rows
    if "split" in data and ("result" in data or "results" in data):
        return metric_rows(data, "remote")
    return []


def index(root):
    rows = []
    for host in ("ws", "h200"):
        manifest = root / f"{host}-artifacts.json"
        if not manifest.exists():
            continue
        for artifact in json.loads(manifest.read_text()):
            if "archive" not in artifact or not artifact["archive"].endswith(".json"):
                continue
            try:
                data = json.loads((root / artifact["archive"]).read_text())
            except ValueError:
                continue
            for row in observations(data):
                rows.append({"host": host, "artifact": artifact["original"],
                             "archive": artifact["archive"], "sha256": artifact["sha256"], **row})
    (root / "remote-metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (root / "remote-metrics.csv").open("w", newline="") as stream:
        fields = ["host", "artifact", "archive", "sha256", "task", "split", "condition",
                  "metric", "value", "sample_count", "seed", "flags"]
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                          for key, value in row.items()} for row in rows)
    print(json.dumps({"remote_metric_rows": len(rows)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    index(parser.parse_args().root)
