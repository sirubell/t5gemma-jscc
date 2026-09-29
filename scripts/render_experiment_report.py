#!/usr/bin/env python3
"""Render verified experiment-records-v1 evidence locally, without loading models.

PNG panels and their exact long-form CSV data are retained with input hashes.
No transfer, upload, pruning, model import, or checkpoint deserialization occurs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import random
import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

# Direct script execution must import this checkout rather than an installed copy.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 0
LIMITS = (
    "Descriptive 95% percentile intervals condition on these weights, development "
    "items and fixed channel draws; they omit training-seed variation and multiplicity "
    "correction. No equivalence, adoption or noninferiority threshold is inferred."
)


def _number(value: Any) -> float | None:
    if isinstance(value, dict) and "value" in value:
        value = value["value"]
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    return float(value) if math.isfinite(value) else None


def _percentile(values: list[float], p: float) -> float:
    position = (len(values) - 1) * p
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def paired_group_bootstrap(left: list[dict], right: list[dict], metric: str) -> dict:
    """Difference left-minus-right; sample source groups, retain all their rows.

    Item membership and item-to-source mapping must match exactly. Missing data
    blocks the specified analysis instead of changing its population.
    """
    def index(rows: list[dict]) -> dict:
        result = {}
        for row in rows:
            item = row.get("item_id")
            source = row.get("source_id")
            if item is None or source is None or not str(source):
                raise ValueError("paired bootstrap requires item_id and source_id")
            if item in result:
                raise ValueError("duplicate paired item_id")
            value = _number(row.get(metric))
            if value is None:
                raise ValueError(f"missing paired metric {metric}")
            result[item] = (source, value)
        return result
    a, b = index(left), index(right)
    if not a or a.keys() != b.keys():
        raise ValueError("paired item membership differs or is empty")
    groups: dict[str, list[float]] = defaultdict(list)
    for item in sorted(a, key=str):
        if a[item][0] != b[item][0]:
            raise ValueError("paired item source_id mapping differs")
        groups[str(a[item][0])].append(a[item][1] - b[item][1])
    ordered = sorted(groups)
    sums = [sum(groups[g]) for g in ordered]
    sizes = [len(groups[g]) for g in ordered]
    rng = random.Random(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = [rng.randrange(len(ordered)) for _ in ordered]
        draws.append(sum(sums[i] for i in sampled) / sum(sizes[i] for i in sampled))
    draws.sort()
    return {
        "estimate": sum(sums) / sum(sizes), "low": _percentile(draws, .025),
        "high": _percentile(draws, .975), "rows": len(a), "groups": len(groups),
        "group_sizes": {g: len(groups[g]) for g in ordered},
        "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
        "weighting": "row-weighted within paired source_id group resamples",
        "limits": LIMITS,
    }


def _scalars(value: Any, prefix: str = ""):
    if isinstance(value, dict) and "value" in value:
        yield prefix, _number(value), value.get("reason") or ""
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _scalars(child, f"{prefix}.{key}".strip("."))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from _scalars(child, f"{prefix}.{i}")
    elif isinstance(value, bool):
        yield prefix, int(value), "boolean flag (0=false, 1=true)"
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, _number(value), "" if _number(value) is not None else "nonfinite"
    elif value is None:
        yield prefix, None, "unavailable"


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def _font(size: int):
    # Pillow's bundled scalable font keeps the report independent of OS fonts.
    return ImageFont.load_default(size=size)


def _plot(path: Path, title: str, rows: list[dict], *, x_label: str, categorical: bool = False) -> None:
    """Draw individual series; nulls break lines and remain explicit red marks."""
    width, height = 1450, 900
    im = Image.new("RGB", (width, height), "#fbfcfe")
    draw = ImageDraw.Draw(im)
    draw.text((40, 24), title, fill="#172b4d", font=_font(20))
    draw.text((40, 60), "Observed records only | crosses = missing/failed | dashed = phase boundaries",
              fill="#42526e", font=_font(15))
    left, right, top, bottom = 105, 1025, 120, 775
    valid_x = [float(r["x"]) for r in rows if _number(r.get("x")) is not None]
    valid_y = [float(r[k]) for r in rows for k in ("value", "low", "high") if _number(r.get(k)) is not None]
    xmin, xmax = (min(valid_x), max(valid_x)) if valid_x else (0., 1.)
    ymin, ymax = (min(valid_y), max(valid_y)) if valid_y else (0., 1.)
    if xmax == xmin:
        xmax = xmin + 1
    if ymax == ymin:
        ymin, ymax = ymin - .5, ymax + .5
    pad = (ymax - ymin) * .08
    ymin, ymax = ymin - pad, ymax + pad
    if categorical:
        ymin = min(0., ymin)
        xmax, xmin = xmax + .5, xmin - .5
    def px(x):
        return left + (x - xmin) / (xmax - xmin) * (right - left)
    def py(y):
        return bottom - (y - ymin) / (ymax - ymin) * (bottom - top)
    for i in range(6):
        y = ymin + (ymax - ymin) * i / 5
        yy = py(y)
        draw.line((left, yy, right, yy), fill="#dfe5ed")
        draw.text((8, yy - 7), f"{y:.5g}", fill="#42526e", font=_font(13))
        x = xmin + (xmax - xmin) * i / 5
        if not categorical:
            draw.text((px(x) - 15, bottom + 12), f"{x:.6g}", fill="#42526e", font=_font(13))
    draw.line((left, top, left, bottom, right, bottom), fill="#52647a", width=2)
    draw.text((left, bottom + 45), x_label, fill="#172b4d", font=_font(18))
    series: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        series[str(row["series"])].append(row)
    colors = ["#0065b3", "#d14900", "#00856a", "#8146ba", "#bd2854", "#627000"]
    for n, (label, points) in enumerate(sorted(series.items())):
        color = colors[n % len(colors)]
        draw.multiline_text((1045, 120 + n * 76), "\n".join(textwrap.wrap(label, 46)), fill=color, font=_font(12))
        last, phase = None, None
        for row in points:
            x, y = _number(row.get("x")), _number(row.get("value"))
            if phase is not None and phase != row.get("phase") and x is not None:
                for y0 in range(top, bottom, 12):
                    draw.line((px(x), y0, px(x), y0 + 5), fill="#8796a8")
                draw.text((px(x) + 3, top + 5 + n * 15), str(row.get("phase")), fill=color, font=_font(12))
                last = None
            phase = row.get("phase")
            if x is None or y is None:
                xx = px(x) if x is not None else left + 12 + n * 12
                yy = bottom - 12 - n * 7
                draw.line((xx - 4, yy - 4, xx + 4, yy + 4), fill="#c42336", width=2)
                draw.line((xx - 4, yy + 4, xx + 4, yy - 4), fill="#c42336", width=2)
                last = None
                continue
            point = (px(x), py(y))
            if _number(row.get("low")) is not None and _number(row.get("high")) is not None:
                draw.line((point[0], py(row["low"]), point[0], py(row["high"])), fill=color, width=2)
            if categorical:
                draw.line((point[0], bottom, point[0], point[1]), fill=color, width=18)
                draw.multiline_text((point[0] - 25, bottom + 8), "\n".join(textwrap.wrap(str(row.get("site_id", x)), 12)), fill=color, font=_font(11))
            elif last is not None and row.get("connect", True):
                draw.line((*last, *point), fill=color, width=2)
            if row.get("learner_kind") == "initialization":
                draw.text((point[0] + 4, point[1] - 15), "init", fill=color, font=_font(11))
            draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=color)
            last = point
    if not valid_y:
        draw.text((left + 30, top + 70), "UNAVAILABLE — no numeric observations", fill="#c42336", font=_font(24))
    draw.text((40, 860), "Exact values, units, identities and missing reasons are in the companion CSV.",
              fill="#42526e", font=_font(15))
    im.save(path)


def _category(metric: str) -> str:
    if any(token in metric for token in ("objective", "weighted", "raw", "loss")):
        return "objective"
    if any(token in metric for token in ("lr", "learning_rate")):
        return "learning-rate"
    if any(token in metric for token in ("grad", "clip", "update_l2", "nonfinite", "skipped")):
        return "gradient-and-update"
    if any(token in metric for token in ("memory", "allocated", "reserved")):
        return "memory"
    if any(token in metric for token in ("time", "elapsed", "throughput", "seconds")):
        return "timing"
    return "accounting"


def _slug(value: str) -> str:
    readable = "".join(c if c.isalnum() or c in "-_" else "-" for c in value).strip("-")[:160]
    return f"{readable}-{hashlib.sha256(value.encode()).hexdigest()[:16]}"


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _at(value: dict, *paths: str) -> Any:
    for path in paths:
        current: Any = value
        for key in path.split("."):
            current = current.get(key) if isinstance(current, dict) else None
        if current is not None:
            return current
    return None


def _update_rows(events: list[dict]) -> list[dict]:
    rows = []
    for sequence, event in enumerate(events):
        if event["event_type"] != "update":
            continue
        payload = event["payload"]
        step = _number(_at(payload, "axes.completed_steps", "axes.completed_updates", "completed_steps", "completed_step"))
        tokens = _number(_at(payload, "exposure.source_valid_tokens_total", "exposure.cumulative_source_valid_tokens", "exposure.cumulative_valid_tokens", "axes.source_valid_tokens", "source_valid_tokens"))
        kind = str(_at(payload, "objective.kind", "objective_kind", "phase_kind") or event["phase_id"])
        kind = "local" if "local" in kind or "reconstruction" in kind else "combined"
        for metric, value, reason in _scalars(payload):
            rows.append({"run_id": event["run_id"], "phase": event["phase_id"], "task": event["task"],
                         "sequence": sequence, "attempted_step": payload["attempted_step"], "objective_kind": kind, "completed_step": step,
                         "source_valid_tokens": tokens, "metric": metric, "value": value,
                         "reason": reason, "units": event["units"], "identity": event["identity"]})
    expanded = []
    previous = {}
    for row in rows:
        key = (row["run_id"], row["phase"], row["metric"])
        prior = previous.get(key)
        if prior is not None and row["attempted_step"] > prior + 1:
            expanded.append({**row, "attempted_step": prior + 1, "completed_step": prior + 1,
                             "source_valid_tokens": None, "value": None, "reason": "missing update record"})
        expanded.append(row)
        previous[key] = row["attempted_step"]
    return expanded


def _observation_rows(events: list[dict]) -> tuple[list[dict], list[dict]]:
    rows, observations = [], []
    for event in events:
        if event["event_type"] != "observation":
            continue
        payload = event["payload"]
        request = payload["request"]
        conditions = list(payload["conditions"])
        present = [c["condition"] for c in conditions]
        for missing in request["conditions"]:
            if missing not in present:
                unavailable = {"value": None, "reason": "missing condition"}
                objective = request["purpose"] == "objective"
                metric_names = ("cider",) if request["task"] == "coco" else ("normalized_accuracy", "raw_accuracy")
                conditions.append({"condition": missing, "requested": request["expected_items"],
                                   "completed": 0, "failed": None, "denominator": 0, "items": [],
                                   "metrics": {} if objective else {name: unavailable for name in metric_names},
                                   "objectives": {component: {field: unavailable for field in ("raw", "weighted", "numerator", "denominator")}
                                                  for component in ("K", "R")} if objective else {}})
        for condition in conditions:
            observations.append({"event": event, "request": request, "condition": condition})
            metrics = {**condition.get("metrics", {}),
                       **{"objective." + k: v for k, v in condition.get("objectives", {}).items()}}
            if not metrics:
                metrics = {"unavailable": {"value": None, "reason": payload.get("status", "missing condition metrics")}}
            for metric, value, reason in _scalars(metrics):
                rows.append({"run_id": event["run_id"], "phase": event["phase_id"], "task": event["task"],
                             "site": request.get("site", request.get("site_id")), "role": request.get("role"),
                             "purpose": request.get("purpose"), "objective_kind": request.get("objective_kind"), "site_step": request.get("site_step"),
                             "learner_kind": request.get("learner_kind"), "condition": condition["condition"],
                             "checkpoint": request.get("checkpoint"), "step": request.get("step"),
                             "metric": metric, "value": value if condition["completed"] == condition["requested"] and condition["failed"] == 0 else None,
                             "partial_value": value, "reason": reason or ("incomplete condition" if condition["completed"] != condition["requested"] or condition["failed"] != 0 else ""),
                             "status": payload["status"], "requested": condition["requested"],
                             "completed": condition["completed"], "failed": condition["failed"],
                             "denominator": condition["denominator"], "observation_identity": payload["identity"],
                             "request": request, "units": event["units"]})
    return rows, observations


def _comparison_key(request: dict, *, vanilla: bool = False) -> str:
    # Vanilla bypass has no codec insertion or channel draw. Those fields must
    # truthfully differ; input/scoring membership and runtime remain matched.
    fields = ("source", "data", "panel", "task", "scorer", "layout", "precision", "backend", "expected_items")
    details = ("prompt", "native_policy")
    if not vanilla:
        fields += ("parent", "site", "protocol", "noise", "comparison")
        details += ("draws",)
    return json.dumps({**{field: request.get(field) for field in fields},
                       "details": {key: request.get("details", {}).get(key) for key in details}}, sort_keys=True)


def _paired_series(row: dict) -> str:
    identity = hashlib.sha256(row["cohort"].encode()).hexdigest()[:12]
    return (f'{row["shared_run"]} - {row["reference_run"]} | {row["site"]} | '
            f'{row["condition"]} | {row["contrast"]} | cohort {identity}')


def _comparisons(observations: list[dict]) -> tuple[list[dict], list[dict]]:
    candidates = [o for o in observations if o["request"].get("purpose") == "task"
                  and o["request"]["task"] == "hellaswag"]
    results, blocked = [], []
    for left in candidates:
        request = left["request"]
        if request.get("learner_kind") != "shared":
            continue
        allowed = ("initialization", "vanilla") if request.get("role") == "heldout" else ("specialist",)
        for kind in allowed:
            vanilla = kind == "vanilla"
            cohort = _comparison_key(request, vanilla=vanilla)
            matches = [right for right in candidates
                       if right["request"].get("learner_kind") == kind
                       and _comparison_key(right["request"], vanilla=vanilla) == cohort
                       and (vanilla or right["condition"]["condition"] == left["condition"]["condition"])
                       and (kind != "specialist" or right["request"].get("site_step") == request.get("site_step"))]
            base = {"site": request["site"], "condition": left["condition"]["condition"],
                    "step": request.get("site_step"), "global_step": request.get("step"),
                    "phase": left["event"]["phase_id"], "contrast": "shared-minus-" + kind,
                    "shared_run": left["event"]["run_id"], "cohort": cohort}
            if len(matches) != 1:
                blocked.append({**base, "reason": "ambiguous matched observations" if matches else "no compatible comparison observation",
                                "candidate_count": len(matches),
                                "candidate_identities": [r["event"]["payload"]["identity"] for r in matches]})
                continue
            right = matches[0]
            base["reference_run"] = right["event"]["run_id"]
            for metric in ("normalized_correct", "raw_correct"):
                try:
                    for side in (left, right):
                        c = side["condition"]
                        if side["event"]["payload"]["status"] != "complete" or c["failed"] != 0 or c["completed"] != c["requested"]:
                            raise ValueError("paired observation incomplete")
                    estimate = paired_group_bootstrap(left["condition"]["items"], right["condition"]["items"], metric)
                except (ValueError, KeyError) as error:
                    blocked.append({**base, "metric": metric, "reason": str(error)})
                else:
                    results.append({**base, "metric": metric, **estimate})
    return results, blocked


def _explicit_comparisons(events: list[dict], manifest: dict) -> tuple[list[dict], list[dict]]:
    """Resolve requested scientific contrasts by exact observation digests."""
    if not isinstance(manifest, dict) or set(manifest) != {"schema", "contrasts"} or manifest["schema"] != "experiment-comparisons-v1" or not isinstance(manifest["contrasts"], list):
        raise ValueError("invalid experiment-comparisons-v1 manifest")
    indexed: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        if event["event_type"] == "observation":
            indexed[event["payload"]["identity"]].append(event)
    ids = [c.get("id") for c in manifest["contrasts"] if isinstance(c, dict)]
    results, blocked = [], []
    for contrast in manifest["contrasts"]:
        base = {"contrast": contrast.get("id") if isinstance(contrast, dict) else None,
                "requested_contrast": contrast}
        try:
            if not isinstance(contrast, dict) or set(contrast) != {"id", "left_observation", "right_observation", "metrics", "allowed_control_differences"}:
                raise ValueError("invalid requested contrast fields")
            if not isinstance(contrast["id"], str) or not contrast["id"].strip() or ids.count(contrast["id"]) != 1:
                raise ValueError("contrast id must be nonempty and unique")
            allowed = contrast["allowed_control_differences"]
            if not isinstance(allowed, list) or any(not isinstance(k, str) for k in allowed) or len(allowed) != len(set(allowed)) or not set(allowed) <= {"architecture", "initialization", "objective"}:
                raise ValueError("unsupported allowed control difference; training_data/exposure/schedule cannot vary")
            bound = []
            for key in ("left_observation", "right_observation"):
                identity = contrast[key]
                if not isinstance(identity, str) or len(indexed.get(identity, [])) != 1:
                    raise ValueError(f"{key} must bind exactly one known observation event")
                bound.append(indexed[identity][0])
            left, right = bound
            lp, rp = left["payload"], right["payload"]
            lq, rq = lp["request"], rp["request"]
            if any(p["status"] != "complete" or p["request"]["purpose"] != "task" for p in (lp, rp)):
                raise ValueError("requested contrast requires complete task observations")
            common = ("task", "source", "data", "panel", "site", "role", "site_step", "protocol", "noise", "scorer", "layout", "precision", "backend", "expected_items", "conditions", "details")
            if any(lq[k] != rq[k] for k in common):
                raise ValueError("requested contrast input/scoring/noise/site/exposure identity mismatch")
            if "initialization" not in allowed and lq["parent"] != rq["parent"]:
                raise ValueError("requested contrast parent differs without initialization allowance")
            for control in ("architecture", "initialization", "training_data", "objective", "exposure", "schedule"):
                if control not in allowed and lq["comparison"][control] != rq["comparison"][control]:
                    raise ValueError(f"undeclared comparison control difference: {control}")
            metrics = contrast["metrics"]
            supported = {"normalized_correct", "raw_correct"} if lq["task"] == "hellaswag" else {"cider"}
            if not isinstance(metrics, list) or not metrics or any(not isinstance(m, str) for m in metrics) or len(set(metrics)) != len(metrics) or not set(metrics) <= supported:
                raise ValueError("unsupported contrast metrics for task")
            if [c["condition"] for c in lp["conditions"]] != lq["conditions"] or [c["condition"] for c in rp["conditions"]] != rq["conditions"]:
                raise ValueError("requested contrast missing condition")
            cohort = json.dumps({"contrast": contrast, "left_request": lq, "right_request": rq}, sort_keys=True)
            base.update({"shared_run": left["run_id"], "reference_run": right["run_id"],
                         "left_run": left["run_id"], "right_run": right["run_id"], "site": lq["site"],
                         "step": lq["site_step"], "global_step": lq["step"], "phase": left["phase_id"],
                         "cohort": cohort, "analysis": "explicit-left-minus-right"})
            pending = []
            for lc, rc in zip(lp["conditions"], rp["conditions"], strict=True):
                if any(c["failed"] != 0 or c["completed"] != c["requested"] for c in (lc, rc)):
                    raise ValueError("requested contrast contains incomplete condition")
                for metric in metrics:
                    estimate = paired_group_bootstrap(lc["items"], rc["items"], metric)
                    pending.append({**base, "condition": lc["condition"], "metric": metric, **estimate})
            results.extend(pending)
        except (ValueError, KeyError, TypeError) as error:
            blocked.append({**base, "reason": str(error)})
    return results, blocked


def _worst_sites(quality: list[dict], comparisons: list[dict]) -> list[dict]:
    result = []
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in quality:
        if (row["learner_kind"] == "shared" and row["role"] == "trained"
                and row["purpose"] == "task" and row["metric"] in ("normalized_accuracy", "raw_accuracy", "cider")
                and row["status"] == "complete" and _number(row["value"]) is not None):
            groups[(row["run_id"], str(row["checkpoint"]), str(row["condition"]), row["metric"])].append(row)
    for key, values in groups.items():
        minimum = min(values, key=lambda row: row["value"])
        result.append({"summary": "worst observed trained-site absolute performance", "cohort": key,
                       "site": minimum["site"], "value": minimum["value"], "step": minimum["step"],
                       "phase": minimum["phase"], "metric": minimum["metric"],
                       "observed_sites": sorted(str(v["site"]) for v in values),
                       "scope": "observed sites only; consult completeness for missing sites"})
    gaps: dict[tuple, list[dict]] = defaultdict(list)
    for row in comparisons:
        if row["contrast"] == "shared-minus-specialist" and row.get("analysis") != "explicit-left-minus-right" and "l14" not in row["site"]:
            gaps[(row["shared_run"], str(row["condition"]), row["step"], row["metric"])].append(row)
    for key, values in gaps.items():
        minimum = min(values, key=lambda row: row["estimate"])
        result.append({"summary": "largest observed matched trained-site degradation (minimum shared-minus-specialist)",
                       "cohort": key, "site": minimum["site"], "value": minimum["estimate"],
                       "step": minimum["step"], "phase": minimum["phase"], "metric": minimum["metric"],
                       "scope": "observed matched sites only"})
    return result


def render_local_report(record_directory: str | Path, output_directory: str | Path) -> dict:
    """Verify every input inventory before writing a local, model-free report."""
    from jscc.experiment_records import completion_status, read_events, verify_inventory

    root, output = Path(record_directory).resolve(), Path(output_directory).resolve()
    runs = [root] if (root / "metrics.jsonl").exists() else sorted(p.parent for p in root.glob("*/metrics.jsonl"))
    if not runs:
        raise ValueError("no run metrics.jsonl found")
    if output == root or any(output == run or run.is_relative_to(output) for run in runs):
        raise ValueError("output must not overwrite input records")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty; preserve prior report artifacts")
    events, completions, inputs = [], [], []
    inventories = {}
    for run in runs:
        inventory_path, manifest_path = run / "inventory.json", run / "manifest.json"
        inventory = json.loads(inventory_path.read_text())
        verify_inventory(run, inventory)
        # Bind the consumed event log to the verified inventory, not just a hash
        # produced after reading an unverified file.
        if "metrics.jsonl" not in {a["path"] for a in inventory["artifacts"]}:
            raise ValueError("metrics.jsonl missing from verified inventory")
        current = read_events(run / "metrics.jsonl")
        manifest = json.loads(manifest_path.read_text())
        status = completion_status(manifest, current, inventory, run)
        completion_path = run / "completion.json"
        if completion_path.exists() and json.loads(completion_path.read_text()) != status:
            raise ValueError("saved completion disagrees with recomputed evidence")
        completions.append(status)
        inventories[manifest["run_id"]] = inventory
        events.extend(current)
        paths = {run / a["path"] for a in inventory["artifacts"]}
        paths.update({inventory_path, manifest_path})
        if completion_path.exists():
            paths.add(completion_path)
        inputs.extend({"path": str(p), "sha256": _digest(p), "bytes": p.stat().st_size} for p in sorted(paths))
    updates = _update_rows(events)
    quality, observations = _observation_rows(events)
    missing_runs = {c["run_id"] for c in completions if c["missing_observations"]}
    for row in quality:
        row["connect"] = row["run_id"] not in missing_runs
    comparisons, blocked = _comparisons(observations)
    comparison_manifest = None
    comparison_path = root / "comparisons.json"
    if comparison_path.exists():
        comparison_manifest = json.loads(comparison_path.read_text())
        explicit, explicit_blocked = _explicit_comparisons(events, comparison_manifest)
        comparisons.extend(explicit)
        blocked.extend(explicit_blocked)
        inputs.append({"path": str(comparison_path), "sha256": _digest(comparison_path), "bytes": comparison_path.stat().st_size})
    worst = _worst_sites(quality, comparisons)
    costs = []
    failures = []
    footprints = []
    for event in events:
        p = event["payload"]
        if event["event_type"] == "cost":
            costs.append({"run_id": event["run_id"], "phase": event["phase_id"], "task": event["task"],
                          **p, "value": _number(p["seconds"]), "reason": p["seconds"]["reason"],
                          "units": "seconds", "identity": event["identity"]})
        elif event["event_type"] == "footprint":
            inventory = inventories[event["run_id"]]
            retained = {a["path"]: a for a in inventory["artifacts"] if a["kind"] == "tensor"}
            omitted = {a["path"]: a for a in inventory["omitted_tensors"]}
            for metric, unit in (("parameter_count", "parameters"), ("checkpoint_bytes", "bytes")):
                value = _number(p[metric])
                reason = p[metric]["reason"]
                known = {**retained, **omitted}
                reconciled = (all(r in known for r in p["checkpoint_refs"])
                              and len({(known[r]["sha256"], known[r]["bytes"]) for r in p["checkpoint_refs"]}) == len(p["checkpoint_refs"])
                              and value == sum(known[r]["bytes"] for r in p["checkpoint_refs"]))
                if metric == "checkpoint_bytes" and value is not None and not reconciled:
                    value, reason = None, "declared checkpoint size does not reconcile with referenced tensor inventory"
                footprints.append({"run_id": event["run_id"], "phase": event["phase_id"], "task": event["task"],
                                   "scope": p["scope"], "site_id": p["site_id"], "metric": metric,
                                   "value": value, "declared_value": _number(p[metric]), "reason": reason, "units": unit,
                                   "checkpoint_refs": p["checkpoint_refs"],
                                   "retained_tensor_refs": [retained[r] for r in p["checkpoint_refs"] if r in retained],
                                   "omitted_tensor_refs": [omitted[r] for r in p["checkpoint_refs"] if r in omitted],
                                   "evidence": "retained bytes verified; omitted sizes are inventory declarations only; parameter counts are declared telemetry; no model load"})
        elif event["event_type"] == "failure":
            failures.append({"run_id": event["run_id"], "phase": event["phase_id"], **p})
    # Sum only disjoint recorded rows within their explicitly declared scope.
    # Unknown components keep their total unknown, never an available-subset sum.
    cost_totals = []
    cost_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in costs:
        cost_groups[(row["run_id"], row["attribution"], row["scope"], row["site_id"], row["category"])].append(row)
    for (run_id, attribution, scope, site, category), values in cost_groups.items():
        complete = len(values) == 1 and all(row["value"] is not None for row in values)
        cost_totals.append({"run_id": run_id, "attribution": attribution, "scope": scope, "site_id": site, "category": category,
                            "value": sum(row["value"] for row in values) if complete else None,
                            "reason": "" if complete else "unavailable component or repeated scope needs explicit disjointness receipt",
                            "components": len(values), "units": "seconds", "task": values[0]["task"]})
    output.mkdir(parents=True, exist_ok=True)
    tables = {"updates": updates, "task-quality": quality, "paired-gaps": comparisons,
              "blocked-comparisons": blocked, "worst-trained-sites": worst, "costs": costs,
              "cost-totals": cost_totals, "footprints": footprints, "failures": failures}
    for name, rows in tables.items():
        _write_csv(output / f"{name}.csv", rows)
    panels = []
    def panel(name: str, title: str, rows: list[dict], axis: str, *, categorical: bool = False) -> None:
        labels = list(dict.fromkeys(str(r["series"]) for r in rows))
        pages = [labels[i:i + 8] for i in range(0, len(labels), 8)] or [[]]
        for page, labels_page in enumerate(pages):
            selected = [r for r in rows if str(r["series"]) in labels_page]
            stem = name + (f"-page-{page + 1}" if len(pages) > 1 else "")
            if (output / f"{stem}.csv").exists() or (output / f"{stem}.png").exists():
                raise ValueError(f"duplicate report artifact path: {stem}")
            caption = title + (f" ({page + 1}/{len(pages)})" if len(pages) > 1 else "")
            _write_csv(output / f"{stem}.csv", selected)
            _plot(output / f"{stem}.png", caption, selected, x_label=axis, categorical=categorical)
            panels.append({"path": f"{stem}.png", "table": f"{stem}.csv", "title": caption,
                           "sha256": _digest(output / f"{stem}.png"), "rows": len(selected)})
    metrics = sorted({row["metric"] for row in updates})
    for metric in metrics:
        category = _category(metric)
        if category == "accounting":
            continue
        kinds = ("local", "combined") if category == "objective" else (None,)
        for kind in kinds:
            subset = [r for r in updates if r["metric"] == metric and (kind is None or r["objective_kind"] == kind)]
            if not subset:
                continue
            for axis in (("completed_step", "source_valid_tokens") if category == "objective" else ("completed_step",)):
                for task in sorted({r["task"] for r in subset}):
                    rows = [{**r, "x": r[axis], "series": f'{r["run_id"]} / {r["metric"]}'} for r in subset if r["task"] == task]
                    name = _slug(f"{task}-{category}-{kind or 'all'}-{metric}-{axis}")
                    panel(name, f"{task} / {kind or 'All phases'}: {metric}", rows, axis.replace("_", " "))
    for metric in sorted({r["metric"] for r in quality}):
        for site in sorted({str(r["site"]) for r in quality}):
            for run_id in sorted({r["run_id"] for r in quality}):
                for objective_kind in (None, "local", "combined"):
                    subset = [r for r in quality if r["metric"] == metric and str(r["site"]) == site and r["run_id"] == run_id and r["objective_kind"] == objective_kind]
                    rows = [{**r, "x": r["step"], "series": f'{r["run_id"]} | {r["condition"]}'} for r in subset]
                    if rows:
                        name = _slug(json.dumps(["quality", run_id, site, metric, objective_kind]))
                        panel(name, f"{run_id} / {site} / {objective_kind or 'task'}: {metric} (init marked)", rows, "checkpoint completed step")
    for metric in sorted({"normalized_correct", "raw_correct"} | {r["metric"] for r in comparisons}):
        subset = [r for r in comparisons if r["metric"] == metric]
        rows = [{**r, "x": r["step"], "value": r["estimate"],
                 "series": _paired_series(r)} for r in subset]
        panel(f"paired-{metric}", f"Paired {metric} gaps (interval endpoints in CSV)", rows, "checkpoint completed step")
    for summary in sorted({r["summary"] for r in worst}):
        rows = [{**r, "x": r["step"], "series": str(r["cohort"])} for r in worst if r["summary"] == summary]
        panel(_slug(summary), summary, rows, "checkpoint completed step")
    for attribution in ("first_use", "reuse"):
        for category in sorted({r["category"] for r in cost_totals}):
            for task in sorted({r["task"] for r in cost_totals}):
                subset = [r for r in cost_totals if r["attribution"] == attribution and r["category"] == category and r["task"] == task]
                rows = [{**r, "x": i, "series": f'{r["run_id"]} / {r["scope"]} / {r["site_id"]}', "phase": "cost"}
                        for i, r in enumerate(subset)]
                panel(f"cost-{task}-{attribution}-{_slug(category)}", f"{task}: {attribution} / {category} (seconds)", rows, "declared site/scope (see legend and CSV)", categorical=True)
    for metric in ("parameter_count", "checkpoint_bytes"):
        for task in sorted({r["task"] for r in footprints}):
            subset = [r for r in footprints if r["metric"] == metric and r["task"] == task]
            rows = [{**r, "x": i, "series": f'{r["run_id"]} / {r["scope"]} / {r["site_id"]}'} for i, r in enumerate(subset)]
            panel(_slug(f"footprint-{task}-{metric}"), f"{task}: {metric} (declared scope; no inferred bank sum)", rows, "declared site/scope (see CSV)", categorical=True)
    settings = {"schema": "experiment-local-report-v1", "inputs": inputs, "completion": completions,
                "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
                              "percentiles": [2.5, 97.5], "unit": "paired source_id groups", "limits": LIMITS},
                "renderer": "Pillow; no smoothing; missing points break lines; local/combined objectives separate",
                "cost_limits": "Cost categories and first-use/reuse are separate; overlapping command-wall/train categories are never summed. Bank totals are explicit producer ledger scopes; this report does not infer membership or disjointness. Common production requires its own physical attribution, never replication across sites. Allocation is not hardware-normalized compute.",
                "comparisons_manifest": comparison_manifest, "images": panels}
    (output / "analysis-settings.json").write_text(json.dumps(settings, indent=2, allow_nan=False) + "\n")
    (output / "image-inventory.json").write_text(json.dumps(panels, indent=2) + "\n")
    summaries = "".join(f'<li><b>{html.escape(c["run_id"])}</b>: {html.escape(c["status"])} — {html.escape(", ".join(c["reasons"]))}; {html.escape(c["inventory_mode"])}</li>' for c in completions)
    cards = "".join(f'<section><h2>{html.escape(p["title"])}</h2><a href="{p["table"]}">Plotted CSV</a><img src="{p["path"]}" alt="{html.escape(p["title"])}"></section>' for p in panels)
    links = " ".join(f'<a href="{name}.csv">{name}</a>' for name in tables)
    (output / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Local experiment evidence</title>'
        '<style>body{font:16px system-ui;max-width:1500px;margin:32px auto;padding:16px;color:#172b4d}img{width:100%}section{margin:35px 0;border-top:1px solid #ccd}a{margin-right:14px}pre{white-space:pre-wrap}.warning{color:#9b2432}</style>'
        '<h1>Local experiment evidence</h1><p>Verified input artifacts; model-free local rendering. Completed evidence is distinct from scientific acceptance and tensor recovery.</p>'
        f'<ul>{summaries}</ul><p>{html.escape(LIMITS)}</p>'
        '<p>Objectives use separate local and combined panels. Missing values are not zero; red crosses and phase boundaries break curves. Runs with unresolved mandatory observations show task points without connecting lines because missing digest-only requests cannot be located on the axis. Exact units and reasons remain in CSV. Normalized accuracy is the task scorer metric, not a rescaling against initialization.</p>'
        f'<p>{html.escape(settings["cost_limits"])}</p><p>{links}</p>'
        '<a href="analysis-settings.json">Input hashes and analysis settings</a><a href="image-inventory.json">Image inventory</a>'
        f'<h2>Failures and blocked comparisons</h2><pre class="warning">{html.escape(json.dumps({"failures": failures, "blocked": blocked}, indent=2))}</pre>{cards}</html>')
    return settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = render_local_report(args.records, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "images": len(result["images"]), "completion": result["completion"]}, indent=2))


if __name__ == "__main__":
    main()
