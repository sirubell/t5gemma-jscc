"""Local sharing contrasts with explicit intervention controls and evidence axes."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from scripts.render_experiment_report import LIMITS, _plot, paired_group_bootstrap

from .experiment_records import read_events, validate_observation, write_json_atomic
from .sharing_schedule import HELDOUT_SITE, TRAINED_SITES


MATCHED_CONTROLS = ("architecture", "initialization", "training_data", "objective")
MATCHED_REQUEST = (
    "source",
    "data",
    "panel",
    "site",
    "task",
    "protocol",
    "noise",
    "scorer",
    "layout",
    "precision",
    "backend",
    "conditions",
    "expected_items",
    "site_step",
)


def _paired(left, right):
    a, b = left["request"], right["request"]
    for key in MATCHED_REQUEST:
        if a[key] != b[key]:
            raise ValueError(f"shared/specialist pairing mismatch: {key}")
    for key in MATCHED_CONTROLS:
        if a["comparison"][key] != b["comparison"][key]:
            raise ValueError(f"shared/specialist control mismatch: {key}")
    for key in ("prompt", "native_policy", "draws"):
        if a["details"][key] != b["details"][key]:
            raise ValueError(f"shared/specialist input/noise mismatch: {key}")
    if a["role"] != "trained" or b["role"] != "trained":
        raise ValueError("specialist contrast is only defined at trained sites")
    return {
        key: {"shared": a["comparison"][key], "specialist": b["comparison"][key]}
        for key in ("exposure", "schedule")
    }


def _events(plan, event_files):
    axes, hashes = {}, {}
    for learner, path in (event_files or {}).items():
        if learner not in plan.learners:
            raise ValueError("undeclared event-file learner")
        path = Path(path)
        events = read_events(path)
        hashes[learner] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        updates = [event for event in events if event["event_type"] == "update"]
        if len({event["run_id"] for event in events}) > 1:
            raise ValueError("mixed run IDs in learner event file")
        schedule = plan.learner(learner)
        counts = dict.fromkeys(TRAINED_SITES, 0)
        axes[learner] = {}
        for index, event in enumerate(updates):
            p = event["payload"]
            expected = schedule.update_at(index)
            if (
                p["skipped"]
                or p["nonfinite"]
                or p["completed_step"] != index + 1
                or p["site_id"] != expected.site
                or p["site_step"] != expected.site_local_index + 1
                or p["exposure"]["valid_tokens"] != expected.view.source_tokens
            ):
                raise ValueError("event update differs from completed sharing plan")
            counts[expected.site] += p["exposure"]["valid_tokens"]
            if p["exposure"]["cumulative_valid_tokens"] != sum(counts.values()):
                raise ValueError("event cumulative valid-token count mismatch")
            axes[learner][index + 1] = {
                "valid_tokens": sum(counts.values()),
                "site_valid_tokens": dict(counts),
                "elapsed_wall_seconds": p["timing"]["elapsed_seconds"]["value"],
                "elapsed_wall_reason": p["timing"]["elapsed_seconds"]["reason"],
                "timing_scope": p["timing"]["scope"],
            }
        if len(updates) != schedule.horizon:
            raise ValueError("report requires complete supplied learner update logs")
    return axes, hashes


def _vanilla(output):
    directory = Path(output) / "vanilla"
    path = directory / "receipt.json"
    if not path.exists():
        return {
            "available": False,
            "reason": "vanilla receipt not supplied",
            "bypass": True,
        }
    receipt = json.loads(path.read_text())
    if receipt["status"] != "complete" or receipt["requested"] != receipt["completed"]:
        raise ValueError("incomplete vanilla bypass receipt")
    hashes = receipt["output_hashes"]
    if "compact_no_noise.jsonl" not in hashes:
        raise ValueError("vanilla compact evidence hash missing")
    for name, expected in hashes.items():
        artifact = directory / name
        if Path(name).name != name or not artifact.is_file():
            raise ValueError("invalid vanilla artifact path")
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
            raise ValueError("vanilla output hash mismatch")
    rows = [
        json.loads(line)
        for line in (directory / "compact_no_noise.jsonl").read_text().splitlines()
        if line.strip()
    ]
    request = receipt["request"]
    if (
        not rows
        or len(rows) != receipt["completed"]
        or [r["sample_id"] for r in rows] != request["input_ids"]
        or [r["source_id"] for r in rows] != request["source_family_ids"]
    ):
        raise ValueError("vanilla per-item membership mismatch")
    metrics = {}
    for name, field in (
        ("normalized_accuracy", "normalized_correct"),
        ("raw_accuracy", "raw_correct"),
    ):
        if any(r[field] not in (0, 1) for r in rows):
            raise ValueError("invalid vanilla per-item correctness")
        metrics[name] = sum(r[field] for r in rows) / len(rows)
    return {
        "available": True,
        "bypass": True,
        "label": "vanilla bypass; codec absent",
        "metrics": metrics,
        "items": [
            {
                **row,
                "item_id": str(row["sample_id"]),
                "source_id": str(row["source_id"]),
            }
            for row in rows
        ],
        "receipt_metrics": receipt["metrics"],
        "requested": receipt["requested"],
        "completed": len(rows),
        "source": receipt["source"],
        "data": receipt["data"],
        "request": request,
        "output_hashes": hashes,
        "receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _heldout_context(heldout, initialization, vanilla):
    a, b = heldout["request"], initialization["request"]
    for key in (*[k for k in MATCHED_REQUEST if k != "site_step"],):
        if a[key] != b[key]:
            raise ValueError(f"heldout initialization pairing mismatch: {key}")
    for key in MATCHED_CONTROLS:
        if a["comparison"][key] != b["comparison"][key]:
            raise ValueError(f"heldout initialization control mismatch: {key}")
    for key in ("prompt", "native_policy", "draws"):
        if a["details"][key] != b["details"][key]:
            raise ValueError(f"heldout initialization detail mismatch: {key}")
    if b["role"] != "heldout" or b["site_step"] != 0:
        raise ValueError("heldout initialization must be heldout step zero")
    if vanilla["available"]:
        for key in ("source", "data"):
            if vanilla[key] != a[key]:
                raise ValueError(f"heldout vanilla provenance mismatch: {key}")
    result = []
    for terminal, initial in zip(heldout["conditions"], initialization["conditions"]):
        for metric, field in (
            ("normalized_accuracy", "normalized_correct"),
            ("raw_accuracy", "raw_correct"),
        ):
            row = {
                "condition": terminal["condition"],
                "metric": metric,
                "terminal": terminal["metrics"][metric]["value"],
                "initialization": initial["metrics"][metric]["value"],
                "terminal_minus_initialization": paired_group_bootstrap(
                    terminal["items"], initial["items"], field
                ),
                "initialization_observation": initialization["identity"],
                "terminal_minus_vanilla": None,
                "vanilla_comparison_reason": "vanilla evidence unavailable",
            }
            if vanilla["available"]:
                row["terminal_minus_vanilla"] = paired_group_bootstrap(
                    terminal["items"], vanilla["items"], field
                )
                row["vanilla_comparison_reason"] = (
                    "This codec condition versus the same fixed codec-free vanilla bypass; no equal-noise-condition claim."
                )
                row["vanilla"] = vanilla["metrics"][metric]
            result.append(row)
    return result


def _write_plots(output, curves, contrasts):
    """Use the existing record renderer and retain exactly its plotting inputs."""
    selected = [
        row
        for row in curves
        if row["purpose"] == "task"
        and row["condition"] == "no_noise"
        and row["metric"] == "normalized_accuracy"
    ]
    panels = []
    for field, stem, label in (
        (
            "site_step",
            "sharing-quality-site-local-step",
            "Site-local completed updates",
        ),
        (
            "valid_tokens",
            "sharing-quality-valid-tokens",
            "Learner cumulative valid source tokens",
        ),
        (
            "elapsed_wall_seconds",
            "sharing-quality-elapsed-wall",
            "Learner elapsed wall seconds (recorded scope)",
        ),
    ):
        rows = [
            {
                **row,
                "x": row[field],
                "series": f"{row['learner']} / {row['site']}",
                "learner_kind": row["learner"],
                "phase": "combined",
                "low": None,
                "high": None,
            }
            for row in sorted(
                selected, key=lambda r: (r["learner"], r["site"], r["site_step"])
            )
        ]
        panels.append((stem, "No-noise normalized accuracy", label, False, rows))
    rows = []
    for contrast in contrasts:
        if contrast["metric"] != "normalized_accuracy":
            continue
        interval = contrast["shared_minus_specialist"]
        rows.append(
            {
                "x": len(rows),
                "value": interval["estimate"],
                "low": interval["low"],
                "high": interval["high"],
                "series": contrast["site"],
                "site_id": f"{contrast['site']} {contrast['condition']}",
                "condition": contrast["condition"],
                "shared_observation": contrast["shared_observation"],
                "specialist_observation": contrast["specialist_observation"],
                "phase": "terminal",
                "limits": interval["limits"],
                "replicates": interval["replicates"],
                "seed": interval["seed"],
            }
        )
    panels.append(
        (
            "sharing-terminal-normalized-gaps",
            "Terminal shared minus specialist normalized accuracy",
            "Trained site / channel condition; descriptive 95% source-group intervals",
            True,
            rows,
        )
    )
    artifacts = []
    for stem, title, label, categorical, rows in panels:
        png, csv_path = output / f"{stem}.png", output / f"{stem}.csv"
        _plot(png, title, rows, x_label=label, categorical=categorical)
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        artifacts.append(
            {
                "png": png.name,
                "csv": csv_path.name,
                "rows": len(rows),
                "png_sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
                "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                "missing_axis_rows": sum(row.get("x") is None for row in rows),
            }
        )
    return artifacts


def write_sharing_report(output, observations, plan, *, event_files=None):
    """Write terminal contrasts and all observed quality/objective learning curves.

    Observation payloads use experiment-records-v1. event_files maps exact learner
    names to their logs. Missing logs leave axes unavailable, never estimated from
    plan quotas. This report does not establish checkpoint/campaign completion;
    the campaign's retained inventories and completion gate remain authoritative.
    """
    observations = list(observations)
    lookup = {}
    for row in observations:
        validate_observation(row)
        if row["status"] != "complete" or row["request"]["task"] != "hellaswag":
            raise ValueError("sharing report requires complete HellaSwag observations")
        r = row["request"]
        key = (r["purpose"], r["learner_kind"], r["site"], r["step"])
        if key in lookup:
            raise ValueError("duplicate sharing observation")
        lookup[key] = row
    axes, event_hashes = _events(plan, event_files)
    contrasts = []
    for site in TRAINED_SITES:
        try:
            shared = lookup[("task", "shared", site, 3 * plan.q)]
            specialist = lookup[("task", "specialist", site, plan.q)]
        except KeyError as error:
            raise ValueError("missing terminal shared/specialist task panel") from error
        if (
            shared["request"]["site_step"] != plan.q
            or specialist["request"]["site_step"] != plan.q
        ):
            raise ValueError("terminal comparison requires matched site-local quota")
        differences = _paired(shared, specialist)
        for left, right in zip(shared["conditions"], specialist["conditions"]):
            for metric, item_metric in (
                ("normalized_accuracy", "normalized_correct"),
                ("raw_accuracy", "raw_correct"),
            ):
                interval = paired_group_bootstrap(
                    left["items"], right["items"], item_metric
                )
                contrasts.append(
                    {
                        "site": site,
                        "condition": left["condition"],
                        "metric": metric,
                        "shared": left["metrics"][metric]["value"],
                        "specialist": right["metrics"][metric]["value"],
                        "shared_minus_specialist": interval,
                        "site_step": plan.q,
                        "shared_step": 3 * plan.q,
                        "specialist_step": plan.q,
                        "shared_observation": shared["identity"],
                        "specialist_observation": specialist["identity"],
                        "matched_controls": {
                            key: shared["request"]["comparison"][key]
                            for key in MATCHED_CONTROLS
                        },
                        "intervention_control_refs": differences,
                    }
                )
    try:
        heldout = lookup[("task", "shared", HELDOUT_SITE, 3 * plan.q)]
    except KeyError as error:
        raise ValueError("missing terminal heldout shared task panel") from error
    if (
        heldout["request"]["role"] != "heldout"
        or heldout["request"]["site_step"] != plan.q
    ):
        raise ValueError("invalid heldout report role/site-local checkpoint")
    architectures = {
        row["request"]["comparison"]["architecture"]
        for row in observations
        if row["request"]["learner_kind"] != "vanilla"
    }
    if len(architectures) != 1:
        raise ValueError("sharing report is a single-architecture comparison")
    curves = []
    for row in observations:
        request = row["request"]
        kind, site, step = request["learner_kind"], request["site"], request["step"]
        learner = f"specialist_{site}" if kind == "specialist" else kind
        axis = axes.get(learner, {}).get(step)
        if step == 0:
            axis = {
                "valid_tokens": 0,
                "site_valid_tokens": {},
                "elapsed_wall_seconds": None,
                "elapsed_wall_reason": "initialization observation timing not recorded",
                "timing_scope": "unavailable",
            }
        for condition in row["conditions"]:
            measures = dict(condition["metrics"])
            for name, component in condition["objectives"].items():
                measures[f"{name}_raw"] = component["raw"]
                measures[f"{name}_weighted"] = component["weighted"]
            for metric, value in measures.items():
                curves.append(
                    {
                        "observation": row["identity"],
                        "learner": learner,
                        "site": site,
                        "role": request["role"],
                        "purpose": request["purpose"],
                        "step": step,
                        "site_step": request["site_step"],
                        "condition": condition["condition"],
                        "metric": metric,
                        "value": value["value"],
                        "unavailable_reason": value["reason"],
                        "valid_tokens": axis["valid_tokens"] if axis else None,
                        "site_valid_tokens": (
                            axis["site_valid_tokens"].get(site, 0)
                            if site != HELDOUT_SITE
                            else 0
                        )
                        if axis
                        else None,
                        "elapsed_wall_seconds": axis["elapsed_wall_seconds"]
                        if axis
                        else None,
                        "axis_unavailable_reason": axis["elapsed_wall_reason"]
                        if axis
                        else "learner event log unavailable",
                        "timing_scope": axis["timing_scope"] if axis else "unavailable",
                    }
                )
    normalized = [c for c in contrasts if c["metric"] == "normalized_accuracy"]
    vanilla = _vanilla(output)
    try:
        initial_heldout = lookup[("task", "initialization", HELDOUT_SITE, 0)]
    except KeyError as error:
        raise ValueError("missing heldout initialization context") from error
    heldout_context = _heldout_context(heldout, initial_heldout, vanilla)
    report = {
        "schema": "sharing-report-v1",
        "vanilla_bypass": vanilla,
        "synthetic": plan.synthetic,
        "study_pairing_id": plan.study_pairing_id,
        "q": plan.q,
        "architecture": next(iter(architectures)),
        "terminal_contrasts": contrasts,
        "heldout_shared_only": {
            "observation": heldout["identity"],
            "site": HELDOUT_SITE,
            "step": 3 * plan.q,
            "specialist_comparator": None,
            "conditions": heldout["conditions"],
            "baseline_context": heldout_context,
        },
        "worst_terminal_shared_normalized_accuracy": min(
            c["shared"] for c in normalized
        ),
        "worst_terminal_specialist_normalized_accuracy": min(
            c["specialist"] for c in normalized
        ),
        "largest_terminal_normalized_degradation": max(
            0.0, max(-c["shared_minus_specialist"]["estimate"] for c in normalized)
        ),
        "intervention": {
            "shared_horizon": 3 * plan.q,
            "specialist_horizon": plan.q,
            "description": "Shared optimizer interleaves three sites; specialists each see one site. Per-site exposure is matched by the campaign; optimizer histories, total learner exposure and horizons differ intentionally.",
            "exposure_evidence": "verified supplied update logs"
            if set(axes) == set(plan.learners)
            else "not all learner update logs supplied",
        },
        "limits": LIMITS,
        "selection_claim": "No architecture selection or equivalence claim.",
        "completion_scope": "Complete terminal observation panels; campaign completion and freeze rely on caller-verified inventories.",
        "input_observation_identities": [row["identity"] for row in observations],
        "event_files": event_hashes,
        "learning_curve_rows": len(curves),
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report["plots"] = _write_plots(output, curves, contrasts)
    write_json_atomic(output / "sharing-comparison.json", report)
    with (output / "sharing-learning-curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    (output / "sharing-report.md").write_text(
        "# Sharing comparison\n\n"
        f"One architecture; site-local quota {plan.q}. Shared horizon {3 * plan.q}; specialist horizon {plan.q}.\n\n"
        "Terminal site/condition results and source-group bootstrap intervals are retained in `sharing-comparison.json`. "
        "`sharing-learning-curves.csv` retains every supplied quality/objective observation with measured exposure/time axes when available.\n\n"
        "Four PNG panels show no-noise normalized quality against site-local step, learner valid tokens and elapsed wall time, plus terminal paired gaps. Each has an exact companion CSV; missing measurements remain marked unavailable.\n\n"
        "Layer 14 is shared-only held-out evidence with no specialist comparator. Initialization observations remain separate rows; vanilla bypass evidence is a separate JSON object.\n\n"
        + report["intervention"]["description"]
        + "\n\n"
        + LIMITS
        + "\n\n"
        + report["selection_claim"]
        + "\n",
        encoding="utf-8",
    )
    return report
