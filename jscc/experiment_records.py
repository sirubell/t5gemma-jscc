"""Strict, model-free experiment evidence and durable local JSON records."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

SCHEMA = "experiment-records-v1"
CONDITIONS = ["no_noise", -6, 0, 6, 12, 18]


class RecordError(ValueError):
    """Evidence violates the versioned contract."""


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise RecordError(message)


def _fields(obj: Any, fields: str, label: str) -> None:
    _require(isinstance(obj, dict), f"{label}: expected object")
    _require(set(obj) == set(fields.split()), f"{label}: fields must be {fields}")


def _text(value: Any, label: str) -> None:
    _require(isinstance(value, str) and bool(value.strip()), f"{label}: nonempty string required")


def _count(value: Any, label: str) -> None:
    _require(type(value) is int and value >= 0, f"{label}: nonnegative integer required")


def _number(value: Any, label: str) -> None:
    _require(type(value) in (int, float) and math.isfinite(value), f"{label}: finite number required")


def measure(value: float | int | None, reason: str | None = None) -> dict[str, Any]:
    result = {"value": value, "reason": reason}
    _measure(result)
    return result


def _measure(value: Any) -> None:
    _fields(value, "value reason", "measure")
    if value["value"] is None:
        _text(value["reason"], "unavailable reason")
    else:
        _number(value["value"], "measure")
        _require(value["reason"] is None, "measured value cannot have unavailable reason")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _identity(value: Any) -> None:
    _fields(value, "source config data parent", "identity")
    for key in ("source", "config", "data"):
        _text(value[key], key)
    if value["parent"] is not None:
        _text(value["parent"], "parent")


def observation_identity(request: dict[str, Any]) -> str:
    _fields(request, "source config data parent checkpoint panel site role task protocol noise scorer layout precision backend conditions expected_items learner_kind step site_step purpose objective_kind comparison details", "request")
    for key in set(request) - {"parent", "conditions", "expected_items", "step", "site_step", "objective_kind", "comparison", "details"}:
        _text(request[key], key)
    _require(request["task"] in ("coco", "hellaswag"), "unsupported task")
    _require(request["learner_kind"] in ("shared", "specialist", "initialization", "vanilla"), "unsupported learner kind")
    if request["parent"] is not None:
        _text(request["parent"], "parent")
    _fields(request["details"], "prompt native_policy draws outputs", "request details")
    for key, value in request["details"].items():
        _text(value, key)
    _require(request["purpose"] in ("task", "objective"), "invalid observation purpose")
    _require(request["objective_kind"] is None if request["purpose"] == "task" else request["objective_kind"] in ("local", "combined"), "purpose/objective kind mismatch")
    _fields(request["comparison"], "architecture initialization training_data objective exposure schedule", "comparison controls")
    for key, value in request["comparison"].items():
        _text(value, f"comparison {key}")
    _require(request["role"] in ("trained", "heldout", "diagnostic"), "invalid site role")
    _require(not ((request["site"] in ("l14", "enc_l14") or request["role"] == "heldout") and request["learner_kind"] == "specialist"), "no held-out specialist")
    _count(request["site_step"], "site_step")
    _count(request["step"], "step")
    _count(request["expected_items"], "expected_items")
    _require(request["expected_items"] > 0, "empty panel")
    _require(request["conditions"] == CONDITIONS or request["conditions"] == ["vanilla"], "unsupported condition grid")
    _require((request["conditions"] == ["vanilla"]) == (request["learner_kind"] == "vanilla"), "vanilla bypass identity mismatch")
    return hashlib.sha256(b"codec-observation-v1\0" + canonical_bytes(request)).hexdigest()


def validate_observation(payload: dict[str, Any]) -> None:
    _fields(payload, "request identity status conditions reuse", "observation")
    request = payload["request"]
    _require(payload["identity"] == observation_identity(request), "observation identity mismatch")
    _require(payload["status"] in ("complete", "incomplete", "failed"), "invalid observation status")
    _require(isinstance(payload["conditions"], list), "conditions must be list")
    seen = []
    complete = True
    for row in payload["conditions"]:
        _fields(row, "condition requested completed failed failure_reason denominator metrics items objectives", "condition")
        _require(row["condition"] in request["conditions"] and row["condition"] not in seen, "unknown/duplicate condition")
        seen.append(row["condition"])
        for key in ("requested", "completed", "denominator"):
            _count(row[key], key)
        _require(row["requested"] == request["expected_items"], "requested count differs from panel")
        if row["failed"] is None:
            _text(row["failure_reason"], "unknown failure count reason")
        else:
            _count(row["failed"], "failed")
            _require(row["failure_reason"] is None or isinstance(row["failure_reason"], str), "invalid failure reason")
        _require(row["completed"] + (row["failed"] or 0) <= row["requested"], "counts exceed requested")
        _require(row["denominator"] == row["completed"], "denominator differs from completed items")
        _require(isinstance(row["metrics"], dict), "metrics object required")
        _require(isinstance(row["objectives"], dict), "objectives must be object")
        if request["purpose"] == "task":
            _require(bool(row["metrics"]), "task metrics required")
        else:
            _require(set(row["objectives"]) == {"K", "R"}, "objective observation requires K/R")
        for component in row["objectives"].values():
            _component(component)
        if request["purpose"] == "objective" and (payload["status"] == "complete" or row["completed"] > 0):
            _objective_value(row["objectives"], request["objective_kind"])
        for metric in row["metrics"].values():
            _measure(metric)
        _require(isinstance(row["items"], list) and len(row["items"]) == row["completed"], "per-item count mismatch")
        ids = []
        for item in row["items"]:
            _require(isinstance(item, dict), "item must be object")
            for key in ("item_id", "source_id"):
                _text(item.get(key), key)
            _require(item["item_id"] not in ids, "duplicate item")
            ids.append(item["item_id"])
            if request["purpose"] == "task":
                _task_item(item, request["task"])
        if request["purpose"] == "task":
            metric_fields = {"normalized_accuracy": "normalized_correct", "raw_accuracy": "raw_correct"} if request["task"] == "hellaswag" else {"cider": "cider"}
            _require(set(metric_fields).issubset(row["metrics"]), "required task metrics missing")
            for metric, field in metric_fields.items():
                actual = row["metrics"][metric]["value"]
                if row["completed"]:
                    expected = sum(item[field] for item in row["items"]) / row["completed"]
                    _require(actual is not None and math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-9), "metric/per-item denominator mismatch")
                else:
                    _require(actual is None, "empty observations must be unavailable")
        complete &= row["completed"] == row["requested"] and row["failed"] == 0
    complete &= seen == request["conditions"]
    _require(payload["status"] != "complete" or complete, "false complete observation")
    if payload["reuse"] is not None:
        reuse = payload["reuse"]
        _fields(reuse, "identity status reference acquisition_seconds", "reuse")
        _require(reuse["identity"] == payload["identity"] and reuse["status"] == "complete", "incompatible/incomplete reuse")
        _text(reuse["reference"], "reuse reference")
        _measure(reuse["acquisition_seconds"])
        _require(payload["status"] == "complete", "reuse requires complete observation")


def _task_item(item: dict[str, Any], task: str) -> None:
    _require(isinstance(item.get("tokens"), list) and all(type(t) is int and t >= 0 for t in item["tokens"]), "item tokens required")
    if task == "hellaswag":
        for key in ("raw_correct", "normalized_correct"):
            _require(type(item.get(key)) in (int, float) and item[key] in (0, 1), "per-item correctness required")
        _text(item.get("prediction"), "prediction")
        _number(item.get("score"), "score")
    else:
        for key in ("caption_raw", "caption_clean"):
            _require(isinstance(item.get(key), str), "COCO caption required")
        _number(item.get("cider"), "cider")
        for key in ("eos", "cap_hit"):
            _require(type(item.get(key)) is bool, "COCO generation diagnostic required")


def _component(component: Any) -> None:
    _fields(component, "raw weighted numerator denominator", "component")
    for key in ("raw", "weighted", "numerator"):
        _measure(component[key])
    _count(component["denominator"], "objective denominator")
    raw, numerator = component["raw"]["value"], component["numerator"]["value"]
    if raw is None:
        _require(numerator is None and component["weighted"]["value"] is None, "partial unavailable objective")
    else:
        _require(component["denominator"] > 0 and numerator is not None, "measured objective requires numerator/denominator")
        _require(math.isclose(raw, numerator / component["denominator"], rel_tol=1e-6, abs_tol=1e-9), "objective numerator/denominator mismatch")


def _objective_value(components: dict[str, Any], kind: str) -> float:
    """Require actual K/R evidence for a successful update or observation."""
    r, k = components["R"], components["K"]
    _require(r["raw"]["value"] is not None, "successful objective requires R")
    _require(r["weighted"]["value"] is not None and math.isclose(r["weighted"]["value"], .1 * r["raw"]["value"], rel_tol=1e-6, abs_tol=1e-9), "R weight must be 0.1")
    if kind == "local":
        _require(all(k[key]["value"] is None for key in ("raw", "weighted", "numerator")) and k["denominator"] == 0, "local K must remain unavailable")
        return r["weighted"]["value"]
    _require(k["raw"]["value"] is not None and k["weighted"]["value"] == k["raw"]["value"], "combined K weight must be one")
    return k["weighted"]["value"] + r["weighted"]["value"]


def _update(p: dict[str, Any]) -> None:
    _fields(p, "site_id attempted_step completed_step final_step phase_start_step site_step sweep exposure objective lr_used lr_next gradient nonfinite skipped update_l2 snr timing memory", "update")
    _text(p["site_id"], "site_id")
    for key in ("attempted_step", "completed_step", "final_step", "phase_start_step", "site_step", "sweep"):
        _count(p[key], key)
    _require(p["phase_start_step"] < p["attempted_step"] <= p["final_step"], "attempt outside phase")
    for key in ("nonfinite", "skipped"):
        _require(type(p[key]) is bool, f"{key} must be boolean")
    _require(p["completed_step"] == p["attempted_step"] - int(p["skipped"] or p["nonfinite"]), "invalid completed axis")
    exposure = p["exposure"]
    _fields(exposure, "source_tokens target_tokens sequences padded_tokens valid_tokens cumulative_valid_tokens latent_width hidden_coordinates hidden_coordinates_allocated memory_coordinates memory_coordinates_allocated", "exposure")
    for key, value in exposure.items():
        if key in ("memory_coordinates", "memory_coordinates_allocated"):
            _measure(value)
        else:
            _count(value, key)
    _require(exposure["padded_tokens"] >= exposure["valid_tokens"] and exposure["cumulative_valid_tokens"] >= exposure["valid_tokens"], "invalid token counters")
    _require(exposure["latent_width"] > 0, "latent width must be positive")
    _require(exposure["hidden_coordinates_allocated"] >= exposure["hidden_coordinates"], "invalid allocated hidden coordinates")
    memory = exposure["memory_coordinates"]["value"]
    allocated = exposure["memory_coordinates_allocated"]["value"]
    _require((memory is None) == (allocated is None), "memory coordinate availability mismatch")
    if memory is not None:
        _count(memory, "memory_coordinates")
        _count(allocated, "memory_coordinates_allocated")
        _require(allocated >= memory, "invalid allocated memory coordinates")
    objective = p["objective"]
    _fields(objective, "kind total components", "objective")
    _require(objective["kind"] in ("local", "combined"), "invalid objective kind")
    _measure(objective["total"])
    _require(isinstance(objective["components"], dict) and bool(objective["components"]), "objective components required")
    _require(set(objective["components"]) == {"K", "R"}, "K and R components required")
    for component in objective["components"].values():
        _component(component)
    if not p["skipped"] and not p["nonfinite"]:
        expected_total = _objective_value(objective["components"], objective["kind"])
        _require(objective["total"]["value"] is not None and math.isclose(objective["total"]["value"], expected_total, rel_tol=1e-6, abs_tol=1e-9), "objective total mismatch")
        _require(exposure["sequences"] > 0 and exposure["source_tokens"] == exposure["valid_tokens"], "invalid successful exposure")
    for key in ("lr_used", "lr_next"):
        _require(isinstance(p[key], list) and bool(p[key]), "LR groups required")
        for value in p[key]:
            _number(value, key)
            _require(value >= 0, "negative LR")
    _require(len(p["lr_used"]) == len(p["lr_next"]), "LR group mismatch")
    _fields(p["gradient"], "pre_clip post_clip clip_threshold clipped", "gradient")
    for key in ("pre_clip", "post_clip", "clip_threshold"):
        _measure(p["gradient"][key])
    _require(type(p["gradient"]["clipped"]) is bool, "clipped must be boolean")
    _measure(p["update_l2"])
    sampled = p["attempted_step"] in (p["phase_start_step"] + 1, p["final_step"]) or p["attempted_step"] % 10 == 0
    if not p["skipped"] and not p["nonfinite"]:
        _require((p["update_l2"]["value"] is not None) == sampled, "update-L2 violates first/every10/final cadence")
    _fields(p["snr"], "condition min mean max draw_ref", "snr")
    for key in ("condition", "draw_ref"):
        _text(p["snr"][key], key)
    for key in ("min", "mean", "max"):
        _measure(p["snr"][key])
    _fields(p["timing"], "step_seconds elapsed_seconds data_cache_seconds throughput throughput_denominator scope", "timing")
    for key in ("step_seconds", "elapsed_seconds", "data_cache_seconds", "throughput"):
        _measure(p["timing"][key])
        _require(p["timing"][key]["value"] is None or p["timing"][key]["value"] >= 0, "negative timing measurement")
    for key in ("throughput_denominator", "scope"):
        _text(p["timing"][key], key)
    _fields(p["memory"], "allocated_bytes reserved_bytes reset_scope", "memory")
    for key in ("allocated_bytes", "reserved_bytes"):
        _measure(p["memory"][key])
        if p["memory"][key]["value"] is not None:
            _count(p["memory"][key]["value"], key)
    _text(p["memory"]["reset_scope"], "reset_scope")


def validate_event(event: dict[str, Any]) -> None:
    _fields(event, "schema event_type run_id phase_id task timestamp units identity payload", "event")
    _require(event["schema"] == SCHEMA, "unsupported schema")
    _require(event["task"] in ("coco", "hellaswag"), "unsupported task")
    for key in ("run_id", "phase_id", "timestamp"):
        _text(event[key], key)
    _require(isinstance(event["units"], dict) and bool(event["units"]), "explicit units required")
    _identity(event["identity"])
    p = event["payload"]
    if event["event_type"] == "update":
        _update(p)
    elif event["event_type"] == "observation":
        validate_observation(p)
        _require(p["request"]["task"] == event["task"], "task mismatch")
        for key, value in event["identity"].items():
            _require(p["request"][key] == value, "observation provenance mismatch")
    elif event["event_type"] == "cost":
        _fields(p, "category seconds attribution site_id scope concurrency device_count", "cost")
        for key in ("category", "site_id", "scope", "concurrency"):
            _text(p[key], key)
        _require(p["attribution"] in ("first_use", "reuse"), "invalid attribution")
        _count(p["device_count"], "device_count")
        _measure(p["seconds"])
        _require(p["seconds"]["value"] is None or p["seconds"]["value"] >= 0, "negative cost seconds")
    elif event["event_type"] == "footprint":
        _fields(p, "scope site_id parameter_count checkpoint_bytes checkpoint_refs", "footprint")
        _require(p["scope"] in ("per_site", "bank_total", "shared"), "invalid footprint scope")
        _text(p["site_id"], "footprint site_id")
        for key in ("parameter_count", "checkpoint_bytes"):
            _measure(p[key])
            if p[key]["value"] is not None:
                _count(p[key]["value"], key)
        _require(isinstance(p["checkpoint_refs"], list), "checkpoint refs must be list")
        for reference in p["checkpoint_refs"]:
            _text(reference, "checkpoint reference")
            _require(not Path(reference).is_absolute() and ".." not in Path(reference).parts, "unsafe checkpoint reference")
        _require(len(set(p["checkpoint_refs"])) == len(p["checkpoint_refs"]), "duplicate checkpoint reference")
        _require(p["checkpoint_bytes"]["value"] is None or bool(p["checkpoint_refs"]), "measured checkpoint bytes require references")
    elif event["event_type"] == "failure":
        _fields(p, "reason reference", "failure")
        _text(p["reason"], "failure reason")
        _text(p["reference"], "failure reference")
    else:
        raise RecordError("unsupported event type")
    canonical_bytes(event)


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.partial-", dir=path.parent)
    # Keep the partial file on any failure for diagnosis; never retry automatically.
    with os.fdopen(fd, "wb") as handle:
        written = handle.write(data)
        if written != len(data):
            raise OSError("partial record write")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def write_json_atomic(path: str | Path, value: Any) -> None:
    _atomic_bytes(Path(path), canonical_bytes(value) + b"\n")


def read_events(path: str | Path) -> list[dict[str, Any]]:
    raw = Path(path).read_bytes()
    _require(not raw or raw.endswith(b"\n"), "truncated event log")
    events = []
    for line in raw.splitlines():
        try:
            event = json.loads(line)
        except (ValueError, UnicodeDecodeError) as exc:
            raise RecordError("invalid event JSON") from exc
        validate_event(event)
        events.append(event)
    return events


def append_event(path: str | Path, event: dict[str, Any]) -> dict[str, Any]:
    """Single-writer atomic replacement; errors stop the caller, never retry."""
    validate_event(event)
    path = Path(path)
    old = b""
    if path.exists():
        read_events(path)
        old = path.read_bytes()
    _atomic_bytes(path, old + canonical_bytes(event) + b"\n")
    return artifact_ref(path, path.parent, "metadata")


def artifact_ref(path: str | Path, root: str | Path, kind: str) -> dict[str, Any]:
    path, root = Path(path).resolve(), Path(root).resolve()
    _require(kind in ("metadata", "tensor"), "unsupported artifact kind")
    relative = path.relative_to(root)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {"path": relative.as_posix(), "kind": kind, "sha256": digest.hexdigest(), "bytes": size}


def verify_inventory(root: str | Path, inventory: dict[str, Any]) -> dict[str, Any]:
    _fields(inventory, "schema mode artifacts omitted_tensors", "inventory")
    _require(inventory["schema"] == SCHEMA, "unsupported inventory schema")
    _require(inventory["mode"] in ("metadata-only", "tensor-complete"), "invalid inventory mode")
    _require(isinstance(inventory["artifacts"], list), "artifact list required")
    _require(isinstance(inventory["omitted_tensors"], list), "omitted tensor list required")
    _require(inventory["mode"] != "tensor-complete" or not inventory["omitted_tensors"], "tensor-complete inventory omits tensors")
    omitted_paths = set()
    for omitted in inventory["omitted_tensors"]:
        _fields(omitted, "path kind sha256 bytes", "omitted tensor")
        _require(omitted["kind"] == "tensor", "only tensors may be omitted")
        _artifact_metadata(omitted)
        _require(omitted["path"] not in omitted_paths, "duplicate omitted artifact path")
        omitted_paths.add(omitted["path"])
    root = Path(root).resolve()
    seen = set()
    for artifact in inventory["artifacts"]:
        _fields(artifact, "path kind sha256 bytes", "artifact")
        _artifact_metadata(artifact)
        path = root / artifact["path"]
        _require(not Path(artifact["path"]).is_absolute() and path.resolve().is_relative_to(root), "artifact escapes root")
        _require(artifact["path"] not in seen, "duplicate artifact path")
        seen.add(artifact["path"])
        _count(artifact["bytes"], "artifact bytes")
        _require(artifact == artifact_ref(path, root, artifact["kind"]), "artifact hash/size mismatch")
    _require(not seen.intersection(a["path"] for a in inventory["omitted_tensors"]), "artifact both present and omitted")
    return inventory


def _artifact_metadata(artifact: dict[str, Any]) -> None:
    _text(artifact["path"], "artifact path")
    path = Path(artifact["path"])
    _require(not path.is_absolute() and ".." not in path.parts, "unsafe artifact path")
    _count(artifact["bytes"], "artifact bytes")
    digest = artifact["sha256"]
    _require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "invalid sha256")


def completion_status(manifest: dict[str, Any], events: list[dict[str, Any]], inventory: dict[str, Any], root: str | Path) -> dict[str, Any]:
    """Compute evidence completeness; caller-supplied success is never accepted."""
    _fields(manifest, "schema run_id phases expected_observations expected_artifacts", "manifest")
    _require(manifest["schema"] == SCHEMA, "unsupported manifest schema")
    verify_inventory(root, inventory)
    reasons = []
    updates: dict[str, list[dict[str, Any]]] = {}
    observed = set()
    tensor_refs = {a["path"]: a for a in inventory["artifacts"] + inventory["omitted_tensors"] if a["kind"] == "tensor"}
    for event in events:
        validate_event(event)
        _require(event["run_id"] == manifest["run_id"], "mixed run identities")
        p = event["payload"]
        if event["event_type"] == "update":
            updates.setdefault(event["phase_id"], []).append(p)
            if p["nonfinite"] or p["skipped"]:
                reasons.append("failed/skipped update")
        elif event["event_type"] == "footprint":
            missing = set(p["checkpoint_refs"]) - set(tensor_refs)
            if missing:
                reasons.append("footprint missing tensor references: " + ", ".join(sorted(missing)))
            elif p["checkpoint_bytes"]["value"] is not None:
                content = [(tensor_refs[ref]["sha256"], tensor_refs[ref]["bytes"]) for ref in p["checkpoint_refs"]]
                if len(set(content)) != len(content):
                    reasons.append("footprint overlapping checkpoint content")
                expected_bytes = sum(tensor_refs[ref]["bytes"] for ref in p["checkpoint_refs"])
                if p["checkpoint_bytes"]["value"] != expected_bytes:
                    reasons.append("footprint checkpoint byte count mismatch")
        elif event["event_type"] == "failure":
            reasons.append(p["reason"])
        elif event["event_type"] == "observation" and p["status"] == "complete":
            observed.add(p["identity"])
    phase_ids = set()
    previous_end: int | None = None
    previous_tokens: int | None = None
    for phase in manifest["phases"]:
        _fields(phase, "phase_id updates start_step start_valid_tokens", "phase")
        _text(phase["phase_id"], "phase_id")
        for key in ("updates", "start_step", "start_valid_tokens"):
            _count(phase[key], key)
        _require(phase["updates"] > 0 and phase["phase_id"] not in phase_ids, "invalid/duplicate phase")
        if previous_end is not None:
            _require(phase["start_step"] == previous_end, "phase step continuity mismatch")
            _require(phase["start_valid_tokens"] == previous_tokens, "phase exposure continuity mismatch")
        phase_ids.add(phase["phase_id"])
        rows = updates.get(phase["phase_id"], [])
        expected = list(range(phase["start_step"] + 1, phase["start_step"] + phase["updates"] + 1))
        if [row["completed_step"] for row in rows] != expected or any(row["final_step"] != phase["start_step"] + phase["updates"] or row["phase_start_step"] != phase["start_step"] for row in rows):
            reasons.append(f"missing/invalid update cadence: {phase['phase_id']}")
        cumulative = phase["start_valid_tokens"]
        for row in rows:
            cumulative += row["exposure"]["valid_tokens"]
            if row["exposure"]["cumulative_valid_tokens"] != cumulative:
                reasons.append("inconsistent cumulative token exposure")
        previous_end = phase["start_step"] + phase["updates"]
        previous_tokens = cumulative
    _require(not set(updates) - phase_ids, "undeclared phase")
    missing_observations = sorted(set(manifest["expected_observations"]) - observed)
    available = {item["path"] for item in inventory["artifacts"]}
    missing_artifacts = sorted(set(manifest["expected_artifacts"]) - available)
    if missing_observations:
        reasons.append("missing mandatory observations")
    if missing_artifacts:
        reasons.append("missing mandatory artifacts")
    return {"schema": SCHEMA, "run_id": manifest["run_id"], "status": "incomplete" if reasons else "complete", "reasons": reasons, "missing_observations": missing_observations, "missing_artifacts": missing_artifacts, "inventory_mode": inventory["mode"], "tensor_bytes_verified": inventory["mode"] == "tensor-complete" and any(a["kind"] == "tensor" for a in inventory["artifacts"])}
