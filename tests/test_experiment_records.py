"""Model-free adversarial tests for event and artifact evidence."""
import copy
import json
from pathlib import Path

import pytest

from jscc import experiment_records as records


def update(step=1, final=1, task="hellaswag"):
    m = records.measure
    component = {"raw": m(2), "weighted": m(.2), "numerator": m(8), "denominator": 4}
    unknown = {"raw": m(None, "unmeasured"), "weighted": m(None, "unmeasured"), "numerator": m(None, "unmeasured"), "denominator": 0}
    return {"schema": records.SCHEMA, "event_type": "update", "run_id": "r", "phase_id": "local", "task": task, "timestamp": "2026-09-30T00:00:00Z", "units": {"time": "seconds", "memory": "bytes"}, "identity": {"source": "source", "config": "config", "data": "data", "parent": None}, "payload": {"site_id": "enc_fn", "phase_start_step": 0, "site_step": step, "sweep": step, "attempted_step": step, "completed_step": step, "final_step": final, "exposure": {"source_tokens": 4, "target_tokens": 2, "sequences": 1, "padded_tokens": 6, "valid_tokens": 4, "cumulative_valid_tokens": 4*step, "latent_width": 2, "hidden_coordinates": 8, "hidden_coordinates_allocated": 12, "memory_coordinates_allocated": m(None, "not_applicable"), "memory_coordinates": m(None, "not_applicable")}, "objective": {"kind": "local", "total": m(.2), "components": {"R": component, "K": unknown}}, "lr_used": [0], "lr_next": [.001], "gradient": {"pre_clip": m(1), "post_clip": m(1), "clip_threshold": m(1), "clipped": False}, "nonfinite": False, "skipped": False, "update_l2": m(0) if step in (1, final) or step % 10 == 0 else m(None, "not_sampled"), "snr": {"condition": "no_noise", "min": m(None, "no_noise"), "mean": m(None, "no_noise"), "max": m(None, "no_noise"), "draw_ref": "draw"}, "timing": {"step_seconds": m(1), "elapsed_seconds": m(step), "data_cache_seconds": m(None, "unmeasured"), "throughput": m(4), "throughput_denominator": "valid_tokens/second", "scope": "asynchronous wall"}, "memory": {"allocated_bytes": m(None, "cpu"), "reserved_bytes": m(None, "cpu"), "reset_scope": "update"}}}


def observation(task="hellaswag"):
    event = update(task=task)
    event["event_type"] = "observation"
    request = dict(event["identity"], checkpoint="checkpoint", panel="panel", site="enc_fn", role="trained", task=task, protocol="protocol", noise="noise", scorer="scorer", layout="layout", precision="fp32", backend="cpu", conditions=records.CONDITIONS, expected_items=1, learner_kind="initialization", step=0, site_step=0, purpose="task", objective_kind=None, comparison={k:k for k in ("architecture", "initialization", "training_data", "objective", "exposure", "schedule")}, details={k:k for k in ("prompt", "native_policy", "draws", "outputs")})
    event["payload"] = {"request": request, "identity": records.observation_identity(request), "status": "complete", "reuse": None, "conditions": [{"condition": c, "requested": 1, "completed": 1, "failed": 0, "failure_reason": None, "denominator": 1, "metrics": ({"raw_accuracy": records.measure(1), "normalized_accuracy": records.measure(1)} if task == "hellaswag" else {"cider": records.measure(1)}), "items": [{"item_id": "i", "source_id": "s", "raw_correct": 1, "normalized_correct": 1, "prediction": "a", "score": 1., "tokens": [1], "caption_raw": "a", "caption_clean": "a", "cider": 1., "eos": True, "cap_hit": False}], "objectives": {}} for c in records.CONDITIONS]}
    return event


def inventory():
    return {"schema": records.SCHEMA, "mode": "metadata-only", "artifacts": [], "omitted_tensors": []}


def manifest():
    return {"schema": records.SCHEMA, "run_id": "r", "phases": [{"phase_id": "local", "updates": 1, "start_step": 0, "start_valid_tokens": 0}], "expected_observations": [], "expected_artifacts": []}


@pytest.mark.parametrize("task", ["coco", "hellaswag"])
def test_both_tasks_and_unavailable_zero(task):
    event = update(task=task)
    records.validate_event(event)
    records.validate_event(observation(task))
    assert event["payload"]["lr_used"] == [0]
    event["payload"]["objective"]["components"]["K"]["raw"] = records.measure(0)
    with pytest.raises(records.RecordError):
        records.validate_event(event)


@pytest.mark.parametrize("field", list(update()["payload"]))
def test_every_update_field_required(field):
    event = update()
    del event["payload"][field]
    with pytest.raises(records.RecordError):
        records.validate_event(event)


@pytest.mark.parametrize("mutation", ["unknown", "nan", "denominator", "counter", "l2", "total"])
def test_reject_bad_updates(mutation):
    event = update(step=2, final=3)
    p = event["payload"]
    if mutation == "unknown":
        p["invented"] = 0
    elif mutation == "nan":
        p["timing"]["step_seconds"]["value"] = float("nan")
    elif mutation == "denominator":
        p["objective"]["components"]["R"]["denominator"] = 0
    elif mutation == "counter":
        p["completed_step"] = 3
    elif mutation == "l2":
        p["update_l2"] = records.measure(0)
    else:
        p["objective"]["total"] = records.measure(0)
    with pytest.raises(records.RecordError):
        records.validate_event(event)


def test_observation_identity_and_complete_reuse():
    event = observation()
    p = event["payload"]
    digest = p["identity"]
    p["reuse"] = {"identity": digest, "status": "complete", "reference": "original receipt", "acquisition_seconds": records.measure(2)}
    records.validate_event(event)
    p["request"]["checkpoint"] = "different"
    with pytest.raises(records.RecordError, match="identity mismatch"):
        records.validate_event(event)
    p["identity"] = records.observation_identity(p["request"])
    with pytest.raises(records.RecordError, match="reuse"):
        records.validate_event(event)


@pytest.mark.parametrize("mutation", ["missing", "denominator", "failed", "duplicate", "source"])
def test_observation_incompleteness(mutation):
    event = observation()
    p = event["payload"]
    row = p["conditions"][0]
    if mutation == "missing":
        p["conditions"].pop()
    elif mutation == "denominator":
        row["denominator"] = 0
    elif mutation == "failed":
        row["failed"] = 1
    elif mutation == "duplicate":
        p["conditions"].append(copy.deepcopy(row))
    else:
        del row["items"][0]["source_id"]
    with pytest.raises(records.RecordError):
        records.validate_event(event)


def test_atomic_append_preserves_old_on_replace_failure(tmp_path, monkeypatch):
    path = tmp_path / "metrics.jsonl"
    records.append_event(path, update())
    before = path.read_bytes()
    def fail(*args):
        raise OSError("injected rename failure")
    monkeypatch.setattr(records.os, "replace", fail)
    with pytest.raises(OSError):
        records.append_event(path, update())
    assert path.read_bytes() == before
    assert list(tmp_path.glob("*.partial-*"))


def test_partial_write_retained_not_published(tmp_path, monkeypatch):
    real_fdopen = records.os.fdopen
    class Partial:
        def __init__(self, fd, mode):
            self.handle = real_fdopen(fd, mode)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.handle.close()
        def write(self, data):
            return self.handle.write(data[:3])
    monkeypatch.setattr(records.os, "fdopen", Partial)
    path = tmp_path / "metrics.jsonl"
    with pytest.raises(OSError, match="partial"):
        records.append_event(path, update())
    assert not path.exists()
    assert list(tmp_path.glob("*.partial-*"))[0].read_bytes() == b'{"e'


def test_fsync_failure_does_not_publish(tmp_path, monkeypatch):
    def fail(fd):
        raise OSError("injected fsync failure")
    monkeypatch.setattr(records.os, "fsync", fail)
    with pytest.raises(OSError):
        records.write_json_atomic(tmp_path / "record.json", {"valid": True})
    assert not (tmp_path / "record.json").exists()


def test_read_truncation_rejects_without_repair(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps(update()))
    with pytest.raises(records.RecordError, match="truncated"):
        records.append_event(path, update())


def test_inventory_hash_bytes_and_omitted_tensors(tmp_path):
    path = tmp_path / "tensor.bin"
    path.write_bytes(b"tensor bytes are not loaded")
    inv = inventory()
    inv["artifacts"] = [records.artifact_ref(path, tmp_path, "tensor")]
    records.verify_inventory(tmp_path, inv)
    inv["artifacts"][0]["bytes"] += 1
    with pytest.raises(records.RecordError, match="hash/size"):
        records.verify_inventory(tmp_path, inv)
    inv["omitted_tensors"] = inv.pop("artifacts")
    inv["artifacts"] = []
    inv["mode"] = "tensor-complete"
    with pytest.raises(records.RecordError, match="omits"):
        records.verify_inventory(tmp_path, inv)


def test_completion_is_derived_and_missing_panels_fail(tmp_path):
    m = manifest()
    m["expected_observations"] = [observation()["payload"]["identity"]]
    result = records.completion_status(m, [update()], inventory(), tmp_path)
    assert result["status"] == "incomplete"
    result = records.completion_status(m, [update(), observation()], inventory(), tmp_path)
    assert result["status"] == "complete"
    assert not result["tensor_bytes_verified"]
    m["phases"][0]["updates"] = 2
    assert records.completion_status(m, [update(), observation()], inventory(), tmp_path)["status"] == "incomplete"


def test_no_model_import_in_clean_process():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-c", "import sys; import jscc.experiment_records; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"], cwd=Path(__file__).parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_unknown_failed_count_preserved(tmp_path):
    event = observation()
    row = event["payload"]["conditions"][0]
    row["failed"] = None
    row["failure_reason"] = "adapter interrupted; exact failed count unknown"
    with pytest.raises(records.RecordError, match="complete"):
        records.validate_event(event)
    event["payload"]["status"] = "incomplete"
    records.validate_event(event)
    m = manifest()
    m["expected_observations"] = [event["payload"]["identity"]]
    assert records.completion_status(m, [update(), event], inventory(), tmp_path)["status"] == "incomplete"


def test_staged_phase_global_offsets(tmp_path):
    event = update(step=201, final=201)
    event["payload"]["phase_start_step"] = 200
    m = manifest()
    m["phases"][0].update(start_step=200, start_valid_tokens=800)
    assert records.completion_status(m, [event], inventory(), tmp_path)["status"] == "complete"


def test_objective_observation_keeps_distinct_denominators():
    event = observation()
    p = event["payload"]
    p["request"]["purpose"] = "objective"
    p["request"]["objective_kind"] = "local"
    p["identity"] = records.observation_identity(p["request"])
    for row in p["conditions"]:
        row["metrics"] = {}
        row["objectives"] = update()["payload"]["objective"]["components"]
        row["items"] = [{"item_id": "i", "source_id": "s"}]
    records.validate_event(event)
    assert p["conditions"][0]["denominator"] == 1
    assert p["conditions"][0]["objectives"]["R"]["denominator"] == 4


@pytest.mark.parametrize("task", ["coco", "hellaswag"])
def test_task_metric_requires_per_item_evidence(task):
    event = observation(task)
    row = event["payload"]["conditions"][0]
    metric = "raw_accuracy" if task == "hellaswag" else "cider"
    row["metrics"][metric] = records.measure(0)
    with pytest.raises(records.RecordError, match="per-item"):
        records.validate_event(event)


def test_heldout_specialist_rejected():
    request = observation()["payload"]["request"]
    request.update(role="heldout", learner_kind="specialist")
    with pytest.raises(records.RecordError, match="held-out"):
        records.observation_identity(request)


def test_failed_update_never_completes(tmp_path):
    event = update()
    event["payload"].update(nonfinite=True, completed_step=0)
    assert records.completion_status(manifest(), [event], inventory(), tmp_path)["status"] == "incomplete"


def test_inventory_path_escape_and_missing(tmp_path):
    inv = inventory()
    inv["artifacts"] = [{"path": "../outside", "kind": "metadata", "sha256": "a"*64, "bytes": 1}]
    with pytest.raises(records.RecordError, match="unsafe"):
        records.verify_inventory(tmp_path, inv)
    inv["artifacts"][0]["path"] = "missing"
    with pytest.raises(FileNotFoundError):
        records.verify_inventory(tmp_path, inv)


def test_duplicate_and_sparse_updates_fail_completion(tmp_path):
    assert records.completion_status(manifest(), [update(), update()], inventory(), tmp_path)["status"] == "incomplete"
    m = manifest()
    m["phases"][0]["updates"] = 3
    events = [update(1, 3), update(3, 3)]
    assert records.completion_status(m, events, inventory(), tmp_path)["status"] == "incomplete"


def objective_observation(kind="local"):
    event = observation()
    p = event["payload"]
    p["request"].update(purpose="objective", objective_kind=kind)
    p["identity"] = records.observation_identity(p["request"])
    for row in p["conditions"]:
        row["metrics"] = {}
        row["objectives"] = copy.deepcopy(update()["payload"]["objective"]["components"])
        if kind == "combined":
            row["objectives"]["K"] = {"raw": records.measure(1), "weighted": records.measure(1), "numerator": records.measure(2), "denominator": 2}
        row["items"] = [{"item_id": "i", "source_id": "s"}]
    return event


@pytest.mark.parametrize("kind", ["local", "combined"])
def test_all_unavailable_objective_cannot_complete(kind, tmp_path):
    event = objective_observation(kind)
    records.validate_event(event)
    p = event["payload"]
    for row in p["conditions"]:
        for component in row["objectives"].values():
            component.update({key: records.measure(None, "missing") for key in ("raw", "weighted", "numerator")})
            component["denominator"] = 0
    m = manifest()
    m["expected_observations"] = [p["identity"]]
    with pytest.raises(records.RecordError, match="requires R"):
        records.completion_status(m, [update(), event], inventory(), tmp_path)


@pytest.mark.parametrize("kind,component,weight", [("local", "R", 2), ("combined", "R", 2), ("combined", "K", .1)])
def test_objective_observation_weights_checked(kind, component, weight):
    event = objective_observation(kind)
    event["payload"]["conditions"][0]["objectives"][component]["weighted"] = records.measure(weight)
    with pytest.raises(records.RecordError, match="weight"):
        records.validate_event(event)


def test_objective_request_kind_and_controls_are_identity_bound():
    request = objective_observation()["payload"]["request"]
    original = records.observation_identity(request)
    request["objective_kind"] = "combined"
    assert records.observation_identity(request) != original
    request["comparison"]["initialization"] = "other retained initialization"
    assert records.observation_identity(request) != original
    del request["comparison"]["schedule"]
    with pytest.raises(records.RecordError, match="comparison controls"):
        records.observation_identity(request)


@pytest.mark.parametrize("start_step,start_tokens", [(0, 4), (2, 4), (1, 0), (1, 8)])
def test_crossphase_reset_overlap_gap_exposure_rejected(tmp_path, start_step, start_tokens):
    m = manifest()
    m["phases"].append({"phase_id": "second", "updates": 1, "start_step": start_step, "start_valid_tokens": start_tokens})
    second = update(step=start_step+1, final=start_step+1)
    second["phase_id"] = "second"
    second["payload"]["phase_start_step"] = start_step
    second["payload"]["exposure"]["cumulative_valid_tokens"] = start_tokens + 4
    with pytest.raises(records.RecordError, match="continuity"):
        records.completion_status(m, [update(), second], inventory(), tmp_path)


def test_crossphase_valid_global_continuation(tmp_path):
    m = manifest()
    m["phases"].append({"phase_id": "second", "updates": 1, "start_step": 1, "start_valid_tokens": 4})
    second = update(step=2, final=2)
    second["phase_id"] = "second"
    second["payload"]["phase_start_step"] = 1
    assert records.completion_status(m, [update(), second], inventory(), tmp_path)["status"] == "complete"


def footprint(checkpoint_bytes=3, refs=None):
    event = update()
    event["event_type"] = "footprint"
    event["payload"] = {"scope": "per_site", "site_id": "enc_fn", "parameter_count": records.measure(10), "checkpoint_bytes": records.measure(checkpoint_bytes), "checkpoint_refs": ["state.bin"] if refs is None else refs}
    return event


def test_duplicate_omitted_paths_rejected(tmp_path):
    inv = inventory()
    ref = {"path": "state.bin", "kind": "tensor", "sha256": "a" * 64, "bytes": 3}
    inv["omitted_tensors"] = [ref, dict(ref, sha256="b" * 64, bytes=7)]
    with pytest.raises(records.RecordError, match="duplicate omitted"):
        records.verify_inventory(tmp_path, inv)


@pytest.mark.parametrize("scope", ["per_site", "bank_total", "shared"])
def test_footprint_retained_and_omitted_counts(tmp_path, scope):
    path = tmp_path / "state.bin"
    path.write_bytes(b"abc")
    ref = records.artifact_ref(path, tmp_path, "tensor")
    inv = inventory()
    inv["artifacts"] = [ref]
    event = footprint()
    event["payload"]["scope"] = scope
    assert records.completion_status(manifest(), [update(), event], inv, tmp_path)["status"] == "complete"
    inv["artifacts"] = []
    inv["omitted_tensors"] = [ref]
    path.unlink()
    result = records.completion_status(manifest(), [update(), event], inv, tmp_path)
    assert result["status"] == "complete"
    assert not result["tensor_bytes_verified"]


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_footprint_invalid_measured_counts(value):
    event = footprint()
    event["payload"]["parameter_count"] = records.measure(value) if type(value) is not bool else {"value": value, "reason": None}
    with pytest.raises(records.RecordError):
        records.validate_event(event)


def test_footprint_unavailable_vs_zero():
    event = footprint()
    event["payload"].update(parameter_count=records.measure(0), checkpoint_bytes=records.measure(None, "not saved"), checkpoint_refs=[])
    records.validate_event(event)
    event["payload"]["checkpoint_bytes"] = records.measure(0)
    with pytest.raises(records.RecordError, match="require references"):
        records.validate_event(event)


def test_footprint_duplicate_refs_rejected():
    with pytest.raises(records.RecordError, match="duplicate checkpoint"):
        records.validate_event(footprint(refs=["state.bin", "state.bin"]))


def test_footprint_missing_or_mismatched_refs_incomplete(tmp_path):
    event = footprint()
    assert records.completion_status(manifest(), [update(), event], inventory(), tmp_path)["status"] == "incomplete"
    inv = inventory()
    inv["omitted_tensors"] = [{"path": "state.bin", "kind": "tensor", "sha256": "a" * 64, "bytes": 4}]
    assert records.completion_status(manifest(), [update(), event], inv, tmp_path)["status"] == "incomplete"


def test_footprint_content_aliases_incomplete(tmp_path):
    inv = inventory()
    inv["omitted_tensors"] = [{"path": path, "kind": "tensor", "sha256": "a" * 64, "bytes": 3} for path in ("first.bin", "alias.bin")]
    result = records.completion_status(manifest(), [update(), footprint(6, ["first.bin", "alias.bin"])], inv, tmp_path)
    assert result["status"] == "incomplete"
    assert "footprint overlapping checkpoint content" in result["reasons"]


@pytest.mark.parametrize("group,key", [("timing", "step_seconds"), ("timing", "elapsed_seconds"), ("timing", "data_cache_seconds"), ("timing", "throughput"), ("memory", "allocated_bytes"), ("memory", "reserved_bytes"), ("exposure", "memory_coordinates"), ("exposure", "memory_coordinates_allocated")])
def test_negative_physical_update_measurements_rejected(group, key):
    event = update()
    event["payload"][group][key] = records.measure(-1)
    with pytest.raises(records.RecordError):
        records.validate_event(event)


def test_negative_cost_rejected():
    event = update()
    event["event_type"] = "cost"
    event["payload"] = {"category": "training", "seconds": records.measure(-1), "attribution": "first_use", "site_id": "enc_fn", "scope": "phase", "concurrency": "none", "device_count": 1}
    with pytest.raises(records.RecordError, match="negative cost"):
        records.validate_event(event)
