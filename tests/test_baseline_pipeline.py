"""Connected tiny CPU producer/save/reload/evaluate/records/report acceptance."""
import copy
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch

from jscc.activation_replay import canonical_digest
from jscc.baseline_protocol import BaselineLearner, comparison_refs, run_baseline
from jscc.evaluation import evaluate_checkpoint
from jscc.experiment_records import completion_status, read_events
from jscc.experiment_state import CheckpointRef, open_state, restore_state
from jscc.models.codec import Codec
from jscc.models.split_model import resolve_encoder_site
from jscc.runtime import model_inputs
from test_core import batch, config, toy_model


def setup_pipeline(monkeypatch, *, cell="R-none", final_step=4, model=None, run_id="cpu"):
    if model is None:
        torch.manual_seed(17)
        model = toy_model(where="after_final_norm")
        architecture, norm = {"R-none": ("residual_mlp", "none"), "R-LN": ("residual_mlp", "both"),
                              "D-none": ("direct_affine", "none"), "D-LN": ("direct_outer_ln", "both")}[cell]
        model.codec = Codec(8, {**config(), "architecture": architecture, "layernorm": norm,
                               "n_res_blocks": 2 if cell.startswith("R") else 0, "snr_film": False})
    learner = BaselineLearner(model, run_id=run_id, task="hellaswag", synthetic=True,
        identity={"source": "source", "config": "config", "data": "data", "parent": None},
        pairing_id="paired-cpu", effective_batch=64, final_step=final_step)
    model.split.pop("index", None)
    model.base.enc.config = SimpleNamespace(hidden_size=8)
    site = resolve_encoder_site(model.base, model.split, "tiny-random-cpu")
    metadata = dict(source_identity="source", config_identity="config", data_identity="data", parent_identity=None,
                    initialization_identity=canonical_digest({k: v.tolist() for k, v in model.codec.state_dict().items()}),
                    stream_identity="fixed-64-sequence-stream", protocol_identity="synthetic-mechanics", lineage=[],
                    site=asdict(site), model_state_contract="codec-only-stateless-channel-v1")
    request = {"schema": "codec-observation-v1", "checkpoint_sha256": "pending", "target_site": asdict(site),
               "target_role": "trained", "task": "hellaswag", "input_ids": [0, 1], "source_family_ids": ["family-0", "family-1"],
               "prompt_policy": "tiny-native", "noise": {"seed": 0, "namespace": "tiny-task"},
               "layout": "two-full-padded-sequences", "backend": "eager", "precision": "fp32",
               "scorer": "tiny-logit-argmax", "reference_corpus": "synthetic", "source": "source", "expected_items": 2,
               "settings": {"num_samples": 2}, "data_settings": {}, "conditions": ["no_noise", -6, 0, 6, 12, 18],
               "config": "config", "data": "data", "parent": None, "panel": "tiny-task-panel", "protocol": "synthetic-mechanics",
               "learner_kind": "single_site_v2", "step": 0, "comparison": comparison_refs(learner, metadata, "combined")}
    task_batch = batch()
    def adapter(model, processor, settings, output, condition, data_settings):
        kwargs, _ = model_inputs(task_batch, model)
        with torch.no_grad():
            logits = model(**kwargs).logits[:, 0, :]
        rows = []
        for i, scores in enumerate(logits.tolist()):
            prediction = int(logits[i].argmax())
            rows.append({"sample_id": i, "source_id": f"family-{i}", "tokens": [prediction],
                         "raw_scores": scores, "denominators": [1] * len(scores), "normalized_scores": scores,
                         "normalized_prediction": prediction, "raw_correct": int(prediction == int(task_batch['labels'][i, 0])),
                         "normalized_correct": int(prediction == int(task_batch['labels'][i, 0]))})
        (output / f"compact_{condition}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        return {"acc": sum(r["raw_correct"] for r in rows) / len(rows)}
    monkeypatch.setattr("jscc.evaluation.evaluate_hellaswag", adapter)
    def assess(state, output):
        actual = {**request, "checkpoint_sha256": state.reference.sha256,
                  "step": state.payload["metadata"]["completed_updates"],
                  "parent": state.payload["metadata"]["parent_identity"],
                  "comparison": state.payload["metadata"]["comparison_controls"]}
        return evaluate_checkpoint(state, actual, model=model, processor=None, output=output)
    training: dict = {k: v.repeat(32, 1) for k, v in batch().items()}
    training.update(row_ids=torch.arange(64), source_family_ids=[f"source-{i}" for i in range(64)])
    return learner, metadata, request, training, assess


@pytest.mark.parametrize("cell", ["D-none", "D-LN", "R-none", "R-LN"])
def test_real_pipeline_inventory_restore_and_report(monkeypatch, tmp_path, cell):
    obj, meta, request, b, assess = setup_pipeline(monkeypatch, cell=cell)
    output = tmp_path / cell
    result = run_baseline(obj, output=output, metadata=meta, update_batches=lambda *_: [b],
                          validation_batches=lambda _: [b], assess=assess, task_request=request)
    assert result["status"] == "complete" and result["tensor_bytes_verified"]
    events = read_events(output / "metrics.jsonl")
    updates = [e for e in events if e["event_type"] == "update"]
    observations = [e for e in events if e["event_type"] == "observation"]
    assert len(updates) == 4 and len(observations) == 8
    assert sum(len(e["payload"]["conditions"]) for e in observations) == 48
    assert updates[0]["payload"]["update_l2"]["value"] == 0  # true warmup zero-LR call
    assert updates[-1]["payload"]["update_l2"]["value"] > 0
    inventory = json.loads((output / "inventory.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert completion_status(manifest, events, inventory, output)["status"] == "complete"
    state = open_state(CheckpointRef(**result["checkpoints"][-1]), expected={**meta, "completed_updates": 4})
    clone = copy.deepcopy(obj.model.codec)
    optimizer = torch.optim.AdamW(clone.parameters(), lr=1e-4)
    from jscc.experiment_schedule import build_baseline_scheduler
    scheduler = build_baseline_scheduler(optimizer)
    restore_state(state, model=clone, optimizer=optimizer, scheduler=scheduler, scaler=obj.scaler)
    assert scheduler.last_epoch == 4
    for name, value in clone.state_dict().items():
        torch.testing.assert_close(value, obj.model.codec.state_dict()[name], rtol=0, atol=0)
    if cell == "D-none":
        from scripts.render_experiment_report import render_local_report
        render_local_report(output, tmp_path / "report")
        assert (tmp_path / "report/index.html").exists()
    (output / "step_000004.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        completion_status(manifest, events, inventory, output)


@pytest.mark.parametrize("failure", ["log", "save", "evaluation", "guard", "ambiguous_update"])
def test_connected_failures_never_complete_or_retry(monkeypatch, tmp_path, failure):
    obj, meta, request, b, assess = setup_pipeline(monkeypatch)
    output = tmp_path / failure
    if failure == "log":
        from jscc import baseline_protocol
        original = baseline_protocol.append_event
        def append(path, event):
            if event["event_type"] == "update":
                raise OSError("injected log failure")
            return original(path, event)
        monkeypatch.setattr(baseline_protocol, "append_event", append)
    elif failure == "save":
        from jscc import experiment_state
        original = experiment_state.save_state
        def save(path, **kwargs):
            if kwargs["metadata"]["completed_updates"] == 1:
                raise OSError("injected save failure")
            return original(path, **kwargs)
        monkeypatch.setattr(experiment_state, "save_state", save)
    elif failure == "evaluation":
        def failed_assessment(*_):
            raise RuntimeError("injected evaluation failure")
        assess = failed_assessment
    elif failure == "ambiguous_update":
        original = obj.optimizer.step
        def step(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("ambiguous optimizer receipt")
        monkeypatch.setattr(obj.optimizer, "step", step)
    def guard():
        if failure == "guard" and obj.completed == 1:
            raise RuntimeError("allocation deadline")
    with pytest.raises((OSError, RuntimeError)):
        run_baseline(obj, output=output, metadata=meta, update_batches=lambda *_: [b],
                     validation_batches=lambda _: [b], assess=assess, task_request=request, resource_guard=guard)
    assert obj.failed
    assert json.loads((output / "completion.json").read_text())["status"] == "incomplete"
    assert (output / "failure.json").exists()
    with pytest.raises(RuntimeError, match="terminal"):
        obj.update([b])
    if failure == "ambiguous_update":
        assert obj.completed == 0 and obj.attempted == 1
        assert all(state["step"] == 1 for state in obj.optimizer.state.values())


def test_false_control_declaration_rejected_before_work(monkeypatch, tmp_path):
    obj, meta, request, b, assess = setup_pipeline(monkeypatch)
    meta["comparison"] = {**request["comparison"], "architecture": "invented-matched-control"}
    with pytest.raises(ValueError, match="actual config/state"):
        run_baseline(obj, output=tmp_path / "bad", metadata=meta, update_batches=lambda *_: [b],
                     validation_batches=lambda _: [b], assess=assess, task_request=request)
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("failed", [False, True])
def test_allocation_outcome_is_bound_to_connected_completion(monkeypatch, tmp_path, failed):
    from scripts.encfn_baseline import bind_allocation_result
    obj, meta, request, b, assess = setup_pipeline(monkeypatch)
    output = tmp_path / "run"
    run_baseline(obj, output=output, metadata=meta, update_batches=lambda *_: [b],
                 validation_batches=lambda _: [b], assess=assess, task_request=request)
    receipt = {"terminal_reason": "deadline" if failed else None,
               "allocations": [{"charged_device_seconds": 3.5, "devices": 1}]}
    status = bind_allocation_result(output, receipt, RuntimeError("deadline") if failed else None)
    assert status["status"] == ("incomplete" if failed else "complete")
    assert "allocation.json" in json.loads((output / "manifest.json").read_text())["expected_artifacts"]
    events = read_events(output / "metrics.jsonl")
    assert any(e["event_type"] == "cost" and e["payload"]["category"] == "gpu_allocation" for e in events)
    assert any(e["event_type"] == "failure" for e in events) == failed
