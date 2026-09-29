import copy

import pytest
import torch

from jscc.baseline_protocol import BaselineLearner, objective_settings
from jscc.training import batch_losses
from jscc.training_objectives import aggregate_batch_losses
from test_core import batch, toy_model


def learner(**kwargs):
    model = toy_model(where="after_final_norm")
    model.codec.film = None
    return BaselineLearner(model, run_id="cpu", task="hellaswag",
                           identity={"source": "s", "config": "c", "data": "d", "parent": None},
                           pairing_id="pair", synthetic=True, effective_batch=2, final_step=2, **kwargs)


def test_online_updates_are_real_and_teacher_stays_frozen():
    obj = learner()
    before = copy.deepcopy(obj.model.codec.state_dict())
    event = obj.update([batch()])
    assert event["payload"]["lr_used"] == [0.0]
    assert obj.scheduler.last_epoch == 1
    assert all(state["step"] == 1 for state in obj.optimizer.state.values())
    assert all(torch.equal(before[k], v) for k, v in obj.model.codec.state_dict().items())
    event = obj.update([batch()])
    assert event["payload"]["completed_step"] == 2
    assert event["payload"]["objective"]["components"]["K"]["denominator"] == 5
    assert any(not torch.equal(before[k], v) for k, v in obj.model.codec.state_dict().items())
    assert all(p.grad is None for p in obj.model.base.parameters())
    assert not obj.model.base.training
    with pytest.raises(RuntimeError, match="terminal"):
        obj.update([batch()])


def test_local_weight_and_missing_k_and_online_gradient_parity():
    obj = learner()
    source = batch()
    from jscc.runtime import model_inputs
    kwargs, _ = model_inputs(source, obj.model)
    with torch.no_grad(), obj.model.transmission(bypass=True):
        captured = []
        hook = obj.model.base.enc.norm.register_forward_hook(lambda _m, _a, o: captured.append(o.detach().clone()))
        obj.model(**kwargs)
        hook.remove()
    shard = {**source, "activation": captured[0]}
    functional, _, draw = obj._values(source, "combined", 1, 0, "train")
    (functional["nmse"] * .1).backward()
    expected = [p.grad.clone() if p.grad is not None else None for p in obj.parameters]
    obj.optimizer.zero_grad(set_to_none=True)
    local, _, local_draw = obj._values(shard, "local", 1, 0, "train")
    local["loss"].backward()
    assert draw == local_draw
    torch.testing.assert_close(local["loss"], functional["nmse"] * .1)
    for p, want in zip(obj.parameters, expected):
        if want is not None:
            torch.testing.assert_close(p.grad, want)
    event = obj.update([shard], kind="local")
    k = event["payload"]["objective"]["components"]["K"]
    assert k["raw"] == {"value": None, "reason": "unmeasured"}
    assert k["denominator"] == 0


def test_validation_six_conditions_keeps_rng_and_mode():
    obj = learner()
    state = torch.get_rng_state().clone()
    records = obj.validate([batch()])
    assert torch.equal(state, torch.get_rng_state())
    assert obj.model.training and not obj.model.base.training
    assert [r["condition"] for r in records] == ["no_noise", -6, 0, 6, 12, 18]
    assert all(r["objective"]["components"]["K"]["denominator"] == 5 for r in records)


def test_skipped_step_fail_stop_preserves_attempted_exposure(monkeypatch):
    obj = learner()
    monkeypatch.setattr(obj.scaler, "step", lambda optimizer: None)
    with pytest.raises(FloatingPointError, match="skipped"):
        obj.update([batch()])
    assert obj.attempted == 1 and obj.completed == 0 and obj.offset == 2
    assert obj.scheduler.last_epoch == 0
    with pytest.raises(RuntimeError, match="terminal"):
        obj.update([batch()])


def test_nonfinite_and_empty_sequences_stop():
    obj = learner()
    with torch.no_grad():
        next(obj.model.codec.parameters()).fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="nonfinite"):
        obj.update([batch()])
    assert obj.completed == 0 and obj.attempted == 1
    obj = learner()
    bad = batch()
    bad["attention_mask"][0] = 0
    with pytest.raises(ValueError, match="empty real sequence"):
        obj.update([bad])


def test_unequal_microbatch_denominators_and_gradients():
    from jscc.models.channel import IdentityChannel
    model = toy_model(where="after_final_norm", channel=IdentityChannel())
    model.codec.film = None
    other = copy.deepcopy(model)
    first = {k: v[:1] for k, v in batch().items()}
    second = batch()
    joined = {k: torch.cat((first[k], second[k])) for k in first}
    settings = objective_settings("combined")
    values = [batch_losses(model, b, settings, None, return_stats=True) for b in [first, second]]
    aggregate = aggregate_batch_losses(values, settings)
    aggregate["loss"].backward()
    full = batch_losses(other, joined, settings, None, return_stats=True)
    full["loss"].backward()
    assert int(aggregate["kl_denominator"]) == 8
    assert int(aggregate["hidden_denominator"]) == 3
    torch.testing.assert_close(aggregate["loss"], full["loss"])
    for p, q in zip(model.codec.parameters(), other.codec.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def test_runner_executes_updates_saves_and_mandatory_measurements(tmp_path):
    from jscc.baseline_protocol import run_baseline, CONDITIONS
    model = toy_model(where="after_final_norm")
    model.codec.film = None
    obj = BaselineLearner(model, run_id="runner", task="hellaswag",
        identity={"source": "s", "config": "c", "data": "d", "parent": None},
        pairing_id="pair", synthetic=True, effective_batch=64, final_step=4)
    b = {k: v.repeat(32, 1) for k, v in batch().items()}
    meta = dict(source_identity="s", config_identity="c", parent_identity=None,
                initialization_identity="i", stream_identity="stream", protocol_identity="p", lineage=[])
    calls = []
    def assess(state, output):
        calls.append(state.payload["metadata"]["completed_updates"])
        return {"status": "complete", "conditions": [{"condition": c} for c in CONDITIONS]}
    result = run_baseline(obj, output=tmp_path / "run", metadata=meta,
        update_batches=lambda step, kind: [b], validation_batches=lambda kind: [b], assess=assess)
    assert result["completed_updates"] == 4
    assert len(result["checkpoints"]) == 5
    assert calls == [0, 2, 4]
    assert all((tmp_path / "run" / f"objective_{step:06d}.json").exists() for step in range(5))


def test_runner_missing_assessment_stops_before_training(tmp_path):
    from jscc.baseline_protocol import run_baseline
    import json
    model = toy_model(where="after_final_norm")
    model.codec.film = None
    obj = BaselineLearner(model, run_id="runner", task="hellaswag",
        identity={"source": "s", "config": "c", "data": "d", "parent": None},
        pairing_id="pair", synthetic=True, effective_batch=64, final_step=4)
    meta = dict(source_identity="s", config_identity="c", parent_identity=None,
                initialization_identity="i", stream_identity="stream", protocol_identity="p", lineage=[])
    with pytest.raises(RuntimeError, match="assessment incomplete"):
        run_baseline(obj, output=tmp_path / "run", metadata=meta,
                     update_batches=lambda *_: pytest.fail("no training allowed"),
                     validation_batches=lambda _: [batch()], assess=lambda *_: {"status": "incomplete"})
    result = json.loads((tmp_path / "run/completion.json").read_text())
    assert result["status"] == "incomplete" and result["completed_updates"] == 0
    assert len(result["checkpoints"]) == 1


def test_site_provenance_does_not_change_paired_draws():
    from jscc.baseline_protocol import batch_identity
    b = batch()
    assert batch_identity(b) == batch_identity({**b, "site_id": "enc_fn", "activation": torch.randn(2, 3, 8)})


def test_task_reuse_requires_exact_complete_observation(monkeypatch, tmp_path):
    from jscc.baseline_protocol import validate_task_receipt, CONDITIONS
    from jscc.evaluation import codec_observation_identity
    from test_evaluation_evidence import observation_fixture
    _, request, _, _ = observation_fixture(monkeypatch, tmp_path)
    receipt = {"status": "complete", "request": request, "identity": codec_observation_identity(request),
               "conditions": [{"condition": c, "status": "complete", "requested": 1, "completed": 1,
                               "failed": 0, "denominator": 1, "metrics": {"acc": 1.0},
                               "items": [{"sample_id": 7, "source_id": "source-7"}]} for c in CONDITIONS]}
    validate_task_receipt(receipt, request)
    for field, value in [("panel", "other-panel"), ("scorer", "other-scorer"),
                         ("noise", {"seed": 9, "namespace": "other"})]:
        changed = {**request, field: value}
        previous = {**receipt, "request": changed, "identity": codec_observation_identity(changed)}
        with pytest.raises(ValueError, match="identity differs"):
            validate_task_receipt(previous, request)
    with pytest.raises(ValueError, match="conditions incomplete"):
        validate_task_receipt({**receipt, "conditions": []}, request)
    for field, value in [("failed", None), ("completed", 0), ("status", "incomplete")]:
        rows = copy.deepcopy(receipt["conditions"])
        rows[0][field] = value
        with pytest.raises(ValueError, match="accounting incomplete"):
            validate_task_receipt({**receipt, "conditions": rows}, request)
