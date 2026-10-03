"""Diagnostic audit capture changes bookkeeping only, with bound prepared inputs."""
import copy
from dataclasses import FrozenInstanceError

import pytest
import torch

from jscc import baseline_protocol
from jscc.baseline_protocol import BaselineLearner, batch_identity, bind_batch_identity
from test_core import batch, toy_model


def make_learner(model, policy="full", final_step=400):
    model.codec.film = None
    return BaselineLearner(model, run_id="audit-cpu", task="hellaswag", synthetic=True,
        identity={"source": "s", "config": "c", "data": "d", "parent": None},
        pairing_id="same-audit-noise", effective_batch=2, final_step=final_step, audit_policy=policy)


def test_sparse_step_two_keeps_exact_gradients_parameters_moments_and_actual_noise(monkeypatch):
    torch.manual_seed(37)
    model = toy_model(where="after_final_norm")
    full = make_learner(model)
    sparse = make_learner(copy.deepcopy(model), "sparse_first_final")
    batches = [batch(), batch()]
    batches[1]["input_ids"][0, 0] = 11
    bindings = [bind_batch_identity(b) for b in batches]
    original_randn = torch.randn
    actual_noise = []
    def snapshot_randn(*args, **kwargs):
        value = original_randn(*args, **kwargs)
        actual_noise.append(value.detach().clone())
        return value
    monkeypatch.setattr(torch, "randn", snapshot_randn)
    original_clip = torch.nn.utils.clip_grad_norm_
    raw_gradients = []
    def snapshot_gradients(parameters, *args, **kwargs):
        parameters = list(parameters)
        raw_gradients.append([p.grad.detach().clone() if p.grad is not None else None for p in parameters])
        return original_clip(parameters, *args, **kwargs)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", snapshot_gradients)
    for step, (b, binding) in enumerate(zip(batches, bindings), 1):
        full_event = full.update([b], batch_identities=[binding])
        full_draw = copy.deepcopy(full.last_draws)
        sparse_event = sparse.update([b], batch_identities=[binding])
        assert full_event["payload"]["objective"] == sparse_event["payload"]["objective"]
        assert full_event["payload"]["gradient"] == sparse_event["payload"]["gradient"]
        assert full_event["payload"]["snr"]["min"] == sparse_event["payload"]["snr"]["min"]
        assert full_event["payload"]["snr"]["mean"] == sparse_event["payload"]["snr"]["mean"]
        assert full_event["payload"]["snr"]["max"] == sparse_event["payload"]["snr"]["max"]
        assert full_draw[0]["key"] == sparse.last_draws[0]["key"]
        assert torch.equal(actual_noise[-2], actual_noise[-1])
        for first, second in zip(raw_gradients[-2], raw_gradients[-1]):
            if first is None:
                assert second is None
            else:
                assert second is not None and torch.equal(first, second)
        for first, second in zip(full.parameters, sparse.parameters):
            assert torch.equal(first, second)
            for name, value in full.optimizer.state[first].items():
                other = sparse.optimizer.state[second][name]
                assert torch.equal(value, other) if torch.is_tensor(value) else value == other
        assert bool(sparse.last_draws[0]["draws"]) == (step == 1)
        assert sparse_event["payload"]["audit"]["noise_capture"] == (step == 1)
        assert "audit" not in full_event["payload"]


def test_bound_identity_hashes_once_before_timer_and_sparse_avoids_epsilon_digest(monkeypatch):
    b = batch()
    original_digest = baseline_protocol.tensor_digest
    hashed = []
    def count_digest(value):
        hashed.append(value)
        return original_digest(value)
    monkeypatch.setattr(baseline_protocol, "tensor_digest", count_digest)
    binding = bind_batch_identity(b)
    assert len(hashed) == 3
    learner = make_learner(toy_model(where="after_final_norm"), "sparse_first_final")
    learner.update([b], batch_identities=[binding])
    assert len(hashed) == 3
    def forbid_digest(_):
        pytest.fail("uncaptured step must not copy/hash epsilon")
    monkeypatch.setattr("jscc.models.channel.tensor_digest", forbid_digest)
    learner.update([b], batch_identities=[binding])
    assert len(hashed) == 3
    assert learner.last_draws[0]["draws"] == []


def test_binding_is_complete_and_rejects_tensor_and_metadata_mutations():
    b = {**batch(), "pixel_values": torch.randn(2, 3, 4, 4),
         "activation": torch.randn(2, 3, 8), "site_id": "enc_fn", "view": {"crop": [0, 1]}}
    binding = bind_batch_identity(b, expected_identity=batch_identity(b))
    assert binding.validate(b) == batch_identity(b)
    with pytest.raises(FrozenInstanceError):
        setattr(binding, "view_sha256", "changed")
    changed = {**b, "activation": b["activation"].clone()}
    with pytest.raises(ValueError, match="changed"):
        binding.validate(changed)
    alias = b["pixel_values"].detach()
    alias.add_(1)
    with pytest.raises(ValueError, match="changed"):
        binding.validate(b)
    fresh = bind_batch_identity(b)
    b["view"]["crop"].append(2)
    with pytest.raises(ValueError, match="changed"):
        fresh.validate(b)
    with pytest.raises(ValueError, match="mismatch"):
        bind_batch_identity(b, expected_identity="wrong")


def test_sparse_captures_final_and_rejects_invalid_tokens_before_attempt():
    learner = make_learner(toy_model(where="after_final_norm"), "sparse_first_final", final_step=3)
    b = batch()
    binding = bind_batch_identity(b)
    for capture in [True, False, True]:
        event = learner.update([b], batch_identities=[binding])
        assert event["payload"]["audit"]["noise_capture"] == capture
    learner = make_learner(toy_model(where="after_final_norm"))
    with pytest.raises(TypeError, match="bound identity"):
        learner.update([b], batch_identities=[binding.view_sha256])
    assert learner.attempted == 0
    with pytest.raises(ValueError, match="count"):
        learner.update([b], batch_identities=[])
    assert learner.attempted == 0
    with pytest.raises(ValueError, match="audit policy"):
        make_learner(toy_model(where="after_final_norm"), "silent")


def test_default_keeps_original_identity_hashing_and_full_capture(monkeypatch):
    learner = make_learner(toy_model(where="after_final_norm"))
    original_identity = baseline_protocol.batch_identity
    calls = []
    def count_identity(b):
        calls.append(b)
        return original_identity(b)
    monkeypatch.setattr(baseline_protocol, "batch_identity", count_identity)
    monkeypatch.setattr(baseline_protocol, "bind_batch_identity",
                        lambda *_: pytest.fail("default must not opt into prepared identity binding"))
    event = learner.update([batch()])
    assert len(calls) == 2
    assert learner.last_draws[0]["draws"]
    assert "audit" not in event["payload"]
