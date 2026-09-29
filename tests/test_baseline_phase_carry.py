import random

import numpy as np
import torch
from torch.amp.grad_scaler import GradScaler

from jscc.experiment_schedule import build_baseline_scheduler
from jscc.experiment_state import branch_metadata, open_state, restore_state, save_state


def test_real_cpu_200_to_201_of_400_full_state_carry(tmp_path):
    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)

    def components():
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        scheduler = build_baseline_scheduler(optimizer)
        scaler = GradScaler("cpu")
        return model, optimizer, scheduler, scaler

    model, optimizer, scheduler, scaler = components()

    def step(parts, n):
        model, optimizer, scheduler, scaler = parts
        optimizer.zero_grad()
        x = torch.randn(64, 2) * (random.random() + float(np.random.rand()))
        reconstruction = model(x).square().mean() * 0.1
        loss = reconstruction if n <= 200 else reconstruction + model(x).mean().square()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        lr = optimizer.param_groups[0]["lr"]
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        return loss.detach(), lr

    original = (model, optimizer, scheduler, scaler)
    for n in range(1, 201):
        step(original, n)
    meta = dict(
        source_identity="source",
        config_identity="config",
        parent_identity=None,
        initialization_identity="init",
        stream_identity="ordered-mask-view-noise",
        protocol_identity="baseline-v2",
        completed_updates=200,
        phase="reconstruction-prefix",
        lineage=[],
    )
    ref = save_state(
        tmp_path / "prefix.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        metadata=meta,
        stream_state={"completed_updates": 200, "offset": 12800},
    )
    boundary = open_state(ref, expected=meta)
    branch = branch_metadata(boundary, phase="staged")
    assert branch["parent_identity"] == ref.sha256
    assert branch["lineage"][-1]["completed_updates"] == 200
    assert branch["phase"] == "staged"
    assert boundary.payload["metadata"]["parent_identity"] is None
    expected = [step(original, n) for n in range(201, 401)]
    restored = components()
    stream = restore_state(
        open_state(ref, expected=meta),
        model=restored[0],
        optimizer=restored[1],
        scheduler=restored[2],
        scaler=restored[3],
    )
    assert stream["offset"] == 12800
    actual = [step(restored, n) for n in range(201, 401)]
    assert all(
        torch.equal(a[0], b[0]) and a[1] == b[1] for a, b in zip(expected, actual)
    )
    assert actual[0][1] > 0 and restored[1].param_groups[0]["lr"] == 0
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, restored[0].state_dict()[name])
    assert scheduler.state_dict() == restored[2].state_dict()
    assert scaler.state_dict() == restored[3].state_dict()
    for key, value in optimizer.state_dict()["state"].items():
        for name, tensor in value.items():
            assert torch.equal(tensor, restored[1].state_dict()["state"][key][name])


def test_real_local_prefix_online_branch_preserves_pairing_and_update(tmp_path):
    import copy
    import pytest
    from jscc.baseline_protocol import BaselineLearner, run_baseline, CONDITIONS
    from jscc.experiment_schedule import Segment
    from jscc.experiment_state import save_state, open_state
    from jscc.runtime import model_inputs
    from test_core import toy_model, batch

    model = toy_model(where="after_final_norm")
    model.codec.film = None
    b = {k: v.repeat(32, 1) for k, v in batch().items()}
    def make(model, pairing="shared-pair"):
        return BaselineLearner(model, run_id="carry-cpu", task="hellaswag", synthetic=True,
            identity={"source": "s", "config": "c", "data": "d", "parent": None}, pairing_id=pairing)
    continuous = make(model)
    kwargs, _ = model_inputs(b, model)
    captured = []
    with torch.no_grad(), model.transmission(bypass=True):
        handle = model.base.enc.norm.register_forward_hook(lambda _m, _a, o: captured.append(o.detach().clone()))
        model(**kwargs)
        handle.remove()
    local = {**b, "activation": captured[0], "site_id": "enc_fn"}
    for _ in range(200):
        continuous.update([local], kind="local")
    metadata = dict(source_identity="s", config_identity="c", parent_identity=None,
                    initialization_identity="i", stream_identity="stream", protocol_identity="p", lineage=[],
                    phase="reconstruction-prefix", completed_updates=200, noise_identity=continuous.noise_identity)
    reference = save_state(tmp_path / "parent.pt", model=model.codec, optimizer=continuous.optimizer,
        scheduler=continuous.scheduler, scaler=continuous.scaler, metadata=metadata,
        stream_state={"completed_updates": 200, "offset": 12800, "source_valid_tokens": continuous.valid_tokens})
    parent = open_state(reference, expected=metadata)
    def clone_model():
        clone = toy_model(where="after_final_norm")
        clone.codec.film = None
        clone.load_state_dict(model.state_dict())
        return clone
    mismatched = make(clone_model(), "changed-pair")
    segment = Segment("synthetic", "staged", 200, 201, (201,), (200, 201))
    with pytest.raises(ValueError, match="pairing/noise"):
        run_baseline(mismatched, output=tmp_path / "rejected", metadata=metadata,
                     update_batches=lambda *_: [b], validation_batches=lambda _: [b], assess=lambda *_: None,
                     segment=segment, parent=parent)
    assert not (tmp_path / "rejected").exists()
    branched = make(clone_model())
    expected = continuous.update([b], kind="combined")
    expected_weights = copy.deepcopy(model.codec.state_dict())
    events = []
    branched.event_sink = events.append
    run_baseline(branched, output=tmp_path / "branch", metadata=metadata,
        update_batches=lambda *_: [b], validation_batches=lambda _: [b],
        assess=lambda *_: {"status": "complete", "conditions": [{"condition": c} for c in CONDITIONS]},
        segment=segment, parent=parent)
    actual = next(event for event in events if event["event_type"] == "update")
    assert actual["payload"]["snr"] == expected["payload"]["snr"]
    assert actual["payload"]["lr_used"] == expected["payload"]["lr_used"]
    assert actual["payload"]["objective"] == expected["payload"]["objective"]
    for name, value in branched.model.codec.state_dict().items():
        torch.testing.assert_close(value, expected_weights[name], rtol=0, atol=0)
    for actual_state, expected_state in zip(branched.optimizer.state.values(), continuous.optimizer.state.values()):
        for key in actual_state:
            torch.testing.assert_close(actual_state[key], expected_state[key], rtol=0, atol=0)
    assert branched.offset == continuous.offset == 12864
