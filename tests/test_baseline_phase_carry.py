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
