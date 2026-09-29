from dataclasses import replace

import pytest
import torch

from jscc.experiment_schedule import build_baseline_scheduler
from jscc.experiment_state import open_state, restore_state, save_state


def components():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scheduler = build_baseline_scheduler(optimizer)
    return model, optimizer, scheduler


def metadata(n=0):
    return dict(
        source_identity="source",
        config_identity="config",
        parent_identity=None,
        initialization_identity="init",
        stream_identity="stream",
        protocol_identity="protocol",
        completed_updates=n,
        phase="reconstruction",
        lineage=[],
    )


def save(path, model, optimizer, scheduler, n=0):
    return save_state(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        metadata=metadata(n),
        stream_state=dict(completed_updates=n, offset=n * 64),
    )


def test_identity_and_parameter_map_rejections_before_mutation(tmp_path):
    model, optimizer, scheduler = components()
    ref = save(tmp_path / "state.pt", model, optimizer, scheduler)
    for field in (
        "source_identity",
        "config_identity",
        "parent_identity",
        "stream_identity",
        "initialization_identity",
    ):
        request = metadata()
        request[field] = "different"
        with pytest.raises(ValueError, match="compatibility"):
            open_state(ref, expected=request)
    validated = open_state(ref, expected=metadata())
    other = torch.nn.Linear(2, 2)
    reversed_optimizer = torch.optim.AdamW(
        list(reversed(list(other.parameters()))), lr=2e-4
    )
    before = other.weight.clone()
    with pytest.raises(ValueError, match="parameter-map"):
        restore_state(
            validated,
            model=other,
            optimizer=reversed_optimizer,
            scheduler=build_baseline_scheduler(reversed_optimizer),
            scaler=None,
        )
    assert torch.equal(before, other.weight)


def test_immutable_publication_and_corrupt_partial_save(tmp_path, monkeypatch):
    import jscc.experiment_state as state

    model, optimizer, scheduler = components()
    path = tmp_path / "state.pt"
    ref = save(path, model, optimizer, scheduler)
    with pytest.raises(FileExistsError):
        save(path, model, optimizer, scheduler)
    assert open_state(ref, expected=metadata())
    path.write_bytes(path.read_bytes()[:30])
    with pytest.raises(ValueError, match="corrupt"):
        open_state(ref, expected=metadata())
    with pytest.raises(ValueError, match="corrupt"):
        open_state(replace(ref, size=30), expected=metadata())

    def fail(*args):
        raise OSError("injected partial save")

    monkeypatch.setattr(state.os, "fsync", fail)
    with pytest.raises(OSError):
        save(tmp_path / "failed.pt", model, optimizer, scheduler)
    assert not (tmp_path / "failed.pt").exists()
    assert not list(tmp_path.glob(".incomplete-*"))


def test_inconsistent_stream_and_missing_state_rejected(tmp_path):
    model, optimizer, scheduler = components()
    with pytest.raises(ValueError, match="stream"):
        save_state(
            tmp_path / "bad.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            metadata=metadata(),
            stream_state={"offset": 64, "completed_updates": 0},
        )


def test_hash_correct_but_missing_required_state_rejected(tmp_path):
    import hashlib
    from jscc.experiment_state import CheckpointRef

    model, optimizer, scheduler = components()
    ref = save(tmp_path / "state.pt", model, optimizer, scheduler)
    payload = torch.load(ref.path, weights_only=False)
    del payload["rng"]["numpy"]
    invalid = tmp_path / "invalid.pt"
    torch.save(payload, invalid)
    data = invalid.read_bytes()
    ref = CheckpointRef(str(invalid), hashlib.sha256(data).hexdigest(), len(data))
    with pytest.raises(ValueError, match="invalid checkpoint"):
        open_state(ref, expected=metadata())
