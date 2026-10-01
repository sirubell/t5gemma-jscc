"""Tiny sharing metadata fixtures; no pretrained weights or real activations."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from typing import Any

import pytest

from jscc.models.split_model import ResolvedSite
from jscc.sharing_state import (
    SITES, build_sharing_state, create_study_freeze, validate_sharing_state,
    verify_study_freeze,
)


def policy():
    return {name: {"site": asdict(ResolvedSite(
        "after_final_norm" if name == "enc_fn" else "after_layer",
        None if name == "enc_fn" else int(name[5:]), "tiny-revision", 26, 8,
        "encoder.norm" if name == "enc_fn" else f"encoder.layers.{name[5:]}")),
        "role": "heldout_after_freeze" if name == "enc_l14" else "trained"}
        for name in (*SITES, "enc_l14")}


def fixture_state(count=5, learner="shared"):
    sharing, stream = build_sharing_state(synthetic=True, learner=learner,
        ordered_view_identities=[f"view-{i}" for i in range(4)],
        completed_updates=count, source_valid_per_view=[3, 5, 7, 9],
        target_valid_per_view=[2, 4, 6, 8], padded_per_view=[4, 6, 8, 10], batch_size=2)
    sites = policy()
    if learner != "shared":
        name = learner.removeprefix("specialist_")
        sites = {name: sites[name]}
    metadata = {"completed_updates": count, "phase": "both", "sharing": sharing,
                "evaluation_sites": sites}
    if learner != "shared":
        metadata["site"] = next(iter(sites.values()))["site"]
    return {"schema": "experiment-state-v2",
            "kind": "shared_encoder_v2" if learner == "shared" else "single_site_v2",
            "metadata": metadata, "stream": stream, "scheduler": {"last_epoch": count}}


def test_rotation_offsets_and_exact_exposure():
    state = fixture_state()
    validate_sharing_state(state)
    sharing = state["metadata"]["sharing"]
    assert sharing["completed_per_site"] == {"enc_l9": 1, "enc_l19": 2, "enc_fn": 2}
    assert sharing["source_valid_per_site"] == {"enc_l9": 3, "enc_l19": 8, "enc_fn": 8}
    assert sharing["target_valid_per_site"] == {"enc_l9": 2, "enc_l19": 6, "enc_fn": 6}
    assert state["stream"]["next_site"] == "enc_l9"
    assert state["stream"]["offset"] == 10
    assert fixture_state(12)["stream"]["next_site"] is None


@pytest.mark.parametrize("learner", [f"specialist_{site}" for site in SITES])
def test_specialists_have_only_own_exposure_and_evaluation(learner):
    state = fixture_state(4, learner)
    validate_sharing_state(state)
    own = learner.removeprefix("specialist_")
    assert state["metadata"]["sharing"]["completed_per_site"][own] == 4
    assert sum(state["metadata"]["sharing"]["completed_per_site"].values()) == 4
    assert set(state["metadata"]["evaluation_sites"]) == {own}


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(kind="single_site_v2"),
    lambda p: p["metadata"].update(phase="reconstruction"),
    lambda p: p["metadata"]["sharing"].update(horizon=400),
    lambda p: p["metadata"]["sharing"].update(q=True),
    lambda p: p["metadata"]["sharing"].update(next_site="enc_fn"),
    lambda p: p["metadata"]["sharing"]["completed_per_site"].update(enc_l9=2),
    lambda p: p["metadata"]["sharing"]["source_valid_per_site"].update(enc_l9=99),
    lambda p: p["metadata"]["sharing"]["trained_sites"].reverse(),
    lambda p: p["stream"]["view_offsets"].update(enc_l9=2),
    lambda p: p["stream"].update(offset=64),
    lambda p: p["scheduler"].update(last_epoch=4),
    lambda p: p["metadata"]["evaluation_sites"]["enc_l14"].update(role="trained"),
    lambda p: p["metadata"]["evaluation_sites"]["enc_fn"]["site"].update(module_path="decoder.norm"),
])
def test_drift_rejected(mutation):
    state = fixture_state()
    mutation(state)
    with pytest.raises(ValueError):
        validate_sharing_state(state)


def test_repeated_declared_views_allowed():
    state = fixture_state()
    state["metadata"]["sharing"]["ordered_view_identities"] = ["same-view"] * 4
    validate_sharing_state(state)


def test_production_horizon_and_batch_size():
    sharing, stream = build_sharing_state(synthetic=False, learner="shared",
        ordered_view_identities=[str(i) for i in range(200)], completed_updates=600,
        source_valid_per_view=[100] * 200, target_valid_per_view=[90] * 200,
        padded_per_view=[128] * 200, batch_size=64)
    assert sharing["horizon"] == 600
    assert stream["offset"] == 38400
    assert sharing["source_valid_per_site"] == dict.fromkeys(SITES, 20000)


def test_freeze_does_not_accept_arbitrary_claims(tmp_path):
    with pytest.raises(ValueError, match="four learners"):
        create_study_freeze(tmp_path / "freeze.json", protocol_identity="p", recipe_identity="r",
            architecture_decision="owner-decision", site_policy=policy(), trained_runs={},
            obligation_plan={"task": ["t"], "objective": ["o"], "geometry": ["g"]}, synthetic=True)
    assert not (tmp_path / "freeze.json").exists()


def test_freeze_bytes_verified_before_metadata(tmp_path):
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps({"status": "complete"}))
    reference = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path.write_text("tampered")
    with pytest.raises(ValueError, match="bytes changed"):
        verify_study_freeze(reference)


def test_builder_does_not_mutate_inputs():
    state = fixture_state()
    original = deepcopy(state)
    validate_sharing_state(state)
    assert state == original


def test_freeze_rechecks_inventory_bytes(tmp_path):
    from jscc.experiment_records import artifact_ref
    from jscc.sharing_state import LEARNERS
    root = tmp_path / "shared"
    root.mkdir()
    tensor = root / "state.pt"
    tensor.write_bytes(b"tiny-checkpoint-bytes")
    inventory = {"schema": "experiment-records-v1", "mode": "tensor-complete",
                 "artifacts": [artifact_ref(tensor, root, "tensor")], "omitted_tensors": []}
    paths = {"inventory": inventory, "manifest": {}, "events": None}
    refs = {}
    for key, value in paths.items():
        path = root / f"{key}.json"
        path.write_text("" if value is None else json.dumps(value))
        refs[key] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    run = {"root": str(root), **refs}
    tensor.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash/size mismatch"):
        create_study_freeze(tmp_path / "freeze.json", protocol_identity="p", recipe_identity="r",
            architecture_decision="decision", site_policy=policy(),
            trained_runs={learner: run for learner in LEARNERS},
            obligation_plan={"task": ["t"], "objective": ["o"], "geometry": ["g"]}, synthetic=True)


def test_synthetic_multiple_quarters():
    sharing, _ = build_sharing_state(synthetic=True, learner="shared",
        ordered_view_identities=["repeated"] * 8, completed_updates=24,
        source_valid_per_view=[3] * 8, target_valid_per_view=[2] * 8,
        padded_per_view=[4] * 8, batch_size=2)
    assert sharing["q"] == 8
    assert sharing["horizon"] == 24


def test_freeze_publication_is_immutable(tmp_path, monkeypatch):
    # Isolate publication from evidence validation (covered by corrupt inventory
    # and incomplete-run tests); the campaign integration exercises real runs.
    import jscc.sharing_state as module
    calls = []
    def verified_runs(runs, **bindings):
        calls.append((runs, bindings))
        return ["a" * 64]
    monkeypatch.setattr(module, "_verified_runs", verified_runs)
    kwargs: dict[str, Any] = dict(protocol_identity="protocol", recipe_identity="recipe",
        architecture_decision="approved-architecture", site_policy=policy(), trained_runs={"verified": True},
        obligation_plan={"task": ["task-panel"], "objective": ["objective-panel"], "geometry": ["atlas"]},
        synthetic=True)
    path = tmp_path / "freeze.json"
    reference = create_study_freeze(path, **kwargs)
    before = path.read_bytes()
    receipt = verify_study_freeze(reference, checkpoint_sha256="a" * 64,
                                 obligation="atlas", site_policy=policy(), protocol_identity="protocol")
    assert receipt["recipe_identity"] == "recipe"
    assert len(calls) == 2
    assert calls[-1][1]["architecture_decision"] == "approved-architecture"
    with pytest.raises(FileExistsError):
        create_study_freeze(path, **kwargs)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".freeze-*"))
    with pytest.raises(ValueError, match="absent from freeze"):
        verify_study_freeze(reference, checkpoint_sha256="b" * 64)
    with pytest.raises(ValueError, match="undeclared"):
        verify_study_freeze(reference, obligation="unplanned-atlas")


def state_components(horizon):
    import torch
    from jscc.sharing_schedule import sharing_lr_factor
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda n: sharing_lr_factor(n, horizon))
    return model, optimizer, scheduler


def advance_components(components, steps=1):
    import torch
    model, optimizer, scheduler = components
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        model(torch.tensor([[1., -2.]])).square().sum().backward()
        optimizer.step()
        scheduler.step()


def complete_metadata(state):
    return {"source_identity": "source", "config_identity": "config", "parent_identity": None,
            "initialization_identity": "init", "stream_identity": "stream",
            "protocol_identity": "protocol", "lineage": [], **state["metadata"]}


@pytest.mark.parametrize("learner,count", [("shared", 0), ("shared", 5), ("specialist_enc_l9", 2),
                                           ("specialist_enc_l19", 2), ("specialist_enc_fn", 2)])
def test_actual_sharing_save_open_restore_and_continue(tmp_path, learner, count):
    import torch
    from jscc.experiment_state import open_state, restore_state, save_state
    fixture = fixture_state(count, learner)
    metadata = complete_metadata(fixture)
    horizon = fixture["metadata"]["sharing"]["horizon"]
    components = state_components(horizon)
    advance_components(components, count)
    model, optimizer, scheduler = components
    reference = save_state(tmp_path / "state.pt", model=model, optimizer=optimizer,
        scheduler=scheduler, scaler=None, metadata=metadata, stream_state=fixture["stream"])
    validated = open_state(reference, expected=metadata)
    assert validated.payload["kind"] == fixture["kind"]
    other = state_components(horizon)
    offsets = restore_state(validated, model=other[0], optimizer=other[1], scheduler=other[2], scaler=None)
    assert offsets == fixture["stream"]
    assert other[2].state_dict() == scheduler.state_dict()
    advance_components(components)
    advance_components(other)
    assert all(torch.equal(value, other[0].state_dict()[key]) for key, value in model.state_dict().items())
    assert optimizer.param_groups[0]["lr"] == other[1].param_groups[0]["lr"]


def test_actual_sharing_reuses_rng_and_optimizer_validation(tmp_path):
    import torch
    from jscc.experiment_state import CheckpointRef, open_state, save_state
    fixture = fixture_state(1)
    metadata = complete_metadata(fixture)
    model, optimizer, scheduler = components = state_components(12)
    advance_components(components)
    reference = save_state(tmp_path / "valid.pt", model=model, optimizer=optimizer,
        scheduler=scheduler, scaler=None, metadata=metadata, stream_state=fixture["stream"])
    original = torch.load(reference.path, weights_only=False)
    for label, mutation in (
        ("rng", lambda p: p["rng"].pop("numpy")),
        ("optimizer", lambda p: p["optimizer"]["state"].clear()),
        ("lineage", lambda p: p["metadata"].update(lineage=None)),
        ("kind", lambda p: p.update(kind="single_site_v2")),
    ):
        payload = deepcopy(original)
        mutation(payload)
        path = tmp_path / f"{label}.pt"
        torch.save(payload, path)
        data = path.read_bytes()
        corrupt = CheckpointRef(str(path), hashlib.sha256(data).hexdigest(), len(data))
        with pytest.raises(ValueError, match="invalid checkpoint"):
            open_state(corrupt, expected=metadata)


def test_ordinary_state_keeps_400_bound_and_64_offset(tmp_path):
    from jscc.experiment_state import open_state, save_state
    model, optimizer, scheduler = components = state_components(404)
    advance_components(components, 400)
    metadata = complete_metadata(fixture_state(0))
    del metadata["sharing"]
    del metadata["evaluation_sites"]
    metadata["completed_updates"] = 400
    metadata["phase"] = "reconstruction"
    reference = save_state(tmp_path / "ordinary.pt", model=model, optimizer=optimizer,
        scheduler=scheduler, scaler=None, metadata=metadata,
        stream_state={"completed_updates": 400, "offset": 400 * 64})
    assert open_state(reference, expected=metadata).payload["kind"] == "single_site_v2"
    with pytest.raises(ValueError, match="stream/update"):
        save_state(tmp_path / "offset.pt", model=model, optimizer=optimizer,
            scheduler=scheduler, scaler=None, metadata=metadata,
            stream_state={"completed_updates": 400, "offset": 400 * 2})
    metadata["completed_updates"] = 401
    with pytest.raises(ValueError, match="completed update"):
        save_state(tmp_path / "over-limit.pt", model=model, optimizer=optimizer,
            scheduler=scheduler, scaler=None, metadata=metadata,
            stream_state={"completed_updates": 401, "offset": 401 * 64})


def test_shared_600_horizon_does_not_relax_ordinary_bound(tmp_path):
    from jscc.experiment_state import open_state, save_state
    # Production-sized counters on tiny CPU tensors test schema capacity only.
    sharing, stream = build_sharing_state(synthetic=False, learner="shared",
        ordered_view_identities=["view"] * 200, completed_updates=600,
        source_valid_per_view=[100] * 200, target_valid_per_view=[90] * 200,
        padded_per_view=[128] * 200, batch_size=64)
    metadata = complete_metadata(fixture_state(0))
    metadata.update(completed_updates=600, sharing=sharing)
    model, optimizer, scheduler = components = state_components(600)
    advance_components(components, 600)
    reference = save_state(tmp_path / "shared600.pt", model=model, optimizer=optimizer,
        scheduler=scheduler, scaler=None, metadata=metadata, stream_state=stream)
    assert open_state(reference, expected=metadata).payload["metadata"]["completed_updates"] == 600
