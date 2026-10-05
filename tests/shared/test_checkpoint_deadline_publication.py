"""Tiny CPU save/publication ordering; no public admission or GPU execution."""
from dataclasses import asdict
import json

import pytest

from jscc.activation_replay import canonical_digest
from jscc.experiment_records import write_manifest
from jscc.experiment_state import CheckpointRef, open_state
from jscc.qualification import state_exact
from jscc.shared_state import RecipeTrainingState
from jscc.shared_training import run_recipe_training
from shared.test_recipe_resume import setup


def invoke(state, batch, output, **kwargs):
    state.learner.model.memory_codec = None
    state.learner.model.universal_codec = False
    return run_recipe_training(
        state.learner.model, None, plan=state.plan, batches=[batch] * state.plan.q,
        metadata=state.metadata, output=output, synthetic=True,
        data_ids={"rows": "tiny"}, **kwargs,
    )


@pytest.mark.parametrize("role", ["initialization", "pre_reset", "terminal"])
def test_deadline_during_durable_save_publishes_recoverable_state_before_failure(
    tmp_path, monkeypatch, role
):
    state, batch, _ = setup(tmp_path / "setup")
    expired = [False]
    published = []
    published_states = []
    observations = []
    real_save = RecipeTrainingState.save

    def crossing_save(owner, path):
        saved = real_save(owner, path)
        if str(path).endswith(f"_{role}.pt"):
            expired[0] = True
        return saved

    monkeypatch.setattr(RecipeTrainingState, "save", crossing_save)

    def guard():
        if expired[0]:
            raise TimeoutError("deadline crossed during durable save")

    def publish(saved, saved_role):
        # Campaign publication must precede the durable inventory callback.
        campaign = json.loads((tmp_path / "run/campaign.json").read_text())
        assert campaign["checkpoints"][-1] == {"role": saved_role, **asdict(saved.reference)}
        assert campaign["completed_updates"] == saved.payload["metadata"]["completed_updates"]
        open_state(saved.reference, expected=saved.payload["metadata"])
        published.append({"role": saved_role, **asdict(saved.reference)})
        published_states.append(saved)
        write_manifest(tmp_path / "inventory.json", {"checkpoints": published})

    def observe(model, processor, **kwargs):
        assert not expired[0], "no scoring after deadline"
        observations.append(kwargs["request"])
        return {"status": "complete", "request_identity": canonical_digest(kwargs["request"])}

    with pytest.raises(TimeoutError, match="during durable save"):
        invoke(state, batch, tmp_path / "run", guard=guard, on_checkpoint=publish, observe=observe)
    inventory = json.loads((tmp_path / "inventory.json").read_text())["checkpoints"]
    latest = inventory[-1]
    assert latest["role"] == role
    reference = CheckpointRef(**{key: latest[key] for key in ("path", "sha256", "size")})
    saved = open_state(reference, expected=published_states[-1].payload["metadata"])
    campaign = json.loads((tmp_path / "run/campaign.json").read_text())
    assert campaign["status"] == "incomplete"
    assert campaign["error"]["type"] == "TimeoutError"
    assert campaign["completed_updates"] == saved.payload["metadata"]["completed_updates"]
    assert not (tmp_path / "run/freeze.json").exists()
    events = [json.loads(line) for line in (tmp_path / "run/metrics.jsonl").read_text().splitlines()]
    assert len([event for event in events if event["event_type"] == "update"]) == campaign["completed_updates"]
    assert all(request["checkpoint_sha256"] != reference.sha256 for request in observations)
    recovered, _, _ = setup(tmp_path / "recovered")
    recovered.restore(reference)
    restored = recovered.save(tmp_path / "restored.pt")
    for key in ("model", "optimizer", "scheduler", "scaler", "rng", "stream"):
        assert state_exact(restored.payload[key], saved.payload[key]), key

    if role == "terminal":
        # Fresh explicit finalization restores the terminal state and emits zero updates.
        finalized, final_batch, _ = setup(tmp_path / "finalized")
        result = invoke(finalized, final_batch, tmp_path / "finalized-run",
                        guard=lambda: None, observe=observe, resume=reference, finalize_terminal=True)
        assert result["status"] == "training_frozen" and result["segment_updates"] == 0
        assert result["completed_updates"] == state.plan.horizon
        terminal = result["checkpoints"][-1]
        final_saved = finalized._open(CheckpointRef(**{key: terminal[key] for key in ("path", "sha256", "size")}))
        for key in ("model", "optimizer", "scheduler", "scaler", "rng", "stream"):
            assert state_exact(final_saved.payload[key], saved.payload[key]), key
        assert (tmp_path / "finalized-run/metrics.jsonl").read_bytes() == b""


def test_inventory_publication_failure_remains_explicit(tmp_path):
    state, batch, _ = setup(tmp_path / "setup")
    callbacks = []

    def fail(saved, role):
        callbacks.append(role)
        raise OSError("inventory publication failed")

    with pytest.raises(OSError, match="inventory publication failed"):
        invoke(state, batch, tmp_path / "run", guard=lambda: None, on_checkpoint=fail,
               observe=lambda *args, **kwargs: pytest.fail("no scoring after publication failure"))
    campaign = json.loads((tmp_path / "run/campaign.json").read_text())
    assert callbacks == ["initialization"]
    assert campaign["status"] == "incomplete" and campaign["completed_updates"] == 0
    assert campaign["error"]["type"] == "OSError"
    assert len(campaign["checkpoints"]) == 1
    assert not (tmp_path / "run/freeze.json").exists()
    assert (tmp_path / "run/metrics.jsonl").read_bytes() == b""


@pytest.mark.parametrize("when", ["before_save", "after_save_without_callback"])
def test_deadline_guards_remain_before_save_and_after_optional_publication(tmp_path, monkeypatch, when):
    state, batch, _ = setup(tmp_path / "setup")
    expired = [when == "before_save"]
    real_save = RecipeTrainingState.save
    saves = []

    def crossing_save(owner, path):
        saved = real_save(owner, path)
        saves.append(saved)
        expired[0] = True
        return saved

    monkeypatch.setattr(RecipeTrainingState, "save", crossing_save)

    def guard():
        if expired[0]:
            raise TimeoutError("deadline")

    with pytest.raises(TimeoutError, match="deadline"):
        invoke(state, batch, tmp_path / "run", guard=guard,
               observe=lambda *args, **kwargs: pytest.fail("no scoring after deadline"))
    campaign = json.loads((tmp_path / "run/campaign.json").read_text())
    assert campaign["completed_updates"] == 0 and campaign["status"] == "incomplete"
    assert not (tmp_path / "run/freeze.json").exists()
    assert (tmp_path / "run/metrics.jsonl").read_bytes() == b""
    if when == "before_save":
        assert saves == [] and campaign["checkpoints"] == []
        assert list((tmp_path / "run").glob("*.pt")) == []
    else:
        assert len(saves) == 1
        assert campaign["checkpoints"] == [{"role": "initialization", **asdict(saves[0].reference)}]
