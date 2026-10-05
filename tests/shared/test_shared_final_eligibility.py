"""EVAL-1 focused mock joins; no public asset admission or model execution claim."""

import json
from types import SimpleNamespace
from pathlib import Path
import pytest
from jscc import shared_evaluation as adapter, panel_evaluation as engine


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def obligation(tmp_path, status="pending_prior_segment"):
    engine_path = tmp_path / "training"
    write(engine_path / "campaign.json", {})
    eligibility = {
        "case_id": "case",
        "scope": "initial",
        "checkpoint": {"engine_path": str(engine_path)},
        "observations_status_recorded": True,
        "observations_status": "complete",
        "observations": [{"step": 4, "state": "pre_reset", "status": status}],
    }
    state = SimpleNamespace(
        payload={
            "metadata": {
                "segment_ancestry": [],
                "recipe_plan": {"description": {"locations": ["enc_fn"]}},
            }
        }
    )
    return eligibility, state


@pytest.mark.parametrize("mode", ["pending_prior_segment", "missing"])
def test_serial_scoring_complete_remains_final_ineligible(tmp_path, monkeypatch, mode):
    eligibility, state = obligation(tmp_path, "complete" if mode == "missing" else mode)
    request = {"sites": ["enc_fn"], "conditions": ["no_noise"], "expected_items": 2}
    attempt = tmp_path / "evaluation"
    attempt.mkdir()
    monkeypatch.setattr(adapter, "validate_reserved_attempt", lambda *a: None)
    monkeypatch.setattr(adapter, "finish_attempt", lambda *a, **k: None)
    monkeypatch.setattr(engine, "_reopen_recipe", lambda s: None)
    monkeypatch.setattr(adapter, "_guard", lambda *a, **k: lambda: None)
    monkeypatch.setattr(adapter, "initialize_recipe", lambda *a, **k: (None, None))

    def score(*a, **k):
        write(
            k["output"] / "result.json",
            {
                "status": "complete",
                "panels": [{"status": "complete", "completed_items": 2}],
            },
        )

    monkeypatch.setattr(engine, "run_recipe_evaluation", score)
    monkeypatch.setattr(engine, "_verify_recipe_result", lambda *a, **k: None)
    adapter._case_evaluation(
        attempt,
        eligibility,
        request,
        state,
        {},
        None,
        {"synthetic_cpu": True},
        timeout_seconds=10,
        device_uuid=None,
    )
    result = json.loads((attempt / "result.json").read_text())
    assert result["status"] == "complete"
    assert result["panels"]["completed"] == 1
    assert result["final_eligible"] is False


@pytest.mark.parametrize("parallel", [None, {"devices": ["cpu"], "workers": 1}])
@pytest.mark.parametrize("final_eligible", [False, True])
def test_aggregate_gate_requires_each_case_verified_eligibility(
    tmp_path, monkeypatch, parallel, final_eligible
):
    root = tmp_path / "root"
    root.mkdir()
    case = root / "cases/case"
    case.mkdir(parents=True)
    monkeypatch.setattr(
        adapter,
        "_selection",
        lambda *a: (
            root,
            {"identity": "recipe"},
            [{"case_id": "case", "path": str(case)}],
        ),
    )
    monkeypatch.setattr(adapter, "_policies", lambda *a: ({}, {"case": {}}, None))
    monkeypatch.setattr(
        adapter,
        "_planned_counts",
        lambda *a: (
            True,
            adapter._counts(
                {"sites": ["enc_fn"], "conditions": ["no_noise"], "expected_items": 2}
            ),
        ),
    )
    monkeypatch.setattr(adapter, "source_state", lambda: {})
    monkeypatch.setattr(adapter, "runtime_identity", lambda: {})
    monkeypatch.setattr("jscc.panel_workers.validate_parallel_controls", lambda x: x)
    eligibility = {
        "case_id": "case",
        "scope": "initial",
        "checkpoint": {},
        "request_identity": "request",
    }
    request = {"sites": ["enc_fn"], "conditions": ["no_noise"], "expected_items": 2}
    monkeypatch.setattr(
        adapter,
        "recipe_evaluation_target",
        lambda *a, **k: (eligibility, request, None, None, None, None),
    )

    def reserve(*a, **k):
        target = case / "evaluations/evaluation-0001"
        target.mkdir(parents=True)
        return target

    monkeypatch.setattr(adapter, "reserve_attempt", reserve)

    def completed(attempt, *a, **k):
        write(
            attempt / "result.json",
            {
                "status": "complete",
                "final_eligible": final_eligible,
                "panels": {
                    "requested": 1,
                    "completed": 1,
                    "failed": 0,
                    "pending": 0,
                    "requested_items": 2,
                    "observed_items": 2,
                },
            },
        )

    monkeypatch.setattr(adapter, "_case_evaluation", completed)
    output = adapter.evaluate_recipe_run(root, selected="case", parallel=parallel)
    assert json.loads((output / "results.json").read_text())["status"] == "complete"
    assert (
        json.loads((output / "gate.json").read_text())["final_eligible"]
        is final_eligible
    )


@pytest.mark.parametrize("mode", ["pending_prior_segment", "missing"])
def test_parallel_supplies_same_observation_eligibility(tmp_path, monkeypatch, mode):
    eligibility, state = obligation(tmp_path, "complete" if mode == "missing" else mode)
    eligibility.update(source={}, runtime={})
    monkeypatch.setattr(engine.core, "validate_parallel_controls", lambda x: x)
    monkeypatch.setattr(engine, "worker_checkpoint_reference", lambda x: {})
    captured = {}

    class Captured(Exception):
        pass

    def plan(**kw):
        captured.update(kw)
        raise Captured

    monkeypatch.setattr(engine.core, "make_plan", plan)
    with pytest.raises(Captured):
        engine.evaluate_parallel_recipe(
            root=tmp_path,
            eligibility=eligibility,
            request={
                "sites": ["enc_fn"],
                "conditions": ["no_noise"],
                "input_ids": [1, 2],
                "noise": {},
            },
            state=state,
            output=tmp_path / "parallel",
            controls={},
            attempt_id="0001",
            checkpoint=None,
            expected_step=None,
            timeout_seconds=10,
        )
    assert captured["identities"]["training_observation_eligible"] is False


def test_corrupt_completed_observation_checkpoint_still_fails_trust(tmp_path):
    eligibility, state = obligation(tmp_path, "complete")
    checkpoint = tmp_path / "corrupt.pt"
    checkpoint.write_bytes(b"corrupt checkpoint")
    write(
        Path(eligibility["checkpoint"]["engine_path"])
        / "observation_000004_pre_reset/result.json",
        {
            "checkpoint": {
                "path": str(checkpoint),
                "size": len(checkpoint.read_bytes()),
                "sha256": "a" * 64,
            }
        },
    )
    with pytest.raises(ValueError):
        engine.verified_observations(eligibility, state)


def test_completed_observation_artifact_is_verified_before_acceptance(
    tmp_path, monkeypatch
):
    eligibility, state = obligation(tmp_path, "complete")
    receipt = {
        "checkpoint": {"path": "bound-checkpoint", "size": 1, "sha256": "a" * 64},
        "status": "complete",
    }
    write(
        Path(eligibility["checkpoint"]["engine_path"])
        / "observation_000004_pre_reset/result.json",
        receipt,
    )
    verified = []
    monkeypatch.setattr(engine, "_open_bound_reference", lambda ref: state)
    monkeypatch.setattr(
        engine,
        "_verify_recipe_result",
        lambda result, s, **k: verified.append((result, k)),
    )
    monkeypatch.setattr(engine, "_own_recipe_observation", lambda *a: True)
    assert engine.verified_observations(eligibility, state) is True
    assert verified == [(receipt, {"observation": True})]
