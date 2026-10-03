"""Execute accepted cadence on the connected tiny real CPU module pipeline."""

from pathlib import Path

import pytest
import torch

from jscc.baseline_protocol import comparison_refs, run_baseline
from jscc.experiment_records import read_events
from jscc.experiment_schedule import Segment, baseline_lr_factor
from jscc.experiment_state import CheckpointRef, open_state
from jscc.runtime import model_inputs
from test_baseline_pipeline import setup_pipeline


def updates_at(output):
    return [
        event["payload"]
        for event in read_events(output / "metrics.jsonl")
        if event["event_type"] == "update"
    ]


def assert_cadence(result, output, *, start, stop, checkpoints, objectives, tasks):
    assert result["status"] == "complete" and result["tensor_bytes_verified"]
    assert [Path(ref["path"]).name for ref in result["checkpoints"]] == [
        f"step_{step:06d}.pt" for step in checkpoints
    ]
    assert [row["step"] for row in result["objective_observations"]] == objectives
    assert [row["step"] for row in result["task_assessments"]] == tasks
    observations = [
        event
        for event in read_events(output / "metrics.jsonl")
        if event["event_type"] == "observation"
    ]
    assert len(observations) == len(objectives) + len(tasks)
    assert all(len(event["payload"]["conditions"]) == 6 for event in observations)
    updates = updates_at(output)
    assert [row["completed_step"] for row in updates] == list(
        range(start + 1, stop + 1)
    )
    assert all(row["exposure"]["sequences"] == 64 for row in updates)
    assert sum(row["exposure"]["sequences"] for row in updates) == (stop - start) * 64
    for step, row in zip(range(start + 1, stop + 1), updates):
        assert row["lr_used"] == pytest.approx([2e-4 * baseline_lr_factor(step - 1)])
        assert row["lr_next"] == pytest.approx([2e-4 * baseline_lr_factor(step)])
        sampled = step == start + 1 or step % 10 == 0 or step == stop
        assert (row["update_l2"]["value"] is not None) == sampled
        if not sampled:
            assert row["update_l2"]["reason"] == "sampled_first_every10_final"
    return updates


def test_full_400_and_shared_200_prefix_both_continuations(monkeypatch, tmp_path):
    learner, metadata, request, batch, assess = setup_pipeline(
        monkeypatch, final_step=400
    )
    output = tmp_path / "combined"
    result = run_baseline(
        learner,
        output=output,
        metadata=metadata,
        update_batches=lambda *_: [batch],
        validation_batches=lambda _: [batch],
        assess=assess,
        task_request=request,
    )
    updates = assert_cadence(
        result,
        output,
        start=0,
        stop=400,
        checkpoints=[0, 100, 200, 300, 400],
        objectives=[0, 100, 200, 300, 400],
        tasks=[0, 200, 400],
    )
    assert updates[0]["update_l2"]["value"] == 0
    assert updates[-1]["lr_next"] == [0]
    initial_ref = CheckpointRef(**result["checkpoints"][0])
    initial = open_state(initial_ref, expected={**metadata, "completed_updates": 0})
    initial_reuse = {
        "state": initial,
        "receipt": result["task_assessments"][0]["receipt"],
        "root": output / "evaluations" / "step_000000",
    }

    prefix, prefix_meta, prefix_request, prefix_batch, prefix_assess = setup_pipeline(
        monkeypatch, final_step=400, run_id="cpu-prefix"
    )
    prefix_request["comparison"] = comparison_refs(prefix, prefix_meta, "local")
    kwargs, _ = model_inputs(prefix_batch, prefix.model)
    captured = []
    with torch.no_grad(), prefix.model.transmission(bypass=True):
        handle = prefix.model.base.enc.norm.register_forward_hook(
            lambda _module, _args, value: captured.append(value.detach().clone())
        )
        try:
            prefix.model(**kwargs)
        finally:
            handle.remove()
    local_batch = {**prefix_batch, "activation": captured[0], "site_id": "enc_fn"}
    prefix_output = tmp_path / "prefix"
    prefix_result = run_baseline(
        prefix,
        output=prefix_output,
        metadata=prefix_meta,
        update_batches=lambda *_: [local_batch],
        validation_batches=lambda _: [local_batch],
        assess=prefix_assess,
        task_request=prefix_request,
        segment=Segment(
            "R-none", "reconstruction-prefix", 0, 200, (50, 100, 150, 200), (100, 200)
        ),
        reused_assessments={0: initial_reuse},
    )
    prefix_updates = assert_cadence(
        prefix_result,
        prefix_output,
        start=0,
        stop=200,
        checkpoints=[0, 50, 100, 150, 200],
        objectives=[0, 50, 100, 150, 200],
        tasks=[0, 100, 200],
    )
    assert all(
        row["objective"]["components"]["K"]["raw"]["value"] is None
        for row in prefix_updates
    )
    parent = open_state(
        CheckpointRef(**prefix_result["checkpoints"][-1]),
        expected={**prefix_meta, "completed_updates": 200},
    )
    parent_reuse = {
        "state": parent,
        "receipt": prefix_result["task_assessments"][-1]["receipt"],
        "root": prefix_output / "evaluations" / "step_000200",
    }
    for strategy, kind, points, assessments in (
        ("reconstruction", "local", (300, 400), (400,)),
        ("staged", "combined", (250, 300, 350, 400), (300, 400)),
    ):
        branch, branch_meta, branch_request, branch_batch, branch_assess = (
            setup_pipeline(monkeypatch, final_step=400, run_id="cpu-" + strategy)
        )
        branch_request["comparison"] = comparison_refs(branch, branch_meta, kind)
        training = local_batch if kind == "local" else branch_batch
        branch_output = tmp_path / strategy
        branch_result = run_baseline(
            branch,
            output=branch_output,
            metadata=branch_meta,
            update_batches=lambda *_: [training],
            validation_batches=lambda _: [training],
            assess=branch_assess,
            task_request=branch_request,
            parent=parent,
            segment=Segment("R-none", strategy, 200, 400, points, assessments),
            reused_assessments={200: parent_reuse},
        )
        branch_updates = assert_cadence(
            branch_result,
            branch_output,
            start=200,
            stop=400,
            checkpoints=[200, *points],
            objectives=[200, *points],
            tasks=[200, *assessments],
        )
        assert all(
            (row["objective"]["components"]["K"]["raw"]["value"] is None)
            == (kind == "local")
            for row in branch_updates
        )
        assert branch_updates[0]["lr_used"] == pytest.approx(
            [2e-4 * baseline_lr_factor(200)]
        )
        assert branch.offset == 400 * 64
        assert all(state["step"] == 400 for state in branch.optimizer.state.values())
        assert branch_result["task_assessments"][0]["reuse"] == parent.reference.sha256
