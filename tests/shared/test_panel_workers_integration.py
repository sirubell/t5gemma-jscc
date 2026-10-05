import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import pytest
from jscc import panel_workers as panels


ATOL = 1e-8
RTOL = 1e-6


def load_runtime(plan, device):
    return {"pid": os.getpid(), "device": device, "load_count": 1}


def full_panel_score(runtime, plan, panel, output):
    # Mock epsilon depends solely on the recorded science seed/namespace.
    assert runtime["load_count"] == 1
    seed = hashlib.sha256(json.dumps(plan["noise"], sort_keys=True).encode()).digest()
    rng = random.Random(int.from_bytes(seed[:8], "big"))
    epsilon = [rng.gauss(0, 1) for _ in plan["item_ids"]]
    if panel["condition"] == "no_noise":
        epsilon = [0.0 for _ in epsilon]
    time.sleep((3 - panel["index"]) * 0.1)
    return {"item_ids": plan["item_ids"],
            "scores": {"mock_task_score": sum(epsilon) + panel["site"]},
            "noise_digest": hashlib.sha256(json.dumps(epsilon).encode()).hexdigest(),
            "pid": runtime["pid"], "finished": time.monotonic_ns()}


def one_failure(runtime, plan, panel, output):
    if panel["index"] == 0:
        raise panels.PanelScoringError("mock task scoring failed")
    return full_panel_score(runtime, plan, panel, output)


def make_plan(tmp_path):
    checkpoint = tmp_path / "durable-checkpoint.bin"
    checkpoint.write_bytes(b"fixed mock durable checkpoint")
    return panels.make_plan(case_id="one-case", checkpoint=dict(panels.file_reference(checkpoint), role="terminal", step=16),
        scope="initial", sites=[0, 1], conditions=["no_noise", 0], item_ids=["a", "b", "c"],
        identities={"source": "accepted101R4", "runtime": "cpu-mock", "input": "full-corpus-three-items", "scorer": "full-panel-mock-v1",
                    "run": "/recorded/run/one-case", "training_attempt": "/recorded/run/one-case/attempts/training-1", "training_observation_eligible": True},
        noise={"seed": 42, "namespace": "paired-fixed-view"})


def execute(root, attempt_id, plan, workers, scorer=full_panel_score, parent=None):
    attempt = panels.reserve_attempt(root, attempt_id, plan, parent=parent)
    metadata = json.loads((attempt / "attempt.json").read_text())
    inherited = {r["panel_id"] for r in metadata["inherited"]}
    selected = [p["id"] for p in plan["panels"] if p["id"] not in inherited]
    before = panels.collect_results(attempt, plan)
    assert before["counts"]["selected"] == len(plan["panels"])
    states = panels.run_workers(plan, attempt, panels.assignments_for(plan, devices=["cpu"] * workers, workers=workers, panel_ids=selected),
        runtime_factory=load_runtime, score_panel=scorer, publish_partial=True)
    return attempt, states, panels.collect_results(attempt, plan, worker_states=states, final=True)


def test_real_processes_ordered_full_panels_and_paired_noise(tmp_path):
    plan = make_plan(tmp_path)
    serial, _, left = execute(tmp_path / "runs", "serial", plan, 1)
    parallel, states, right = execute(tmp_path / "runs", "parallel", plan, 2)
    assert len({s["pid"] for s in states.values()}) == 2
    assert [r["panel_id"] for r in left["rows"]] == [r["panel_id"] for r in right["rows"]]
    assert right["counts"]["completed"] == 4 and right["counts"]["started"] == 4
    assert right["observed_items"] == right["expected_items"] == 12
    assert right["final_eligible"] and not right["incomplete"]
    for a, b in zip(left["rows"], right["rows"], strict=True):
        assert a["receipt"]["item_ids"] == b["receipt"]["item_ids"] == plan["item_ids"]
        assert a["receipt"]["noise_digest"] == b["receipt"]["noise_digest"]
        assert math.isclose(a["receipt"]["scores"]["mock_task_score"], b["receipt"]["scores"]["mock_task_score"], abs_tol=ATOL, rel_tol=RTOL)
    # Paired draws stay identical at the two sites within each condition.
    assert right["rows"][0]["receipt"]["noise_digest"] == right["rows"][1]["receipt"]["noise_digest"]
    assert right["rows"][2]["receipt"]["noise_digest"] == right["rows"][3]["receipt"]["noise_digest"]
    values = [json.loads(Path(r["receipt"]["result"]["path"]).read_text()) for r in right["rows"]]
    assert values[1]["finished"] < values[0]["finished"]  # arrival order differs from plan order
    snapshots = [json.loads(p.read_text()) for p in sorted((parallel / "snapshots").glob("*.json"))]
    assert any(s["counts"]["completed"] > 0 and s["incomplete"] and s["counts"]["running"] > 0 for s in snapshots)
    assert panels.collect_results(parallel, plan, final=True) == right


def test_partial_failure_and_fresh_linked_retry_keeps_parent_bytes(tmp_path):
    plan = make_plan(tmp_path)
    first, states, partial = execute(tmp_path / "runs", "first", plan, 2, scorer=one_failure)
    assert partial["counts"]["completed"] == 3 and partial["counts"]["failed"] == 1
    assert partial["incomplete"] and not partial["final_eligible"]
    before = {str(p.relative_to(first)): p.read_bytes() for p in first.rglob("*.json")}
    retry, _, complete = execute(tmp_path / "runs", "retry", plan, 1, parent=first)
    assert complete["counts"]["completed"] == 4 and complete["retry_count"] == 1
    assert complete["final_eligible"] and not complete["incomplete"]
    assert len(list((retry / "workers").rglob("receipts/*.json"))) == 1
    assert before == {str(p.relative_to(first)): p.read_bytes() for p in first.rglob("*.json")}
    with pytest.raises(FileExistsError):
        panels.reserve_attempt(tmp_path / "runs", "retry", plan, parent=first)
