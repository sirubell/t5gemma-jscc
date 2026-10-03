"""Actual tiny CPU baseline arms feed immutable explicit report contrasts."""
import copy
import csv
import json
import shutil

import pytest

from jscc.baseline_protocol import run_baseline, write_baseline_comparisons
from jscc.experiment_records import artifact_ref, observation_identity, read_events
from jscc.experiment_schedule import Segment
from scripts.render_experiment_report import _explicit_comparisons, render_local_report
from test_baseline_pipeline import setup_pipeline


@pytest.fixture(scope="module")
def actual_campaign(tmp_path_factory):
    campaign = tmp_path_factory.mktemp("actual-baseline-comparisons")
    with pytest.MonkeyPatch.context() as monkeypatch:
        for cell in ("D-none", "D-LN", "R-none", "R-LN"):
            learner, metadata, request, batch, assess = setup_pipeline(
                monkeypatch, cell=cell, run_id=cell)
            result = run_baseline(
                learner, output=campaign / cell, metadata=metadata,
                update_batches=lambda *_: [batch], validation_batches=lambda _: [batch],
                assess=assess, task_request=request,
                segment=Segment(cell, "both", 0, 4, (1, 2, 3, 4), (0, 2, 4)))
            assert result["status"] == "complete" and result["tensor_bytes_verified"]
    return campaign


def test_actual_four_arm_campaign_renders_all_explicit_contrasts(actual_campaign, tmp_path):
    manifest = write_baseline_comparisons(actual_campaign, steps=(0, 2, 4))
    assert len(manifest["contrasts"]) == 12
    assert json.loads((actual_campaign / "comparisons.json").read_text()) == manifest
    output = tmp_path / "report"
    render_local_report(actual_campaign, output)
    with (output / "paired-gaps.csv").open() as stream:
        rows = [row for row in csv.DictReader(stream) if row["analysis"] == "explicit-left-minus-right"]
    assert len(rows) == 12 * 6 * 2
    assert {row["contrast"] for row in rows} == {row["id"] for row in manifest["contrasts"]}
    with (output / "blocked-comparisons.csv").open() as stream:
        blocked = list(csv.DictReader(stream))
    assert not [row for row in blocked if row.get("contrast") in {r["id"] for r in manifest["contrasts"]}]
    assert (output / "index.html").is_file()


@pytest.mark.parametrize("problem", ["missing", "corrupted", "incomplete"])
def test_actual_campaign_rejects_unusable_arms(actual_campaign, tmp_path, problem):
    campaign = tmp_path / "campaign"
    shutil.copytree(actual_campaign, campaign)
    (campaign / "comparisons.json").unlink(missing_ok=True)
    if problem == "missing":
        shutil.rmtree(campaign / "D-LN")
    elif problem == "corrupted":
        (campaign / "D-LN" / "step_000004.pt").write_bytes(b"corrupted checkpoint")
    else:
        # Keep an actual valid event log but remove required terminal evidence.
        path = campaign / "D-LN" / "metrics.jsonl"
        events = read_events(path)
        events = [event for event in events if not (
            event["event_type"] == "observation" and event["payload"]["request"]["step"] == 4)]
        path.write_text("".join(json.dumps(event) + "\n" for event in events))
        inventory_path = path.parent / "inventory.json"
        inventory = json.loads(inventory_path.read_text())
        for artifact in inventory["artifacts"]:
            if artifact["path"] == "metrics.jsonl":
                artifact.update(artifact_ref(path, path.parent, artifact["kind"]))
        inventory_path.write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="tensor-complete" if problem == "incomplete" else None):
        write_baseline_comparisons(campaign, steps=(0, 2, 4))
    assert not (campaign / "comparisons.json").exists()


@pytest.mark.parametrize("field,value", [("noise", {"seed": 999, "namespace": "tiny-task"}),
                                         ("settings", {"num_samples": 2, "batch_size": 99})])
def test_actual_inline_requests_reject_changed_task_controls(actual_campaign, field, value):
    manifest = write_baseline_comparisons(actual_campaign, steps=(0, 2, 4))
    events = [event for cell in ("D-none", "D-LN", "R-none", "R-LN")
              for event in read_events(actual_campaign / cell / "metrics.jsonl")]
    contrast = copy.deepcopy(manifest["contrasts"][0])
    target = next(event["payload"] for event in events
                  if event["event_type"] == "observation" and event["payload"]["identity"] == contrast["right_observation"])
    request = target["request"]
    prefix = "inline-execution-request-v1:"
    assert request["details"]["outputs"].startswith(prefix)
    execution = json.loads(request["details"]["outputs"].removeprefix(prefix))
    execution[field] = value
    request["details"]["outputs"] = prefix + json.dumps(execution)
    target["identity"] = observation_identity(request)
    contrast["right_observation"] = target["identity"]
    rows, blocked = _explicit_comparisons(events, {"schema": "experiment-comparisons-v1", "contrasts": [contrast]})
    assert not rows and len(blocked) == 1
    assert "input/scoring/noise/site/exposure identity mismatch" in blocked[0]["reason"]
