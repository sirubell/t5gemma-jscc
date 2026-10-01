import csv
import importlib.util
from pathlib import Path

import pytest

from jscc.experiment_records import append_event, observation_identity
from jscc.sharing_report import write_sharing_report
from jscc.sharing_schedule import BatchView, SharingPlan, TRAINED_SITES

_fixture_path = Path(__file__).parent / "fixtures/experiment_records/build_fixture.py"
_spec = importlib.util.spec_from_file_location("sharing_report_fixture", _fixture_path)
assert _spec is not None and _spec.loader is not None
fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixture)


def plan():
    view = BatchView("a" * 64, "b" * 64, "c" * 64, 4, 32, 16, 40)
    return SharingPlan((view,) * 4, "synthetic-sharing", synthetic=True)


def observations():
    result = []
    for site in (*TRAINED_SITES, "enc_l14"):
        for kind in ("shared", "specialist") if site != "enc_l14" else ("shared",):
            step = 12 if kind == "shared" else 4
            row = fixture.observation(kind, "combined", site, kind, step)["payload"]
            row["request"]["site_step"] = 4
            row["request"]["comparison"]["schedule"] = kind + "-horizon"
            row["request"]["comparison"]["exposure"] = kind + "-exposure"
            row["request"]["details"]["outputs"] = (
                kind + "-different-checkpoint-outputs"
            )
            row["identity"] = observation_identity(row["request"])
            result.append(row)
    initial = fixture.observation(
        "initialization", "combined", "enc_l14", "initialization", 0
    )["payload"]
    result.append(initial)
    return result


def test_paired_report_keeps_intended_intervention_and_no_equivalence(tmp_path):
    result = write_sharing_report(tmp_path, observations(), plan())
    assert len(result["terminal_contrasts"]) == 36
    contrast = result["terminal_contrasts"][0]
    interval = contrast["shared_minus_specialist"]
    assert interval["low"] <= 0 <= interval["high"]
    assert "No equivalence" in interval["limits"]
    assert interval["replicates"] == 10000
    assert contrast["intervention_control_refs"]["schedule"] == {
        "shared": "shared-horizon",
        "specialist": "specialist-horizon",
    }
    assert result["heldout_shared_only"]["specialist_comparator"] is None
    assert len(result["heldout_shared_only"]["baseline_context"]) == 12
    assert all(c["site"] != "enc_l14" for c in result["terminal_contrasts"])
    with (tmp_path / "sharing-learning-curves.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["valid_tokens"] == ""
    assert rows[0]["axis_unavailable_reason"] == "learner event log unavailable"
    assert (tmp_path / "sharing-report.md").is_file()


@pytest.mark.parametrize(
    "key", ["initialization", "architecture", "training_data", "objective"]
)
def test_controls_must_match(tmp_path, key):
    rows = observations()
    rows[1]["request"]["comparison"][key] = "different"
    rows[1]["identity"] = observation_identity(rows[1]["request"])
    with pytest.raises(ValueError, match="control mismatch"):
        write_sharing_report(tmp_path, rows, plan())
    assert not (tmp_path / "sharing-comparison.json").exists()


@pytest.mark.parametrize("key", ["noise", "panel", "layout", "site_step"])
def test_native_pairing_controls_must_match(tmp_path, key):
    rows = observations()
    rows[1]["request"][key] = 3 if key == "site_step" else "different"
    rows[1]["identity"] = observation_identity(rows[1]["request"])
    with pytest.raises(ValueError, match="pairing mismatch|site-local quota"):
        write_sharing_report(tmp_path, rows, plan())


def test_source_family_mismatch_blocks_bootstrap(tmp_path):
    rows = observations()
    rows[1]["conditions"][0]["items"][0]["source_id"] = "another-family"
    with pytest.raises(ValueError, match="source_id mapping"):
        write_sharing_report(tmp_path, rows, plan())


def test_real_event_axes_are_not_inferred_from_plan(tmp_path):
    campaign = plan()
    paths = {}
    for learner in campaign.learners:
        path = tmp_path / (learner + ".jsonl")
        schedule = campaign.learner(learner)
        for index in range(schedule.horizon):
            planned = schedule.update_at(index)
            event = fixture.update(
                learner,
                "combined",
                index + 1,
                final=schedule.horizon,
                site=planned.site,
            )
            event["payload"]["site_step"] = planned.site_local_index + 1
            append_event(path, event)
        paths[learner] = path
    report = write_sharing_report(
        tmp_path / "report", observations(), campaign, event_files=paths
    )
    assert (
        report["intervention"]["exposure_evidence"] == "verified supplied update logs"
    )
    with (tmp_path / "report/sharing-learning-curves.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    shared = next(
        row for row in rows if row["learner"] == "shared" and row["site"] == "enc_l9"
    )
    specialist = next(row for row in rows if row["learner"] == "specialist_enc_l9")
    assert shared["valid_tokens"] == "384" and specialist["valid_tokens"] == "128"
    assert shared["site_valid_tokens"] == specialist["site_valid_tokens"] == "128"
    assert shared["elapsed_wall_seconds"] == "3.0"
    assert specialist["elapsed_wall_seconds"] == "1.0"


def test_missing_terminal_panel_rejected(tmp_path):
    with pytest.raises(
        ValueError, match="missing terminal|missing heldout initialization"
    ):
        write_sharing_report(tmp_path, observations()[:-1], plan())


def test_vanilla_is_separate_verified_bypass_evidence(tmp_path):
    import hashlib
    import json

    directory = tmp_path / "vanilla"
    directory.mkdir()
    rows = [
        {
            "sample_id": str(i),
            "source_id": f"family-{i // 2}",
            "normalized_correct": 1,
            "raw_correct": 0,
        }
        for i in range(4)
    ]
    compact = directory / "compact_no_noise.jsonl"
    compact.write_text("".join(json.dumps(row) + "\n" for row in rows))
    receipt = {
        "status": "complete",
        "requested": 4,
        "completed": 4,
        "metrics": {"accuracy": 1},
        "source": "synthetic-source",
        "data": "synthetic-data",
        "request": {
            "input_ids": [str(i) for i in range(4)],
            "source_family_ids": [f"family-{i // 2}" for i in range(4)],
        },
        "output_hashes": {
            compact.name: hashlib.sha256(compact.read_bytes()).hexdigest()
        },
    }
    (directory / "receipt.json").write_text(json.dumps(receipt))
    report = write_sharing_report(tmp_path, observations(), plan())
    assert report["vanilla_bypass"]["bypass"] is True
    assert report["vanilla_bypass"]["metrics"]["normalized_accuracy"] == 1
    assert (
        report["heldout_shared_only"]["baseline_context"][0]["terminal_minus_vanilla"]
        is not None
    )
    compact.write_text("tampered\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        write_sharing_report(tmp_path, observations(), plan())


def test_plot_artifacts_retain_exact_values_and_missing_axes(tmp_path):
    from PIL import Image

    source = observations()
    result = write_sharing_report(tmp_path, source, plan())
    assert len(result["plots"]) == 4
    for artifact in result["plots"]:
        with Image.open(tmp_path / artifact["png"]) as image:
            image.verify()
        assert (tmp_path / artifact["csv"]).is_file()
    with (tmp_path / "sharing-quality-site-local-step.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    point = next(row for row in rows if row["observation"] == source[0]["identity"])
    assert float(point["x"]) == source[0]["request"]["site_step"]
    assert (
        float(point["value"])
        == source[0]["conditions"][0]["metrics"]["normalized_accuracy"]["value"]
    )
    with (tmp_path / "sharing-quality-elapsed-wall.csv").open() as stream:
        elapsed = list(csv.DictReader(stream))
    assert all(row["x"] == "" for row in elapsed)
    with (tmp_path / "sharing-terminal-normalized-gaps.csv").open() as stream:
        gaps = list(csv.DictReader(stream))
    assert len(gaps) == 18
    interval = result["terminal_contrasts"][0]["shared_minus_specialist"]
    assert float(gaps[0]["value"]) == interval["estimate"]
    assert float(gaps[0]["low"]) == interval["low"]
    assert float(gaps[0]["high"]) == interval["high"]
