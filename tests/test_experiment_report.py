"""CPU-only scientific analysis and report evidence tests."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import runpy

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("experiment_report", ROOT / "scripts/render_experiment_report.py")
assert spec is not None and spec.loader is not None
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_bootstrap_row_weighted_group_resampling_and_pair_mapping():
    left = [{"item_id": str(i), "source_id": "large" if i < 3 else "small", "score": float(i < 3)} for i in range(4)]
    right = [{**r, "score": 0.0} for r in left]
    result = report.paired_group_bootstrap(left, right, "score")
    assert result["estimate"] == .75  # not the group-weighted .5
    assert result["group_sizes"] == {"large": 3, "small": 1}
    assert result["replicates"] == 10000 and result["seed"] == 0
    assert result["low"] == 0 and result["high"] == 1
    assert report.paired_group_bootstrap(list(reversed(left)), right, "score") == result
    right[0]["source_id"] = "wrong"
    with pytest.raises(ValueError, match="mapping"):
        report.paired_group_bootstrap(left, right, "score")


@pytest.mark.parametrize("change", ["missing_source", "missing_item", "duplicate", "missing_score"])
def test_bootstrap_blocks_incomplete_pairing(change):
    left = [{"item_id": "1", "source_id": "source", "score": 1.0}]
    right = [dict(left[0])]
    if change == "missing_source":
        del right[0]["source_id"]
    elif change == "missing_item":
        right[0]["item_id"] = "2"
    elif change == "duplicate":
        right.append(dict(right[0]))
    else:
        right[0]["score"] = None
    with pytest.raises(ValueError):
        report.paired_group_bootstrap(left, right, "score")


def _helper():
    return runpy.run_path(str(ROOT / "tests/test_experiment_records.py"))


def _write_run(path):
    from jscc import experiment_records as records
    helpers = _helper()
    event = helpers["update"]()
    path.mkdir()
    records.append_event(path / "metrics.jsonl", event)
    inventory = {"schema": records.SCHEMA, "mode": "metadata-only", "artifacts": [records.artifact_ref(path / "metrics.jsonl", path, "metadata")], "omitted_tensors": []}
    manifest = {"schema": records.SCHEMA, "run_id": event["run_id"], "phases": [{"phase_id": event["phase_id"], "updates": 2, "start_step": 0, "start_valid_tokens": 0}], "expected_observations": [], "expected_artifacts": ["metrics.jsonl"]}
    records.write_json_atomic(path / "inventory.json", inventory)
    records.write_json_atomic(path / "manifest.json", manifest)
    return event


def test_report_verified_inputs_missing_values_and_incomplete_manifest(tmp_path):
    source = tmp_path / "run"
    _write_run(source)
    output = tmp_path / "report"
    result = report.render_local_report(source, output)
    assert result["completion"][0]["status"] == "incomplete"
    assert len(result["inputs"]) == 3
    assert result["images"]
    with Image.open(output / result["images"][0]["path"]) as image:
        assert image.size == (1450, 900)
    updates = (output / "updates.csv").read_text()
    assert "not_sampled" in updates or "unavailable" in updates or "cpu" in updates.lower()
    assert "incomplete" in (output / "index.html").read_text()
    settings = json.loads((output / "analysis-settings.json").read_text())
    assert settings["bootstrap"]["replicates"] == 10000
    with pytest.raises(ValueError, match="empty"):
        report.render_local_report(source, output)
    (source / "metrics.jsonl").write_text("tampered\n")
    with pytest.raises(ValueError, match="hash/size"):
        report.render_local_report(source, tmp_path / "bad-report")
    assert not (tmp_path / "bad-report").exists()


def test_plot_preserves_missing_points_in_csv_and_draws_png(tmp_path):
    rows = [{"series": "sample", "x": 1, "value": 2, "phase": "local"},
            {"series": "sample", "x": 2, "value": None, "phase": "local", "reason": "failed"},
            {"series": "sample", "x": 3, "value": 4, "phase": "combined"}]
    report._write_csv(tmp_path / "data.csv", rows)
    report._plot(tmp_path / "plot.png", "Missing points", rows, x_label="completed step")
    assert ",,local,failed" in (tmp_path / "data.csv").read_text()
    assert (tmp_path / "plot.png").stat().st_size > 1000


def test_full_synthetic_campaign_report(tmp_path):
    fixture = runpy.run_path(str(ROOT / "tests/fixtures/experiment_records/build_fixture.py"))
    source = fixture["build"](tmp_path / "campaign")
    output = tmp_path / "report"
    result = report.render_local_report(source, output)
    statuses = {r["run_id"]: r["status"] for r in result["completion"]}
    assert statuses["coco"] == "incomplete"
    assert statuses["shared"] == statuses["specialist"] == "complete"
    assert statuses["staged"] == "complete"
    names = [p["path"] for p in result["images"]]
    for category in ("objective-local", "objective-combined", "learning-rate", "gradient-and-update", "timing", "memory", "quality", "paired", "cost"):
        assert any(category in name for name in names), category
    import csv
    with (output / "paired-gaps.csv").open() as stream:
        comparisons = list(csv.DictReader(stream))
    assert comparisons
    assert any(r["contrast"] == "shared-minus-specialist" for r in comparisons)
    assert all(r["contrast"] != "shared-minus-specialist" for r in comparisons if r["site"] == "enc_l14")
    assert any(r["contrast"] == "shared-minus-initialization" and r["site"] == "enc_l14" for r in comparisons)
    assert "bank_total" in (output / "costs.csv").read_text()
    explicit = [r for r in comparisons if r.get("analysis") == "explicit-left-minus-right"]
    assert len(explicit) == 12
    assert {r["condition"] for r in explicit} == {"no_noise", "-6", "0", "6", "12", "18"}
    assert any(Path(r["path"]).name == "comparisons.json" for r in result["inputs"])
    assert result["comparisons_manifest"]["schema"] == "experiment-comparisons-v1"
    with (output / "footprints.csv").open() as stream:
        footprints = list(csv.DictReader(stream))
    assert {r["units"] for r in footprints} == {"parameters", "bytes"}
    bank_bytes = [r for r in footprints if r["scope"] == "bank_total" and r["metric"] == "checkpoint_bytes"]
    assert len(bank_bytes) == 1 and float(bank_bytes[0]["value"]) == 24
    assert len(json.loads(bank_bytes[0]["retained_tensor_refs"])) == 3
    assert any("footprint" in image["path"] for image in result["images"])
    with (output / "task-quality.csv").open() as stream:
        quality = list(csv.DictReader(stream))
    assert any(r["value"] == "" and r["partial_value"] for r in quality if r["run_id"] == "coco")
    for image in result["images"]:
        assert image["sha256"] == report._digest(output / image["path"])
        assert (output / image["table"]).exists()
        with (output / image["table"]).open() as stream:
            plotted = list(csv.DictReader(stream))
        assert len({r["series"] for r in plotted}) <= 8
    assert len({p["sha256"] for p in result["inputs"]}) > 1


def _comparison_observation(kind, *, run=None, role="trained"):
    controls = {name: "controlled-" + name for name in ("architecture", "initialization", "training_data", "objective", "exposure", "schedule")}
    request: dict = {name: "matched-" + name for name in ("source", "data", "parent", "panel", "site", "task", "protocol", "noise", "scorer", "layout", "precision", "backend")}
    request.update({"task": "hellaswag", "purpose": "task", "objective_kind": None,
                    "learner_kind": kind, "role": role, "expected_items": 2,
                    "site_step": 6, "step": 18 if kind == "shared" else 6,
                    "comparison": controls, "details": {"prompt": "prompt", "native_policy": "five-shot", "draws": "draws"}})
    condition = {"condition": "no_noise", "requested": 2, "completed": 2, "failed": 0,
                 "items": [{"item_id": str(i), "source_id": str(i), "normalized_correct": i, "raw_correct": i} for i in range(2)]}
    return {"request": request, "condition": condition,
            "event": {"run_id": run or kind, "phase_id": "combined", "payload": {"status": "complete", "identity": run or kind}}}


@pytest.mark.parametrize("control", ["architecture", "initialization", "training_data", "objective", "exposure", "schedule"])
def test_pairing_requires_all_immutable_comparison_controls(control):
    shared, specialist = _comparison_observation("shared"), _comparison_observation("specialist")
    assert len(report._comparisons([shared, specialist])[0]) == 2
    specialist["request"]["comparison"][control] = "different"
    results, blocked = report._comparisons([shared, specialist])
    assert not results
    assert blocked[0]["reason"] == "no compatible comparison observation"


def test_pairing_blocks_ambiguous_matches():
    observations = [_comparison_observation("shared"), _comparison_observation("specialist", run="ref-a"),
                    _comparison_observation("specialist", run="ref-b")]
    results, blocked = report._comparisons(observations)
    assert not results
    assert blocked[0]["candidate_count"] == 2
    assert blocked[0]["reason"] == "ambiguous matched observations"


def test_truthful_vanilla_bypass_matches_shared_heldout_and_checks_membership():
    shared = _comparison_observation("shared", role="heldout")
    vanilla = _comparison_observation("vanilla", role="diagnostic")
    vanilla["request"].update({"site": "not_applicable", "noise": "not_applicable", "parent": None})
    vanilla["request"]["details"]["draws"] = "not_applicable"
    vanilla["condition"]["condition"] = "vanilla"
    results, _ = report._comparisons([shared, vanilla])
    assert len(results) == 2
    assert all(r["contrast"] == "shared-minus-vanilla" for r in results)
    vanilla["condition"]["items"][0]["item_id"] = "different"
    results, blocked = report._comparisons([shared, vanilla])
    assert not results
    assert any("membership" in r["reason"] for r in blocked)


def test_paired_series_separates_shared_arms_and_cohorts():
    a = {"shared_run": "shared-a", "reference_run": "reference", "site": "enc_l9",
         "condition": "no_noise", "contrast": "shared-minus-specialist", "cohort": "cohort-a"}
    assert report._paired_series(a) != report._paired_series({**a, "shared_run": "shared-b"})
    assert report._paired_series(a) != report._paired_series({**a, "cohort": "cohort-b"})
    assert "shared-a" in report._paired_series(a) and "reference" in report._paired_series(a)


def test_slug_collision_resistance_and_defensive_output_guard(tmp_path, monkeypatch):
    assert report._slug("quality-run.a") != report._slug("quality-run-a")
    assert report._slug("quality-run.a") == report._slug("quality-run.a")
    source = tmp_path / "run"
    _write_run(source)
    monkeypatch.setattr(report, "_slug", lambda _: "forced-collision")
    with pytest.raises(ValueError, match="duplicate report artifact path"):
        report.render_local_report(source, tmp_path / "report")


def _explicit_fixture(*, task="hellaswag"):
    fixture = runpy.run_path(str(ROOT / "tests/fixtures/experiment_records/build_fixture.py"))
    left = fixture["observation"]("baseline-direct", "combined", "enc_fn", "specialist", 12, task=task)
    right = fixture["observation"]("baseline-residual", "combined", "enc_fn", "specialist", 12, task=task)
    manifest = {"schema": "experiment-comparisons-v1", "contrasts": [{"id": "direct-minus-residual",
                "left_observation": left["payload"]["identity"], "right_observation": right["payload"]["identity"],
                "metrics": ["cider"] if task == "coco" else ["normalized_correct", "raw_correct"],
                "allowed_control_differences": ["architecture", "initialization"]}]}
    return left, right, manifest


def test_explicit_baseline_contrast_all_six_conditions_and_coco():
    for task, expected in (("hellaswag", 12), ("coco", 6)):
        left, right, manifest = _explicit_fixture(task=task)
        rows, blocked = report._explicit_comparisons([left, right], manifest)
        assert not blocked and len(rows) == expected
        assert {r["condition"] for r in rows} == {"no_noise", -6, 0, 6, 12, 18}
        assert all(r["analysis"] == "explicit-left-minus-right" for r in rows)
        assert all(r["left_run"] == "baseline-direct" for r in rows)
        assert all(r["replicates"] == 10000 for r in rows)


@pytest.mark.parametrize("problem", ["missing", "ambiguous", "duplicate_id", "unsupported_control", "undeclared_difference", "noise", "membership", "incomplete"])
def test_explicit_requested_contrast_blocks_incompatible_evidence(problem):
    left, right, manifest = _explicit_fixture()
    events = [left, right]
    if problem == "missing":
        manifest["contrasts"][0]["left_observation"] = "unknown"
    elif problem == "ambiguous":
        events.append(left)
    elif problem == "duplicate_id":
        manifest["contrasts"].append(dict(manifest["contrasts"][0]))
    elif problem == "unsupported_control":
        manifest["contrasts"][0]["allowed_control_differences"].append("exposure")
    elif problem == "undeclared_difference":
        right["payload"]["request"]["comparison"]["schedule"] = "different"
    elif problem == "noise":
        right["payload"]["request"]["noise"] = "different"
    elif problem == "membership":
        right["payload"]["conditions"][0]["items"][0]["source_id"] = "different"
    else:
        right["payload"]["status"] = "incomplete"
    rows, blocked = report._explicit_comparisons(events, manifest)
    assert not rows
    assert blocked and all(r["reason"] for r in blocked)
    assert all(r["contrast"] == "direct-minus-residual" for r in blocked)


@pytest.mark.parametrize("task", ["coco", "hellaswag"])
@pytest.mark.parametrize("purpose", ["task", "objective"])
def test_missing_condition_retains_correct_metric_family(task, purpose):
    from jscc.experiment_records import validate_event
    fixture = runpy.run_path(str(ROOT / "tests/fixtures/experiment_records/build_fixture.py"))
    if purpose == "objective":
        event = fixture["objective_observation"]("run", "combined", 12, task)
        expected = {f"objective.{component}.{field}" for component in ("K", "R") for field in ("raw", "weighted", "numerator", "denominator")}
    else:
        event = fixture["observation"]("run", "combined", "enc_fn", "specialist", 12, task=task)
        expected = {"cider"} if task == "coco" else {"normalized_accuracy", "raw_accuracy"}
    event["payload"]["conditions"].pop()
    event["payload"]["status"] = "incomplete"
    validate_event(event)
    rows, _ = report._observation_rows([event])
    missing = [row for row in rows if row["condition"] == 18]
    assert {row["metric"] for row in missing} == expected
    assert all(row["value"] is None and "missing condition" in row["reason"] for row in missing)
