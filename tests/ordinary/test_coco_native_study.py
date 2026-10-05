"""CPU coverage for future COCO plan bounds and fixed-exposure estimates."""

import jscc
from jscc.checkpoint_readers import coco_native_verify_plan as verify_plan, coco_native_verify_training_completion as verify_training_completion
import copy
import json
from pathlib import Path

import pytest
import yaml
from jscc.config import load_config

from jscc.checkpoint_readers import coco_native_SPLITS as SPLITS, coco_native_digest as digest, coco_native_estimate as estimate, coco_native_scenarios as scenarios, coco_native_validate_measurements as validate_measurements
from model_helpers import prepare_coco_native_fixture as prepare


def receipt():
    return {"version": "coco-native-measurements-v1", "status": "PASS",
            "selection_basis": "capacity_and_throughput_before_quality",
            "training_batch_size": 8, "gradient_accumulation": 4,
            "evaluation_batch_size": 8, "evidence_files": {"receipt.json": "pending"},
            "source_files": {"train.py": "pending"}, "vanilla_generation_seconds_per_image": 0.3,
            "routes": {split: {"status": "PASS", "timing_representative": split,
                "timing_kind": "measured", "seconds_per_update": 4.0,
                "generation_seconds_per_image": 0.5, "selection_seconds": 300} for split in SPLITS}}


def test_fixed_presentations_not_fixed_updates():
    value = receipt()
    assert validate_measurements(value) == 6000
    value["gradient_accumulation"] = 2
    assert validate_measurements(value) == 12000


@pytest.mark.parametrize("mutation", [
    lambda r: r["routes"].pop("dec_l24"),
    lambda r: r["routes"]["dec_l24"].update(status="NOT_RUN"),
    lambda r: r["routes"]["dec_l24"].update(seconds_per_update=float("nan")),
    lambda r: r.update(training_batch_size=7),
    lambda r: r.update(selection_basis="best_quality"),
    lambda r: r.update(source_files={}),
])
def test_incomplete_or_invalid_receipts_rejected(mutation):
    value = receipt()
    mutation(value)
    with pytest.raises(ValueError):
        validate_measurements(value)


def test_proxy_must_name_measured_representative():
    value = receipt()
    value["routes"]["enc_l4"].update(timing_kind="conservative_proxy", timing_representative="enc_emb")
    assert validate_measurements(value) == 6000
    value["routes"]["enc_emb"]["timing_kind"] = "conservative_proxy"
    with pytest.raises(ValueError, match="actually measured"):
        validate_measurements(value)


def test_budget_not_silently_truncated():
    value = receipt()
    value["routes"]["enc_emb"]["seconds_per_update"] = 100
    timing = estimate(value, 6000)
    assert not timing["all_jobs_within_48h"]
    assert timing["rows"][0]["train_job_minutes"] > 2880


def test_prepare_no_submission_and_complete_configs(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.py").write_text("fixture source")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "receipt.json").write_text('{}')
    value = receipt()
    value["source_files"]["train.py"] = digest(source / "train.py")
    value["evidence_files"]["receipt.json"] = digest(evidence / "receipt.json")
    path = tmp_path / "measurements.json"
    path.write_text(json.dumps(value))
    out = tmp_path / "plan"
    card = prepare(path, source, evidence, out)
    assert card["execution_authorized"] is False
    assert card["runs"] == 13 and card["panels"] == 53
    assert card["selection_policy"]["locked"] is False
    assert card["selection_policy"]["historical_policy"] == "periodic_validation"
    configs = [load_config(p) for p in (out / "configs").glob("*.yaml")]
    assert len(configs) == 13
    for config in configs:
        train = config["training"]
        assert train["batch_size"] * train["gradient_accumulation"] * train["max_steps"] == 192000
        assert train["max_steps"] == train["schedule_steps"] == 6000
        assert train["streamed_backward"] is True
        assert train["valid_only_kl"] is True
        assert config["evaluation"]["snrs"] == ["no_noise", -6, 6, 18]
        assert config["evaluation"]["mode"] == "codec_only"
    assert yaml.safe_load((out / "vanilla.yaml").read_text())["mode"] == "vanilla_only"
    manifest = json.loads((out / "manifest.json").read_text())
    assert len({r["evaluation_link"] for r in manifest["runs"]}) == 13
    for relative, expected in json.loads((out / "PREPARED-FILES.json").read_text()).items():
        assert digest(out / relative) == expected
    tampered = copy.deepcopy(value)
    tampered["source_files"]["train.py"] = "wrong"
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="Changed"):
        prepare(path, source, evidence, tmp_path / "another")




def test_scenarios_keep_exposure_and_eval_denominators_explicit():
    rows = scenarios(receipt())
    assert len(rows) == 6
    assert {(r["presentations_per_run"], r["images_per_panel"]) for r in rows} == {
        (p, n) for p in (64000, 128000, 192000) for n in (512, 2000)}
    assert all(r["updates"] * r["effective_batch"] == r["presentations_per_run"] for r in rows)
    assert not any(r["execution_authorized"] for r in rows)


def ready_plan(tmp_path, authorized=False):
    root = Path(jscc.__file__).resolve().parents[1]
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "receipt.json").write_text('{}')
    value = receipt()
    paths = [root / name for name in ("experiment.py", "jscc/experiment.py", "jscc/checkpoint_readers.py", "pyproject.toml", "uv.lock")]
    paths.extend((root / "jscc").rglob("*.py"))
    value["source_files"] = {str(path.relative_to(root)): digest(path) for path in paths}
    value["evidence_files"]["receipt.json"] = digest(evidence / "receipt.json")
    measurements = tmp_path / "measurements.json"
    measurements.write_text(json.dumps(value))
    output = tmp_path / "plan"
    prepare(measurements, root, evidence, output)
    if authorized:
        card_path = output / "APPROVAL-CARD.json"
        card = json.loads(card_path.read_text())
        card.update(execution_authorized=True, approval_reference="TEST-ONLY", site_four_gpu_cap_verified=True)
        card["selection_policy"]["locked"] = True
        card_path.write_text(json.dumps(card))
        refresh_bindings(output)
    return output


def refresh_bindings(output):
    path = output / "PREPARED-FILES.json"
    values = json.loads(path.read_text())
    path.write_text(json.dumps({relative: digest(output / relative) for relative in values}))




def test_execution_rejects_current_unauthorized_plan(tmp_path):
    output = ready_plan(tmp_path)
    with pytest.raises(ValueError, match="not authorized"):
        verify_plan(output, digest(output / "PREPARED-FILES.json"), require_authorized=True)
    assert not (output / "execution").exists()


def test_execution_requires_submitted_frozen_manifest(tmp_path):
    output = ready_plan(tmp_path, authorized=True)
    with pytest.raises(ValueError, match="frozen hash"):
        verify_plan(output, "wrong", require_authorized=True)




def test_tampered_config_blocked_before_execution(tmp_path):
    output = ready_plan(tmp_path, authorized=True)
    config = next((output / "configs").glob("*.yaml"))
    config.write_text(config.read_text() + "# changed")
    with pytest.raises(ValueError, match="Changed prepared"):
        verify_plan(output, digest(output / "PREPARED-FILES.json"), require_authorized=True)




@pytest.mark.parametrize("field", ["site_four_gpu_cap_verified", "selection_locked"])
def test_execution_needs_site_cap_and_selection_freeze(tmp_path, field):
    output = ready_plan(tmp_path, authorized=True)
    path = output / "APPROVAL-CARD.json"
    card = json.loads(path.read_text())
    if field == "selection_locked":
        card["selection_policy"]["locked"] = False
    else:
        card[field] = False
    path.write_text(json.dumps(card))
    refresh_bindings(output)
    with pytest.raises(ValueError):
        verify_plan(output, digest(output / "PREPARED-FILES.json"), require_authorized=True)
    assert not (output / "execution").exists()




def save_completed_training(output, index, **overrides):
    import torch
    run = output / f"fixture-run-{index}"
    run.mkdir()
    completion = {"reason": "max_steps", "step": 6000,
                  "optimizer_updates": 6000, "presentations": 192000}
    completion.update(overrides)
    (run / "completion.json").write_text(json.dumps(completion))
    torch.save({"step": 6000, "codec": {}, "optimizer": {}, "config": {}}, run / "step_006000.pt")
    (output / f"links/{index:04d}.txt").write_text(str(run))
    return run


@pytest.mark.parametrize("overrides", [
    {"reason": "time_budget"}, {"step": 5999}, {"optimizer_updates": 5999},
    {"presentations": 191696},
])
def test_zero_exit_training_with_partial_completion_fails(tmp_path, monkeypatch, overrides):
    output = ready_plan(tmp_path, authorized=True)
    save_completed_training(output, 0, **overrides)
    with pytest.raises(ValueError, match="Incomplete COCO"):
        verify_training_completion(output, json.loads((output / "manifest.json").read_text()), 0,
                                   json.loads((output / "APPROVAL-CARD.json").read_text()))


@pytest.mark.parametrize("fault", ["missing", "wrong_step"])
def test_checkpoint_gate_before_eval(tmp_path, monkeypatch, fault):
    import torch
    import jscc.study_task
    output = ready_plan(tmp_path, authorized=True)
    run = save_completed_training(output, 1)
    checkpoint = run / "step_006000.pt"
    if fault == "missing":
        checkpoint.unlink()
    else:
        torch.save({"step": 5999}, checkpoint)
    monkeypatch.setattr(jscc.study_task, "run_task", lambda *a, **kw: pytest.fail("evaluator must not run"))
    with pytest.raises(ValueError, match="checkpoint"):
        verify_training_completion(output, json.loads((output / "manifest.json").read_text()), 1, json.loads((output / "APPROVAL-CARD.json").read_text()))


def test_full_training_verifies_actual_coco_schema(tmp_path, monkeypatch):
    output = ready_plan(tmp_path, authorized=True)
    save_completed_training(output, 0)
    row = verify_training_completion(output, json.loads((output / "manifest.json").read_text()), 0,
                                     json.loads((output / "APPROVAL-CARD.json").read_text()))
    assert row["completion"]["presentations"] == 192000


def test_prepare_rejects_nondivisible_microbatch(tmp_path):
    output = ready_plan(tmp_path)
    path = tmp_path / "measurements.json"
    value = json.loads(path.read_text())
    value.update(training_batch_size=32, gradient_accumulation=1)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Microbatch must divide"):
        prepare(path, Path(jscc.__file__).resolve().parents[1], tmp_path / "evidence", output.parent / "batch32")
    value.update(training_batch_size=16, gradient_accumulation=2)
    path.write_text(json.dumps(value))
    card = prepare(path, Path(jscc.__file__).resolve().parents[1], tmp_path / "evidence", output.parent / "batch16")
    assert card["updates"] == 6000


def test_prepare_removes_inherited_time_budget(tmp_path, monkeypatch):
    import model_helpers as study
    original = study.expand_study
    def with_timer(*args, **kwargs):
        plan = original(*args, **kwargs)
        for run in plan.runs:
            run.config["training"]["max_minutes"] = 1
        return plan
    monkeypatch.setattr(study, "expand_study", with_timer)
    output = ready_plan(tmp_path)
    for config in (output / "configs").glob("*.yaml"):
        assert "max_minutes" not in load_config(config)["training"]


def add_numerical_limitation(value):
    value["routes"]["enc_fn"].update(status="PASS_EXECUTION_WITH_NUMERICAL_LIMITATION",
        execution_checks_passed=True, numerical_limitations=[{
            "raw_status": "BLOCKED", "check": "Q33 fixed V2", "failed_coordinates": 5,
            "total_coordinates": 262144, "max_logit_delta": 0.00117588,
            "max_probability_delta": 8.63e-5, "all_argmax_equal": True}],
        numerical_review_file="independent-review.json")
    value["evidence_files"]["independent-review.json"] = "test-placeholder"


@pytest.mark.parametrize("fault", ["missing_review", "missing_list", "execution_not_passed", "hidden_pass"])
def test_numerical_exception_requires_explicit_review(fault):
    value = receipt()
    add_numerical_limitation(value)
    row = value["routes"]["enc_fn"]
    if fault == "missing_review":
        value["evidence_files"].pop("independent-review.json")
    elif fault == "missing_list":
        row["numerical_limitations"] = []
    elif fault == "execution_not_passed":
        row["execution_checks_passed"] = False
    else:
        row["status"] = "PASS"
    with pytest.raises(ValueError, match="numerical"):
        validate_measurements(value)


def test_prepared_numerical_exception_requires_separate_card_acceptance(tmp_path, monkeypatch):
    ready_plan(tmp_path)
    measurements = tmp_path / "measurements.json"
    value = json.loads(measurements.read_text())
    add_numerical_limitation(value)
    raw = tmp_path / "evidence/independent-review.json"
    raw.write_text(json.dumps({"raw_status": "BLOCKED", "decision": "proposal_with_limitation"}))
    value["evidence_files"]["independent-review.json"] = digest(raw)
    measurements.write_text(json.dumps(value))
    output = tmp_path / "limited-plan"
    card = prepare(measurements, Path(jscc.__file__).resolve().parents[1], tmp_path / "evidence", output)
    assert card["accept_numerical_limitations"] is False
    assert card["numerical_limitations"][0]["split"] == "enc_fn"
    assert json.loads(raw.read_text())["raw_status"] == "BLOCKED"
    card.update(execution_authorized=True, approval_reference="TEST", site_four_gpu_cap_verified=True)
    card["selection_policy"]["locked"] = True
    card_path = output / "APPROVAL-CARD.json"
    card_path.write_text(json.dumps(card))
    refresh_bindings(output)
    with pytest.raises(ValueError, match="explicitly accept"):
        verify_plan(output, digest(output / "PREPARED-FILES.json"), require_authorized=True)
    assert not (output / "execution").exists()
    original = copy.deepcopy(card["numerical_limitations"])
    card["numerical_limitations"] = []
    card_path.write_text(json.dumps(card))
    refresh_bindings(output)
    with pytest.raises(ValueError, match="preserve"):
        verify_plan(output, digest(output / "PREPARED-FILES.json"), require_authorized=True)
    card["numerical_limitations"] = original
    card["accept_numerical_limitations"] = True
    card_path.write_text(json.dumps(card))
    refresh_bindings(output)
    save_completed_training(output, 0)
    verify_plan(output, digest(output / "PREPARED-FILES.json"), require_authorized=True)
    assert verify_training_completion(output, json.loads((output / "manifest.json").read_text()), 0, card)["completion"]["step"] == card["updates"]


@pytest.mark.parametrize("presentations,updates", [(64000, 1000), (128000, 2000), (192000, 3000)])
def test_mutually_exclusive_receipt_budgets(tmp_path, presentations, updates):
    ready_plan(tmp_path)
    path = tmp_path / "measurements.json"
    value = json.loads(path.read_text())
    value.update(training_batch_size=16, gradient_accumulation=4, presentations_per_run=presentations)
    assert validate_measurements(value) == updates
    path.write_text(json.dumps(value))
    card = prepare(path, Path(jscc.__file__).resolve().parents[1], tmp_path / "evidence", tmp_path / "alternative")
    assert card["presentations_per_run"] == presentations
    assert card["updates"] == card["horizon"] == updates
    assert card["execution_authorized"] is False
    assert card["fixed_checkpoint"] == f"step_{updates:06d}.pt"


@pytest.mark.parametrize("budget", [64001, 0, -1, 64000.0, True])
def test_invalid_chosen_budget_rejected(budget):
    value = receipt()
    value.update(training_batch_size=16, gradient_accumulation=4, presentations_per_run=budget)
    with pytest.raises(ValueError):
        validate_measurements(value)


def test_completion_requires_exact_selected_budget(tmp_path):
    from jscc.checkpoint_readers import coco_native_verify_training_completion as verify_training_completion
    import torch
    run = tmp_path / "run"
    run.mkdir()
    (tmp_path / "link.txt").write_text(str(run))
    completion = {"reason": "max_steps", "step": 2000, "optimizer_updates": 2000, "presentations": 128000}
    (run / "completion.json").write_text(json.dumps(completion))
    torch.save({"step": 2000}, run / "step_002000.pt")
    card = {"updates": 2000, "presentations_per_run": 128000, "fixed_checkpoint": "step_002000.pt"}
    manifest = {"runs": [{"run_link": "link.txt"}]}
    assert verify_training_completion(tmp_path, manifest, 0, card)["completion"]["presentations"] == 128000
    completion["presentations"] = 192000
    (run / "completion.json").write_text(json.dumps(completion))
    with pytest.raises(ValueError, match="presentations"):
        verify_training_completion(tmp_path, manifest, 0, card)


@pytest.mark.parametrize("indices", [[], [0, 2], [5, 2], [2, 5, 5], [5, 13], [True, 5]])
def test_selected_indices_invalid(indices):
    value = receipt()
    value["selected_indices"] = indices
    with pytest.raises(ValueError, match="selected_indices"):
        validate_measurements(value)


def test_selected_six_scope_and_execution_refusal(tmp_path):
    ready_plan(tmp_path)
    path = tmp_path / "measurements.json"
    value = json.loads(path.read_text())
    value["selected_indices"] = [0, 2, 5, 6, 8, 11]
    path.write_text(json.dumps(value))
    output = tmp_path / "six"
    card = prepare(path, Path(jscc.__file__).resolve().parents[1], tmp_path / "evidence", output)
    assert card["runs"] == 6 and card["panels"] == 25
    assert len(json.loads((output / "manifest.json").read_text())["runs"]) == 13
    assert [row["split"] for row in card["estimates"]["rows"]] == [SPLITS[i] for i in value["selected_indices"]]
    assert [row["split"] for row in card["estimates"]["rows"]] == [SPLITS[i] for i in value["selected_indices"]]
    verify_plan(output, digest(output / "PREPARED-FILES.json"))



def test_coco_recorded_completion_uses_bound_attempt_and_fixed_checkpoint(tmp_path):
    import torch
    from compatibility.test_outer_ln_plan import recorded_training_fixture
    from jscc.checkpoint_readers import coco_native_verify_training_completion as verify_training_completion
    logical, run, config, _ = recorded_training_fixture(tmp_path)
    (tmp_path / 'run.txt').write_text(str(logical))
    card = {'updates': 5000, 'presentations_per_run': 640000, 'fixed_checkpoint': 'step_005000.pt'}
    (run / 'completion.json').write_text(json.dumps({'reason': 'max_steps', 'step': 5000,
        'optimizer_updates': 5000, 'presentations': 640000}))
    checkpoint = run / 'step_005000.pt'
    torch.save({'step': 5000, 'config': config}, checkpoint)
    manifest = {'runs': [{'run_link': 'run.txt'}]}
    result = verify_training_completion(tmp_path, manifest, 0, card)
    assert result['run'] == str(run) and result['logical_root'] == str(logical)
    assert result['checkpoint'] == str(checkpoint)
    torch.save({'step': 5000, 'config': {}}, checkpoint)
    with pytest.raises(ValueError, match='checkpoint configuration'):
        verify_training_completion(tmp_path, manifest, 0, card)
