"""Public ordinary roots with tiny real CPU training and fixture task scoring."""

import jscc

from jscc import run_records as _body_run_records
import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from jscc import evaluation, ordinary_runs, training
from jscc.config import load_config
from jscc.data import TaskData
from contracts.test_core import batch, toy_model
from ordinary.test_coco import coco_model, image_batch

REPO = Path(jscc.__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def fixture_task(tmp_path, monkeypatch, task):
    config = load_config(REPO / f"configs/experiments/ordinary/{task}.yaml")
    config["model"].update(device="cpu", dtype="float32")
    config["training"].update(max_steps=2, schedule_steps=2, batch_size=1,
        gradient_accumulation=1, eval_every=1, selection_steps=[1, 2], min_steps=0,
        patience=None, save_steps=[1, 2], log_every=1, validation_batches=1,
        record_coco_presentations=False, streamed_backward=False)
    for key in ("expected_attention_backend", "presentation_stream", "paired_randomness", "feature_summary"):
        config["training"].pop(key, None)
    config["evaluation"].pop("attention_backend", None)
    config["evaluation"].update(snrs=["no_noise"], vanilla=False, noise_seed=0, num_samples=None)
    def build(_):
        return None, coco_model() if task == "coco" else toy_model("enc")
    sample = {**image_batch(), "labels": torch.tensor([[5, 6]])} if task == "coco" else batch()
    monkeypatch.setattr(training, "build_model", build)
    monkeypatch.setattr(evaluation, "build_model", build)
    data = TaskData(train=[sample], validation=[sample], ids={"fixture": [11, 12]})
    monkeypatch.setattr(training, "load_data", lambda *args: data)
    monkeypatch.setattr(evaluation, "load_data", lambda *args, **kwargs: data)
    def score(model, processor, *args):
        # A real tiny forward after checkpoint reload; external task scoring is mocked.
        value = training.batch_losses(model, sample, config["training"], None)["loss"].item()
        assert torch.isfinite(torch.tensor(value))
        return {"fixture_loss": value, "num_samples": sample["labels"].shape[0]}
    monkeypatch.setattr(evaluation, "evaluate_coco", score)
    monkeypatch.setattr(evaluation, "evaluate_hellaswag", score)
    if task == 'hellaswag':
        config['training'].pop('record_coco_presentations', None)
    authored = tmp_path / f"{task}.yaml"
    authored.write_text(yaml.safe_dump(config))
    return config, authored


@pytest.mark.parametrize("task", ["coco", "hellaswag"])
def test_public_singleton_train_and_evaluate_frozen_yaml(tmp_path, monkeypatch, task):
    from request_helpers import invoke
    config, authored = fixture_task(tmp_path, monkeypatch, task)
    root = tmp_path / "exact-root"
    link = tmp_path / "run-link.txt"
    invoke(tmp_path, 'ordinary', 'train', {'root': str(root), 'attempt': '0001', 'success_link': str(link)}, config)
    config["run"]["output_dir"] = str(root)
    assert link.read_text().strip() == str(root)
    assert _body_run_records.bound_config(root) == config
    target = read(root / "checkpoint.json")
    attempt = root / "attempts/training-0001"
    assert Path(target["path"]) == attempt / "best.pt"
    assert target["role"] == "ordinary-best"
    assert read(attempt / "attempt.json")["status"] == "completed"
    metadata = read(attempt / "run.json")
    assert metadata["setup_identity"]["initial_communication"]["codec"]
    assert read(attempt / "data_ids.json") == {"fixture": [11, 12]}
    assert read(attempt / "completion.json")["optimizer_updates"] == 2
    assert not (root / "best.pt").exists()  # Authoritative bytes stay in their attempt.
    authored.write_text("task: invalid-later-edit\n")
    invoke(tmp_path, 'ordinary', 'evaluate', {'root': str(root), 'attempt': '0001', 'evaluation': {}})
    output = root / "evaluations/evaluation-0001"
    result = read(output / "results.json")
    assert result["task"] == task and result["optimizer_step"] == target["step"]
    assert result["checkpoint_sha256"] == target["sha256"]
    assert len(result["conditions"]) == 1
    assert _body_run_records.bound_config(root) == config
    assert read(output / "attempt.json")["status"] == "completed"
    with pytest.raises(FileExistsError):
        invoke(tmp_path, 'ordinary', 'evaluate', {'root': str(root), 'attempt': '0001', 'evaluation': {}})
    with pytest.raises(FileExistsError):
        ordinary_runs.train_task(config, root)
    supplemental = ordinary_runs.evaluate_run(root, "step_000001.pt", expected_step=1, attempt_id="step1")
    assert read(supplemental / "attempt.json")["request"]["role"] == "supplemental"
    assert read(supplemental / "results.json")["optimizer_step"] == 1


@pytest.mark.parametrize("error,status", [(RuntimeError("setup failed"), "failed"),
                                         (KeyboardInterrupt(), "interrupted")])
def test_failed_setup_retains_reserved_attempt(tmp_path, monkeypatch, error, status):
    config, authored = fixture_task(tmp_path, monkeypatch, "hellaswag")
    root = tmp_path / "failed"
    def fail(_):
        assert (root / "case.json").is_file()
        assert read(root / "attempts/training-0001/attempt.json")["status"] == "running"
        raise error
    monkeypatch.setattr(training, "build_model", fail)
    with pytest.raises(type(error)):
        ordinary_runs.train_task(config, root, authored_config=authored)
    assert read(root / "attempts/training-0001/attempt.json")["status"] == status
    assert not (root / "checkpoint.json").exists()
    with pytest.raises(FileExistsError):
        ordinary_runs.train_task(config, root)
    with pytest.raises(FileNotFoundError):
        ordinary_runs.evaluate_run(root)


def test_evaluation_failure_keeps_training_and_partial_results(tmp_path, monkeypatch):
    config, authored = fixture_task(tmp_path, monkeypatch, "hellaswag")
    config["evaluation"]["snrs"] = ["no_noise", 0]
    root = ordinary_runs.train_task(config, tmp_path / "root")
    original = evaluation.evaluate_hellaswag
    def fail(model, processor, settings, output, condition, data_settings):
        if condition == 0:
            raise RuntimeError("later scoring failed")
        return original(model, processor, settings, output, condition, data_settings)
    monkeypatch.setattr(evaluation, "evaluate_hellaswag", fail)
    with pytest.raises(RuntimeError, match="later scoring"):
        ordinary_runs.evaluate_run(root)
    output = root / "evaluations/evaluation-0001"
    assert read(output / "attempt.json")["status"] == "failed"
    assert len(read(output / "results.json")["conditions"]) == 1
    assert read(root / "attempts/training-0001/attempt.json")["status"] == "completed"
    with pytest.raises(FileExistsError):
        ordinary_runs.evaluate_run(root)


def test_case_reservations_references_and_bound_mutation(tmp_path):
    authored = tmp_path / "task.yaml"
    design = tmp_path / "model.yaml"
    authored.write_text("model_config: model.yaml\n")
    design.write_text("model: frozen-reference-context\n")
    config = load_config(REPO / "configs/experiments/ordinary/hellaswag.yaml")
    root = _body_run_records.reserve_root(tmp_path / "comparison", kind="independent-cases")
    first = _body_run_records.bind_case(root, config, authored_config=authored, case_id="width32")
    changed = copy.deepcopy(config)
    changed["codec"]["bottleneck_dim"] = 64
    second = _body_run_records.bind_case(root, changed, case_id="width64")
    assert _body_run_records.bound_config(first)["codec"] != _body_run_records.bound_config(second)["codec"]
    assert [item["path"] for item in read(first / "case.json")["authored"]] == [str(authored), str(design)]
    with pytest.raises(FileExistsError):
        _body_run_records.bind_case(root, config, case_id="width32")
    with pytest.raises(ValueError, match="identifier"):
        _body_run_records.bind_case(root, config, case_id="../escape")
    attempt = _body_run_records.reserve_attempt(first, "collection", attempt_id="manual")
    _body_run_records.finish_attempt(attempt, "failed", details={"missing": 1})
    assert not (second / "attempts").exists()
    (first / "config.yaml").write_text("task: silently-mutated\n")
    with pytest.raises(ValueError, match="configuration changed"):
        _body_run_records.bound_config(first)


def test_resume_and_unsupported_reject_before_reservation(tmp_path):
    config = load_config(REPO / "configs/experiments/ordinary/hellaswag.yaml")
    root = tmp_path / "root"
    with pytest.raises(FileNotFoundError, match="case.json"):
        ordinary_runs.train_task(config, root, resume="last.pt")
    with pytest.raises(ValueError, match="COCO/HellaSwag"):
        ordinary_runs.train_task({"task": "joint"}, root)
    assert not root.exists()


def test_checkpoint_tampering_missing_and_wrong_step_fail_before_model(tmp_path, monkeypatch):
    config, _ = fixture_task(tmp_path, monkeypatch, "hellaswag")
    root = ordinary_runs.train_task(config, tmp_path / "root")
    def unexpected(_):
        pytest.fail("ineligible checkpoint reached model construction")
    monkeypatch.setattr(evaluation, "build_model", unexpected)
    with pytest.raises(ValueError, match="optimizer step"):
        ordinary_runs.evaluate_run(root, "step_000001.pt", expected_step=2)
    target = Path(read(root / "checkpoint.json")["path"])
    state = torch.load(target, weights_only=True)
    state["best"] = 123
    torch.save(state, target)
    with pytest.raises(ValueError, match="identity changed"):
        ordinary_runs.evaluate_run(root)
    target.unlink()
    with pytest.raises(ValueError, match="missing eligible"):
        ordinary_runs.evaluate_run(root)
    assert not (root / "evaluations").exists()


def test_engine_reservation_excludes_concurrent_invocations(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    config = load_config(REPO / "configs/smoke/hellaswag_cpu.yaml")
    root = _body_run_records.reserve_root(tmp_path / "root")
    case = _body_run_records.bind_case(root, config)
    output = _body_run_records.reserve_attempt(case, "training")
    def claim():
        try:
            _body_run_records.validate_reserved_attempt(output, "training")
            return "claimed"
        except FileExistsError:
            return "blocked"
    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(lambda _: claim(), range(2))) == ["blocked", "claimed"]
    assert (output / ".engine-started").is_dir()


def test_checkpoint_replaced_between_wrapper_and_engine_is_rejected(tmp_path, monkeypatch):
    config, _ = fixture_task(tmp_path, monkeypatch, "hellaswag")
    root = ordinary_runs.train_task(config, tmp_path / "root")
    original = evaluation.evaluate
    def replaced(*args, **kwargs):
        target = Path(kwargs["expected_checkpoint"]["path"])
        state = torch.load(target, weights_only=True)
        state["codec"][next(iter(state["codec"]))].add_(1)
        torch.save(state, target)
        return original(*args, **kwargs)
    monkeypatch.setattr(evaluation, "evaluate", replaced)
    monkeypatch.setattr(evaluation, "build_model", lambda _: pytest.fail("replaced checkpoint loaded"))
    with pytest.raises(ValueError, match="checkpoint bytes"):
        ordinary_runs.evaluate_run(root)
    assert read(root / "evaluations/evaluation-0001/attempt.json")["status"] == "failed"


def test_later_result_write_failure_retains_prior_condition(tmp_path, monkeypatch):
    config, _ = fixture_task(tmp_path, monkeypatch, "hellaswag")
    config["evaluation"]["snrs"] = ["no_noise", 0]
    root = ordinary_runs.train_task(config, tmp_path / "root")
    original = evaluation.write_json_atomic
    def fail_later(path, payload):
        if len(payload["conditions"]) == 2:
            raise OSError("replacement failed")
        original(path, payload)
    monkeypatch.setattr(evaluation, "write_json_atomic", fail_later)
    with pytest.raises(OSError, match="replacement failed"):
        ordinary_runs.evaluate_run(root)
    output = root / "evaluations/evaluation-0001"
    assert len(read(output / "results.json")["conditions"]) == 1
    assert read(output / "attempt.json")["status"] == "failed"


def test_custom_channel_scalar_and_extra_state_survive_identity_capture(tmp_path, monkeypatch):
    from jscc.models.channel import AWGNChannel
    class ScalarChannel(AWGNChannel):
        def __init__(self):
            super().__init__()
            self.register_buffer("gain", torch.tensor(0.5))
        def get_extra_state(self):
            return {"protocol": "fixture-channel"}
        def set_extra_state(self, state):
            assert state == {"protocol": "fixture-channel"}
    config, _ = fixture_task(tmp_path, monkeypatch, "hellaswag")
    monkeypatch.setattr(training, "build_model", lambda _: (None, toy_model("enc", channel=ScalarChannel())))
    root = ordinary_runs.train_task(config, tmp_path / "root")
    attempt = root / "attempts/training-0001"
    identity = read(attempt / "run.json")["setup_identity"]["initial_communication"]["channel"]
    assert identity["gain"]["shape"] == [] and identity["gain"]["dtype"] == "torch.float32"
    assert identity["gain"]["sha256"]
    assert identity["_extra_state"]["identity_available"] is False
    state = torch.load(attempt / "best.pt", weights_only=True)
    assert state["channel"]["gain"].item() == 0.5
    assert state["channel"]["_extra_state"] == {"protocol": "fixture-channel"}
