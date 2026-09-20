import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from scripts import evaluate_prompt_alignment as evaluation


def saved_run(tmp_path):
    config = {
        "task": "hellaswag", "training": {"max_steps": 500}, "model": {"dtype": "bfloat16"},
        "channel": {"normalize_power": True}, "codec": {"bottleneck_dim": 512},
        "split": {"stack": "enc", "where": "after_layer", "index": 19},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    (tmp_path / "run.json").write_text(json.dumps({
        "resolved_config_sha256": evaluation.sha(tmp_path / "config.yaml"),
        "training_source_sha256": "source-digest", "source": {"revision": None, "dirty": None},
    }))
    torch.save({"config": config, "step": 500}, tmp_path / "last.pt")
    (tmp_path / "completion.json").write_text(json.dumps({
        "status": "FULL_BUDGET_COMPLETED", "step": 500, "optimizer_updates": 500,
        "source_verified": True, "presentation_budget_verified": True,
        "final_checkpoint": {"file": "last.pt", "step": 500,
                             "sha256": evaluation.sha(tmp_path / "last.pt")},
    }))
    return {"run_id": "enc_l19-raw", "run_path": str(tmp_path),
            "checkpoint": "last.pt", "expected_step": 500}, config


def test_validated_checkpoint_binds_hash_source_config_and_exact_step(tmp_path):
    record, config = saved_run(tmp_path)
    actual, state, receipt = evaluation.validate_run(record, "source-digest")
    assert actual == config and state["step"] == 500
    assert receipt["checkpoint_sha256"] == evaluation.sha(tmp_path / "last.pt")
    assert receipt["split"] == "enc_l19"
    with pytest.raises(ValueError, match="step/update count"):
        evaluation.validate_run({**record, "expected_step": 499}, "source-digest")
    with pytest.raises(ValueError, match="Training source"):
        evaluation.validate_run(record, "different")
    with pytest.raises(ValueError, match="Checkpoint hash"):
        evaluation.validate_run({**record, "checkpoint_sha256": "different"}, "source-digest")
    changed = copy.deepcopy(config)
    changed["split"]["index"] = 9
    torch.save({"config": changed, "step": 500}, tmp_path / "last.pt")
    completion_path = tmp_path / "completion.json"
    completion = json.loads(completion_path.read_text())
    completion["final_checkpoint"]["sha256"] = evaluation.sha(tmp_path / "last.pt")
    completion_path.write_text(json.dumps(completion))
    with pytest.raises(ValueError, match="Checkpoint config"):
        evaluation.validate_run(record, "source-digest")


@pytest.mark.parametrize("split,expected", [
    ({"stack": "enc", "where": "after_layer", "index": 9}, "enc_l9"),
    ({"stack": "enc", "where": "after_layer", "index": 19}, "enc_l19"),
    ({"stack": "enc", "where": "after_final_norm"}, "enc_fn"),
])
def test_all_authorized_splits(split, expected):
    assert evaluation.split_name({"split": split}) == expected


def test_decoder_is_not_authorized():
    with pytest.raises(ValueError, match="encoder"):
        evaluation.split_name({"split": {"stack": "dec", "where": "after_final_norm"}})


def test_compact_predictions_use_per_candidate_denominators():
    group = [{"sample_id": 2, "source_id": "src", "gold": 1, "prompt_sha256": "prompt",
              "denominator": d} for d in (1, 10, 1, 1)]
    row = evaluation.compact_row(group, [-1., -2., -3., -4.], "raw", 0,
                                  {"run_id": "run", "split": "enc_fn"}, "fixture")
    assert row["raw_prediction"] == 0 and not row["raw_correct"]
    assert row["normalized_prediction"] == 1 and row["normalized_correct"]
    assert row["raw_margin"] == -1.
    assert row["normalized_margin"] == pytest.approx(.8)
    assert row["normalized_ranking"] == [1, 0, 2, 3]


def test_cli_passes_only_inference_arguments(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(evaluation, "run", lambda *args: calls.append(args))
    evaluation.main(["--runs-json", "runs.json", "--prepared", "prepared", "--output", str(tmp_path)])
    assert calls == [(Path("runs.json"), Path("prepared"), tmp_path)]
    with pytest.raises(SystemExit):
        evaluation.main(["--train"])


def test_wrong_frozen_hash_rejected_before_tensor_processing(tmp_path):
    (tmp_path / "encoded-requests.jsonl").write_text("[]\n")
    (tmp_path / "batch-manifest.json").write_text("[]\n")
    (tmp_path / "completion.json").write_text(json.dumps({
        "status": "PASS_CPU_NO_MODEL",
        "encoded_requests_sha256": evaluation.sha(tmp_path / "encoded-requests.jsonl"),
        "batch_manifest_sha256": evaluation.sha(tmp_path / "batch-manifest.json"),
    }))
    with pytest.raises(ValueError, match="authorized FROZEN"):
        evaluation.validate_prepared(tmp_path)


@pytest.mark.parametrize("field,value,message", [
    ("status", "PARTIAL", "full budget"),
    ("optimizer_updates", 499, "step/update count"),
    ("source_verified", False, "not verified"),
    ("presentation_budget_verified", False, "not verified"),
])
def test_incomplete_training_is_never_evaluated(tmp_path, field, value, message):
    record, _ = saved_run(tmp_path)
    path = tmp_path / "completion.json"
    completion = json.loads(path.read_text())
    completion[field] = value
    path.write_text(json.dumps(completion))
    with pytest.raises(ValueError, match=message):
        evaluation.validate_run(record, "source-digest")


def test_final_checkpoint_hash_bound_to_training_receipt(tmp_path):
    record, _ = saved_run(tmp_path)
    path = tmp_path / "completion.json"
    completion = json.loads(path.read_text())
    completion["final_checkpoint"]["sha256"] = "wrong"
    path.write_text(json.dumps(completion))
    with pytest.raises(ValueError, match="completion hash"):
        evaluation.validate_run(record, "source-digest")
