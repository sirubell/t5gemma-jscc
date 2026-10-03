"""Evaluation provenance and pinned task recipes without model/dataset downloads."""
import copy
import json
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from jscc.evaluation import evaluate_hellaswag


@pytest.mark.parametrize("revision", [None, "dataset-commit"])
def test_harness_recipe_and_raw_evidence(monkeypatch, tmp_path, revision):
    recipe = {"task": "hellaswag", "dataset_path": "Rowan/hellaswag",
              "dataset_kwargs": {"trust_remote_code": False}, "doc_to_choice": "choices",
              "metadata": {"version": 1.0}}
    original = copy.deepcopy(recipe)
    manager = SimpleNamespace(task_index={"hellaswag": SimpleNamespace(yaml_path="fixture.yaml")})
    monkeypatch.setattr("lm_eval.tasks.TaskManager", lambda: manager)
    monkeypatch.setattr("lm_eval.tasks._yaml_loader.load_yaml", lambda *args, **kwargs: recipe)
    adapter = object()
    captured = {}
    monkeypatch.setattr("lm_eval.api.registry.get_model", lambda _: lambda **kwargs: adapter)

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        return {"results": {"hellaswag": {"acc,none": 0.5, "acc_norm,none": 0.6}},
                "n-samples": {"hellaswag": {"effective": 2}}, "git_hash": "harness-commit",
                "configs": {"hellaswag": kwargs["tasks"][0]},
                "samples": {"hellaswag": [{"doc_id": np.int64(3), "resps": [[-2.0, False]],
                                            "doc": {"ctx": "original context"}}]}}

    monkeypatch.setattr("lm_eval.evaluator.simple_evaluate", fake_evaluate)
    settings = {"batch_size": 8, "num_fewshot": 5, "num_samples": 2}
    metrics = evaluate_hellaswag(SimpleNamespace(base=object(), split={"stack": "enc"}), object(), settings,
                                tmp_path, -6, {"name": "Rowan/hellaswag", "revision": revision})
    assert recipe == original
    used = captured["tasks"][0]
    assert used["dataset_kwargs"].get("revision") == revision
    assert used["doc_to_choice"] == "choices"
    assert captured["log_samples"] is True
    assert captured["random_seed"] == 0
    assert captured["fewshot_random_seed"] == captured["numpy_random_seed"] == 1234
    assert captured["torch_random_seed"] == 0
    assert metrics == {"acc": 0.5, "acc_norm": 0.6, "num_fewshot": 5,
                       "num_samples": {"effective": 2}}
    metadata = json.loads((tmp_path / "harness_-6.json").read_text())
    assert metadata["git_hash"] == "harness-commit"
    assert "samples" not in metadata
    sample = json.loads((tmp_path / "samples_-6.jsonl").read_text())
    assert sample["doc_id"] == 3
    assert sample["doc"]["ctx"] == "original context"
    assert sample["task"] == "hellaswag"


def test_each_condition_retains_timing_and_completed_metrics(monkeypatch, tmp_path):
    from contextlib import nullcontext
    from jscc import evaluation

    settings = {"snrs": ["no_noise", 0], "vanilla": False}
    state = {"config": {"evaluation": settings, "task": "hellaswag", "seed": 0,
                        "data": {"name": "Rowan/hellaswag"}, "split": {"stack": "enc"}},
             "data_ids": {}, "step": 20}
    model = SimpleNamespace(load_communication_state=lambda _: None, eval=lambda: None,
                            transmission=lambda *args, **kwargs: nullcontext(),
                            channel_uses={"hidden": 10})
    monkeypatch.setattr(evaluation.torch, "load", lambda *args, **kwargs: state)
    monkeypatch.setattr(evaluation, "build_model", lambda _: (object(), model))
    monkeypatch.setattr(evaluation, "load_data", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluation, "new_run", lambda *args: tmp_path)
    monkeypatch.setattr(evaluation, "save_config", lambda *args: None)
    ticks = iter([10.0, 12.0, 20.0, 23.0])
    monkeypatch.setattr(evaluation.time, "perf_counter", lambda: next(ticks))
    seen = []

    def fake_harness(model, processor, settings, output, condition, data_settings):
        seen.append(condition)
        if condition == 0:
            assert json.loads((output / "results.json").read_text())["conditions"][0]["acc"] == 0.5
        return {"acc": 0.5}

    monkeypatch.setattr(evaluation, "evaluate_hellaswag", fake_harness)
    (tmp_path / "best.pt").write_bytes(b"fixture checkpoint bytes")
    evaluation.evaluate(tmp_path)
    results = json.loads((tmp_path / "results.json").read_text())
    assert seen == ["no_noise", 0]
    assert [row["elapsed_seconds"] for row in results["conditions"]] == [2.0, 3.0]
    assert results["conditions"][0]["channel_uses_real"] == {"hidden": 10}


def test_installed_recipe_keeps_callable_preprocessing_through_factory(monkeypatch):
    from datasets import Dataset
    from lm_eval.tasks import TaskManager

    manager = TaskManager()
    monkeypatch.setattr("lm_eval.tasks.TaskManager", lambda: manager)
    monkeypatch.setattr("lm_eval.api.registry.get_model", lambda _: lambda **kwargs: object())
    # Intercept only task construction, after the real factory resolves/merges the recipe.
    # This avoids downloading a dataset while exercising installed task-loading semantics.
    monkeypatch.setattr("lm_eval.tasks._factory.ConfigurableTask", lambda config: SimpleNamespace(config=config))

    def fake_evaluate(**kwargs):
        task = manager._load_spec(kwargs["tasks"][0])
        cfg = cast(Any, task).config
        assert callable(cfg["process_docs"])
        assert cfg["dataset_kwargs"]["revision"] == "pinned-revision"
        rows = Dataset.from_list([{"ctx_a": "A person", "ctx_b": "runs",
                                  "activity_label": "Running", "endings": [" home", " away"],
                                  "label": "0"}])
        processed = cfg["process_docs"](rows)
        assert processed[0]["choices"] == ["home", "away"]
        assert int(processed[0]["label"]) == 0
        return {"results": {"hellaswag": {"acc,none": 0.5, "acc_norm,none": 0.5}}}

    monkeypatch.setattr("lm_eval.evaluator.simple_evaluate", fake_evaluate)
    evaluate_hellaswag(SimpleNamespace(base=object(), split={"stack": "enc"}), object(),
                       {"batch_size": 2, "num_fewshot": 5, "num_samples": 1},
                       data_settings={"name": "Rowan/hellaswag", "revision": "pinned-revision"})


def observation_fixture(monkeypatch, tmp_path):
    from contextlib import nullcontext
    from dataclasses import asdict
    import torch
    from jscc.experiment_state import open_state, save_state
    from jscc.models.split_model import ResolvedSite
    site = ResolvedSite("after_layer", 9, "revision", 24, 8, "encoder.layers.9")
    monkeypatch.setattr("jscc.models.split_model.resolve_encoder_site", lambda *args: site)
    metadata = {"snapshot_role": "initialization", "completed_updates": 0,
                "model_state_contract": "codec-only-stateless-channel-v1",
                "site": asdict(site), "config_identity": "config", "source_identity": "source",
                "protocol_identity": "protocol", "parent_identity": None,
                "initialization_identity": "init", "stream_identity": "stream",
                "phase": "both", "lineage": [], "data_identity": "data",
                "comparison_controls": {key: "synthetic-" + key for key in
                    ("architecture", "initialization", "training_data", "objective", "exposure", "schedule")}}
    codec = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(codec.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    reference = save_state(tmp_path / "fixture.pt", model=codec, optimizer=optimizer,
                           scheduler=scheduler, scaler=None, metadata=metadata,
                           stream_state={"completed_updates": 0, "offset": 0})
    state = open_state(reference, expected=metadata)
    request = {"schema": "codec-observation-v1", "checkpoint_sha256": reference.sha256,
               "target_site": asdict(site), "target_role": "trained", "task": "hellaswag",
               "input_ids": [7], "source_family_ids": ["source-7"], "prompt_policy": "native",
               "noise": {"seed": 12, "namespace": "fixture"}, "layout": "full", "backend": "eager", "precision": "fp32",
               "scorer": "pinned", "reference_corpus": "pinned", "source": "source",
               "expected_items": 1, "settings": {"num_samples": 1}, "data_settings": {},
               "conditions": ["no_noise", -6, 0, 6, 12, 18], "config": "config", "data": "data",
               "parent": None, "panel": "initialization-task", "protocol": "protocol",
               "learner_kind": "single_site_v2", "step": 0,
               "comparison": {key: "synthetic-" + key for key in
                              ("architecture", "initialization", "training_data", "objective", "exposure", "schedule")}}
    loaded = []
    model = SimpleNamespace(base=object(), codec=SimpleNamespace(load_state_dict=loaded.append, state_dict=codec.state_dict),
                            training=True, eval=lambda: None, train=lambda value: None,
                            channel=SimpleNamespace(replay=lambda *args: nullcontext(), parameters=lambda: iter(())),
                            transmission=lambda *args: nullcontext(), at_site=lambda *args, **kwargs: nullcontext(),
                            channel_uses={"hidden": 8}, valid_payload_counts={"hidden": 4})
    return state, request, model, loaded


def test_full_observation_identity_binds_noise_checkpoint_and_site(monkeypatch, tmp_path):
    from jscc.evaluation import codec_observation_identity
    _, request, _, _ = observation_fixture(monkeypatch, tmp_path)
    original = codec_observation_identity(request)
    for field, value in (("checkpoint_sha256", "other"), ("scorer", "other"),
                         ("source", "other"), ("noise", {"seed": 13, "namespace": "fixture"}), ("input_ids", [8])):
        altered = {**request, field: value}
        assert codec_observation_identity(altered) != original
    with pytest.raises(ValueError, match="six conditions"):
        codec_observation_identity({**request, "conditions": ["no_noise"]})


def test_v2_initialization_gate_precedes_weights_and_preserves_split(monkeypatch, tmp_path):
    from jscc.evaluation import evaluate_checkpoint
    state, request, model, loaded = observation_fixture(monkeypatch, tmp_path)
    request["target_role"] = "heldout_after_freeze"
    with pytest.raises(ValueError, match="saved site"):
        evaluate_checkpoint(state, request, model=model, processor=None, output=tmp_path / "observation")
    assert loaded == []
    assert not (tmp_path / "observation").exists()


def test_six_condition_observation_retains_failure_and_restores_rng(monkeypatch, tmp_path):
    import torch
    from jscc import evaluation
    state, request, model, loaded = observation_fixture(monkeypatch, tmp_path)
    initial_rng = torch.get_rng_state().clone()

    def adapter(model, processor, settings, output, condition, data_settings):
        torch.rand(4)
        if condition == 6:
            raise RuntimeError("synthetic interrupted condition")
        (output / f"compact_{condition}.jsonl").write_text(json.dumps(
            {"sample_id": 7, "source_id": "source-7", "raw_scores": [1, 0], "denominators": [1, 1],
             "tokens": [4], "normalized_prediction": 0, "normalized_scores": [1, 0],
             "raw_correct": 1, "normalized_correct": 1}) + "\n")
        return {"acc": 1.0}

    monkeypatch.setattr(evaluation, "evaluate_hellaswag", adapter)
    receipt = evaluation.evaluate_checkpoint(state, request, model=model, processor=None,
                                            output=tmp_path / "observation")
    assert len(loaded) == 2 and set(loaded[0]) == {"weight", "bias"}
    assert torch.equal(initial_rng, torch.get_rng_state())
    assert receipt["status"] == "incomplete"
    assert len(receipt["conditions"]) == 4
    failed = receipt["conditions"][3]
    assert failed["completed"] == 0 and failed["failed"] is None
    assert failed["denominator"] == 0 and failed["unobserved"] == 1
    assert receipt["conditions"][0]["items"][0]["source_id"] == "source-7"
    assert receipt["conditions"][-1]["completed"] == 0
    assert json.loads((tmp_path / "observation" / "observation.json").read_text()) == receipt


def test_shared_heldout_requires_bound_freeze_before_weight_load(monkeypatch, tmp_path):
    from dataclasses import asdict
    from jscc.evaluation import validate_evaluation_target
    from jscc.models.split_model import ResolvedSite
    state, request, model, loaded = observation_fixture(monkeypatch, tmp_path)
    site = ResolvedSite("after_layer", 14, "revision", 24, 8, "encoder.layers.14")
    monkeypatch.setattr("jscc.models.split_model.resolve_encoder_site", lambda *args: site)
    state.payload["kind"] = "shared_encoder_v2"
    request["learner_kind"] = "shared_encoder_v2"
    request.update(target_site=asdict(site), target_role="heldout_after_freeze")
    state.payload["metadata"]["evaluation_sites"] = {
        "enc_l14": {"site": asdict(site), "role": "heldout_after_freeze"}}
    with pytest.raises(ValueError, match="freeze receipt"):
        validate_evaluation_target(state, request, model)
    assert loaded == []


def test_legacy_entrypoint_rejects_versioned_checkpoint(monkeypatch, tmp_path):
    from jscc import evaluation
    monkeypatch.setattr(evaluation.torch, "load", lambda *args, **kwargs: {"schema": "experiment-state-v2"})
    monkeypatch.setattr(evaluation, "build_model", lambda config: pytest.fail("must gate before model construction"))
    with pytest.raises(ValueError, match="Versioned checkpoints require"):
        evaluation.evaluate(tmp_path)


def test_item_mismatch_retains_actual_denominator(monkeypatch, tmp_path):
    from jscc import evaluation
    state, request, model, _ = observation_fixture(monkeypatch, tmp_path)

    def adapter(model, processor, settings, output, condition, data_settings):
        (output / f"compact_{condition}.jsonl").write_text(json.dumps(
            {"sample_id": 999, "source_id": "actual-source"}) + "\n")
        return {"acc": 0.0}

    monkeypatch.setattr(evaluation, "evaluate_hellaswag", adapter)
    receipt = evaluation.evaluate_checkpoint(state, request, model=model, processor=None,
                                            output=tmp_path / "observation")
    assert receipt["status"] == "incomplete"
    for condition in receipt["conditions"]:
        assert condition["completed"] == condition["denominator"] == 1
        assert condition["items"][0]["sample_id"] == 999
        assert condition["metrics"]["acc"] == 0.0


def test_records_projection_identity_and_unavailable_measure(monkeypatch, tmp_path):
    import hashlib
    from jscc.evaluation import codec_observation_identity, observation_event_payload
    _, request, _, _ = observation_fixture(monkeypatch, tmp_path)
    receipt = {"request": request, "identity": codec_observation_identity(request),
               "status": "incomplete", "reuse": None, "conditions": [
                   {"condition": "no_noise", "requested": 1, "completed": 0, "failed": None,
                    "denominator": 0, "metrics": {}, "items": [], "unobserved": 1,
                    "error": {"message": "adapter interrupted"}}]}
    payload = observation_event_payload(receipt)
    expected = hashlib.sha256(b"codec-observation-v1\0" + json.dumps(
        payload["request"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert expected == payload["identity"]
    assert set(payload) == {"request", "identity", "status", "conditions", "reuse"}
    projected = payload["request"]
    assert projected["learner_kind"] == "initialization"
    assert projected["site"] == "enc_l9"
    assert json.loads(projected["details"]["outputs"].removeprefix("inline-execution-request-v1:")) == request
    row = payload["conditions"][0]
    assert row["failed"] is None
    assert row["failure_reason"] == "Per-item failure count unavailable"
    assert row["metrics"]["unobserved_items"] == {"value": 1, "reason": None}
    assert row["metrics"]["task_score"] == {"value": None, "reason": "adapter interrupted"}


def test_records_projection_retains_item_values(monkeypatch, tmp_path):
    from jscc.evaluation import codec_observation_identity, observation_event_payload
    _, request, _, _ = observation_fixture(monkeypatch, tmp_path)
    item = {"sample_id": 7, "source_id": "family-7", "raw_scores": [-1.0, -2.0], "denominators": [2, 3],
            "tokens": [4, 5], "raw_correct": 0, "normalized_correct": 0,
            "normalized_prediction": 1, "normalized_scores": [-0.5, -0.4]}
    receipt = {"request": request, "identity": codec_observation_identity(request),
               "status": "incomplete", "conditions": [
                   {"condition": "no_noise", "requested": 1, "completed": 1, "failed": 0,
                    "denominator": 1, "metrics": {"acc": 0.0, "num_samples": {"effective": 1}},
                    "items": [item], "unobserved": 0}]}
    row = observation_event_payload(receipt)["conditions"][0]
    assert row["items"][0] == {**item, "item_id": "7", "prediction": "1", "score": -0.4}
    assert row["metrics"]["acc"] == {"value": 0.0, "reason": None}



@pytest.mark.parametrize("tamper", ["metadata", "tensor", "corrupt", "delete"])
def test_evaluation_reopens_checkpoint_before_mutation(monkeypatch, tmp_path, tamper):
    from pathlib import Path
    from jscc.evaluation import evaluate_checkpoint
    state, request, model, loaded = observation_fixture(monkeypatch, tmp_path)
    if tamper == "metadata":
        state.payload["metadata"]["site"]["index"] = 8
    elif tamper == "tensor":
        state.payload["model"]["weight"].add_(1)
    elif tamper == "corrupt":
        Path(state.reference.path).write_bytes(b"corrupt")
    else:
        Path(state.reference.path).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        evaluate_checkpoint(state, request, model=model, processor=None, output=tmp_path / "observation")
    assert loaded == []
    assert not (tmp_path / "observation").exists()


def test_observation_comparison_requires_explicit_controls(monkeypatch, tmp_path):
    from jscc.evaluation import codec_observation_identity
    _, request, _, _ = observation_fixture(monkeypatch, tmp_path)
    del request["comparison"]["exposure"]
    with pytest.raises(ValueError, match="comparison control"):
        codec_observation_identity(request)


def test_actual_harness_document_tokens_are_joined(tmp_path):
    from jscc.evaluation import _observation_items
    (tmp_path / "compact_no_noise.jsonl").write_text(json.dumps(
        {"sample_id": 7, "source_id": "source-7", "normalized_prediction": 1}) + "\n")
    (tmp_path / "documents.jsonl").write_text(json.dumps(
        {"sample_id": 7, "requests": [{"continuation_token_ids": [3]},
                                      {"continuation_token_ids": [4, 5]}]}) + "\n")
    items, ids, families = _observation_items(tmp_path, "hellaswag", "no_noise")
    assert ids == [7] and families == ["source-7"]
    assert items[0]["tokens"] == [4, 5]
    assert items[0]["tokens_policy"] == "normalized-prediction-continuation-v1"


def test_records_coco_projection_uses_actual_generation_fields(monkeypatch, tmp_path):
    from jscc.evaluation import codec_observation_identity, observation_event_payload
    _, request, _, _ = observation_fixture(monkeypatch, tmp_path)
    request["task"] = "coco"
    item = {"image_id": 7, "token_ids": [0, 4, 1], "raw_caption": "Cat\nother",
            "caption": "Cat", "cider": 0.4, "eos": True, "truncated": False}
    receipt = {"request": request, "identity": codec_observation_identity(request),
               "status": "incomplete", "conditions": [
                   {"condition": "no_noise", "requested": 1, "completed": 1, "failed": 0,
                    "denominator": 1, "metrics": {"cider": 0.4}, "items": [item], "unobserved": 0}]}
    row = observation_event_payload(receipt)["conditions"][0]
    assert row["items"][0] == {**item, "item_id": "7", "source_id": "7", "tokens": [0, 4, 1],
                               "caption_raw": "Cat\nother", "caption_clean": "Cat", "cap_hit": False}
    assert row["metrics"]["cider"] == {"value": 0.4, "reason": None}


def test_checkpoint_evaluation_restores_codec_and_stops_after_failure(monkeypatch, tmp_path):
    import random
    import numpy as np
    import torch
    from jscc import evaluation
    state, request, model, _ = observation_fixture(monkeypatch, tmp_path)
    model.codec = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.codec.weight.fill_(42)
        model.codec.bias.fill_(-7)
    original = {key: value.clone() for key, value in model.codec.state_dict().items()}
    from jscc.experiment_state import capture_rng
    rng = capture_rng()
    seen = []

    def adapter(*args):
        seen.append(args[4])
        random.random()
        np.random.rand()
        torch.rand(2)
        with torch.no_grad():
            model.codec.weight.add_(10)
        raise RuntimeError("adapter stopped")

    monkeypatch.setattr(evaluation, "evaluate_hellaswag", adapter)
    receipt = evaluation.evaluate_checkpoint(state, request, model=model, processor=None,
                                            output=tmp_path / "observation")
    assert seen == ["no_noise"]
    assert len(receipt["conditions"]) == 1
    assert evaluation._state_payload_equal(original, model.codec.state_dict())
    assert evaluation._state_payload_equal(rng, capture_rng())
    evaluation.verify_observation_artifacts(receipt, tmp_path / "observation")


@pytest.mark.parametrize("tamper", ["bytes", "items", "inventory", "count"])
def test_external_observation_verifies_actual_artifacts(monkeypatch, tmp_path, tamper):
    from jscc import evaluation
    state, request, model, _ = observation_fixture(monkeypatch, tmp_path)

    def adapter(model, processor, settings, output, condition, data_settings):
        item = {"sample_id": 7, "source_id": "source-7", "raw_scores": [-1., -2.],
                "denominators": [1, 1], "normalized_scores": [-1., -2.],
                "normalized_prediction": 0, "raw_correct": 1, "normalized_correct": 1,
                "tokens": [4]}
        (output / f"compact_{condition}.jsonl").write_text(json.dumps(item) + "\n")
        return {"acc": 1.0}

    monkeypatch.setattr(evaluation, "evaluate_hellaswag", adapter)
    root = tmp_path / "observation"
    receipt = evaluation.evaluate_checkpoint(state, request, model=model, processor=None, output=root)
    evaluation.verify_observation_artifacts(receipt, root)
    if tamper == "bytes":
        (root / "no_noise" / "compact_no_noise.jsonl").write_text("{}\n")
    elif tamper == "inventory":
        (root / "no_noise" / "extra.json").write_text("{}")
    elif tamper == "items":
        receipt["conditions"][0]["items"][0]["tokens"] = [9]
    else:
        receipt["conditions"][0]["completed"] = 0
    (root / "observation.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        evaluation.verify_observation_artifacts(receipt, root)


@pytest.mark.parametrize("field", ["source", "config", "parent", "data", "comparison"])
def test_evaluation_rejects_unbound_controls_before_load(monkeypatch, tmp_path, field):
    from jscc.evaluation import evaluate_checkpoint
    state, request, model, loaded = observation_fixture(monkeypatch, tmp_path)
    if field == "comparison":
        request[field]["architecture"] = "different"
    else:
        request[field] = "different"
    with pytest.raises(ValueError, match="identity mismatch|comparison controls"):
        evaluate_checkpoint(state, request, model=model, processor=None, output=tmp_path / "observation")
    assert loaded == []
    assert not (tmp_path / "observation").exists()
