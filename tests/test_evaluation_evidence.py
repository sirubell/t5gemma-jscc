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
