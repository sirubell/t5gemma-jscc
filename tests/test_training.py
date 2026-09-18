import copy
import json
from pathlib import Path

import torch
import pytest

from jscc.config import load_config
from jscc.data import TaskData
from jscc import training
from test_core import batch, toy_model


@pytest.mark.parametrize("stack", ["enc", "dec"])
def test_training_checkpoint_and_resume(tmp_path, monkeypatch, stack):
    config = load_config(Path(__file__).parents[1] / "configs/tasks/hellaswag.yaml")
    config["run"].update(output_dir=str(tmp_path), name="test")
    config["model"].update(device="cpu", dtype="float32")
    config["training"].update(max_steps=2, eval_every=1, gradient_accumulation=2,
                                validation_batches=1, patience=None, save_steps=[1], log_every=1)
    models = []
    def build(config):
        model = toy_model(stack)
        models.append(model)
        return None, model
    monkeypatch.setattr(training, "build_model", build)
    monkeypatch.setattr(training, "load_data", lambda *args: TaskData(train=[batch()], validation=[batch()], ids={"test": [1]}))
    clock = [0.0]
    def monotonic():
        clock[0] += 1.0
        return clock[0]
    original_validate = training.validate
    def validate(*args):
        clock[0] += 100.0  # Deliberate validation overhead must enter wall throughput.
        return original_validate(*args)
    monkeypatch.setattr(training.time, "monotonic", monotonic)
    monkeypatch.setattr(training, "validate", validate)
    run = training.train(copy.deepcopy(config))
    run_metadata = json.loads((run / "run.json").read_text())
    assert set(run_metadata["source"]) == {"revision", "dirty"}
    assert run_metadata["training_budget"] == {"max_steps": 2, "schedule_steps": 2}
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    train_rows = [row for row in rows if row["phase"] == "train"]
    second = train_rows[1]
    assert second["interval_updates"] == 1
    assert second["interval_wall_seconds"] == second["elapsed_seconds"] - train_rows[0]["elapsed_seconds"]
    assert second["interval_wall_seconds"] > second["seconds_per_update"] + 100
    assert second["interval_wall_updates_per_second"] == pytest.approx(1 / second["interval_wall_seconds"])
    assert all(row["validation_seconds"] >= 100 for row in rows if row["phase"] == "validation")
    last = torch.load(run / "last.pt", weights_only=True)
    assert last["step"] == 2
    assert last["scheduler"]["last_epoch"] == 2
    assert all(value["step"] == 2 for value in last["optimizer"]["state"].values())
    assert "base" not in last and (run / "best.pt").is_file()
    restored = toy_model(stack)
    restored.load_communication_state(last)
    assert (last["memory_codec"] is not None) == (stack == "dec")
    hidden = torch.randn(2, 3, 8)
    assert torch.equal(restored.codec.encode(hidden), models[0].codec.encode(hidden))
    previous = (run / "step_000001.pt").read_bytes()
    resumed = training.train(copy.deepcopy(config), resume=run / "step_000001.pt")
    assert resumed != run
    assert (run / "step_000001.pt").read_bytes() == previous
    new_state = torch.load(resumed / "last.pt", weights_only=True)
    assert new_state["step"] == 2
    assert new_state["scheduler"]["last_epoch"] == 2
    resumed_rows = [json.loads(line) for line in (resumed / "metrics.jsonl").read_text().splitlines()]
    resumed_train = next(row for row in resumed_rows if row["phase"] == "train")
    assert resumed_train["interval_updates"] == 1  # Excludes the update from the earlier run.
