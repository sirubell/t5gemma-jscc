import copy
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
    run = training.train(copy.deepcopy(config))
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
