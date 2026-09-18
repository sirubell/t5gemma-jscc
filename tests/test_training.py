import copy
import json
from pathlib import Path

import torch
import pytest

from jscc.config import load_config
from jscc.data import TaskData
from jscc import training
from jscc.models.channel import IdentityChannel
from jscc.runtime import prepare_trainable_parameters, precision_telemetry, promote_optimizer_state
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


def test_trainable_communication_state_is_fp32_with_bf16_backbone():
    torch.manual_seed(12)
    model = toy_model("enc")
    model.to(dtype=torch.bfloat16)
    prepare_trainable_parameters(model)
    assert all(parameter.dtype == torch.bfloat16 for parameter in model.base.parameters())
    assert all(parameter.dtype == torch.float32 for parameter in model.codec.parameters())

    optimizer = torch.optim.AdamW(model.codec.parameters(), lr=1e-3)
    values = training.batch_losses(
        model, batch(), {"temperature": 1.0, "kl_weight": 1.0, "mse_weight": 0.1}, None
    )
    values["loss"].backward()
    optimizer.step()
    promote_optimizer_state(optimizer)
    assert all(value.dtype == torch.float32
               for state in optimizer.state.values()
               for value in state.values()
               if torch.is_tensor(value) and value.is_floating_point())
    telemetry = precision_telemetry(model, optimizer)
    assert telemetry["trainable_parameters"] == {"float32": len(list(model.codec.parameters()))}
    assert telemetry["frozen_parameters"] == {"bfloat16": len(list(model.base.parameters()))}
    # AdamW stores step, exp_avg and exp_avg_sq for each parameter.
    assert telemetry["optimizer_state"] == {"float32": 3 * len(list(model.codec.parameters()))}


def test_checkpoint_reload_keeps_communication_dtype_and_values():
    torch.manual_seed(13)
    model = toy_model("enc")
    model.to(dtype=torch.bfloat16)
    prepare_trainable_parameters(model)
    state = {"codec": {key: value.detach().clone() for key, value in model.codec.state_dict().items()},
             "channel": model.channel.state_dict(), "memory_codec": None}

    restored = toy_model("enc")
    restored.to(dtype=torch.bfloat16)
    prepare_trainable_parameters(restored)
    restored.load_communication_state(state)
    assert all(parameter.dtype == torch.float32 for parameter in restored.codec.parameters())
    for key, value in state["codec"].items():
        torch.testing.assert_close(restored.codec.state_dict()[key], value)


def test_effective_batch_objective_matches_concatenated_batch():
    settings = {"temperature": 1.0, "kl_weight": 1.0, "mse_weight": 0.1}
    first = {"input_ids": torch.tensor([[1, 2, 3]]),
             "attention_mask": torch.tensor([[1, 1, 1]]),
             "labels": torch.tensor([[6, 7, -100]])}
    second = {"input_ids": torch.tensor([[4, 5, 0]]),
              "attention_mask": torch.tensor([[1, 1, 0]]),
              "labels": torch.tensor([[9, 10, 11]])}

    torch.manual_seed(14)
    accumulated = toy_model("enc", channel=IdentityChannel()).train()
    concatenated = toy_model("enc", channel=IdentityChannel()).train()
    concatenated.load_state_dict(accumulated.state_dict())
    values = [training.batch_losses(accumulated, item, settings, None, return_stats=True)
              for item in (first, second)]
    training.aggregate_batch_losses(values, settings)["loss"].backward()
    accumulated_gradients = []
    for parameter in accumulated.codec.parameters():
        assert parameter.grad is not None
        accumulated_gradients.append(parameter.grad.detach().clone())

    combined = {key: torch.cat([first[key], second[key]], dim=0)
                for key in ("input_ids", "attention_mask", "labels")}
    training.batch_losses(concatenated, combined, settings, None)["loss"].backward()
    concatenated_gradients = []
    for parameter in concatenated.codec.parameters():
        assert parameter.grad is not None
        concatenated_gradients.append(parameter.grad.detach().clone())
    for actual, expected in zip(accumulated_gradients, concatenated_gradients):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
