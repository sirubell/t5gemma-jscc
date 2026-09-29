import copy
from pathlib import Path

import pytest
import torch

from jscc import training
from jscc.config import load_config, validate_config
from scripts.experiments.main_weight_screen_train import screen_config
from test_core import batch, toy_model


@pytest.mark.parametrize("hidden_weight", [.05, .5, 5.])
def test_explicit_stream_weights_preserve_pooled_denominators_and_gradients(hidden_weight):
    settings = {"temperature": 1., "kl_weight": 1., "mse_weight": .1,
                "loss_weights": {"kl": 1., "hidden": hidden_weight, "memory": .05}}
    model = toy_model("dec")
    parts = [batch(), batch()]
    parts[1]["labels"][0] = -100  # differing valid target/sample denominator
    params = [p for p in model.parameters() if p.requires_grad]
    values = [training.batch_losses(model, b, settings, None, return_stats=True) for b in parts]
    pooled = training.aggregate_batch_losses(values, settings)
    assert torch.allclose(pooled["loss"], pooled["kl"] + hidden_weight*pooled["hidden"] + .05*pooled["memory"])
    expected = torch.autograd.grad(pooled["loss"], params)
    model.zero_grad(set_to_none=True)
    denominators = training.effective_batch_denominators(model, parts, "cpu")
    for b in parts:
        value = training.batch_losses(model, b, settings, None, return_stats=True)
        training.scaled_batch_loss(value, settings, denominators).backward()
    for parameter, reference in zip(params, expected):
        torch.testing.assert_close(parameter.grad, reference, atol=2e-6, rtol=2e-5)


def test_w005_matches_original_objective_and_both_codec_gradients():
    model = toy_model("dec")
    legacy = {"temperature": 1., "kl_weight": 1., "mse_weight": .1}
    values = training.batch_losses(model, batch(), legacy, None, return_stats=True)
    original = values["kl"] + .1 * values["nmse"]
    params = [p for p in model.parameters() if p.requires_grad]
    expected = torch.autograd.grad(original, params, retain_graph=True)
    explicit = {**legacy, "loss_weights": {"kl": 1., "hidden": .05, "memory": .05}}
    actual = training.aggregate_batch_losses([values], explicit)
    torch.testing.assert_close(actual["loss"], original)
    grads = torch.autograd.grad(actual["loss"], params)
    for result, reference in zip(grads, expected):
        torch.testing.assert_close(result, reference)


def test_explicit_weights_not_renormalized_for_missing_microbatch_memory():
    parameter = torch.tensor(2., requires_grad=True)
    item: training.BatchValues = {"loss": parameter, "kl": parameter, "nmse": parameter,
            "kl_numerator": parameter, "hidden_numerator": parameter,
            "kl_denominator": torch.tensor(1.), "hidden_denominator": torch.tensor(1.),
            "memory_numerator": None, "memory_denominator": None}
    settings = {"loss_weights": {"kl": 1., "hidden": .05, "memory": .05}}
    actual = training.scaled_batch_loss(item, settings,
        {"kl": torch.tensor(2.), "hidden": torch.tensor(2.), "memory": torch.tensor(2.), "stream_count": 2})
    torch.testing.assert_close(actual, parameter / 2 * 1.05)


@pytest.mark.parametrize("bad", [{"hidden": .5}, {"kl": 1., "hidden": float("nan"), "memory": .05},
                                {"kl": 1., "hidden": -1., "memory": .05}])
def test_invalid_explicit_weight_schema_rejected(bad):
    cfg = load_config(Path(__file__).parents[1] / "configs/tasks/hellaswag.yaml")
    cfg["training"]["loss_weights"] = bad
    with pytest.raises(ValueError, match="loss_weights"):
        validate_config(cfg)


def test_screen_recipe_schedule_and_preflight():
    path = Path(__file__).parents[1] / "tests/fixtures/main_weight_screen/base-dec_l20.yaml"
    base = load_config(path)
    before = copy.deepcopy(base)
    config = screen_config(base, Path("/tmp/screen"), "W5")
    assert base == before
    assert config["training"]["max_steps"] == 1000
    assert config["training"]["schedule_steps"] == 10000
    assert config["training"]["warmup_ratio"] == .05
    assert config["training"]["selection_steps"] == [250, 500, 1000]
    assert config["training"]["loss_weights"] == {"kl": 1., "hidden": 5., "memory": .05}
    preflight = screen_config(base, Path("/tmp/preflight"), "W005", True)
    assert preflight["training"]["max_steps"] == 2
    assert preflight["training"]["presentation_stream"]["total_presentations"] == 128


def test_discarded_preflight_executes_two_nonzero_updates_and_fp32_adam(tmp_path, monkeypatch):
    import json
    from torch.utils.data import DataLoader, Dataset
    from jscc.data import TaskData
    from scripts.experiments import main_weight_screen_train as runner

    cfg = load_config(Path(__file__).parents[1] / "configs/smoke/hellaswag_cpu.yaml")
    cfg["run"].update(output_dir=str(tmp_path / "out" / "runs"), name="cpu-preflight", wandb_project=None)
    cfg["model"].update(device="cpu", dtype="float32")
    cfg["training"].update(max_steps=2, schedule_steps=10000, batch_size=64, gradient_accumulation=1,
        save_steps=[2], eval_every=100, streamed_backward=True, valid_only_kl=True,
        loss_weights={"kl": 1., "hidden": .5, "memory": .05}, patience=None)
    monkeypatch.setattr(runner, "screen_config", lambda *args: copy.deepcopy(cfg))
    monkeypatch.setattr(training, "build_model", lambda cfg: (None, toy_model("dec")))
    samples = [{k: v[0].clone() for k, v in batch().items()} for _ in range(512)]
    train_samples = [{**samples[0], "row_ids": torch.tensor(i+1000)} for i in range(128)]
    class Rows(Dataset):
        def __init__(self, rows):
            self.rows = rows
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, index):
            return self.rows[index]
    data = TaskData(train=DataLoader(Rows(train_samples), batch_size=64),
                    validation=DataLoader(Rows(samples), batch_size=64),
                    ids={"validation_split": "train", "validation_rows": list(range(512)),
                         "train_rows": list(range(1000, 1128))})
    monkeypatch.setattr(training, "load_data", lambda *args: data)
    report = runner.run_screen(cfg, tmp_path / "out", "W05", preflight=True, seconds=60)
    assert report["optimizer_calls"] == 2
    assert report["selection_presentations"] == 0
    assert report["training_presentations"] == 128
    rows = [json.loads(line) for line in (tmp_path / "out/preflight_updates.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and all(row["lr"] > 0 for row in rows)
    assert all(row["adam_states_fp32_finite"] for row in rows)
    assert (tmp_path / "out/init.pt").is_file()


def test_frozen_e3_order_uses_second_shuffle_and_rejects_wrong_membership():
    from scripts.experiments.evening_gradients import selection_ids, validate_fixture
    from scripts.experiments.main_weight_screen_train import frozen_selection_order
    ids = {"validation_split": "train", "validation_rows": list(range(512)),
           "train_rows": list(range(1000, 1128))}
    first_order = selection_ids(ids)
    frozen = {"rows": validate_fixture([{"row_id": i} for i in first_order], ids)}
    actual_order = [r["row_id"] for r in frozen["rows"]]
    assert actual_order != first_order  # Reproduces the faulty preflight guard.
    assert set(actual_order) == set(first_order)
    assert frozen_selection_order(ids, frozen) == actual_order
    broken = copy.deepcopy(frozen)
    broken["rows"][0]["row_id"] = 99999
    with pytest.raises(ValueError, match="selection128"):
        frozen_selection_order(ids, broken)


def test_actual_frozen_e3_fixture_order_regression():
    import json
    from scripts.experiments.evening_gradients import selection_ids
    from scripts.experiments.main_weight_screen_train import frozen_selection_order
    root = Path(__file__).parents[1] / "tests/fixtures/main_weight_screen"
    fixture = json.loads((root / "fixture.json").read_text())
    ids = json.loads((root / "saved-data-ids.json").read_text())
    expected = [r["row_id"] for r in fixture["rows"]]
    assert selection_ids(ids) != expected
    assert set(selection_ids(ids)) == set(expected)
    assert frozen_selection_order(ids, fixture) == expected
