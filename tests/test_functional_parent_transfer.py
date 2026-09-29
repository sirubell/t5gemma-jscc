"""Real CPU terminal receipts and fresh-state functional reset control."""

import copy
import json
from pathlib import Path

import pytest
import torch

from jscc.activation_replay import file_digest
from jscc.config import load_config
from jscc.data import TaskData
from jscc.local_reconstruction import parent_recipe_sha256, validate_phase_transfer
from jscc.models.channel import AWGNChannel
from jscc.models.codec import Codec
from jscc.models.split_model import SplitModel
from jscc.presentation import PresentationSampler
from jscc.training import train
from model_helpers import tiny_backbone
from scripts.hellaswag_two_stage import check_plan, prepare_plan


@pytest.fixture
def functional_parent(tmp_path, monkeypatch):
    import jscc.training as training

    config = load_config(Path(__file__).parents[1] / "configs/tasks/hellaswag.yaml")
    config["model"].update(name="tiny", revision="fixed", device="cpu", dtype="float32")
    config["codec"].update(hidden_dim=16, bottleneck_dim=8, n_res_blocks=1)
    config["data"].update(name="fixture", revision="fixed", max_length=8, num_validation=1, num_train=2)
    config["data"].pop("prompt_policy", None)
    config["run"].update(name="e1-cpu", output_dir=str(tmp_path))
    config["training"].update(max_steps=2, schedule_steps=2, batch_size=2,
                              gradient_accumulation=1, eval_every=2, min_steps=2,
                              save_steps=[2], selection_steps=[2], patience=None,
                              feature_summary=None, log_every=1)
    config["training"]["presentation_stream"].update(total_presentations=4, start_presentation=0)
    ids = {"train_rows": [1, 2], "validation_rows": [3], "validation_split": "train"}

    def build(cfg):
        torch.manual_seed(cfg["seed"])
        model = SplitModel(tiny_backbone(num_hidden_layers=10), Codec(16, cfg["codec"]),
                           AWGNChannel(), cfg["split"], cfg["channel"])
        return None, model

    def batch():
        return {"input_ids": torch.tensor([[3, 4, 5, 0], [6, 7, 0, 0]]),
                "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
                "labels": torch.tensor([[8, 9, -100], [10, -100, -100]]),
                "row_ids": torch.tensor([1, 2])}

    class Batches:
        def __init__(self, cfg):
            stream = cfg["training"]["presentation_stream"]
            self.sampler = PresentationSampler(ids["train_rows"], stream["total_presentations"],
                                               stream["seed"], start=stream["start_presentation"])

        def __iter__(self):
            for rows in self.sampler.actual_ids.split(2):
                value = batch()
                value["row_ids"] = rows.clone()
                yield value

    def data(cfg, _processor, saved_ids=None):
        assert saved_ids is None or saved_ids == ids
        return TaskData(train=Batches(cfg), validation=[batch()], ids=ids)

    monkeypatch.setattr(training, "build_model", build)
    monkeypatch.setattr(training, "load_data", data)
    run = train(config)
    checkpoint = run / "step_000002.pt"
    phase2 = copy.deepcopy(config)
    phase2["run"]["name"] = "e-cpu"
    phase2["training"].update(max_steps=1, schedule_steps=1, eval_every=1, min_steps=1,
                              save_steps=[1], selection_steps=[1])
    phase2["training"]["presentation_stream"].update(total_presentations=2, start_presentation=4)
    phase2["training"]["phase_transfer"] = {
        "parent_kind": "functional", "parent_recipe_sha256": parent_recipe_sha256(config),
        "parent_step": 2, "parent_checkpoint_sha256": file_digest(checkpoint)}
    return checkpoint, phase2


def test_real_functional_terminal_loads_weights_and_resets_optimizer(functional_parent):
    checkpoint, config = functional_parent
    transfer = validate_phase_transfer(checkpoint, config)
    assert transfer["metadata"]["parent_kind"] == "functional"
    assert transfer["metadata"]["parent_checkpoint_sha256"] == file_digest(checkpoint)
    run = train(config, initial_checkpoint=checkpoint)
    initial = torch.load(run / "initial_communication.pt", weights_only=True)
    parent = torch.load(checkpoint, weights_only=True)
    for key, tensor in parent["codec"].items():
        assert torch.equal(initial["hidden"][key], tensor)
    state = torch.load(run / "step_000001.pt", weights_only=True)
    assert state["step"] == 1 and state["scheduler"]["last_epoch"] == 1
    assert all(int(value["step"]) == 1 for value in state["optimizer"]["state"].values())
    for field in ("optimizer_state_loaded", "scheduler_state_loaded", "scaler_state_loaded"):
        assert state["phase_transfer"][field] is False
    assert json.loads((run / "completion.json").read_text())["status"] == "FULL_BUDGET_COMPLETED"


@pytest.mark.parametrize("mutation,match", [
    ("digest", "SHA-256"), ("kind", "parent_kind"), ("recipe", "parent recipe"),
    ("stream", "exact presentations"), ("partial", "partial"),
    ("source", "source differs"), ("data", "data IDs"), ("config", "recipe receipt"),
    ("missing", "lacks"), ("local", "completed B"),
])
def test_functional_transfer_rejects_unbound_or_changed_evidence(functional_parent, mutation, match):
    checkpoint, config = functional_parent
    transfer = config["training"]["phase_transfer"]
    if mutation == "digest":
        transfer["parent_checkpoint_sha256"] = "0" * 64
    elif mutation == "kind":
        transfer["parent_kind"] = "invented"
    elif mutation == "recipe":
        transfer["parent_recipe_sha256"] = "0" * 64
    elif mutation == "stream":
        config["training"]["presentation_stream"]["start_presentation"] = 0
    elif mutation == "local":
        transfer.pop("parent_kind")
    elif mutation == "missing":
        checkpoint.with_name("completion.json").unlink()
    elif mutation == "config":
        with checkpoint.with_name("config.yaml").open("a") as stream:
            stream.write("\n# altered\n")
    else:
        name = {"partial": "completion.json", "source": "run.json", "data": "data_ids.json"}[mutation]
        path = checkpoint.with_name(name)
        value = json.loads(path.read_text())
        if mutation == "partial":
            value["status"] = "PARTIAL"
        elif mutation == "source":
            value["training_source_sha256"] = "0" * 64
        else:
            value["train_rows"] = [9, 10]
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match=match):
        validate_phase_transfer(checkpoint, config)


def test_optional_reset_plan_matches_phase_recipes(tmp_path):
    import yaml

    base = str(Path(__file__).parents[1] / "configs/tasks/hellaswag.yaml")
    result = prepare_plan(base, tmp_path / "plan", total_updates=200, local_updates=160,
                          functional_updates=40, include_reset_control=True)
    assert set(result["recipes"]) == {"A", "B", "C", "D", "E1", "E"}
    assert result["arms"]["E"] == result["arms"]["C"]
    assert result["arms"]["E1"] == result["arms"]["B"]
    path = Path(result["recipes"]["E"])
    config = load_config(path)
    config["training"]["lr"] *= 2
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="C phase-two"):
        check_plan(result["recipes"])
