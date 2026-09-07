"""Exercise every supplied split/bottleneck plan entry with a small CPU backbone."""
from pathlib import Path

import pytest
import torch

from jscc.models.channel import build_channel
from jscc.models.codec import Codec
from jscc.models.split_model import SplitModel
from jscc.studies import PlannedRun, expand_study
from jscc.training import batch_losses
from model_helpers import tiny_backbone


CONFIGS = Path(__file__).parents[1] / "configs/studies"
RUNS = [run for study in ("splits", "bottleneck")
        for run in expand_study(CONFIGS / f"{study}.yaml").runs]


@pytest.mark.parametrize("planned", RUNS, ids=lambda run: f"{run.task}-{run.experiment}")
def test_sweep_entry_forward_backward_and_generation(planned: PlannedRun):
    torch.manual_seed(7)
    config = planned.config
    # Preserve all selected layer indices and bottleneck widths. Hidden width,
    # vocabulary and image resolution are tiny: this is not a VRAM/quality test.
    base = tiny_backbone(num_hidden_layers=26)
    codec = Codec(16, {**config["codec"], "hidden_dim": 16})
    model = SplitModel(base, codec, build_channel(config["channel"]),
                       config["split"], config["channel"])
    input_ids = torch.tensor([[2, 31, 31, 31, 31, 4]]) if planned.task == "coco" else torch.tensor([[2, 4, 5]])
    batch = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids),
             "labels": torch.tensor([[5, 6, 1]])}
    if planned.task == "coco":
        batch["pixel_values"] = torch.randn(1, 1, 3, 8, 8)
    losses = batch_losses(model, batch, config["training"], 0.0)
    assert all(torch.isfinite(value).item() for value in losses.values())
    losses["loss"].backward()
    grad = codec.encoder[0].get_parameter("weight").grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in base.parameters())
    inputs = {key: value for key, value in batch.items() if key != "labels"}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].flatten(0, 1)
    model.eval()
    with torch.no_grad(), model.transmission(0.0):
        generated = model.generate(**inputs, max_new_tokens=2, decoder_start_token_id=0)
    assert generated.shape[0] == 1 and generated.shape[1] >= 2
