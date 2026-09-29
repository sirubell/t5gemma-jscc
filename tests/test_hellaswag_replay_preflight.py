"""Offline checks for the one-batch replay preflight."""

import copy

import pytest
import torch

from jscc.config import load_config
from jscc.models.channel import AWGNChannel
from jscc.models.codec import Codec
from jscc.models.split_model import SplitModel
from scripts import hellaswag_replay_preflight as preflight
from model_helpers import tiny_backbone


def _recipe():
    return {"task": "hellaswag", "seed": 0,
            "model": {"device": "cpu", "dtype": "float32"},
            "split": {"stack": "enc", "where": "after_layer", "index": 9},
            "codec": {"hidden_dim": 16, "bottleneck_dim": 8, "n_res_blocks": 1,
                      "activation": "gelu", "layernorm": "none", "snr_film": False},
            "channel": {"type": "awgn", "kwargs": {}, "normalize_power": True,
                        "train_noise": True, "train_snr_range": [-6.0, 18.0],
                        "clean_film_snr": 18.0},
            "training": {"batch_size": 2, "gradient_accumulation": 1,
                         "paired_randomness": {"policy": "step-microbatch-stream-v1", "seed": 0},
                         "kl_weight": 0.0, "mse_weight": 0.1}}


def _model(config):
    return SplitModel(tiny_backbone(num_hidden_layers=10),
                      Codec(16, config["codec"]), AWGNChannel(),
                      config["split"], config["channel"])


def _batch():
    return {"input_ids": torch.tensor([[3, 4, 5, 0], [6, 7, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
            "labels": torch.tensor([[8, 9, -100], [10, -100, -100]]),
            "row_ids": torch.tensor([1, 2])}


@torch.enable_grad()
def test_one_batch_online_capture_replay_parity():
    torch.manual_seed(7)
    config = _recipe()
    model = _model(config)
    report = preflight.compare_one_batch(model, _batch(), config)
    assert report["activation_shape"] == [2, 4, 16]
    assert set(report["conditions"]) == {"no_noise", "fixed_awgn"}
    for condition in report["conditions"].values():
        assert condition["passed"]
        assert condition["activation_equal"]
        assert condition["mask_equal"]
        assert condition["noise_draws_equal"]
        assert condition["valid_samples_online"] == condition["valid_samples_replay"] == 2
        assert condition["gradients"]
    assert all(parameter.grad is None for parameter in model.base.parameters())


@torch.enable_grad()
def test_bad_replay_result_is_reported_before_failure(monkeypatch):
    config = _recipe()
    model = _model(config)
    original = preflight.local_batch_values
    completed = {}

    def wrong_replay(*args, **kwargs):
        values = copy.copy(original(*args, **kwargs))
        values["nmse"] = values["nmse"] + 0.1
        return values

    monkeypatch.setattr(preflight, "local_batch_values", wrong_replay)
    with pytest.raises(AssertionError, match="parity failed"):
        preflight.compare_one_batch(model, _batch(), config,
            on_condition=lambda label, value: completed.update({label: value}))
    assert completed["no_noise"]["passed"] is False
    assert completed["no_noise"]["nmse_abs_error"] > 0.09


def test_reject_incomplete_or_rebatched_probe():
    config = _recipe()
    with pytest.raises(ValueError, match="complete declared microbatch"):
        preflight.compare_one_batch(_model(config),
            {key: value[:1] for key, value in _batch().items()}, config)


@pytest.mark.parametrize(("field", "key", "replacement"), [
    ("protocol", None, "other-protocol"),
    ("seed", None, 3),
    ("model", "name", "other-model"),
    ("model", "revision", "other-revision"),
    ("model", "dtype", "float32"),
    ("split", "index", 8),
    ("codec", "hidden_dim", 1024),
    ("codec", "bottleneck_dim", 256),
    ("channel", "normalize_power", False),
    ("data", "revision", "other-revision"),
    ("data", "prompt_policy", {"version": "other"}),
    ("evaluation", "num_fewshot", 0),
    ("training", "lr", 0.001),
    ("training", "batch_size", 16),
])
def test_scientific_baseline_drift_is_rejected(field, key, replacement):
    config = load_config(preflight.BASELINE_PATH)
    if key is None:
        config[field] = replacement
    else:
        config[field][key] = replacement
    with pytest.raises(ValueError):
        preflight.check_preflight_recipe(config)


def test_bounded_training_caps_keep_pinned_scientific_identity():
    config = load_config(preflight.BASELINE_PATH)
    training = config["training"]
    training.update(max_steps=2, schedule_steps=2, eval_every=2,
                    min_steps=2, save_steps=[2], selection_steps=[2],
                    validation_batches=1)
    training["presentation_stream"].update(total_presentations=128, start_presentation=0)
    training["paired_randomness"]["audit_steps"] = [1, 2]
    config["run"]["name"] = "bounded-probe"
    budget = preflight.check_preflight_recipe(config)
    assert budget["updates"] == 2
    assert budget["presentations"] == 128
    assert budget["scientific_identity_sha256"]
    assert budget["baseline_resolved_sha256"]
