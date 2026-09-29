"""Scoped SDPA policy: CPU dispatch flags, restoration and construction guards."""
from pathlib import Path

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from jscc.config import load_config, validate_config
from jscc.models.split_model import build_model
from test_coco import coco_model, image_batch


def flags():
    return (torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.math_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled(), torch.backends.cuda.cudnn_sdp_enabled())


@pytest.mark.parametrize("method", ["forward", "generate"])
@pytest.mark.parametrize("raises", [False, True])
def test_calls_scope_and_restore_policy(monkeypatch, method, raises):
    model = coco_model()
    model.sdpa_backend_policy = "flash_math"
    seen = []

    def operation(**kwargs):
        seen.append(flags())
        if raises:
            raise RuntimeError("sentinel")
        return "result"

    monkeypatch.setattr(model.base, method, operation)
    with sdpa_kernel(SDPBackend.MATH):
        before = flags()
        if raises:
            with pytest.raises(RuntimeError, match="sentinel"):
                getattr(model, method)()
        else:
            assert getattr(model, method)() == "result"
        assert flags() == before
    assert seen == [(True, True, False, False)]


def test_auto_preserves_outer_policy_and_direct_base_requires_context(monkeypatch):
    model = coco_model()
    assert model.sdpa_backend_policy == "auto"
    monkeypatch.setattr(model.base, "forward", lambda **kwargs: flags())
    with sdpa_kernel(SDPBackend.MATH):
        expected = flags()
        assert model() == expected
        model.sdpa_backend_policy = "flash_math"
        assert model.base() == expected
        with model.attention_context():
            assert model.base() == (True, True, False, False)
        assert flags() == expected


@torch.no_grad()
def test_real_generation_encoder_pass_is_inside_policy():
    model = coco_model()
    model.sdpa_backend_policy = "flash_math"
    seen = []
    handle = model.base.get_encoder().register_forward_pre_hook(
        lambda module, args: seen.append(flags()))
    try:
        with sdpa_kernel(SDPBackend.MATH):
            before = flags()
            output = model.generate(**image_batch(), max_new_tokens=2, do_sample=False)
            assert flags() == before
        assert output.shape[0] == 1
        assert seen and all(state == (True, True, False, False) for state in seen)
    finally:
        handle.remove()


@pytest.mark.parametrize("policy", ["bad", None, 1, []])
def test_bad_policy_rejected_before_loading(policy):
    config = load_config(Path(__file__).parents[1] / "configs/tasks/coco.yaml")
    config["model"]["sdpa_backend_policy"] = policy
    with pytest.raises(ValueError, match="sdpa_backend_policy"):
        validate_config(config)
    # Only a model section: any attempt to load or inspect task/model name is a bug.
    with pytest.raises(ValueError, match="sdpa_backend_policy"):
        build_model({"model": {"sdpa_backend_policy": policy}})


def test_coco_runtime_policy_does_not_change_hellaswag():
    recipes = Path(__file__).parents[1] / "configs/tasks"
    assert load_config(recipes / "coco.yaml")["model"]["sdpa_backend_policy"] == "flash_math"
    other = load_config(recipes / "hellaswag.yaml")
    assert other["model"].get("sdpa_backend_policy", "auto") == "auto"
    other["model"]["sdpa_backend_policy"] = "flash_math"
    with pytest.raises(ValueError, match="only for task=coco"):
        validate_config(other)
    with pytest.raises(ValueError, match="only for task=coco"):
        build_model(other)
