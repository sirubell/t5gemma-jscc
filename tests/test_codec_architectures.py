"""Explicit architecture semantics, expansion, and saved-state compatibility."""
from pathlib import Path

import pytest
import torch
from torch import nn
import yaml

from jscc.config import load_config, resolve_codec_configs, validate_config
from jscc.models.codec import Codec


def settings(architecture="residual_mlp", blocks=2, bottleneck=8):
    return {"architecture": architecture, "hidden_dim": 16,
            "bottleneck_dim": bottleneck, "n_res_blocks": blocks,
            "activation": "gelu", "dropout": 0.0, "layernorm": "none",
            "snr_film": False, "film_hidden": 4}


@pytest.mark.parametrize("architecture,blocks", [
    ("residual_mlp", 2), ("residual_mlp", 0), ("direct_affine", 0),
])
@pytest.mark.parametrize("bottleneck", [256, 512, 1152, 2304])
def test_compression_and_expansion_forward_backward_and_state(architecture, blocks, bottleneck):
    config = settings(architecture, blocks, bottleneck)
    codec = Codec(1152, config)
    hidden = torch.randn(2, 3, 1152)
    encoded = codec.encode(hidden)
    assert encoded.shape == (2, 3, bottleneck)
    reconstructed = codec.decode(encoded, -6.0)
    assert reconstructed.shape == hidden.shape
    reconstructed.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in codec.parameters())
    restored = Codec(1152, config)
    restored.load_state_dict(codec.state_dict(), strict=True)
    torch.testing.assert_close(restored.decode(restored.encode(hidden), 18.0),
                               reconstructed, rtol=0, atol=0)


def test_direct_affine_is_one_biased_linear_per_half():
    codec = Codec(16, settings("direct_affine", 0))
    assert len(codec.encoder) == len(codec.decoder) == 1
    encoder, decoder = codec.encoder[0], codec.decoder[0]
    assert isinstance(encoder, nn.Linear)
    assert isinstance(decoder, nn.Linear)
    assert encoder.bias is not None and decoder.bias is not None
    assert not any(isinstance(module, nn.LayerNorm) for module in codec.modules())
    assert codec.film is None
    x = torch.randn(2, 3, 16)
    encoded = torch.nn.functional.linear(x, encoder.weight, encoder.bias)
    expected = torch.nn.functional.linear(encoded, decoder.weight, decoder.bias)
    torch.testing.assert_close(codec.decode(codec.encode(x), None), expected, rtol=0, atol=0)


def test_zero_block_is_factorized_affine_with_no_internal_normalization():
    codec = Codec(16, settings(blocks=0))
    assert len(codec.encoder) == len(codec.decoder) == 2
    assert all(isinstance(layer, nn.Linear) for layer in [*codec.encoder, *codec.decoder])
    assert not any(isinstance(module, nn.LayerNorm) for module in codec.modules())
    x = torch.randn(2, 3, 16)
    # Affine halves preserve midpoint interpolation, including their biases.
    y = torch.randn_like(x)
    torch.testing.assert_close(codec.encode((x + y) / 2),
                               (codec.encode(x) + codec.encode(y)) / 2)


@pytest.mark.parametrize("bottleneck", [256, 512, 1152, 2304])
def test_native_dimension_direct_parameter_count(bottleneck):
    with torch.device("meta"):
        codec = Codec(1152, settings("direct_affine", 0, bottleneck))
    assert sum(p.numel() for p in codec.parameters()) == 2 * 1152 * bottleneck + bottleneck + 1152


@pytest.mark.parametrize("blocks", [0, 2])
@pytest.mark.parametrize("film", [False, True])
def test_omitted_architecture_preserves_residual_checkpoint_keys_and_initialization(blocks, film):
    explicit = {**settings(blocks=blocks), "snr_film": film, "layernorm": "both"}
    historical = {key: value for key, value in explicit.items() if key != "architecture"}
    torch.manual_seed(713)
    old = Codec(16, historical)
    torch.manual_seed(713)
    new = Codec(16, explicit)
    # Historical Sequential indices are checkpoint format, not just naming.
    expected_keys = {f"{norm}.{kind}" for norm in ("input_norm", "output_norm")
                     for kind in ("weight", "bias")}
    for half in ("encoder", "decoder"):
        expected_keys.update(f"{half}.{index}.{kind}"
                             for index in (0, blocks + 1) for kind in ("weight", "bias"))
        expected_keys.update(f"{half}.{index}.{part}.{kind}"
                             for index in range(1, blocks + 1)
                             for part in ("norm", "fc1", "fc2") for kind in ("weight", "bias"))
    if film:
        expected_keys.update(f"film.{index}.{kind}"
                             for index in (0, 2) for kind in ("weight", "bias"))
    assert old.state_dict().keys() == new.state_dict().keys() == expected_keys
    for key, tensor in old.state_dict().items():
        torch.testing.assert_close(tensor, new.state_dict()[key], rtol=0, atol=0)
    new.load_state_dict(old.state_dict(), strict=True)
    hidden = torch.randn(2, 3, 16)
    torch.testing.assert_close(old.decode(old.encode(hidden), 6.0),
                               new.decode(new.encode(hidden), 6.0), rtol=0, atol=0)
    assert "architecture" not in historical


@pytest.mark.parametrize("override,match", [
    ({"architecture": "typo"}, "codec.architecture"),
    ({"layernorm": "both"}, "codec.layernorm"),
    ({"snr_film": True}, "codec.snr_film"),
    ({"n_res_blocks": 2}, "codec.n_res_blocks"),
    ({"dropout": 0.1}, "codec.dropout"),
])
def test_invalid_direct_designs_fail_in_constructor_and_config_validation(override, match):
    config = settings("direct_affine", 0) | override
    with pytest.raises(ValueError, match=match):
        Codec(16, config)
    task = load_config(Path(__file__).parents[1] / "configs/tasks/hellaswag.yaml")
    task["codec"] = config
    with pytest.raises(ValueError, match=match):
        validate_config(task)


def test_direct_memory_overrides_are_validated_before_model_construction():
    task = load_config(Path(__file__).parents[1] / "configs/tasks/hellaswag.yaml")
    task["codec"] = {**settings("direct_affine", 0), "memory": {"layernorm": "both"}}
    with pytest.raises(ValueError, match="codec.layernorm"):
        validate_config(task)


@pytest.mark.parametrize("name", ["residual", "zero_block", "direct_affine"])
@pytest.mark.parametrize("task", ["coco", "hellaswag"])
def test_named_designs_compose_for_both_tasks(name, task, tmp_path):
    root = Path(__file__).parents[1] / "configs"
    recipe = yaml.safe_load((root / "tasks" / f"{task}.yaml").read_text())
    recipe["model_config"] = str(root / "models" / f"enc_l9_{name}.yaml")
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(recipe))
    resolved = load_config(path)
    main, memory = resolve_codec_configs(resolved["codec"])
    for codec_config in (main, memory):
        with torch.device("meta"):
            codec = Codec(1152, codec_config)
        assert sum(p.numel() for p in codec.parameters()) > 0
    assert resolved["split"] == {"stack": "enc", "where": "after_layer", "index": 9}
    assert resolved["channel"]["normalize_power"] is True
