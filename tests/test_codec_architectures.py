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


@pytest.mark.parametrize("architecture,blocks", [("residual_mlp", 2), ("direct_affine", 0)])
def test_four_cells_preserve_equal_core_initialization(architecture, blocks):
    plain = settings(architecture, blocks)
    normalized = {**plain, "layernorm": "both"}
    if architecture == "direct_affine":
        normalized["architecture"] = "direct_outer_ln"
    torch.manual_seed(91)
    left = Codec(16, plain)
    torch.manual_seed(91)
    right = Codec(16, normalized)
    for key, value in left.state_dict().items():
        torch.testing.assert_close(value, right.state_dict()[key], rtol=0, atol=0)
    assert isinstance(right.input_norm, nn.LayerNorm)
    assert isinstance(right.output_norm, nn.LayerNorm)
    assert sum(isinstance(m, nn.LayerNorm) for m in right.modules()) == 2 + 2 * blocks
    x = torch.randn(2, 3, 16)
    right.decode(right.encode(x), None).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in right.parameters())


def routed_model():
    from jscc.models.channel import build_channel
    from jscc.models.split_model import SplitModel
    from model_helpers import tiny_backbone
    base = tiny_backbone(num_hidden_layers=2)
    config = {"type": "identity", "kwargs": {}, "normalize_power": False, "clean_film_snr": 18.0}
    return SplitModel(base, Codec(16, settings("direct_affine", 0)), build_channel(config),
                      {"stack": "enc", "where": "after_layer", "index": 0}, config)


def test_encoder_routing_is_post_final_norm_single_and_exception_safe():
    from jscc.models.split_model import resolve_encoder_site, stack_module
    model = routed_model()
    site = resolve_encoder_site(model.base, {"stack": "enc", "where": "after_final_norm"}, "tiny-v1")
    authorization = {"sites": {"enc_fn": "trained"}}
    stack = stack_module(model.base, "enc")
    raw = []
    handle = stack.norm.register_forward_pre_hook(lambda module, args: raw.append(args[0].detach()))
    batch = {"input_ids": torch.tensor([[2, 4, 5]]), "attention_mask": torch.ones(1, 3, dtype=torch.long),
             "labels": torch.tensor([[5, 6, 1]])}
    original_split = model.split
    codec = model.codec
    with pytest.raises(RuntimeError, match="deliberate"):
        with model.at_site(site, purpose="train", authorization=authorization), model.transmission():
            output = model(**batch)
            torch.testing.assert_close(model.activation, stack.norm.forward(raw[-1]))
            assert model.channel_uses["hidden"] == 1 * 3 * 8
            output.logits.square().mean().backward()
            assert model.codec is codec
            assert model.codec.encoder[0].weight.grad is not None
            assert all(p.grad is None for p in model.base.parameters())
            with pytest.raises(RuntimeError, match="nest"):
                with model.at_site(site, purpose="capture", authorization=authorization):
                    pass
            raise RuntimeError("deliberate")
    handle.remove()
    assert model.split is original_split
    assert model._encoder_valid_mask is None
    with model.transmission():
        model(**batch)
        assert model.channel_uses["hidden"] == 24
    assert len(stack.norm._forward_hooks) == 0
    assert len(stack.layers[0]._forward_hooks) == 1


@pytest.mark.parametrize("role,purpose,match", [
    ("diagnostic", "train", "trained"), ("heldout", "capture", "freeze"),
    (None, "evaluate", "declared"),
])
def test_encoder_routing_rejects_undeclared_or_unfrozen_roles(role, purpose, match):
    from jscc.models.split_model import resolve_encoder_site
    model = routed_model()
    site = resolve_encoder_site(model.base, {"stack": "enc", "where": "after_layer", "index": 1}, "tiny-v1")
    with pytest.raises(ValueError, match=match):
        with model.at_site(site, purpose=purpose, authorization={"sites": {site.site_id: role}}):
            pass


@pytest.mark.parametrize("route", ["encoder", "forward", "generate"])
def test_route_cannot_change_during_direct_encoder_forward_and_recovers_exception(route):
    from jscc.models.split_model import resolve_encoder_site, stack_module
    model = routed_model()
    encoder = stack_module(model.base, "enc")
    site = resolve_encoder_site(model.base, {"stack": "enc", "where": "after_final_norm"}, "tiny-v1")
    authorization = {"sites": {site.site_id: "trained"}}

    def fail_inside(module, args):
        with model.at_site(site, purpose="capture", authorization=authorization):
            pass

    def run():
        inputs = {"input_ids": torch.tensor([[2, 4, 5]])}
        if route == "encoder":
            return encoder(**inputs)
        if route == "forward":
            return model(**inputs, labels=torch.tensor([[5, 6, 1]]))
        return model.generate(**inputs, max_new_tokens=2, decoder_start_token_id=0)

    assert not model.base._forward_hooks and not model.base._forward_pre_hooks
    hook = encoder.layers[0].register_forward_pre_hook(fail_inside)
    with pytest.raises(RuntimeError, match="during a forward"):
        run()
    hook.remove()
    assert model._forward_active == 0
    with model.at_site(site, purpose="capture", authorization=authorization), model.transmission(bypass=True):
        run()
    assert model._forward_active == 0
    assert not model.base._forward_hooks and not model.base._forward_pre_hooks


def test_resolved_sites_distinguish_raw_final_and_normalized_and_reject_forgery():
    from dataclasses import replace
    from jscc.models.split_model import resolve_encoder_site
    model = routed_model()
    raw = resolve_encoder_site(model.base, {"stack": "enc", "where": "after_layer", "index": 1}, "tiny-v1")
    final = resolve_encoder_site(model.base, {"stack": "enc", "where": "after_final_norm"}, "tiny-v1")
    assert raw.module_path != final.module_path
    assert raw.site_id == "enc_l1" and final.site_id == "enc_fn"
    for spec in ({"stack": "enc", "where": "after_layer", "index": 2},
                 {"stack": "enc", "where": "after_layer", "index": True},
                 {"stack": "enc", "where": "after_final_norm", "index": 1},
                 {"stack": "dec", "where": "after_final_norm"}):
        with pytest.raises(ValueError):
            resolve_encoder_site(model.base, spec, "tiny-v1")
    with pytest.raises(ValueError, match="metadata"):
        with model.at_site(replace(final, hidden_dim=32), purpose="capture",
                           authorization={"sites": {"enc_fn": "trained"}}):
            pass


@pytest.mark.parametrize("task", ["coco", "hellaswag"])
def test_direct_outer_ln_named_design_composes_without_promoting_default(task, tmp_path):
    root = Path(__file__).parents[1] / "configs"
    recipe = yaml.safe_load((root / "tasks" / f"{task}.yaml").read_text())
    recipe["model_config"] = str(root / "models/codec_direct_outer_ln.yaml")
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(recipe))
    config = load_config(path)
    assert config["split"] == {"stack": "enc", "where": "after_final_norm"}
    codec = Codec(16, config["codec"])
    restored = Codec(16, config["codec"])
    restored.load_state_dict(codec.state_dict(), strict=True)
    x = torch.randn(2, 3, 16)
    expected = codec.output_norm(codec.decoder(codec.encoder(codec.input_norm(x))))
    torch.testing.assert_close(restored.decode(restored.encode(x), None), expected, rtol=0, atol=0)
    assert load_config(root / "tasks" / f"{task}.yaml")["codec"].get("architecture", "residual_mlp") == "residual_mlp"
