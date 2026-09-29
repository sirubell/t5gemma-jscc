"""CPU-only E5 family wiring and replay controls; no training-quality claims."""
import copy

import pytest
import torch
from torch import nn

from jscc.models import split_model
from jscc.models.channel import AWGNChannel, normalize_power
from jscc.models.codec import Codec, ResidualBlock
from jscc.presentation import PresentationSampler, tensor_digest
from scripts.experiments.evening_pilot import (EXPECTED_PARAMETERS, FAMILIES, codec_factory,
    codec_family_context, pilot_config, projection_roles, selection_indices)


def codec_config(width=8, bottleneck=3):
    return dict(hidden_dim=width, bottleneck_dim=bottleneck, layernorm="none",
                snr_film=False, dropout=0., n_res_blocks=2, activation="gelu", film_hidden=4)


@pytest.mark.parametrize("family", FAMILIES)
def test_exact_full_size_parameter_counts(family):
    codec = codec_factory(1152, codec_config(1152, 512), family)
    assert sum(p.numel() for p in codec.parameters()) == EXPECTED_PARAMETERS[family]
    assert codec.film is None
    assert isinstance(codec.input_norm, nn.Identity)
    assert isinstance(codec.output_norm, nn.Identity)


def test_same_role_projections_and_independent_global_rng():
    torch.manual_seed(73)
    before = torch.get_rng_state().clone()
    families = {f: codec_factory(8, codec_config(), f) for f in FAMILIES}
    assert torch.equal(before, torch.get_rng_state())
    shared = {}
    for family, codec in families.items():
        for role, module in projection_roles(codec, family).items():
            key = role, tuple(module.weight.shape)
            hashes = {k: tensor_digest(v) for k, v in module.state_dict().items()}
            if key in shared:
                assert hashes == shared[key]
            shared[key] = hashes
    # At D=H all three encoder/decoder bottleneck projections can match.
    assert len(shared) == 4


def test_shallow_has_gelu_and_no_ln_or_skip_and_residual_is_production():
    affine = codec_factory(8, codec_config(), "affine_core")
    shallow = codec_factory(8, codec_config(), "shallow_nonlinear")
    residual = codec_factory(8, codec_config(), "current_residual")
    assert isinstance(residual, Codec)
    assert sum(isinstance(m, nn.LayerNorm) for m in residual.modules()) == 4
    assert sum(isinstance(m, nn.LayerNorm) for m in shallow.modules()) == 0
    assert sum(isinstance(m, nn.GELU) for m in shallow.modules()) == 2
    assert len(affine.encoder) == len(affine.decoder) == 1
    for trunk in (residual.encoder, residual.decoder):
        for block in list(trunk.children())[1:-1]:
            assert isinstance(block, ResidualBlock)
            assert torch.equal(block.fc2.bias, torch.zeros_like(block.fc2.bias))


def test_context_restores_production_factory_even_on_error():
    original = split_model.Codec
    with pytest.raises(RuntimeError), codec_family_context("affine_core"):
        assert type(split_model.Codec(8, codec_config())).__name__ == "SimpleCodec"
        raise RuntimeError("abort")
    assert split_model.Codec is original


def test_matching_data_snr_epsilon_with_different_parameter_counts():
    replay = []
    for family in FAMILIES:
        codec = codec_factory(8, codec_config(), family)
        ids = PresentationSampler(list(range(80)), 128, seed=0).actual_ids
        channel = AWGNChannel()
        hidden = torch.arange(64 * 5 * 8).reshape(64, 5, 8).float() / 100
        mask = torch.ones((64, 5), dtype=torch.bool)
        z = normalize_power(codec.encode(hidden), mask)
        with channel.replay(0, "train:1:0", capture=True):
            out = channel.transmit(z, torch.zeros((64, 1, 1)), stream="hidden")
        assert out.shape == (64, 5, 3)
        codec.decode(out, None).sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in codec.parameters())
        replay.append((ids.tolist(), channel.draw_summaries))
    assert replay[0] == replay[1] == replay[2]


def base_config():
    return {"codec": codec_config(1152, 512), "protocol": "corrected-baseline-v2-native-decoder-inputs",
            "task": "hellaswag", "seed": 0, "model": {"dtype": "bfloat16"},
            "split": {"stack": "enc", "where": "after_layer", "index": 9},
            "data": {"max_length": 512, "num_validation": 512,
                     "prompt_policy": {"mode": "five_shot", "source_max_length": 2048}},
            "channel": {"type": "awgn", "train_noise": True, "train_snr_range": [-6, 18]},
            "run": {}, "training": {"kl_weight": 1., "mse_weight": .1, "grad_clip": 1., "weight_decay": .01}}


def test_recipe_preserves_original_pool_and_objective(tmp_path):
    original = base_config()
    frozen = copy.deepcopy(original)
    cfg = pilot_config(original, tmp_path, "affine_core")
    assert original == frozen
    assert cfg["data"] == original["data"]
    settings = cfg["training"]
    assert (settings["max_steps"], settings["schedule_steps"], settings["batch_size"], settings["gradient_accumulation"]) == (500, 500, 64, 1)
    assert settings["warmup_ratio"] * settings["schedule_steps"] == 25
    assert settings["save_steps"] == [250, 500]
    assert settings["presentation_stream"]["total_presentations"] == 32000
    assert settings["paired_randomness"]["audit_steps"] == list(range(1, 501))
    assert settings["grad_clip"] == original["training"]["grad_clip"]
    assert settings["weight_decay"] == original["training"]["weight_decay"]
    ids = list(range(512))
    selected = selection_indices(ids)
    assert selected == selection_indices(ids)
    assert len(set(selected)) == 128
    assert ids == list(range(512))


@pytest.mark.parametrize("key,value", [("snr_film", True), ("layernorm", "both"), ("dropout", .1)])
def test_factory_rejects_unshared_policy(key, value):
    cfg = codec_config()
    cfg[key] = value
    with pytest.raises(ValueError):
        codec_factory(8, cfg, "shallow_nonlinear")


@pytest.mark.parametrize("family", FAMILIES)
def test_initial_and_updated_state_reload_forward_parity(family, tmp_path):
    from test_core import toy_model, batch
    from jscc.runtime import model_inputs
    torch.manual_seed(17)
    model = toy_model("enc")
    model.codec = codec_factory(8, codec_config(), family)
    inputs = batch()
    kwargs, _ = model_inputs(inputs, model)
    for update in (0, 1):
        if update:
            optimizer = torch.optim.AdamW(model.codec.parameters(), lr=2e-4)
            optimizer.zero_grad()
            with model.transmission(None, encoder_mask=inputs["attention_mask"]):
                model(**kwargs).logits.square().mean().backward()
            optimizer.step()
        path = tmp_path / f"{family}-{update}.pt"
        torch.save({"codec": model.codec.state_dict(), "channel": model.channel.state_dict(),
                    "memory_codec": None, "step": update}, path)
        restored = toy_model("enc")
        restored.base.load_state_dict(model.base.state_dict())
        restored.codec = codec_factory(8, codec_config(), family, seed=123)
        restored.load_communication_state(torch.load(path, weights_only=True))
        for snr in (None, -6.):
            torch.manual_seed(98)
            with model.transmission(snr, encoder_mask=inputs["attention_mask"]):
                expected = model(**kwargs).logits
            torch.manual_seed(98)
            with restored.transmission(snr, encoder_mask=inputs["attention_mask"]):
                actual = restored(**kwargs).logits
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("family", FAMILIES)
def test_production_initialization_evidence_accepts_family(family, tmp_path):
    from test_core import toy_model
    from jscc.presentation import initialization_evidence
    model = toy_model("enc")
    model.codec = codec_factory(8, codec_config(), family)
    evidence = initialization_evidence(model, tmp_path)
    assert evidence["hidden"]["internal_layernorm_count"] == (4 if family == "current_residual" else 0)
    assert evidence["hidden"]["parameter_count"] == sum(p.numel() for p in model.codec.parameters())
    assert (tmp_path / "initial_communication.pt").is_file()
