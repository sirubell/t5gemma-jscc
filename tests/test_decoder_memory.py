"""Receiver-only memory transmission on real tiny T5Gemma-2, including KV caching."""
import copy

import pytest
import torch
from model_helpers import tiny_backbone

from jscc.models.channel import AWGNChannel
from jscc.models.codec import Codec
from jscc.models.split_model import SplitModel
from jscc.training import batch_losses


def make_model(index=0, layernorm="both", memory_layernorm=None):
    config = {"hidden_dim": 16, "bottleneck_dim": 8, "n_res_blocks": 1,
              "activation": "gelu", "layernorm": layernorm, "snr_film": False}
    if memory_layernorm is not None:
        config["memory"] = {"layernorm": memory_layernorm}
    return SplitModel(tiny_backbone(3), Codec(16, config), AWGNChannel(),
                      {"stack": "dec", "where": "after_layer", "index": index},
                      {"normalize_power": True, "clean_film_snr": 18.0})


def inputs():
    return {"input_ids": torch.tensor([[2, 3, 4]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}


def test_both_codecs_get_task_gradients_and_backbone_is_frozen():
    model = make_model().train()
    loss = batch_losses(model, {**inputs(), "labels": torch.tensor([[5, 6, 1]])},
                        {"temperature": 1, "kl_weight": 1, "mse_weight": 0}, 0)
    loss["loss"].backward()
    assert model.memory_codec is not None
    for codec in (model.codec, model.memory_codec):
        grad = codec.encoder[0].get_parameter("weight").grad
        assert grad is not None and grad.abs().sum() > 0
    assert all(p.grad is None for p in model.base.parameters())


def test_memory_codec_can_override_boundary_layernorm():
    model = make_model(layernorm="none", memory_layernorm="both")
    assert model.codec.config["layernorm"] == "none"
    assert model.memory_codec is not None
    assert model.memory_codec.config["layernorm"] == "both"
    # The nested override inherits the rest of the flat codec design.
    assert model.memory_codec.config["bottleneck_dim"] == model.codec.config["bottleneck_dim"]
    assert model.memory_codec.config["n_res_blocks"] == model.codec.config["n_res_blocks"]


def test_flat_codec_config_keeps_memory_codec_identical():
    model = make_model(layernorm="none")
    assert model.memory_codec is not None
    assert model.memory_codec.config["layernorm"] == "none"
    assert model.memory_codec.config == model.codec.config


@torch.no_grad()
def test_receiver_layers_share_memory_and_uncached_forward_retransmits():
    model = make_model().eval()
    received = []
    def record(module, args, kwargs):
        received.append(kwargs["encoder_hidden_states"])
    handles = [layer.self_attn.register_forward_pre_hook(record, with_kwargs=True)
               for layer in model.base.get_decoder().layers[1:]]
    try:
        with model.transmission(0):
            for _ in range(2):
                model(**inputs(), decoder_input_ids=torch.tensor([[0, 5]]), use_cache=False)
        assert received[0] is received[1]
        assert received[2] is received[3]
        assert not torch.equal(received[0], received[2])
        assert model.channel_uses == {"hidden": 2 * 2 * 8, "memory": 2 * 3 * 8}
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def test_generation_transmits_memory_once_and_new_sequence_starts_fresh():
    model = make_model().eval()
    with model.transmission(0):
        for i in range(2):
            model.generate(**inputs(), decoder_start_token_id=0, max_new_tokens=3,
                           min_new_tokens=3, do_sample=False, num_beams=1)
            assert model.channel_uses["memory"] == (i + 1) * 3 * 8
            assert model.channel_uses["hidden"] == (i + 1) * 3 * 8


@torch.no_grad()
def test_final_decoder_layer_needs_no_memory_transmission():
    model = make_model(index=2).eval()
    assert model.memory_codec is None
    with model.transmission(0):
        model(**inputs(), decoder_input_ids=torch.tensor([[0, 5]]), use_cache=False)
    assert model.channel_uses == {"hidden": 2 * 8, "memory": 0}


@torch.no_grad()
def test_bypass_matches_backbone_and_old_decoder_state_cannot_load_silently():
    model = make_model().eval()
    # Independent backbone with the same weights and no transmission hooks.
    reference = tiny_backbone(3).eval()
    reference.load_state_dict(model.base.state_dict())
    batch = {**inputs(), "decoder_input_ids": torch.tensor([[0, 5]]), "use_cache": False}
    with model.transmission(bypass=True):
        torch.testing.assert_close(model(**batch).logits, reference(**batch).logits, rtol=0, atol=0)
    assert model.channel_uses == {"hidden": 0, "memory": 0}
    state = {"codec": model.codec.state_dict(), "channel": model.channel.state_dict()}
    with pytest.raises(ValueError, match="no receiver-memory codec"):
        model.load_communication_state(state)
    assert model.memory_codec is not None
    state["memory_codec"] = copy.deepcopy(model.memory_codec.state_dict())
    restored = make_model()
    restored.load_communication_state(state)
    assert restored.memory_codec is not None
    for key, value in state["memory_codec"].items():
        torch.testing.assert_close(restored.memory_codec.state_dict()[key], value)


def test_beam_generation_requires_explicit_future_channel_semantics():
    model = make_model().eval()
    with pytest.raises(ValueError, match="num_beams=1"):
        model.generate(**inputs(), num_beams=2, decoder_start_token_id=0)
