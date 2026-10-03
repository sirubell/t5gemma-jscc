"""Real tiny26 T5Gemma2 boundaries, role gradients and canonical state."""
from copy import deepcopy
from typing import cast

import pytest
import torch

from jscc.encoder_online_reconstruction import (capture_online_original13_targets,
                                               original13_reconstruction_batch_values)
from jscc.encoder_stages_plan import ORIGINAL13_SITES
from jscc.models.channel import AWGNChannel
from jscc.models.codec import Codec
from jscc.models.split_model import SplitModel, resolve_original13_site, stack_module
from jscc.original13_routing import export_codec, load_codec, observation_state, restore_runtime_state
from jscc.runtime import model_inputs, prepare_trainable_parameters
from jscc.training import batch_losses
from model_helpers import tiny_backbone


def make_model(dtype=torch.float32):
    torch.set_num_threads(1)
    torch.manual_seed(41)
    codec = Codec(16, dict(architecture="direct_affine", hidden_dim=16, bottleneck_dim=8,
                          n_res_blocks=0, activation="gelu", dropout=0., layernorm="none", snr_film=False))
    model = SplitModel(cast(torch.nn.Module, tiny_backbone(26)).to(dtype=dtype), codec, AWGNChannel(),
        {"stack": "enc", "where": "after_final_norm"},
        {"normalize_power": True, "clean_film_snr": 18.})
    prepare_trainable_parameters(model)
    return model.train().enable_universal_codec()


def batch(sequences=2):
    return {"input_ids": torch.tensor([[2, 3, 4], [2, 5, 0]]).repeat(sequences // 2, 1),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]).repeat(sequences // 2, 1),
            "labels": torch.tensor([[7, 8, 1, -100], [9, 1, -100, -100]]).repeat(sequences // 2, 1)}


def settings(site, functional=False):
    decoder = site.startswith("dec_")
    return {"temperature": 1., "loss_weights": {"kl": float(functional),
        "hidden": .05 if decoder else .1, "memory": .05 if decoder else 0.},
        "requires_memory": decoder}


@pytest.mark.parametrize("site_id", ORIGINAL13_SITES)
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_real_routes_targets_payload_and_backward(site_id, dtype, monkeypatch):
    model, data = make_model(dtype), batch()
    site = resolve_original13_site(model.base, site_id, "tiny-original13")
    original = model.split.copy()
    before_hooks = len(model._memory_handles)
    with model.at_site(site, purpose="train", authorization={"sites": {site_id: "trained"}}):
        assert model.requires_memory_transmission == site_id.startswith("dec_")
        hidden, memory = capture_online_original13_targets(model, data, site)
        assert not hidden.requires_grad
        # The online target must equal the actual clean full-forward boundary.
        kwargs, _ = model_inputs(data, model)
        with torch.no_grad(), model.transmission(bypass=True):
            model(**kwargs)
        torch.testing.assert_close(hidden, model.activation, rtol=0, atol=0)
        if memory is not None:
            with torch.no_grad():
                expected_memory = model.base.get_encoder()(input_ids=data["input_ids"],
                    attention_mask=data["attention_mask"]).last_hidden_state
            torch.testing.assert_close(memory, expected_memory, rtol=0, atol=0)
        # Reconstruction must never compute logits or enter the receiver suffix.
        def forbidden(*_args, **_kwargs):
            pytest.fail("reconstruction executed a forbidden model suffix or LM head")
        monkeypatch.setattr(model.base.lm_head, "forward", forbidden)
        if memory is not None:
            assert site.index is not None
            monkeypatch.setattr(stack_module(model.base, "dec").layers[site.index + 1], "forward", forbidden)
        else:
            monkeypatch.setattr(stack_module(model.base, "dec"), "forward", forbidden)
        values = original13_reconstruction_batch_values(model, data, 0., settings(site_id), site=site)
        weights = settings(site_id)["loss_weights"]
        torch.testing.assert_close(values["loss"], weights["hidden"] * cast(torch.Tensor, values.get("hidden")) +
                                   weights["memory"] * cast(torch.Tensor, values.get("memory")))
        values["loss"].backward()
        assert all(p.grad is None for p in model.base.parameters())
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.codec.parameters())
        assert sum(float(p.grad.abs().sum()) for p in model.codec.parameters() if p.grad is not None) > 0
        target = site_id.startswith("dec_")
        assert model.channel_uses_valid == {"hidden": 5 * 8,
                                           "memory": 5 * 8 if target else 0}
        assert model.channel_uses == {"hidden": (8 if target else 6) * 8,
                                     "memory": 6 * 8 if target else 0}
    assert model.split == original and not model._routing_active and model._forward_active == 0
    assert len(model._memory_handles) == before_hooks
    assert model.activation is model.reconstruction is model.memory_reconstruction is None


@pytest.mark.parametrize("site_id", ORIGINAL13_SITES)
def test_functional_gradient_routes_and_clean_prefix(site_id):
    model, data = make_model(), batch()
    site = resolve_original13_site(model.base, site_id, "tiny-original13")
    sent, memories, handles = [], [], []
    transmit = model._transmit
    def observe_transmit(hidden, *args, **kwargs):
        reconstructed = transmit(hidden, *args, **kwargs)
        reconstructed.retain_grad()
        sent.append((args[1], reconstructed))
        return reconstructed
    model._transmit = observe_transmit
    with model.at_site(site, purpose="train", authorization={"sites": {site_id: "trained"}}):
        decoder = site_id.startswith("dec_")
        if decoder:
            def observe_memory(_module, _args, kwargs):
                if not model.bypass:
                    memories.append(kwargs["encoder_hidden_states"])
            for layer in stack_module(model.base, "dec").layers:
                handles.append(layer.self_attn.register_forward_pre_hook(observe_memory, with_kwargs=True))
        objective = settings(site_id, True)
        objective["loss_weights"]["hidden"] = objective["loss_weights"]["memory"] = 0.
        try:
            values = batch_losses(model, data, objective, None, return_stats=True, valid_only_kl=True)
            values["loss"].backward()
            assert [role for role, _ in sent] == (["hidden", "memory"] if decoder else ["hidden"])
            assert all(value.grad is not None and float(value.grad.abs().sum()) > 0 for _, value in sent)
            assert all(p.grad is None for p in model.base.parameters())
            if decoder:
                assert site.index is not None
                assert model.memory_activation is not None and model.memory_reconstruction is not None
                assert all(torch.equal(value, model.memory_activation) for value in memories[:site.index + 1])
                assert all(value is model.memory_reconstruction for value in memories[site.index + 1:])
                assert not torch.equal(model.memory_reconstruction, model.memory_activation)
        finally:
            for handle in handles:
                handle.remove()


@pytest.mark.parametrize("role", ("hidden", "memory"))
def test_separate_reconstruction_role_gradients_share_parameters(role):
    model = make_model()
    site = resolve_original13_site(model.base, "dec_l8", "tiny-original13")
    with model.at_site(site, purpose="train", authorization={"sites": {"dec_l8": "trained"}}):
        values = original13_reconstruction_batch_values(model, batch(), None, settings("dec_l8"), site=site)
        values["weighted_" + role].backward()
    assert model.memory_codec is model.codec
    assert all(p.grad is not None for p in model.codec.parameters())
    assert sum(float(p.grad.abs().sum()) for p in model.codec.parameters() if p.grad is not None) > 0


def test_alias_movement_strict_body_and_idle_restore():
    model = make_model().to("cpu")
    assert model.memory_codec is model.codec
    assert len(list(model.parameters())) == len({id(p) for p in model.parameters()})
    weights = export_codec(model)
    assert set(weights) == {"encoder.0.weight", "encoder.0.bias", "decoder.0.weight", "decoder.0.bias"}
    before = observation_state(model)
    model.eval()
    model.channel_uses["memory"] = 912
    model.channel.draw_summaries = [{"stream": "memory"}]
    restore_runtime_state(model, before)
    assert observation_state(model) == before
    with torch.no_grad():
        next(model.codec.parameters()).add_(1)
    load_codec(model, weights)
    for key, value in export_codec(model).items():
        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
    for bad in ({"codec": weights, "memory_codec": weights}, {"kind": "legacy", **weights},
                {**weights, "memory.encoder.0.weight": weights["encoder.0.weight"]}):
        with pytest.raises(ValueError, match="wrong-kind"):
            load_codec(model, bad)
    bad = deepcopy(weights)
    bad["decoder.0.bias"][0] = float("nan")
    with pytest.raises(ValueError, match="invalid canonical"):
        load_codec(model, bad)
    assert model.memory_codec is model.codec


def test_route_exception_restores_hooks_and_idle_state(monkeypatch):
    model = make_model()
    before = observation_state(model)
    site = resolve_original13_site(model.base, "dec_l24", "tiny-original13")
    def fail(*_args, **_kwargs):
        raise RuntimeError("receiver failure")
    monkeypatch.setattr(stack_module(model.base, "dec").layers[25], "forward", fail)
    with pytest.raises(RuntimeError, match="receiver failure"):
        with model.at_site(site, purpose="evaluate", authorization={"sites": {site.site_id: "trained"}}):
            with pytest.raises(ValueError, match="idle"):
                observation_state(model)
            batch_losses(model, batch(), settings(site.site_id, True), 0.)
    assert model._forward_active == 0 and model._routing_active is False
    restore_runtime_state(model, before)
    assert observation_state(model) == before


def test_generation_reuses_one_memory_transmission():
    model = make_model().eval()
    site = resolve_original13_site(model.base, "dec_l0", "tiny-original13")
    with model.at_site(site, purpose="evaluate", authorization={"sites": {site.site_id: "trained"}}):
        with model.transmission(None), model.observe_payload() as events:
            data = batch()
            model.generate(input_ids=data["input_ids"], attention_mask=data["attention_mask"],
                           max_new_tokens=3, min_new_tokens=3, do_sample=False)
        assert len([event for event in events if event["stream"] == "memory"]) == 1
        assert len([event for event in events if event["stream"] == "hidden"]) == 3
