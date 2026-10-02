"""Online original-six prefix capture and local gradients on tiny CPU weights."""
from dataclasses import asdict
from typing import cast

import pytest
import torch

from jscc.baseline_protocol import objective_settings
from jscc.encoder_online_reconstruction import online_reconstruction_batch_values
from jscc.encoder_pilot_plan import SITE_SPECS, SITES
from jscc.losses import reconstruction_loss_stats
from jscc.models.channel import AWGNChannel
from jscc.models.codec import Codec
from jscc.models.split_model import SplitModel, resolve_encoder_site, stack_module
from jscc.runtime import prepare_trainable_parameters
from jscc.training_objectives import aggregate_batch_losses, effective_batch_denominators, scaled_batch_loss
from model_helpers import tiny_backbone


@pytest.mark.parametrize("site_id", SITES)
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_online_source_capture_and_codec_gradient(site_id, dtype, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    codec = Codec(16, dict(architecture="direct_affine", hidden_dim=16, bottleneck_dim=8,
                          n_res_blocks=0, activation="gelu", dropout=0., layernorm="none", snr_film=False))
    backbone = cast(torch.nn.Module, tiny_backbone(20)).to(dtype=dtype)
    model = SplitModel(backbone, codec, AWGNChannel(),
                       SITE_SPECS[site_id], {"normalize_power": True, "clean_film_snr": 18.})
    prepare_trainable_parameters(model)
    model.train()
    site = resolve_encoder_site(model.base, model.split, "tiny-online-cpu")
    site_identity = asdict(site)
    batch = {"input_ids": torch.tensor([[2, 3, 4], [2, 5, 0]]),
             "attention_mask": torch.tensor([[1, 1, 1], [0, 0, 0]]),
             "labels": torch.tensor([[-100, -100], [7, 1]])}
    stack = stack_module(model.base, "enc")
    expected = []

    def observe_input(_module, args, kwargs):
        expected.append((args[0] if args else kwargs["hidden_states"]).detach().clone())

    def observe_output(_module, _args, output):
        expected.append((output[0] if isinstance(output, tuple) else output).detach().clone())

    if site_id == "enc_emb":
        handle = stack.layers[0].register_forward_pre_hook(observe_input, with_kwargs=True, prepend=True)
    else:
        module = stack.norm if site_id == "enc_fn" else stack.layers[site.index]
        handle = module.register_forward_hook(observe_output, prepend=True)
    try:
        with torch.no_grad(), model.attention_context(), model.transmission(bypass=True):
            model.base.get_encoder()(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    finally:
        handle.remove()
    assert len(expected) == 1

    def forbidden(*_args, **_kwargs):
        pytest.fail("online reconstruction executed decoder/teacher logits or prepared decoder inputs")

    monkeypatch.setattr(model.base, "forward", forbidden)
    monkeypatch.setattr(model.base.get_decoder(), "forward", forbidden)
    monkeypatch.setattr(model.base, "prepare_decoder_input_ids_from_labels", forbidden)
    prefix_calls = []
    hooks = [layer.register_forward_pre_hook(lambda _module, _args, i=i: prefix_calls.append(i))
             for i, layer in enumerate(stack.layers)]
    sent, received = [], []
    original_roundtrip = model._roundtrip

    def roundtrip(activation):
        assert not activation.requires_grad
        assert activation.dtype == dtype
        torch.testing.assert_close(activation, expected[0], rtol=0, atol=0)
        sent.append(activation.detach().clone())
        result = original_roundtrip(activation)
        assert result is not None
        assert result.requires_grad and result.dtype == dtype
        received.append(result.detach().clone())
        return result

    monkeypatch.setattr(model, "_roundtrip", roundtrip)
    try:
        values = online_reconstruction_batch_values(model, batch, torch.full((2, 1, 1), 0.),
                                                     objective_settings("local"), site=site)
    finally:
        for hook in hooks:
            hook.remove()
    if site_id == "enc_emb":
        visited = 0
    elif site_id == "enc_fn":
        visited = 20
    else:
        assert site.index is not None
        visited = site.index + 1
    assert prefix_calls == list(range(visited))
    assert len(sent) == len(received) == 1
    assert model._forward_active == 0
    assert asdict(site) == site_identity and model.split == SITE_SPECS[site_id]
    assert model.activation is model.reconstruction is None
    assert model.memory_activation is model.memory_reconstruction is None
    assert model.channel_uses_valid == {"hidden": 3 * 8, "memory": 0}
    assert "hidden_denominator" in values and "kl_denominator" in values
    assert int(values["hidden_denominator"]) == 1
    assert int(values["kl_denominator"]) == 0 and float(values["kl"]) == 0.
    numerator, denominator = reconstruction_loss_stats(received[0], sent[0], batch["attention_mask"])
    torch.testing.assert_close(values["nmse"], numerator / denominator)
    torch.testing.assert_close(values["loss"], 0.1 * values["nmse"])
    # Combined baseline accounting must also retain precisely the R-only loss.
    settings = objective_settings("combined")
    denominators = effective_batch_denominators(model, [batch], torch.device("cpu"))
    loss = scaled_batch_loss(values, settings, denominators)
    torch.testing.assert_close(loss, values["loss"])
    torch.testing.assert_close(aggregate_batch_losses([values], settings)["loss"], loss)
    loss.backward()
    assert all(p.grad is None for p in model.base.parameters())
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.codec.parameters())
    assert sum(float(p.grad.abs().sum()) for p in model.codec.parameters() if p.grad is not None) > 0.
