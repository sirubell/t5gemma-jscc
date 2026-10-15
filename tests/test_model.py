from typing import Any

import pytest
import torch

from jscc.channel import normalize_power
from tests.helpers import text_batch, tiny_model

SITES = ["enc_emb", "enc_l0", "enc_l1", "enc_l2", "enc_fn", "dec_l0", "dec_l1"]


def logits(model, batch, **kw):
    with model.transmission(encoder_mask=batch["attention_mask"], decoder_mask=batch["labels"] != -100, **kw):
        return model(**batch).logits, model.reconstruction, model.memory_reconstruction


@pytest.mark.parametrize("site", SITES)
def test_every_site_transmits_and_bypass_is_vanilla(site):
    model = tiny_model()
    batch = text_batch()
    vanilla = model.base(**batch).logits  # no site installed yet
    model.set_site(site)
    bypass, rec, _ = logits(model, batch, bypass=True)
    assert rec is None and torch.equal(bypass, vanilla)
    coded, rec, mem = logits(model, batch, snr_db=None)
    assert rec is not None and not torch.allclose(coded, vanilla)
    assert (mem is not None) == site.startswith("dec_")


def test_decoder_site_needs_receiver_layer():
    with pytest.raises(ValueError):
        tiny_model().set_site("dec_l2")


def test_noise_changes_output_and_none_is_deterministic():
    model = tiny_model().set_site("enc_l1")
    batch = text_batch()
    a, _, _ = logits(model, batch, snr_db=None)
    b, _, _ = logits(model, batch, snr_db=None)
    c, _, _ = logits(model, batch, snr_db=torch.full((2, 1, 1), -6.0))
    assert torch.equal(a, b) and not torch.allclose(a, c)


def test_power_normalization_ignores_padding():
    z = torch.randn(2, 5, 4)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    out = normalize_power(z, mask)
    assert torch.allclose(out[0, :3].pow(2).mean(), torch.tensor(1.0), atol=1e-4)
    assert torch.allclose(out[1].pow(2).mean(), torch.tensor(1.0), atol=1e-4)


class Quantized(torch.nn.Module):
    normalizes_power = True

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def forward(self, z, snr_db):
        self.aux_loss = z.pow(2).mean()
        return z * self.scale


def test_custom_channel_owns_power_and_aux_loss():
    model = tiny_model(channel={"type": "tests.test_model:Quantized", "normalize_power": True}).set_site("enc_l1")
    assert not model.normalize
    out, _, _ = logits(model, text_batch(), snr_db=None)
    out.sum().backward()
    assert model.channel.scale.grad is not None and model.channel.aux_loss.ndim == 0


def test_qam_example_trains_through_the_split_model():
    model = tiny_model(channel={"type": "examples.qam_channel:LearnedQAMChannel", "normalize_power": True}).set_site("dec_l0")
    batch = text_batch()
    out, _, _ = logits(model, batch, snr_db=3.0)
    (out.sum() + model.channel.aux_loss).backward()
    channel: Any = model.channel
    assert channel.mapper[0].weight.grad is not None and model.codec.encoder[0].weight.grad is not None
