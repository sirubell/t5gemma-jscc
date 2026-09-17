from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
from torch import nn

from jscc.config import load_config
from jscc.models.channel import AWGNChannel, build_channel, normalize_power
from jscc.models.codec import Codec
from jscc.models.split_model import SplitModel
from jscc.runtime import isolated_rng
from jscc.training import batch_losses, make_scheduler


def config():
    return {"hidden_dim": 8, "bottleneck_dim": 4, "n_res_blocks": 2,
            "activation": "gelu", "dropout": 0.0, "layernorm": "none",
            "snr_film": True, "film_hidden": 4}


class ToyAttention(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.projection = nn.Linear(8, 8)

    def forward(self, hidden, *, encoder_hidden_states):
        return self.projection(hidden + encoder_hidden_states.mean(dim=1, keepdim=True))


class ToyDecoderLayer(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.self_attn = ToyAttention(layer_idx)

    def forward(self, hidden, memory):
        return self.self_attn(hidden, encoder_hidden_states=memory)


class Stack(nn.Module):
    def __init__(self, decoder=False):
        super().__init__()
        self.layers = nn.ModuleList([ToyDecoderLayer(i) if decoder else nn.Linear(8, 8) for i in range(2)])
        self.norm = nn.LayerNorm(8)

    def forward(self, hidden, memory=None):
        for layer in self.layers:
            hidden = layer(hidden) if memory is None else layer(hidden, memory)
        return self.norm(hidden)


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.enc, self.dec = Stack(), Stack(decoder=True)
        self.head = nn.Linear(8, 16)
        self.config = SimpleNamespace(pad_token_id=0, decoder_start_token_id=0)

    def get_encoder(self):
        return self.enc

    def get_decoder(self):
        return self.dec

    def forward(self, input_ids, decoder_input_ids, **kwargs):
        memory = self.enc(self.embedding(input_ids))
        hidden = self.dec(self.embedding(decoder_input_ids), memory)
        return SimpleNamespace(logits=self.head(hidden))


def toy_model(stack="enc", where="after_layer", channel=None):
    return SplitModel(ToyBackbone(), Codec(8, config()), channel or AWGNChannel(),
                      {"stack": stack, "where": where, "index": 0},
                      {"normalize_power": True, "clean_film_snr": 18.0})


def batch():
    return {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
            "labels": torch.tensor([[6, 7, 8], [9, 10, -100]])}


def test_channel_power_snr_and_clean_rng():
    torch.manual_seed(4)
    z = torch.randn(8, 1000, 16)
    z[0] *= 50
    normalized = normalize_power(z)
    assert torch.allclose(normalized.square().mean((1, 2)), torch.ones(8), atol=1e-6)
    assert torch.allclose(normalize_power(z[:1]), normalized[:1])
    channel = AWGNChannel()
    rng = torch.get_rng_state()
    assert torch.equal(channel(normalized, None), normalized)
    assert torch.equal(rng, torch.get_rng_state())
    noise = channel(normalized, 10.0) - normalized
    assert noise.square().mean().item() == pytest.approx(0.1, rel=0.03)


@pytest.mark.parametrize("directory,suffix", [("tasks", ""), ("smoke", "_cpu"), ("smoke", "_h200")])
def test_task_recipes_share_a_film_off_model(directory, suffix):
    root = Path(__file__).parents[1] / "configs"
    coco = load_config(root / directory / f"coco{suffix}.yaml")
    hellaswag = load_config(root / directory / f"hellaswag{suffix}.yaml")
    baseline = load_config(root / "tasks" / "coco.yaml")
    assert coco["model"] == hellaswag["model"]
    for section in ("split", "codec", "channel"):
        assert coco[section] == hellaswag[section] == baseline[section]
    assert coco["codec"]["snr_film"] is False
    assert coco["channel"]["train_noise"] is True


def test_film_off_decoder_ignores_snr_and_has_no_film_parameters():
    codec = Codec(8, {**config(), "layernorm": "both", "snr_film": False}).eval()
    assert codec.film is None
    assert not any(name.startswith("film.") for name in codec.state_dict())
    received = torch.randn(2, 3, 4, requires_grad=True)
    low = codec.decode(received, -6.0)
    torch.testing.assert_close(low, codec.decode(received, 18.0), rtol=0, atol=0)
    torch.testing.assert_close(low, codec.decode(received, None), rtol=0, atol=0)
    low.square().mean().backward()
    assert received.grad is not None and torch.isfinite(received.grad).all()


@pytest.mark.parametrize("norm", ["none", "pre", "post", "both"])
def test_configurable_codec_and_gradient(norm):
    settings = {**config(), "layernorm": norm}
    codec = Codec(8, settings)
    hidden = torch.randn(2, 3, 8, requires_grad=True)
    reconstructed = codec.decode(codec.encode(hidden), 0.0)
    assert reconstructed.shape == hidden.shape
    reconstructed.square().mean().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


@pytest.mark.parametrize("stack", ["enc", "dec"])
@pytest.mark.parametrize("where", ["before_first_layer", "after_layer", "after_final_norm"])
def test_distillation_updates_codec_but_freezes_backbone(stack, where):
    model = toy_model(stack, where).train()
    assert not model.base.training
    settings = {"temperature": 1.0, "kl_weight": 1.0, "mse_weight": 0.1}
    before = {name: p.detach().clone() for name, p in model.base.named_parameters()}
    optimizer = torch.optim.AdamW(model.codec.parameters(), lr=0.01)
    values = batch_losses(model, batch(), settings, 0.0)
    values["loss"].backward()
    encoder_grad = model.codec.encoder[0].get_parameter("weight").grad
    decoder_grad = model.codec.decoder[0].get_parameter("weight").grad
    assert encoder_grad is not None and encoder_grad.abs().sum() > 0
    assert decoder_grad is not None and decoder_grad.abs().sum() > 0
    optimizer.step()
    assert all(p.grad is None and torch.equal(before[name], p) for name, p in model.base.named_parameters())


def test_python_channel_replacement_retains_autograd():
    channel = build_channel({"type": "examples.wireless_channel:WirelessChannel", "kwargs": {"gain": 0.5}})
    model = toy_model(channel=channel)
    values = batch_losses(model, batch(), {"temperature": 1.0, "kl_weight": 1.0, "mse_weight": 0.1}, 0.0)
    values["loss"].backward()
    grad = model.codec.encoder[0].get_parameter("weight").grad
    assert grad is not None and grad.abs().sum() > 0


def test_validation_rng_does_not_change_training_rng():
    before = torch.get_rng_state()
    with isolated_rng(123):
        torch.randn(10)
    assert torch.equal(before, torch.get_rng_state())


def test_scheduler_counts_optimizer_updates():
    parameter = nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = make_scheduler(optimizer, {"max_steps": 10, "warmup_ratio": 0.2, "min_lr_ratio": 0.1})
    rates = []
    for _ in range(10):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    assert rates[1] == pytest.approx(1.0)
    assert rates[-1] == pytest.approx(0.1)
