"""Focused tests for corrected power and valid-payload semantics."""

import torch

from model_helpers import tiny_backbone
from test_core import batch
from jscc.models.channel import AWGNChannel, normalize_power
from jscc.models.codec import Codec
from jscc.models.split_model import SplitModel
from jscc.runtime import prepare_trainable_parameters


def _codec():
    return Codec(16, {"hidden_dim": 16, "bottleneck_dim": 8, "n_res_blocks": 1,
                      "activation": "gelu", "layernorm": "none", "snr_film": False})


def test_masked_sequence_power_ignores_padding_values():
    torch.manual_seed(7)
    latent = torch.randn(2, 4, 5)
    valid = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    changed_padding = latent.clone()
    changed_padding[~valid] = torch.randn_like(changed_padding[~valid]) * 100

    normalized = normalize_power(latent, valid)
    changed = normalize_power(changed_padding, valid)
    torch.testing.assert_close(normalized[valid], changed[valid], rtol=0, atol=1e-6)
    valid_power = (normalized.square() * valid.unsqueeze(-1)).sum((1, 2))
    valid_count = valid.sum(1) * latent.shape[-1]
    torch.testing.assert_close(valid_power / valid_count, torch.ones(2), rtol=0, atol=1e-6)


def test_decoder_token_power_is_prefix_causal():
    torch.manual_seed(11)
    latent = torch.randn(1, 5, 7)
    changed_future = latent.clone()
    changed_future[:, 3:] = torch.randn_like(changed_future[:, 3:]) * 50
    first = normalize_power(latent, token_wise=True)
    second = normalize_power(changed_future, token_wise=True)
    torch.testing.assert_close(first[:, :3], second[:, :3], rtol=0, atol=1e-6)


def test_split_model_keeps_allocated_and_valid_payload_counts():
    config = {"hidden_dim": 8, "bottleneck_dim": 4, "n_res_blocks": 1,
              "activation": "gelu", "dropout": 0.0, "layernorm": "none",
              "snr_film": False}
    model = SplitModel(
        _ToyBackbone(), Codec(8, config), AWGNChannel(),
        {"stack": "enc", "where": "after_layer", "index": 0},
        {"normalize_power": True, "clean_film_snr": 18.0},
    ).eval()
    inputs = batch()
    with model.transmission(None):
        model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
              decoder_input_ids=torch.zeros((2, 3), dtype=torch.long), use_cache=False)
    assert model.channel_uses == {"hidden": 2 * 3 * 4, "memory": 0}
    assert model.channel_uses_valid == {"hidden": 5 * 4, "memory": 0}
    assert model.channel_uses_allocated is model.channel_uses
    assert model.valid_payload_counts is model.channel_uses_valid


class _ToyAttention(torch.nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.projection = torch.nn.Linear(8, 8)

    def forward(self, hidden, *, encoder_hidden_states):
        return self.projection(hidden + encoder_hidden_states.mean(dim=1, keepdim=True))


class _ToyLayer(torch.nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.self_attn = _ToyAttention(layer_idx)

    def forward(self, hidden, memory):
        return self.self_attn(hidden, encoder_hidden_states=memory)


class _ToyStack(torch.nn.Module):
    def __init__(self, decoder=False):
        super().__init__()
        self.layers = torch.nn.ModuleList([_ToyLayer(i) if decoder else torch.nn.Linear(8, 8)
                                           for i in range(2)])
        self.norm = torch.nn.LayerNorm(8)

    def forward(self, hidden, memory=None):
        for layer in self.layers:
            hidden = layer(hidden, memory) if memory is not None else layer(hidden)
        return self.norm(hidden)


class _ToyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 8)
        self.enc, self.dec = _ToyStack(), _ToyStack(decoder=True)
        self.head = torch.nn.Linear(8, 16)
        self.config = type("Config", (), {"pad_token_id": 0, "decoder_start_token_id": 0})()

    def get_encoder(self):
        return self.enc

    def get_decoder(self):
        return self.dec

    def forward(self, input_ids, decoder_input_ids, **kwargs):
        memory = self.enc(self.embedding(input_ids))
        hidden = self.dec(self.embedding(decoder_input_ids), memory)
        return type("Output", (), {"logits": self.head(hidden)})()


@torch.no_grad()
def test_decoder_token_normalization_does_not_change_full_prefix():
    config = {"hidden_dim": 8, "bottleneck_dim": 4, "n_res_blocks": 1,
              "activation": "gelu", "dropout": 0.0, "layernorm": "none",
              "snr_film": False}
    model = SplitModel(
        _ToyBackbone(), Codec(8, config), AWGNChannel(),
        {"stack": "dec", "where": "after_layer", "index": 0},
        {"normalize_power": True, "clean_film_snr": 18.0},
    ).eval()
    source = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(source)
    with model.transmission(None):
        model(input_ids=source, attention_mask=mask,
              decoder_input_ids=torch.tensor([[0, 4]]), use_cache=False)
        assert model.reconstruction is not None
        prefix_reconstruction = model.reconstruction.clone()
    with model.transmission(None):
        model(input_ids=source, attention_mask=mask,
              decoder_input_ids=torch.tensor([[0, 4, 5]]), use_cache=False)
        assert model.reconstruction is not None
        full_reconstruction = model.reconstruction.clone()
    torch.testing.assert_close(prefix_reconstruction, full_reconstruction[:, :2], rtol=0, atol=1e-6)


def test_decoder_mask_can_be_supplied_by_training_context():
    config = {"hidden_dim": 8, "bottleneck_dim": 4, "n_res_blocks": 1,
              "activation": "gelu", "dropout": 0.0, "layernorm": "none",
              "snr_film": False}
    model = SplitModel(
        _ToyBackbone(), Codec(8, config), AWGNChannel(),
        {"stack": "dec", "where": "after_layer", "index": 0},
        {"normalize_power": True, "clean_film_snr": 18.0},
    ).eval()
    source = torch.tensor([[1, 2, 3]])
    with model.transmission(None, decoder_mask=torch.tensor([[1, 1, 0]], dtype=torch.bool)):
        model(input_ids=source, attention_mask=torch.ones_like(source),
              decoder_input_ids=torch.tensor([[0, 4, 0]]), use_cache=False)
    assert model.channel_uses["hidden"] == 3 * 4
    assert model.channel_uses_valid["hidden"] == 2 * 4


def test_real_tiny_model_generation_uses_one_memory_payload_per_sequence():
    config = {"hidden_dim": 16, "bottleneck_dim": 8, "n_res_blocks": 1,
              "activation": "gelu", "layernorm": "none", "snr_film": False}
    model = SplitModel(
        tiny_backbone(3), Codec(16, config), AWGNChannel(),
        {"stack": "dec", "where": "after_layer", "index": 0},
        {"normalize_power": True, "clean_film_snr": 18.0},
    ).eval()
    source = torch.tensor([[2, 3, 4]])
    with model.transmission(None):
        model.generate(input_ids=source, attention_mask=torch.ones_like(source),
                       decoder_start_token_id=0, max_new_tokens=3,
                       min_new_tokens=3, do_sample=False, num_beams=1)
    assert model.channel_uses_valid["memory"] == 3 * 8
    assert model.channel_uses_valid["hidden"] == 3 * 8


@torch.no_grad()
def test_direct_bf16_backbone_evaluation_autocasts_fp32_codec():
    config = {"hidden_dim": 8, "bottleneck_dim": 4, "n_res_blocks": 1,
              "activation": "gelu", "dropout": 0.0, "layernorm": "none",
              "snr_film": False}
    model = SplitModel(
        _ToyBackbone(), Codec(8, config), AWGNChannel(),
        {"stack": "enc", "where": "after_layer", "index": 0},
        {"normalize_power": True, "clean_film_snr": 18.0},
    ).eval()
    model.base.to(dtype=torch.bfloat16)
    prepare_trainable_parameters(model)
    with model.transmission(None):
        output = model.base(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            decoder_input_ids=torch.tensor([[0, 4]]),
            use_cache=False,
        )
    assert output.logits.dtype == torch.bfloat16
    assert all(parameter.dtype == torch.float32 for parameter in model.codec.parameters())
