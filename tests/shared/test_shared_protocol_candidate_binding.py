"""Current primitive assertions extracted without resurrecting legacy execution."""
import copy
import torch
import pytest
from torch import nn
from transformers import T5Gemma2Config, T5Gemma2ForConditionalGeneration
from jscc.models.codec import Codec
from jscc.models.channel import AWGNChannel
from jscc.models.split_model import SplitModel
from jscc.checkpoint_readers import SHARING_PREPARED_CELLS as CELLS


def candidate_config(cell, width):
    return dict(architecture=CELLS[cell][0], hidden_dim=1152, bottleneck_dim=width,
                n_res_blocks=2 if cell == 'R-LN' else 0, activation='gelu',
                layernorm=CELLS[cell][1], snr_film=False, dropout=0.)


@pytest.mark.parametrize('width', [512, 1152, 2304])
def test_frozen_t_forward_gradient_and_state_roundtrip(width):
    torch.set_num_threads(1)
    torch.manual_seed(19)
    codec = Codec(1152, candidate_config('T-LN', width))
    # Independent literal historical topology, never inferred from zero residual blocks.
    reference = nn.Sequential(nn.LayerNorm(1152), nn.Linear(1152, 1152), nn.GELU(),
                              nn.Linear(1152, width), nn.Linear(width, 1152), nn.GELU(),
                              nn.Linear(1152, 1152), nn.LayerNorm(1152))
    for target, source in zip([reference[0], reference[1], reference[3], reference[4], reference[6], reference[7]],
                               [codec.input_norm, codec.encoder[0], codec.encoder[2],
                                codec.decoder[0], codec.decoder[2], codec.output_norm]):
        target.load_state_dict(source.state_dict())
    x = torch.randn(2, 3, 1152, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    actual, expected = codec.decode(codec.encode(x), None), reference(y)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
    for left, right in zip(codec.parameters(), reference.parameters()):
        # Parameter traversal order differs because outer norms are registered first.
        assert left.grad is not None and right.grad is not None
    for target, source in zip([reference[0], reference[1], reference[3], reference[4], reference[6], reference[7]],
                               [codec.input_norm, codec.encoder[0], codec.encoder[2],
                                codec.decoder[0], codec.decoder[2], codec.output_norm]):
        for a, b in zip(target.parameters(), source.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    restored = Codec(1152, candidate_config('T-LN', width))
    restored.load_state_dict(copy.deepcopy(codec.state_dict()), strict=True)
    torch.testing.assert_close(restored.decode(restored.encode(x.detach()), None), actual.detach(), rtol=0, atol=0)
    assert sum(p.numel() for p in codec.parameters()) == 2*1152*width + width + 7*1152 + 2*1152**2
    assert sum(isinstance(m, nn.GELU) for m in codec.modules()) == 2
    assert all(m.bias is not None for m in codec.modules() if isinstance(m, nn.Linear))


def t_model():
    """Real 26-layer Transformer, required D1152, small attention/MLP/vocabulary."""
    torch.manual_seed(29)
    text = dict(vocab_size=32, hidden_size=1152, intermediate_size=16,
                num_hidden_layers=26, num_attention_heads=1, num_key_value_heads=1,
                head_dim=8, query_pre_attn_scalar=8, max_position_embeddings=64,
                layer_types=['full_attention'] * 26)
    vision = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                  num_attention_heads=2, image_size=8, patch_size=4)
    config = T5Gemma2Config(encoder=dict(text_config=text, vision_config=vision,
        mm_tokens_per_image=4, boi_token_index=29, eoi_token_index=30), decoder=text,
        image_token_index=31)
    return SplitModel(T5Gemma2ForConditionalGeneration(config),
        Codec(1152, candidate_config('T-LN', 512)), AWGNChannel(),
        {'stack': 'enc', 'where': 'after_final_norm'},
        {'normalize_power': True, 'clean_film_snr': 18.})
