"""Focused CPU verification for the exact normalization/architecture matrix."""
import copy
from typing import Any

import pytest
import torch
from torch import nn

from jscc.config import validate_codec_architecture
from jscc.models.codec import BackboneRMSNorm, Codec

torch.set_num_threads(1)

# Copied verbatim from the observed installed Transformers source, not a
# reformulated oracle. Provenance remains in the local normalization campaign
# architecture receipt; this portable test has no private file dependency.
REFERENCE_SOURCE = '''class T5Gemma2RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst T5Gemma2 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"
'''
namespace: dict[str, Any] = {'nn': nn, 'torch': torch}
exec(REFERENCE_SOURCE, namespace)
ReferenceRMSNorm = namespace['T5Gemma2RMSNorm']


def config(architecture, norm, width):
    return {'architecture': architecture, 'hidden_dim': 1152,
            'bottleneck_dim': width, 'n_res_blocks': 0, 'activation': 'gelu',
            'dropout': 0., 'layernorm': norm, 'snr_film': False,
            **({'rms_norm_eps': 1e-6} if norm == 'rms_both' else {})}


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.float16])
def test_rms_exact_installed_reference_forward_backward(dtype):
    torch.manual_seed(93)
    actual, expected = BackboneRMSNorm(1152), ReferenceRMSNorm(1152)
    # A nonzero scale catches offset-vs-direct scale and multiply/cast order.
    with torch.no_grad():
        actual.weight.copy_(torch.linspace(-.83, .91, 1152))
    expected.load_state_dict(actual.state_dict())
    x = (torch.randn(2, 3, 1152) * 7.3 + 1.7).to(dtype).requires_grad_()
    y = x.detach().clone().requires_grad_()
    a, b = actual(x), expected(y)
    assert a.dtype == dtype
    assert torch.equal(a, b)
    cotangent = torch.randn_like(a)
    a.backward(cotangent)
    b.backward(cotangent)
    assert x.grad is not None and y.grad is not None
    assert actual.weight.grad is not None and expected.weight.grad is not None
    assert torch.equal(x.grad, y.grad)
    assert torch.equal(actual.weight.grad, expected.weight.grad)
    assert list(actual.state_dict()) == ['weight']
    assert torch.equal(BackboneRMSNorm(1152).weight, torch.zeros(1152))
    assert torch.isfinite(BackboneRMSNorm(1152)(torch.zeros_like(x))).all()


@pytest.mark.parametrize('width', [512, 1152, 2304])
@pytest.mark.parametrize('norm', ['none', 'both', 'rms_both'])
@pytest.mark.parametrize('linear_count', [1, 2])
def test_exact_matrix_boundaries_structure_and_training(width, norm, linear_count):
    architecture = ('two_linear_gelu' if linear_count == 2 else
                    {'none': 'direct_affine', 'both': 'direct_outer_ln',
                     'rms_both': 'direct_outer_rms'}[norm])
    torch.manual_seed(0)
    codec = Codec(1152, config(architecture, norm, width))
    assert codec.film is None
    kind = {'none': nn.Identity, 'both': nn.LayerNorm, 'rms_both': BackboneRMSNorm}[norm]
    assert isinstance(codec.input_norm, kind) and isinstance(codec.output_norm, kind)
    if norm != 'none':
        assert codec.input_norm.weight.shape == codec.output_norm.weight.shape == (1152,)
    for half in [codec.encoder, codec.decoder]:
        linears = [m for m in half if isinstance(m, nn.Linear)]
        assert len(linears) == linear_count
        assert all(m.bias is not None for m in linears)
        assert sum(isinstance(m, nn.GELU) for m in half) == linear_count - 1
        assert all(isinstance(m, (nn.Linear, nn.GELU)) for m in half)
    x = torch.randn(2, 3, 1152, requires_grad=True)
    z = codec.encode(x)
    assert z.shape == (2, 3, width)
    out = codec.decode(z, None)
    assert out.shape == x.shape and torch.isfinite(out).all()
    out.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in codec.parameters())
    assert all(p.grad is not None and p.grad.abs().max() > 0 for p in codec.parameters())
    torch.optim.AdamW(codec.parameters(), lr=2e-4).step()
    assert all(torch.isfinite(p).all() for p in codec.parameters())
    rebuilt = Codec(1152, copy.deepcopy(codec.config))
    rebuilt.load_state_dict(codec.state_dict(), strict=True)
    assert torch.equal(codec.decode(codec.encode(x.detach()), None),
                       rebuilt.decode(rebuilt.encode(x.detach()), None))




@pytest.mark.parametrize('patch', [
    {'hidden_dim': 512}, {'activation': 'relu'}, {'n_res_blocks': 1},
    {'dropout': .1}, {'snr_film': True}, {'layernorm': 'pre'},
    {'bottleneck_dim': 0}, {'bottleneck_dim': True},
    {'layernorm': 'rms_both'}, {'layernorm': 'rms_both', 'rms_norm_eps': 1e-5},
])
def test_matrix_rejects_silent_architecture_changes(patch):
    cfg = config('two_linear_gelu', 'both', 512)
    cfg.update(patch)
    with pytest.raises(ValueError):
        validate_codec_architecture(cfg)


@pytest.mark.parametrize('width', [512, 1152, 2304])
@pytest.mark.parametrize('linear_count', [1, 2])
def test_same_architecture_width_has_identical_linear_initialization_across_norms(width, linear_count):
    states = []
    for norm in ['none', 'both', 'rms_both']:
        architecture = ('two_linear_gelu' if linear_count == 2 else
                        {'none': 'direct_affine', 'both': 'direct_outer_ln',
                         'rms_both': 'direct_outer_rms'}[norm])
        torch.manual_seed(0)
        state = Codec(1152, config(architecture, norm, width)).state_dict()
        states.append({k: v for k, v in state.items() if k.startswith(('encoder.', 'decoder.'))})
    assert states[0].keys() == states[1].keys() == states[2].keys()
    assert all(torch.equal(states[0][k], states[1][k]) and torch.equal(states[0][k], states[2][k])
               for k in states[0])
