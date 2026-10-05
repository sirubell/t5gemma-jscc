"""Current primitive assertions extracted without resurrecting legacy execution."""
import pytest
import torch
from torch import nn
from jscc.models.codec import Codec
from shared.test_shared_protocol_candidate_binding import candidate_config


@pytest.mark.parametrize('width', [512, 1152, 2304])
def test_no_norm_matches_literal_affine_and_gradients(width):
    torch.set_num_threads(1)
    torch.manual_seed(19)
    codec = Codec(1152, candidate_config('D-N', width))
    reference = nn.Sequential(nn.Linear(1152, width), nn.Linear(width, 1152))
    reference[0].load_state_dict(codec.encoder[0].state_dict())
    reference[1].load_state_dict(codec.decoder[0].state_dict())
    x = torch.randn(2, 3, 1152, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    actual = codec.decode(codec.encode(x), None)
    expected = reference(y)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
    for a, b in zip(codec.parameters(), reference.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    assert isinstance(codec.input_norm, nn.Identity)
    assert isinstance(codec.output_norm, nn.Identity)
    assert sum(isinstance(m, nn.Linear) for m in codec.modules()) == 2
    assert not any(isinstance(m, (nn.LayerNorm, nn.GELU)) for m in codec.modules())
    assert sum(p.numel() for p in codec.parameters()) == 2 * 1152 * width + width + 1152
