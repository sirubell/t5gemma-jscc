"""Preflight health must reject a partially disconnected trainable model."""
import pytest
import torch
from scripts.native_preflight_train import checked_gradient_names


def test_missing_gradient_cannot_pass_with_other_nonzero_gradients():
    model = torch.nn.Linear(2, 2)
    model.weight.grad = torch.ones_like(model.weight)
    with pytest.raises(RuntimeError, match='missing trainable gradient: bias'):
        checked_gradient_names(model)
    model.bias.grad = torch.ones_like(model.bias)
    assert checked_gradient_names(model) == ['weight', 'bias']
    model.bias.grad[0] = float('nan')
    with pytest.raises(FloatingPointError, match='nonfinite'):
        checked_gradient_names(model)
