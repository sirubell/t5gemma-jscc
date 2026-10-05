"""Exact checkpoint values are independent of the tensors' storage device."""

import numpy as np
import pytest
import torch

from jscc.qualification import state_exact


def test_tensor_comparison_uses_detached_cpu_values(monkeypatch):
    equal = torch.equal
    comparisons = []

    def checked_equal(left, right):
        assert left.device.type == right.device.type == 'cpu'
        assert not left.requires_grad and not right.requires_grad
        comparisons.append((left, right))
        return equal(left, right)

    monkeypatch.setattr(torch, 'equal', checked_equal)
    left = torch.tensor([1., 2.], requires_grad=True)
    right = left.detach().clone().requires_grad_()
    assert state_exact(left, right)
    assert len(comparisons) == 1
    assert left.requires_grad and right.requires_grad
    assert left.grad is None and right.grad is None


@pytest.mark.parametrize('right', [
    torch.tensor([1., 2.], dtype=torch.float64),
    torch.tensor([[1., 2.]]),
    torch.tensor([1., 2.0000002]),
])
def test_tensor_dtype_shape_and_exact_value_mismatches_are_rejected(right):
    left = torch.tensor([1., 2.])
    assert not state_exact(left, right)
    assert not state_exact(right, left)


def test_recursive_checkpoint_structure_and_values_remain_exact():
    left = {'model': {'weight': torch.tensor([1., 2.])},
            'optimizer': [{'step': torch.tensor(3), 'moments': (torch.tensor([.25]),)}],
            'rng': np.array([4, 5], dtype=np.uint32), 'scaler': None}
    right = {'model': {'weight': torch.tensor([1., 2.])},
             'optimizer': [{'step': torch.tensor(3), 'moments': (torch.tensor([.25]),)}],
             'rng': np.array([4, 5], dtype=np.uint32), 'scaler': None}
    assert state_exact(left, right)
    assert not state_exact(left, {**right, 'extra': None})
    assert not state_exact(left, {**right, 'optimizer': tuple(right['optimizer'])})
    assert not state_exact(left, {**right, 'optimizer': []})
    assert not state_exact(left, {**right, 'model': {'weight': torch.tensor([1., 3.])}})
    assert not state_exact(left, {**right, 'rng': np.array([4, 6], dtype=np.uint32)})
    assert not state_exact(left, {**right, 'rng': np.array([4, 5], dtype=np.int64)})
    assert not state_exact(left, {**right, 'scaler': 0})


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cpu_checkpoint_matches_cuda_live_state_exactly():
    cpu = {'model': [torch.tensor([1., 2.])], 'optimizer': (torch.tensor(3),)}
    cuda = {'model': [cpu['model'][0].cuda()], 'optimizer': (cpu['optimizer'][0].cuda(),)}
    assert state_exact(cpu, cuda)
    assert state_exact(cuda, cpu)
    cuda['model'][0][1] += 1
    assert not state_exact(cpu, cuda)
    assert not state_exact(cuda, cpu)
