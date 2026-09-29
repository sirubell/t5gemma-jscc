"""Pooled32 probe conservation, weight algebra, and isolation on tiny backbones."""
import copy
import random

import numpy as np
import pytest
import torch

from scripts.experiments.evening_gradients import Budget
from scripts.experiments.main_weight_screen_gradients import run_probe, summarize_vectors
from test_core import toy_model


def test_raw_weighted_algebra_includes_cross_stream_total():
    vectors = {'kl': torch.tensor([1., 2.]), 'hidden': torch.tensor([3., -1.]),
               'memory': torch.tensor([-2., 1.])}
    weights = {'kl': 1., 'hidden': .5, 'memory': .05}
    result = summarize_vectors(vectors, weights)
    expected = sum((vectors[k].double()*weights[k] for k in vectors), torch.zeros(2, dtype=torch.float64))
    assert result['total_weighted_norm_before_clip'] == pytest.approx(float(expected.norm()))
    assert result['weighted_pairs']['hidden:memory']['dot'] == pytest.approx(-7*.5*.05)


def test_probe_fixed_denominators_12_vjps_and_no_training_state_pollution():
    model = toy_model('dec')
    model.train()
    before = copy.deepcopy(model.state_dict())
    parameter = next(model.codec.parameters())
    parameter.grad = torch.ones_like(parameter)
    saved_grad = parameter.grad.clone()
    modes = [m.training for m in model.modules()]
    rows = [{'row_id': i, 'input_ids': [1, 2], 'attention_mask': [1, 1],
             'label_ids': [4] if i < 8 else [4, 5]} for i in range(32)]
    random.seed(73)
    np.random.seed(73)
    torch.manual_seed(73)
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    output = {}
    budget = Budget(12)
    run_probe(model, rows, {'temperature': 1., 'kl_weight': 1., 'mse_weight': .1}, 0, .5,
              budget, lambda name, row: output.setdefault(name, []).append(row), {'arm': 'W05'})
    assert budget.calls == 12
    assert budget.teacher_forwards == budget.student_forwards == 4
    assert budget.presentations == 32
    assert random.getstate() == python_state
    np.testing.assert_equal(np.random.get_state(), numpy_state)
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert modes == [m.training for m in model.modules()]
    assert torch.equal(parameter.grad, saved_grad)
    for name, value in before.items():
        torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
    gradients = output['probe_gradients.jsonl']
    assert len(gradients) == 2
    assert all(r['denominators'] == {'kl': 56., 'hidden': 32., 'memory': 32.} for r in gradients)
    final = output['probe_losses.jsonl'][-1]
    assert final['total'] == pytest.approx(sum(final['weighted_components'].values()))
    assert final['global_norm_before_clip'] == pytest.approx(sum(r['total_weighted_norm_before_clip']**2 for r in gradients)**.5)
    assert final['optimizer_updates'] == 0 and not final['clip_applied']
    assert len(output['probe_features.jsonl']) == 64


def test_probe_rejects_incomplete_allowance_before_model_work():
    model = toy_model('dec')
    rows = [{'row_id': i} for i in range(32)]
    with pytest.raises(ValueError, match='12 VJP'):
        run_probe(model, rows, {}, 0, .05, Budget(11), lambda *args: None, {})
