"""E3 denominators, gradient pooling, identity noise, and no-update boundaries."""
import copy

import pytest
import torch

from scripts.experiments.evening_gradients import (
    Budget, ExampleAWGN, component_vjps, feature_rows, gradient_summary,
    run_diagnostic, sample_snrs, validate_fixture,
)
from test_core import toy_model


def test_vjp_structural_zero_and_pooled_direction():
    main = torch.nn.Parameter(torch.tensor([2., -1.]))
    memory = torch.nn.Parameter(torch.tensor([3.]))
    groups = {'main': [('m', main)], 'memory': [('r', memory)]}
    budget = Budget(4)
    result = component_vjps({'kl': (main.square().sum()+memory.square().sum())/7,
                            'hidden': main.square().sum()/2}, groups, budget)
    assert budget.calls == 2
    assert result['hidden']['memory']['structural_zero_parameters'] == ['r']
    torch.testing.assert_close(result['hidden']['memory']['vector'], torch.zeros(1))
    assert main.grad is None and memory.grad is None
    # Pool gradients before cosine: opposite unequal vectors don't average to zero.
    summary = gradient_summary({'kl': torch.tensor([2., 0.]), 'hidden': torch.tensor([-1., 2.])},
                               {'kl': 1., 'hidden': .1})
    assert summary['pairs']['kl:hidden']['dot'] == -2
    assert summary['norms']['weighted_hidden'] == pytest.approx(5**.5/10)
    assert summary['kl_dot_total_weighted_gradient'] == pytest.approx(3.8)


def test_near_zero_has_null_cosine_and_ratio():
    summary = gradient_summary({'kl': torch.zeros(2), 'hidden': torch.ones(2)}, {'kl': 1., 'hidden': .1})
    assert summary['pairs']['kl:hidden']['cosine'] is None
    assert summary['weighted_reconstruction_to_kl_norm_ratio'] is None
    assert summary['ratio_reason'] == 'near_zero_weighted_kl_norm'


def test_budget_counts_failed_attempts_and_stops_before_extra_vjp():
    p = torch.nn.Parameter(torch.ones(1))
    budget = Budget(1)
    with pytest.raises(RuntimeError, match='budget exhausted'):
        component_vjps({'kl': p.sum(), 'hidden': p.sum()}, {'main': [('p', p)]}, budget)
    assert budget.calls == 1


def test_feature_statistics_use_raw_valid_coordinates_per_example():
    original = torch.tensor([[[1., 3.], [999., 999.]], [[2., 4.], [6., 8.]]])
    reconstructed = original * 2
    result = feature_rows(original, reconstructed, torch.tensor([[1, 0], [1, 1]]))
    assert result[0]['original_mean'] == 2
    assert result[0]['original_energy'] == 10
    assert result[0]['valid_coordinates'] == 2
    assert result[0]['nmse'] == pytest.approx(1)
    assert result[1]['valid_coordinates'] == 4
    assert result[1]['nmse'] == pytest.approx(1)


def test_fixed_noise_pairs_row_and_stream_independent_of_batch_order():
    channel = ExampleAWGN()
    z = torch.zeros(2, 3, 4)
    channel.row_ids = [20, 30]
    snrs = sample_snrs(channel.row_ids)
    with channel.replay(0, 'test'):
        hidden = channel.transmit(z, torch.tensor(snrs).reshape(-1, 1, 1), stream='hidden')
        memory = channel.transmit(z, torch.tensor(snrs).reshape(-1, 1, 1), stream='memory')
    channel.row_ids = [30, 20]
    with channel.replay(0, 'test'):
        replay = channel.transmit(z, torch.tensor(sample_snrs(channel.row_ids)).reshape(-1, 1, 1), stream='hidden')
    torch.testing.assert_close(hidden, replay.flip(0), rtol=0, atol=0)
    assert not torch.equal(hidden, memory)
    assert torch.equal(channel(z, None), z)
    assert all(-6 <= snr <= 18 for snr in snrs)


def test_selection_identity_and_seeded_order():
    rows = [{'row_id': i} for i in range(128)]
    ids = {'validation_rows': list(range(128)), 'validation_split': 'train'}
    result = validate_fixture(rows, ids)
    assert result == validate_fixture(rows[::-1], ids)
    assert result != rows
    assert len({r['row_id'] for r in result[:64]}) == 64
    with pytest.raises(ValueError, match='exactly match'):
        validate_fixture(rows[:-1] + [rows[0]], ids)


@pytest.mark.parametrize('stack,expected', [('enc', 32), ('dec', 48)])
def test_full_bounded_toy_route_no_updates_all_rows_and_denominators(stack, expected):
    torch.manual_seed(7)
    model = toy_model(stack)
    model.codec.film = None
    if model.memory_codec is not None:
        model.memory_codec.film = None
    before = copy.deepcopy(model.state_dict())
    rows = [{'row_id': i, 'input_ids': [1, 2] if i % 3 else [1, 2, 3],
             'attention_mask': [1, 1] if i % 3 else [1, 1, 1],
             'label_ids': [4] if i < 32 else [4, 5, 6]} for i in range(128)]
    output = {}
    def write(name, row):
        output.setdefault(name, []).append(row)
    budget = Budget(expected)
    settings = {'temperature': 1., 'kl_weight': 1., 'mse_weight': .1}
    run_diagnostic(model, rows, settings, 0, budget, write, {'route': stack})
    assert budget.calls == expected
    assert budget.teacher_forwards == budget.student_forwards == 32
    assert budget.presentations == 256
    for key, value in before.items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
    assert all(p.grad is None for p in model.parameters())
    pooled = [r for r in output['e3_gradients.jsonl'] if r['scope'] == 'pooled64']
    assert all(r['denominators']['kl'] == 128 for r in pooled)
    assert all(r['denominators']['hidden'] == 64 for r in pooled)
    assert all(r['weights']['hidden'] == (.1 if stack == 'enc' else .05) for r in pooled)
    losses = [r for r in output['e3_losses.jsonl'] if r.get('scope') == 'pooled128']
    assert all(r['components']['kl']['denominator'] == 320 for r in losses)
    features = [r for r in output['e3_features.jsonl'] if 'statistics' in r]
    assert len(features) == 256 * (1 if stack == 'enc' else 2)
    assert {r['row_id'] for r in features} == set(range(128))


def test_unequal_token_microbatches_pool_to_full_objective_gradient():
    from jscc.losses import distillation_loss_stats, reconstruction_loss_stats
    p = torch.nn.Parameter(torch.tensor([.4, -.2]))
    groups = {'main': [('p', p)]}
    labels = torch.tensor([[1, -100, -100], [1, 2, 3], [1, 2, -100]])
    inputs = torch.arange(18, dtype=torch.float32).reshape(3, 3, 2)/10
    teacher = torch.flip(inputs, [-1])
    originals = inputs + .5
    gradients = []
    for indices in [slice(0, 1), slice(1, 3)]:
        kl, _ = distillation_loss_stats(inputs[indices]*p, teacher[indices], labels[indices])
        hidden, _ = reconstruction_loss_stats(originals[indices]*p, originals[indices], labels[indices] != -100)
        components = {'kl': kl/6, 'hidden': hidden/3}
        gradients.append(component_vjps(components, groups, Budget(2)))
    kl, _ = distillation_loss_stats(inputs*p, teacher, labels)
    hidden, _ = reconstruction_loss_stats(originals*p, originals, labels != -100)
    reference = torch.autograd.grad(kl/6 + .1*hidden/3, p)[0]
    pooled = sum(g['kl']['main']['vector'] + .1*g['hidden']['main']['vector'] for g in gradients)
    torch.testing.assert_close(pooled, reference)


def test_selection128_from512_excludes_training_and_is_stable():
    from scripts.experiments.evening_gradients import selection_ids
    ids = {'validation_rows': list(range(512)), 'validation_split': 'train', 'train_rows': [1000]}
    selected = selection_ids(ids)
    assert len(selected) == len(set(selected)) == 128
    assert selected == selection_ids(ids)
    assert selected != list(range(128))
    ids['train_rows'] = [selected[0]]
    with pytest.raises(ValueError, match='overlap'):
        selection_ids(ids)
