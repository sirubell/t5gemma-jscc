import copy
import hashlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from scripts import benchmark_speed as benchmark


@pytest.fixture
def checkpoint_boundary(monkeypatch):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    saved_optimizer = torch.optim.AdamW([parameter], lr=0.003, weight_decay=0.04)
    parameter.grad = torch.ones_like(parameter)
    saved_optimizer.step()
    parameter.grad = None
    state = {
        'config': {
            'task': 'hellaswag',
            'model': {'device': 'cpu', 'dtype': 'float32', 'revision': 'fixed-model'},
            'training': {'batch_size': 1, 'gradient_accumulation': 1,
                         'lr': 0.01, 'weight_decay': 0.1},
            'split': {'stack': 'dec', 'index': 8},
            'codec': {'bottleneck_dim': 4},
            'data': {'selection_seed': 7},
        },
        'data_ids': {'train': [9, 2], 'validation': [1]},
        'optimizer': saved_optimizer.state_dict(),
    }
    calls = {}
    model = SimpleNamespace(parameters=lambda: [parameter],
                            load_communication_state=Mock(), train=Mock(),
                            split=state['config']['split'],
                            codec=SimpleNamespace(config=state['config']['codec']))
    load = Mock(return_value=state)
    monkeypatch.setattr(benchmark.torch, 'load', load)

    def build(config):
        calls['build_config'] = copy.deepcopy(config)
        return 'processor', model

    def data(config, processor, ids):
        calls['data_config'] = copy.deepcopy(config)
        calls['data_ids'] = ids
        assert processor == 'processor'
        return SimpleNamespace(train=[{'fixture': True}])

    updates = []

    def update(model_arg, loader, iterator, optimizer, parameters, settings,
               variant, snr, valid_only_kl, logging_mode='none', capture_log=False):
        assert model_arg is model
        assert parameters == [parameter]
        updates.append({'settings': copy.deepcopy(settings), 'variant': variant,
                        'snr': snr, 'valid_only_kl': valid_only_kl,
                        'logging_mode': logging_mode, 'capture_log': capture_log})
        calls['optimizer'] = optimizer
        batch = {'attention_mask': torch.ones(settings['batch_size'], 3),
                 'labels': torch.ones(settings['batch_size'], 2)}
        return (iterator, [batch] * settings['gradient_accumulation'],
                {'loss': torch.tensor(1.)}, 0.5, None, 0.01)

    sampler = Mock()
    sampler.__enter__ = Mock(return_value=sampler)
    sampler.__exit__ = Mock(return_value=False)
    sampler.summary.return_value = {'minimum_free_gib': None}
    monkeypatch.setattr(benchmark, 'GpuSampler', lambda: sampler)
    monkeypatch.setattr(benchmark, 'build_model', build)
    monkeypatch.setattr(benchmark, 'load_data', data)
    monkeypatch.setattr(benchmark, 'prepare_trainable_parameters', Mock())
    promote = Mock()
    seed = Mock()
    monkeypatch.setattr(benchmark, 'promote_optimizer_state', promote)
    monkeypatch.setattr(benchmark, 'seed_everything', seed)
    monkeypatch.setattr(benchmark, '_one_update', update)
    monkeypatch.setattr(benchmark.torch.cuda, 'is_available', lambda: False)
    profile = Mock(side_effect=lambda *args: args[-2] / 'fixture.trace.json')
    monkeypatch.setattr(benchmark, '_profile_update', profile)
    return SimpleNamespace(state=state, calls=calls, load=load, updates=updates,
                           model=model, profile=profile, seed=seed, promote=promote)


@pytest.mark.parametrize('explicit_defaults', [False, True])
@pytest.mark.parametrize('batch,accumulation,variant', [(16, 2, 'streamed'), (8, 4, 'reference')])
def test_checkpoint_overrides_and_executed_settings_define_identity(
        checkpoint_boundary, tmp_path, explicit_defaults, batch, accumulation, variant):
    boundary = checkpoint_boundary
    if explicit_defaults:
        boundary.state['config']['training'].update(
            {'grad_clip': 0.2, 'kl_weight': 0.6, 'mse_weight': 0.3})
    checkpoint = tmp_path / 'checkpoint.pt'
    result = benchmark.run_repeat(checkpoint, tmp_path, variant, batch, accumulation,
                                  1, 2, 3, 5.0, True, 'low', True)
    boundary.load.assert_called_once_with(checkpoint, map_location='cpu', weights_only=True)
    config = result['effective_config']
    assert config['model'] == {'device': 'cuda', 'dtype': 'bfloat16', 'revision': 'fixed-model'}
    assert config['training']['batch_size'] == batch
    assert config['training']['gradient_accumulation'] == accumulation
    assert config['training']['grad_clip'] == (0.2 if explicit_defaults else 1.0)
    assert config['training']['kl_weight'] == (0.6 if explicit_defaults else 1.0)
    assert config['training']['mse_weight'] == (0.3 if explicit_defaults else 0.1)
    for stage in ('build_config', 'data_config'):
        expected = copy.deepcopy(config)
        if not explicit_defaults:
            for key in ('grad_clip', 'kl_weight', 'mse_weight'):
                expected['training'].pop(key)
        assert boundary.calls[stage] == expected
    assert boundary.calls['data_ids'] is boundary.state['data_ids']
    boundary.model.load_communication_state.assert_called_once_with(boundary.state)
    boundary.seed.assert_called_once_with(9003)
    boundary.promote.assert_called_once_with(boundary.calls['optimizer'])
    optimizer = boundary.calls['optimizer']
    assert optimizer.param_groups[0]['lr'] == 0.003
    assert optimizer.param_groups[0]['weight_decay'] == 0.04
    assert next(iter(optimizer.state.values()))['step'].item() == 1
    assert len(boundary.updates) == 3
    assert all(row['settings'] == config['training'] for row in boundary.updates)
    assert all(row['snr'] == 5.0 and row['valid_only_kl'] for row in boundary.updates)
    assert all(row['variant'] == variant for row in boundary.updates)
    assert boundary.updates[-1]['logging_mode'] == 'low'
    boundary.profile.assert_called_once()
    assert boundary.profile.call_args.args[5] == config['training']
    assert result['effective_batch_size'] == 32
    assert result['measured_examples'] == 64
    assert result['valid_source_tokens'] == 192
    assert result['valid_target_tokens'] == 128
    assert result['allocated_payload'] == result['valid_payload'] == 1280
    assert result['effective_config_hash'] == hashlib.sha256(json.dumps(
        config, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    changed = copy.deepcopy(config)
    changed['model']['revision'] = 'another-model'
    assert result['effective_config_hash'] != hashlib.sha256(json.dumps(
        changed, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def test_cli_reports_checkpoint_and_effective_identity(checkpoint_boundary, tmp_path, monkeypatch):
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'mocked checkpoint bytes')
    output = tmp_path / 'output'
    monkeypatch.setattr(sys, 'argv', ['benchmark_speed.py', '--checkpoint', str(checkpoint),
                                    '--output', str(output), '--variant', 'streamed',
                                    '--warmup', '0', '--measured', '1'])
    benchmark.main()
    result = json.loads((output / 'benchmark-repeats.jsonl').read_text())
    assert result['checkpoint_hash'] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert result['checkpoint_path'] == str(checkpoint.resolve())
    assert result['effective_config']['training'] == checkpoint_boundary.updates[0]['settings']
    assert not {'config_hash', 'config_path'} & result.keys()


@pytest.mark.parametrize('extra,error', [(['--config', 'unused.yaml'], 'unrecognized arguments'),
                                        (['--batch-size', '1'], 'effective batch must remain 32')])
def test_unsupported_config_and_effective_batch_reject_before_work(tmp_path, monkeypatch, capsys,
                                                                 extra, error):
    output = tmp_path / 'uncreated'
    load = Mock(side_effect=AssertionError('checkpoint must not load'))
    run = Mock(side_effect=AssertionError('benchmark must not run'))
    monkeypatch.setattr(benchmark.torch, 'load', load)
    monkeypatch.setattr(benchmark, 'run_repeat', run)
    monkeypatch.setattr(sys, 'argv', ['benchmark_speed.py', '--checkpoint', 'absent.pt',
                                    '--output', str(output), '--variant', 'streamed', *extra])
    with pytest.raises(SystemExit) as caught:
        benchmark.main()
    if extra[0] == '--config':
        assert caught.value.code == 2
        assert error in capsys.readouterr().err
    else:
        assert error in str(caught.value)
    load.assert_not_called()
    run.assert_not_called()
    assert not output.exists()
