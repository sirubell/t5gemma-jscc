"""Explicit configuration binding; CPU preparation never enables production."""
from copy import deepcopy
from pathlib import Path

import pytest

from jscc.config import load_config
from jscc.sharing_accumulation import partition_policy
from jscc.sharing_preparation import _config_contract, require_execution_ready


def pair(micro=32, width=1152):
    config = load_config(Path(__file__).resolve().parents[1] / 'configs/tasks/hellaswag.yaml')
    config['model'].update(device='cpu', dtype='float32', revision='tiny-cpu', numerical_policy='native')
    config['split'] = {'stack': 'enc', 'where': 'after_final_norm'}
    config['codec'].update(architecture='direct_outer_ln', layernorm='both', n_res_blocks=0,
                           snr_film=False, dropout=0.0, bottleneck_dim=width)
    config['training'].update(batch_size=micro, gradient_accumulation=64 // micro, lr=2e-4,
                              weight_decay=.01, grad_clip=1, temperature=1, kl_weight=1, mse_weight=.1)
    config['seed'] = 0
    manifest = dict(synthetic_cpu=True, cell='D-LN', batch_size=64, model_revision='tiny-cpu',
                    execution_partition=partition_policy(micro), selected_bottleneck_dim=width)
    return config, manifest


@pytest.mark.parametrize('micro', [32, 16])
@pytest.mark.parametrize('width', [512, 1152, 2304])
def test_exact_partition_and_explicit_width_binding(micro, width):
    config, manifest = pair(micro, width)
    _config_contract(config, manifest)
    manifest['synthetic_cpu'] = False
    _config_contract(config, manifest)
    with pytest.raises(ValueError, match='production execution disabled'):
        require_execution_ready(manifest)


@pytest.mark.parametrize('change', [
    lambda c, m: c['training'].update(gradient_accumulation=1),
    lambda c, m: c['training'].update(batch_size=64),
    lambda c, m: m.update(batch_size=32),
    lambda c, m: m.update(selected_bottleneck_dim=512),
    lambda c, m: m.pop('selected_bottleneck_dim'),
    lambda c, m: m['execution_partition'].update(padding='trim'),
])
def test_contract_rejects_unbound_changes(change):
    config, manifest = pair()
    original = deepcopy(config)
    change(config, manifest)
    with pytest.raises(ValueError):
        _config_contract(config, manifest)
    assert original['training']['gradient_accumulation'] == 2
