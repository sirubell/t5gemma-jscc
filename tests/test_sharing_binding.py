"""Real reviewed input binding is CPU preparation, not candidate selection."""
import json
from pathlib import Path

import pytest
import torch
import yaml

from jscc.config import load_config
from jscc.sharing_binding import bind_production
from jscc.sharing_preparation import load_prepared, require_execution_ready
from test_sharing_candidate_binding import candidate_config

INPUTS = Path('/Users/tim_c_wang/Documents/Codex/2026-09-30/task-11/sharing-production/inputs/production-inputs.json')
DIGEST = 'feccf3232b67f9a99693c04f82cf9646556b2e31031fe9de09b62434ea623a55'


def test_real_inputs_bind_cpu_only(tmp_path):
    # This integration fixture is intentionally local and fails if evidence is unavailable.
    config = load_config(Path(__file__).resolve().parents[1] / 'configs/tasks/hellaswag.yaml')
    inputs = json.loads(INPUTS.read_text())
    config['model'].update(device='cuda', dtype='bfloat16', revision=inputs['model_revision'],
                           numerical_policy='native', attn_implementation='sdpa')
    config['split'] = {'stack': 'enc', 'where': 'after_final_norm'}
    config['codec'] = candidate_config('T-LN', 512)
    config['training'].update(batch_size=16, gradient_accumulation=4)
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    kwargs = dict(inputs_path=INPUTS, inputs_sha256=DIGEST, config_path=path,
        cell='T-LN', bottleneck_dim=512, microbatch_size=16,
        architecture_decision='CPU fixture only; not scientific selection',
        protocol_id='sharing-production-binding-test', study_pairing_id='binding-fixture',
        recipe_identity='K+0.1R-fixture', hard_cap_seconds=18000, output=tmp_path / 'bound')
    prepared = bind_production(**kwargs)
    manifest = load_prepared(prepared)
    assert (manifest['q'], manifest['batch_size']) == (200, 64)
    assert manifest['initialization_policy']['baseline_state_reused'] is False
    assert manifest['task_template']['settings']['batch_size'] == 16
    assert manifest['task_template']['settings']['numerical_policy'] == 'native'
    original = json.loads((INPUTS.parent / inputs['task_template']['path']).read_text())
    for key in ('input_ids', 'source_family_ids', 'prompt_policy', 'noise', 'scorer'):
        assert manifest['task_template'][key] == original[key]
    assert manifest['updates'] == inputs['updates']
    assert manifest['producers'] == inputs['producers']
    assert not torch.cuda.is_initialized()
    with pytest.raises(ValueError, match='production execution disabled'):
        require_execution_ready(manifest)
    with pytest.raises(FileExistsError):
        bind_production(**kwargs)
    with pytest.raises(ValueError, match='checksum'):
        bind_production(**{**kwargs, 'inputs_sha256': 'wrong'})
