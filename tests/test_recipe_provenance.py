"""Trust-boundary failures and hash-free real recipe update consumers."""
from copy import deepcopy
from typing import cast

import pytest
import torch

from jscc.activation_replay import file_digest
from jscc.baseline_protocol import batch_identity, bind_batch_identity
from jscc.original13_provenance import bind_model_assets, verify_model_assets
from jscc.sharing_protocol import batch_view
from test_recipe_training import setup_recipe


def model_binding(tmp_path):
    for name, value in {'config.json': b'{}', 'tokenizer.json': b'{}',
                        'model.safetensors': b'fixture-weight-bytes'}.items():
        (tmp_path / name).write_bytes(value)
    binding = bind_model_assets(tmp_path, 'fixed')
    return {'model': {'name': str(tmp_path), 'revision': 'fixed', 'asset_binding': binding}}


def test_model_startup_reuses_acquisition_and_explicit_audit_hashes(tmp_path, monkeypatch):
    config = model_binding(tmp_path)
    calls = []
    monkeypatch.setattr('jscc.original13_provenance.file_digest',
        lambda p: calls.append(p.name) or file_digest(p))
    verify_model_assets(config)
    assert calls == []
    verify_model_assets(config, verify_bytes=True)
    assert set(calls) == {'config.json', 'tokenizer.json', 'model.safetensors'}
    target = tmp_path / 'model.safetensors'
    target.write_bytes(b'x' * target.stat().st_size)
    verify_model_assets(config)  # Immutable-cache violation requires explicit rebind/audit.
    with pytest.raises(ValueError, match='asset bytes differ'):
        verify_model_assets(config, verify_bytes=True)
    rebound = bind_model_assets(tmp_path, 'fixed')
    assert rebound['files'] != config['model']['asset_binding']['files']


@pytest.mark.parametrize('mutation', ['size', 'missing', 'revision', 'path', 'inventory', 'tokenizer'])
def test_model_transition_failures(tmp_path, mutation):
    config = model_binding(tmp_path)
    binding = config['model']['asset_binding']
    if mutation == 'size':
        (tmp_path / 'model.safetensors').write_bytes(b'new')
    elif mutation == 'missing':
        (tmp_path / 'tokenizer.json').unlink()
    elif mutation == 'revision':
        config['model']['revision'] = 'different'
    elif mutation == 'path':
        binding['files']['../escape.bin'] = {'bytes': 0, 'sha256': 'a' * 64}
    elif mutation == 'inventory':
        del binding['files']['model.safetensors']
    else:
        binding['tokenizer_files']['config.json'] = {'bytes': 2, 'sha256': 'b' * 64}
    with pytest.raises(ValueError):
        verify_model_assets(config)


def test_bound_real_updates_do_not_hash_and_preserve_noise(tmp_path, monkeypatch):
    left, data, _ = setup_recipe(tmp_path, locations=['enc_emb', 'dec_l3'])
    right, other, _ = setup_recipe(tmp_path, locations=['enc_emb', 'dec_l3'])
    binding = bind_batch_identity(data)
    retained_view = batch_view(data)
    assert batch_view(data, binding=binding) == retained_view
    calls = []
    import jscc.baseline_protocol as baseline
    real = baseline.tensor_digest
    monkeypatch.setattr(baseline, 'tensor_digest', lambda value: calls.append(value.numel()) or real(value))
    monkeypatch.setattr('jscc.sharing_protocol.tensor_digest',
        lambda value: calls.append(value.numel()) or real(value))
    for _ in range(2):
        bound = left.update([data], batch_identities=[binding])
        assert calls == []
        direct = right.update([other])
        assert len(calls) == sum(torch.is_tensor(v) for v in other.values())
        calls.clear()
        for key in ('objective', 'snr', 'site_id', 'recipe_progress'):
            assert bound['payload'][key] == direct['payload'][key]
    for a, b in zip(left.parameters, right.parameters):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize('mutation', ['tensor', 'replacement', 'metadata', 'layout'])
def test_bound_input_mutation_rejected_before_update(tmp_path, mutation):
    learner, data, _ = setup_recipe(tmp_path, locations=['enc_emb', 'dec_l3'])
    binding = bind_batch_identity(data)
    if mutation == 'tensor':
        data['input_ids'].add_(1)
    elif mutation == 'replacement':
        data['input_ids'] = data['input_ids'].clone()
    elif mutation == 'metadata':
        cast(dict, data)['new_metadata'] = 'different'
    else:
        data['input_ids'] = data['input_ids'].t().contiguous().t()
    with pytest.raises(ValueError, match='changed after identity binding'):
        learner.update([data], batch_identities=[binding])
    assert learner.completed == learner.attempted == 0


def test_new_input_wrong_view_rejected_and_mask_digest_retained(tmp_path):
    learner, data, _ = setup_recipe(tmp_path, locations=['enc_emb', 'dec_l3'])
    changed = deepcopy(data)
    changed['input_ids'].add_(1)
    binding = bind_batch_identity(changed)
    with pytest.raises(ValueError, match='differs from ordered sharing plan'):
        learner.update([changed], batch_identities=[binding])
    with pytest.raises(ValueError, match='input/view identity mismatch'):
        bind_batch_identity(changed, expected_identity=batch_identity(data))
    assert learner.completed == 0
