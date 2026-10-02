"""Actual tiny native16 training/freeze/final-scoring lifecycle and failures."""
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest
import torch
import yaml

from jscc.activation_replay import canonical_digest, file_digest
from jscc.baseline_protocol import batch_identity
from jscc.config import load_config
from jscc.sharing_preparation import source_inventory, state_dict_identity, write_prepared
from jscc.sharing_schedule import NATIVE16_LONG_PROTOCOL, NATIVE16_WEIGHT_DECAY, native16_exposure
from jscc.sharing_state import validate_sharing_state, verify_study_freeze
from sharing_fixtures import sharing_model, sharing_batches
from test_shared_lifecycle import geometry_batches, install_adapter, task_template
from test_sharing_native16 import native_batches


def fixture_package(tmp_path):
    source = Path(__file__).resolve().parents[1]
    root = tmp_path / 'prepared'
    root.mkdir()
    model = sharing_model('D-N')
    config = load_config(source / 'configs/tasks/hellaswag.yaml')
    config['model'].update(device='cpu', dtype='float32', revision='tiny-cpu', numerical_policy='native')
    config['codec'] = dict(model.codec.config)
    config['split'] = dict(model.split)
    config['training'].update(batch_size=16, gradient_accumulation=1, lr=2e-4,
        weight_decay=NATIVE16_WEIGHT_DECAY, grad_clip=1, temperature=1, kl_weight=1, mse_weight=.1,
        max_steps=4, schedule_steps=4, patience=None)
    config.update(seed=0, protocol=NATIVE16_LONG_PROTOCOL)
    (root / 'config.yaml').write_text(yaml.safe_dump(config))
    config = load_config(root / 'config.yaml')
    def ref(name):
        return {'path': name, 'sha256': file_digest(root / name)}
    def batch_ref(name, batch):
        torch.save(batch, root / name)
        return {**ref(name), 'view_sha256': batch_identity(batch)}
    updates = [batch_ref(f'update{i}.pt', batch) for i, batch in enumerate(native_batches())]
    geometry = {role: [batch_ref(f'geometry-{role}.pt', values[0])] for role, values in geometry_batches().items()}
    template = task_template()
    template['backend'] = model.base.config.decoder._attn_implementation
    (root / 'task.json').write_text(json.dumps(template))
    torch.save(model.codec.state_dict(), root / 'initial.pt')
    inventory = source_inventory(source)
    with zipfile.ZipFile(root / 'source.zip', 'w') as archive:
        for name in inventory:
            archive.write(source / name, name)
    prepared = write_prepared(root / 'prepared.json', dict(schema='sharing-prepared-v1',
        cell='D-N', synthetic_cpu=True, q=4, batch_size=16, protocol_id=NATIVE16_LONG_PROTOCOL,
        model_revision='tiny-cpu', study_pairing_id='native16-long-test',
        recipe_identity='K+0.1R-native-effective16-fresh-v1', architecture_decision='CPU fixture only',
        hard_cap_seconds=240, exposure_binding=native16_exposure(4),
        initialization_policy={'seed': 0, 'fresh_cpu_codec': True, 'baseline_state_reused': False},
        config=ref('config.yaml'), config_identity=canonical_digest(config), updates=updates,
        validation=[batch_ref('validation.pt', sharing_batches()[0])], task_request=ref('task.json'),
        final_task_request=ref('task.json'), geometry=geometry,
        source_inventory=inventory, source_identity=canonical_digest(inventory), source_archive=ref('source.zip'),
        initialization={**ref('initial.pt'), 'state_identity': state_dict_identity(model.codec.state_dict())},
        data_ids={'synthetic': True}))
    spec = importlib.util.spec_from_file_location('long_study_cli', source / 'scripts/sharing_long_study.py')
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return prepared, cli


def test_train_freeze_and_no_noise_first_final_cli(tmp_path, monkeypatch):
    from jscc import sharing_run, sharing_long_run
    torch.set_num_threads(1)
    calls = install_adapter(monkeypatch)
    prepared, cli = fixture_package(tmp_path)
    monkeypatch.setattr(sharing_run, 'build_model', lambda config: (object(), sharing_model('D-N')))
    monkeypatch.setattr(sharing_long_run, 'build_model', lambda config: (object(), sharing_model('D-N')))
    trained = tmp_path / 'trained'
    result = cli.main(['train', '--prepared', str(prepared), '--output', str(trained), '--timeout-seconds', '240'])
    assert result['status'] == 'training_frozen'
    assert result['completed_updates'] == 24 and result['completed_checkpoints'] == 10
    assert calls == []  # fixed selection objective only; no inspected dev task tuning
    observations = json.loads((trained / 'observations.json').read_text())
    assert len(observations) == 18
    assert all(o['request']['conditions'] == ['no_noise'] for o in observations)
    assert all(o['request']['site'] != 'enc_l14' for o in observations)
    assert verify_study_freeze(result['freeze_reference'])['synthetic']
    states = sorted(trained.rglob('step_*.pt'))
    assert len(states) == 10
    for path in states:
        payload = torch.load(path, map_location='cpu', weights_only=False)
        validate_sharing_state(payload)
        assert payload['optimizer']['param_groups'][0]['weight_decay'] == NATIVE16_WEIGHT_DECAY
        if payload['metadata']['completed_updates'] == payload['metadata']['sharing']['horizon']:
            assert payload['optimizer']['param_groups'][0]['lr'] == 0
    partial = torch.load(trained / 'shared/step_000004.pt', weights_only=False)
    assert partial['metadata']['sharing']['completed_per_site'] == {'enc_l9': 2, 'enc_l19': 1, 'enc_fn': 1}
    final = cli.main(['evaluate', '--prepared', str(prepared), '--training-run', str(trained),
                     '--output', str(tmp_path / 'final'), '--timeout-seconds', '240'])
    assert final['status'] == 'complete' and len(final['completed_panels']) == 42
    assert [condition for _, condition in calls] == ['no_noise'] * 7 + [c for c in (-6, 0, 6, 12, 18) for _ in range(7)]
    shared_hashes = {p['checkpoint'] for p in final['completed_panels'] if p['learner'] == 'shared'}
    assert len(shared_hashes) == 1
    assert sum(p['site'] == 'enc_l14' for p in final['completed_panels']) == 6
    assert not torch.cuda.is_initialized()
    with pytest.raises(ValueError, match='fresh'):
        cli.main(['train', '--prepared', str(prepared), '--output', str(trained), '--timeout-seconds', '240'])
    # Training freeze evidence is immutable. Tampering fails before model loading.
    (trained / 'shared/metrics.jsonl').write_text('')
    monkeypatch.setattr(sharing_long_run, 'build_model', lambda config: pytest.fail('tampered freeze reached model'))
    with pytest.raises(ValueError):
        cli.main(['evaluate', '--prepared', str(prepared), '--training-run', str(trained),
                  '--output', str(tmp_path / 'rejected'), '--timeout-seconds', '240'])


def test_requires_explicit_long_entry_and_timeout(tmp_path, monkeypatch):
    from jscc.sharing_preparation import load_prepared, require_execution_ready
    from jscc.sharing_long_run import configure_long_entry, check_long_deadline
    prepared, _ = fixture_package(tmp_path)
    manifest = load_prepared(prepared)
    monkeypatch.delenv('SHARING_LONG_ENTRY_ID', raising=False)
    with pytest.raises(ValueError, match='sharing_long_study'):
        require_execution_ready(manifest)
    with pytest.raises(ValueError, match='timeout'):
        configure_long_entry(manifest, approved_source=None, timeout_seconds=241)
    configure_long_entry(manifest, approved_source=None, timeout_seconds=240)
    monkeypatch.setenv('SHARING_LONG_DEADLINE_MONOTONIC', '0')
    with pytest.raises(TimeoutError):
        check_long_deadline()
