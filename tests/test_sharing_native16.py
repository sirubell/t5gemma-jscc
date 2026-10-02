"""Native16 route, complete-pool exposure, and actual fresh CPU CLI lifecycle."""
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
from jscc.presentation import PresentationSampler
from jscc.sharing_preparation import _config_contract, load_prepared, source_inventory, state_dict_identity, write_prepared
from jscc.sharing_protocol import batch_view
from jscc.sharing_schedule import NATIVE16_PROTOCOL, NATIVE16_WEIGHT_DECAY, SharingPlan, native16_exposure
from jscc.sharing_state import build_sharing_state, validate_sharing_state, verify_study_freeze
from sharing_fixtures import sharing_model, sharing_batches
from test_shared_lifecycle import geometry_batches, install_adapter, task_template
from test_sharing_schedule import make_plan


def native_batches():
    result = []
    for i, original in enumerate(sharing_batches()):
        result.append({key: value.repeat(8, 1) for key, value in original.items()
                       if key in ('input_ids', 'attention_mask', 'labels') and torch.is_tensor(value)})
        result[-1]['row_ids'] = torch.arange(i * 16, (i + 1) * 16)
    return result


def test_long_schedule_and_state_bind_per_site_exposure():
    from dataclasses import replace
    plan = replace(make_plan(6400, 16), synthetic=False, protocol_id=NATIVE16_PROTOCOL)
    assert plan.learner('shared').horizon == 19200
    assert plan.learner('specialist_enc_l9').horizon == 6400
    assert native16_exposure(6400)['presentations_per_site'] == 102400
    assert plan.learner('shared').update_at(960).lr_factor == 1
    for index in (0, 959, 960, 19199):
        shared = plan.learner('shared').update_at(index)
        specialist = plan.learner('specialist_' + shared.site).update_at(shared.site_local_index)
        assert shared.view == specialist.view and shared.noise_key == specialist.noise_key
    with pytest.raises(ValueError, match='cycling'):
        replace(plan, views=plan.views[:200] * 32)
    with pytest.raises(ValueError, match='production'):
        replace(plan, protocol_id=None)
    with pytest.raises(ValueError):
        replace(plan, views=plan.views[:-1])
    sharing, stream = build_sharing_state(synthetic=False, learner='shared',
        ordered_view_identities=[v.view_sha256 for v in plan.views], completed_updates=19200,
        source_valid_per_view=[100] * 6400, target_valid_per_view=[90] * 6400,
        padded_per_view=[128] * 6400, batch_size=16, protocol_id=NATIVE16_PROTOCOL)
    assert stream['offset'] == 307200
    assert sharing['completed_per_site'] == dict.fromkeys(('enc_l9', 'enc_l19', 'enc_fn'), 6400)


@pytest.mark.parametrize('width', [512, 1152, 2304])
def test_native16_config_is_explicit(width):
    config = load_config(Path(__file__).resolve().parents[1] / 'configs/tasks/sharing_direct_no_norm_b512.yaml')
    config['protocol'] = NATIVE16_PROTOCOL
    config['codec']['bottleneck_dim'] = width
    config['training'].update(batch_size=16, gradient_accumulation=1, weight_decay=NATIVE16_WEIGHT_DECAY,
                              max_steps=6400, schedule_steps=6400, patience=None)
    manifest = dict(synthetic_cpu=False, q=6400, batch_size=16, cell='D-N', selected_bottleneck_dim=width,
        protocol_id=NATIVE16_PROTOCOL, model_revision=config['model']['revision'],
        recipe_identity='K+0.1R-native-effective16-fresh-v1', exposure_binding=native16_exposure(6400),
        initialization_policy={'seed': 0, 'fresh_cpu_codec': True, 'baseline_state_reused': False})
    _config_contract(config, manifest)
    config['training']['gradient_accumulation'] = 4
    with pytest.raises(ValueError, match='optimizer'):
        _config_contract(config, manifest)
    config['training']['gradient_accumulation'] = 1
    manifest['exposure_binding']['presentations_per_site'] = 25600
    with pytest.raises(ValueError, match='exposure'):
        _config_contract(config, manifest)


def test_actual_native16_prepared_cli_lifecycle(tmp_path, monkeypatch):
    from jscc import sharing_run
    torch.set_num_threads(1)
    install_adapter(monkeypatch)
    model = sharing_model('D-N')
    source = Path(__file__).resolve().parents[1]
    root = tmp_path / 'prepared'
    root.mkdir()
    config = load_config(source / 'configs/tasks/hellaswag.yaml')
    config['model'].update(device='cpu', dtype='float32', revision='tiny-cpu', numerical_policy='native')
    config['codec'] = dict(model.codec.config)
    config['split'] = dict(model.split)
    config['training'].update(batch_size=16, gradient_accumulation=1, lr=2e-4,
        weight_decay=NATIVE16_WEIGHT_DECAY, grad_clip=1, temperature=1, kl_weight=1, mse_weight=.1,
        max_steps=4, schedule_steps=4, patience=None)
    config['seed'] = 0
    config['protocol'] = NATIVE16_PROTOCOL
    (root / 'config.yaml').write_text(yaml.safe_dump(config))
    config = load_config(root / 'config.yaml')
    def ref(name):
        return {'path': name, 'sha256': file_digest(root / name)}
    def batch_ref(name, batch):
        torch.save(batch, root / name)
        return {**ref(name), 'view_sha256': batch_identity(batch)}
    updates = [batch_ref(f'update{i}.pt', batch) for i, batch in enumerate(native_batches())]
    validation = [batch_ref('validation.pt', sharing_batches()[0])]
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
        cell='D-N', synthetic_cpu=True, q=4, batch_size=16, protocol_id=NATIVE16_PROTOCOL,
        model_revision='tiny-cpu', study_pairing_id='native16-test',
        recipe_identity='K+0.1R-native-effective16-fresh-v1', architecture_decision='CPU fixture only',
        hard_cap_seconds=240, exposure_binding=native16_exposure(4),
        initialization_policy={'seed': 0, 'fresh_cpu_codec': True, 'baseline_state_reused': False},
        config=ref('config.yaml'), config_identity=canonical_digest(config), updates=updates,
        validation=validation, task_request=ref('task.json'), geometry=geometry,
        source_inventory=inventory, source_identity=canonical_digest(inventory), source_archive=ref('source.zip'),
        initialization={**ref('initial.pt'), 'state_identity': state_dict_identity(model.codec.state_dict())},
        data_ids={'synthetic': True}))
    monkeypatch.setattr(sharing_run, 'build_model', lambda config: (object(), sharing_model('D-N')))
    spec = importlib.util.spec_from_file_location('native16_cli_test', source / 'scripts/shared_codec.py')
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    output = tmp_path / 'executed'
    result = cli.main(['--execute', str(prepared), '--output', str(output)])
    assert result['status'] == 'complete'
    assert result['completed_updates'] == 24 and result['completed_checkpoints'] == 17
    assert verify_study_freeze(result['freeze_reference'])['synthetic']
    for name in SharingPlan(tuple(batch_view(b) for b in native_batches()), 'native16-test',
                            synthetic=True, protocol_id=NATIVE16_PROTOCOL).learners:
        states = sorted((output / name).glob('step_*.pt'))
        payload = torch.load(states[-1], map_location='cpu', weights_only=False)
        validate_sharing_state(payload)
        assert payload['metadata']['sharing']['batch_size'] == 16
        assert payload['optimizer']['param_groups'][0]['weight_decay'] == NATIVE16_WEIGHT_DECAY
        assert payload['optimizer']['param_groups'][0]['lr'] == 0
        assert payload['stream']['offset'] == payload['metadata']['completed_updates'] * 16
    initial = torch.load(output / 'shared/step_000000.pt', weights_only=False)
    final = torch.load(output / 'shared/step_000012.pt', weights_only=False)
    assert any(not torch.equal(initial['model'][k], final['model'][k]) for k in initial['model'])
    with pytest.raises(ValueError, match='fresh'):
        cli.main(['--execute', str(prepared), '--output', str(output)])
    assert load_prepared(prepared)['batch_size'] == 16
    assert not torch.cuda.is_initialized()


def test_complete_pool_crosses_epochs_without_subset_cycle():
    pool = list(range(1000))
    stream = PresentationSampler(pool, 102400, seed=20260920).actual_ids.tolist()
    assert len(set(stream[:1000])) == 1000
    assert len(set(stream[1000:2000])) == 1000
    assert stream[:3200] != stream[3200:6400]


@pytest.mark.parametrize('few_steps_only', [False, True])
def test_actual_native16_readiness_worker(tmp_path, monkeypatch, few_steps_only):
    from copy import deepcopy
    from jscc import evaluation, sharing_qualification as qualification
    from test_sharing_qualification import stress_parent
    from jscc.sharing_accumulation import partition_native64
    torch.set_num_threads(1)
    model = sharing_model('D-N')
    references = []
    for index in range(8):
        batch = partition_native64(stress_parent(index), microbatch_size=16)[0]
        path = tmp_path / f'batch{index}.pt'
        torch.save(batch, path)
        references.append({'path': path.name, 'sha256': file_digest(path), 'view_sha256': batch_identity(batch)})
    initial = deepcopy(model.codec.state_dict())
    manifest = dict(synthetic_cpu=True, cell='D-N', selected_bottleneck_dim=8,
        resolved_config={'codec': dict(model.codec.config), 'model': {'device': 'cpu', 'dtype': 'float32'},
                         'training': {'deterministic_algorithms': True}},
        prepared_identity='CPU-only', source_identity='source', config_identity='config',
        model_revision='tiny-cpu', recipe_identity='native16', architecture_decision='CPU-only',
        protocol_id=NATIVE16_PROTOCOL, q=8, batch_size=16, updates=references, validation=[references[0]],
        geometry={}, task_request={}, data_ids={}, initialization_state=initial,
        initialization={'state_identity': state_dict_identity(initial)},
        task_template=task_template(), study_pairing_id='native16-diagnostic',
        readiness_cases=[{'label': label, 'batch': reference} for label, reference in zip(
            ('representative', 'longest_source', 'longest_target'), references)])
    panel = {'documents': [{'sample_id': i} for i in range(16)],
             'prompts': [{'prompt_id': f'p{i}'} for i in range(16)],
             'fewshots': [{'fewshot_id': f'd{i}'} for i in range(16)]}
    def scorer(model, processor, settings, output, condition, data_settings):
        assert not few_steps_only, 'runnability probe must not perform task evaluation'
        assert settings['num_samples'] == 16
        assert condition in ('no_noise', 0)
        for key, rows in panel.items():
            (output / (key + '.jsonl')).write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return {'candidate_forward_requests': 64}
    monkeypatch.setattr(evaluation, 'evaluate_hellaswag', scorer)
    result = qualification.run_diagnostic(manifest, tmp_path, tmp_path, lambda: None,
                                          expected_panel=panel, model_bundle=(object(), model),
                                          few_steps_only=few_steps_only)
    if few_steps_only:
        assert result['completed_updates'] == 6 and result['scientific_updates'] == 0
        assert result['per_site_updates'] == dict.fromkeys(('enc_l9', 'enc_l19', 'enc_fn'), 2)
        assert result['checkpoint']['exact'] and result['restored_fresh_state']
        assert result['model_loading_seconds'] >= 0 and result['six_updates_seconds'] > 0
        assert result['task_documents'] == 0 and result['objective_panels'] == 0
        qualification.validate_result(result, {'probe_mode': 'native16-runnability-v1',
                                              'binding': qualification.binding_identity(manifest)})
        assert state_dict_identity(model.codec.state_dict()) == state_dict_identity(initial)
        assert not torch.cuda.is_initialized()
        return
    assert result['completed_updates'] == 30 and result['scientific_updates'] == 0
    assert result['shared_updates'] == 6
    assert result['shared_per_site_updates'] == dict.fromkeys(('enc_l9', 'enc_l19', 'enc_fn'), 2)
    qualification.validate_result(result, {'probe_mode': 'native16-readiness-v1',
                                          'binding': qualification.binding_identity(manifest)})
    assert state_dict_identity(model.codec.state_dict()) == state_dict_identity(initial)
    for record in result['sites'].values():
        assert record['guard']['restored_second_update_exact']
        assert record['guard']['nonzero_lr_update_changed_weights']
        assert all(row['exposure']['sequences'] == 16 for row in record['stress_timings'])
    assert not torch.cuda.is_initialized()


def test_native16_probe_cannot_enter_without_owner_payload(tmp_path):
    from jscc.sharing_qualification import _admission
    from test_sharing_qualification import ref
    payload = tmp_path / 'payload.json'
    payload.write_text('{}')
    with pytest.raises(ValueError, match='coordinator-approved'):
        _admission({'probe_mode': 'native16-readiness-v1', 'payload_receipt': ref(payload)})


@pytest.mark.parametrize('defect', ['absent', 'missing_file', 'byte_hash', 'view_hash', 'duplicate', 'batch_size', 'label_rows'])
def test_readiness_inputs_fail_before_model_construction(tmp_path, monkeypatch, defect):
    from copy import deepcopy
    from jscc import sharing_qualification as qualification
    from jscc.models import split_model
    from jscc.sharing_preparation import read_native16_readiness_cases
    cases = []
    for label, batch in zip(('representative', 'longest_source', 'longest_target'), native_batches()):
        path = tmp_path / f'{label}.pt'
        if defect == 'batch_size' and label == 'longest_source':
            batch = {key: value[:8] for key, value in batch.items()}
        if defect == 'label_rows' and label == 'longest_source':
            batch['labels'] = batch['labels'][:8]
        torch.save(batch, path)
        cases.append({'label': label, 'batch': {'path': path.name,
            'sha256': file_digest(path), 'view_sha256': batch_identity(batch)}})
    if defect == 'absent':
        cases = None
    elif defect == 'missing_file':
        cases[1]['batch']['path'] = 'missing.pt'
    elif defect in ('byte_hash', 'view_hash'):
        cases[1]['batch']['sha256' if defect == 'byte_hash' else 'view_sha256'] = '0' * 64
    elif defect == 'duplicate':
        cases[1]['batch'] = deepcopy(cases[0]['batch'])
    with pytest.raises(ValueError):
        read_native16_readiness_cases(tmp_path, cases)
    def forbidden_model(config):
        raise AssertionError('malformed inputs reached model construction')
    monkeypatch.setattr(split_model, 'build_model', forbidden_model)
    with pytest.raises(ValueError):
        qualification.run_diagnostic({'protocol_id': NATIVE16_PROTOCOL, 'readiness_cases': cases},
                                      tmp_path, tmp_path, lambda: None, expected_panel={})
