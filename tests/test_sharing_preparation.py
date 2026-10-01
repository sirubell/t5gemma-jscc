"""CPU-only package verification and connected CLI delegation guards."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import zipfile

import pytest
import torch
import yaml

from jscc.activation_replay import canonical_digest, file_digest
from jscc.baseline_protocol import batch_identity
from jscc.config import load_config
from jscc import sharing_preparation as preparation


def publish(path, manifest):
    manifest = copy.deepcopy(manifest)
    manifest.pop('manifest_identity', None)
    manifest['manifest_identity'] = canonical_digest(manifest)
    path.write_text(json.dumps(manifest))
    return path


@pytest.fixture
def package(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    (source / 'jscc').mkdir(parents=True)
    (source / 'scripts').mkdir()
    for name in ('jscc/example.py', 'scripts/shared_codec.py', 'scripts/sharing_production.py', 'scripts/sharing_qualification.py', 'scripts/render_experiment_report.py',
                 'uv.lock', 'pyproject.toml'):
        (source / name).write_text('# source bytes\n')
    inventory = preparation.source_inventory(source)
    real_inventory = preparation.source_inventory
    monkeypatch.setattr(preparation, 'source_inventory', lambda source_root=None: real_inventory(source))
    root = tmp_path / 'package'
    root.mkdir()
    archive = root / 'source.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        for name in inventory:
            bundle.write(source / name, name)
    config = load_config(Path(__file__).resolve().parents[1] / 'configs/tasks/hellaswag.yaml')
    config['model'].update(device='cpu', dtype='float32', revision='tiny-cpu', numerical_policy='native')
    config['split'] = {'stack': 'enc', 'where': 'after_final_norm'}
    config['codec'].update(architecture='direct_outer_ln', layernorm='both', n_res_blocks=0,
                           snr_film=False, dropout=0.0, bottleneck_dim=4)
    config['training'].update(batch_size=2, gradient_accumulation=1, lr=2e-4,
                              weight_decay=.01, grad_clip=1, temperature=1,
                              kl_weight=1, mse_weight=.1)
    config['seed'] = 0
    config_path = root / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    config = load_config(config_path)
    batch = {'input_ids': torch.tensor([[1, 2, 3], [1, 4, 0]]),
             'attention_mask': torch.tensor([[1, 1, 1], [1, 1, 0]]),
             'labels': torch.tensor([[2, 3], [4, -100]])}
    torch.save(batch, root / 'batch.pt')
    state = {'codec.weight': torch.ones(2, 3)}
    torch.save(state, root / 'initial.pt')
    template = {'task': 'hellaswag', 'expected_items': 2,
                'conditions': ['no_noise', -6, 0, 6, 12, 18],
                'input_ids': [1, 2], 'source_family_ids': ['family1', 'family2'],
                'prompt_policy': 'synthetic-native', 'noise': {'seed': 0, 'namespace': 'eval'},
                'layout': 'native-batch2', 'backend': 'eager', 'precision': 'fp32',
                'scorer': 'tiny-cpu-fixture', 'reference_corpus': 'synthetic',
                'settings': {'num_samples': 2, 'numerical_policy': 'native'}, 'data_settings': {}}
    (root / 'task.json').write_text(json.dumps(template))
    def ref(name):
        return {'path': name, 'sha256': file_digest(root / name)}
    batch_ref = {**ref('batch.pt'), 'view_sha256': batch_identity(batch)}
    geometry_refs = {}
    for partition in ('train', 'selection'):
        geometry_batch = {**batch, 'query_ids': [f'{partition}-1', f'{partition}-2'],
                          'view_ids': ['view1', 'view2'],
                          'token_positions': torch.tensor([[0, 1, 2], [0, 1, 2]]),
                          'token_roles': [['demo', 'query', 'query'], ['demo', 'query', 'pad']]}
        name = f'geometry-{partition}.pt'
        torch.save(geometry_batch, root / name)
        geometry_refs[partition] = [{**ref(name), 'view_sha256': batch_identity(geometry_batch)}]
    manifest = dict(schema=preparation.SCHEMA, cell='D-LN', synthetic_cpu=True, q=4,
                    batch_size=2, protocol_id='sharing-synthetic-test-v1', model_revision='tiny-cpu',
                    study_pairing_id='cpu-pairing', recipe_identity='fixed-recipe',
                    architecture_decision='D-LN', hard_cap_seconds=120,
                    config=ref('config.yaml'), config_identity=canonical_digest(config),
                    updates=[batch_ref] * 4, validation=[batch_ref], task_request=ref('task.json'),
                    geometry=geometry_refs,
                    source_inventory=inventory, source_identity=canonical_digest(inventory),
                    source_archive=ref('source.zip'),
                    initialization={**ref('initial.pt'), 'state_identity': preparation.state_dict_identity(state)},
                    data_ids={'synthetic': True})
    path = publish(root / 'prepared.json', manifest)
    return path, manifest, source


def test_complete_snapshot_roundtrip_and_preview(package):
    path, manifest, _ = package
    verified = preparation.load_prepared(path)
    assert torch.equal(verified['initialization_state']['codec.weight'], torch.ones(2, 3))
    assert verified['task_template']['expected_items'] == 2
    assert verified['resolved_config']['model']['device'] == 'cpu'
    assert preparation.preview(verified)['physical_updates'] == 24
    assert preparation.preview(verified)['task_panels'] == 108
    preparation.require_execution_ready(verified)
    other = preparation.write_prepared(path.parent / 'second.json', manifest)
    assert preparation.load_prepared(other)['prepared_identity'] == verified['prepared_identity']
    with pytest.raises(FileExistsError):
        preparation.write_prepared(other, manifest)


@pytest.mark.parametrize('field,value', [
    ('q', 8), ('batch_size', 64), ('synthetic_cpu', False), ('cell', None),
    ('architecture_decision', ''), ('hard_cap_seconds', 0), ('protocol_id', ''),
    ('updates', []), ('validation', []), ('geometry', {'enc_l14': []}),
    ('source_identity', 'wrong'), ('config_identity', 'wrong'), ('model_revision', 'wrong'),
])
def test_manifest_contract_rejections(package, field, value):
    path, manifest, _ = package
    manifest[field] = value
    publish(path, manifest)
    with pytest.raises(ValueError):
        preparation.load_prepared(path)


@pytest.mark.parametrize('name', ['config.yaml', 'batch.pt', 'task.json', 'initial.pt', 'source.zip'])
def test_byte_tampering_rejected_before_use(package, name):
    path, _, _ = package
    with (path.parent / name).open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        preparation.load_prepared(path)


def test_manifest_source_and_tensor_identity_tampering(package):
    path, manifest, source = package
    raw = json.loads(path.read_text())
    raw['q'] = 8
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='manifest identity'):
        preparation.load_prepared(path)
    manifest['initialization']['state_identity'] = 'bad'
    publish(path, manifest)
    with pytest.raises(ValueError, match='initial codec'):
        preparation.load_prepared(path)
    (source / 'jscc/example.py').write_text('# changed')
    with pytest.raises(ValueError, match='source inventory'):
        preparation.load_prepared(path)


def test_unbound_include_and_reference_escape(package):
    path, manifest, _ = package
    manifest['updates'][0]['path'] = '../source/jscc/example.py'
    publish(path, manifest)
    with pytest.raises(ValueError, match='escapes'):
        preparation.load_prepared(path)
    config_path = path.parent / 'config.yaml'
    config = yaml.safe_load(config_path.read_text())
    config['model_config'] = 'external.yaml'
    config_path.write_text(yaml.safe_dump(config))
    manifest['config']['sha256'] = file_digest(config_path)
    publish(path, manifest)
    with pytest.raises(ValueError, match='complete resolved'):
        preparation.load_prepared(path)


def test_missing_bound_production_acceptance():
    with pytest.raises(ValueError, match='production execution disabled'):
        preparation.require_execution_ready({'synthetic_cpu': False, 'gpu_acceptance': None})


def test_cli_preview_and_executor_delegation(package, monkeypatch, capsys):
    path, _, _ = package
    script = Path(__file__).resolve().parents[1] / 'scripts/shared_codec.py'
    spec = importlib.util.spec_from_file_location('shared_codec_cli_test', script)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert cli.main(['--preview', str(path)])['objective_panels'] == 192
    called = []
    def execute(manifest, root, output):
        called.append((manifest, root, output))
        return {'status': 'delegated-fixture'}
    monkeypatch.setitem(sys.modules, 'jscc.sharing_run', types.SimpleNamespace(execute_prepared=execute))
    output = path.parent / 'fresh-output'
    assert cli.main(['--execute', str(path), '--output', str(output)])['status'] == 'delegated-fixture'
    assert called[0][1:] == (path.parent, output)
    assert called[0][0]['resolved_config']['task'] == 'hellaswag'
    output.mkdir()
    with pytest.raises(ValueError, match='fresh'):
        cli.main(['--execute', str(path), '--output', str(output)])
    assert len(called) == 1
    capsys.readouterr()


@pytest.mark.parametrize('cell,architecture,blocks', [('D-LN', 'direct_outer_ln', 0),
                                                    ('R-LN', 'residual_mlp', 2)])
def test_both_candidates_without_selection_default(package, cell, architecture, blocks):
    path, manifest, _ = package
    config_path = path.parent / 'config.yaml'
    config = yaml.safe_load(config_path.read_text())
    config['codec'].update(architecture=architecture, n_res_blocks=blocks)
    config_path.write_text(yaml.safe_dump(config))
    manifest.update(cell=cell, architecture_decision=cell,
                    config_identity=canonical_digest(load_config(config_path)))
    manifest['config']['sha256'] = file_digest(config_path)
    publish(path, manifest)
    assert preparation.load_prepared(path)['cell'] == cell


def test_wrong_batch_view_binding_and_initial_tensor(package):
    path, manifest, _ = package
    manifest['updates'][0]['view_sha256'] = 'f' * 64
    publish(path, manifest)
    with pytest.raises(ValueError, match='input/view'):
        preparation.load_prepared(path)
    with pytest.raises(ValueError, match='codec tensor state'):
        preparation.state_dict_identity({'metadata': 'not-weights'})


@pytest.mark.parametrize('field,value', [
    ('input_ids', [1, 1]), ('source_family_ids', ['family1']), ('prompt_policy', ''),
    ('noise', {'seed': 0}), ('layout', ''), ('backend', ''), ('precision', 'bf16'),
    ('scorer', ''), ('reference_corpus', ''), ('settings', {'num_samples': 3}),
    ('settings', {'num_samples': 2, 'numerical_policy': 'codec_receiver_fp32'}),
    ('data_settings', None),
])
def test_task_contract_fails_before_model_construction(package, field, value):
    path, manifest, _ = package
    target = path.parent / 'task.json'
    template = json.loads(target.read_text())
    template[field] = value
    target.write_text(json.dumps(template))
    manifest['task_request']['sha256'] = file_digest(target)
    publish(path, manifest)
    with pytest.raises(ValueError):
        preparation.load_prepared(path)


@pytest.mark.parametrize('change', ['missing_roles', 'position_shape', 'invalid_roles', 'partition_overlap'])
def test_geometry_contract_fails_before_model_construction(package, change):
    path, manifest, _ = package
    target = path.parent / 'geometry-selection.pt'
    batch = torch.load(target, weights_only=True)
    if change == 'missing_roles':
        del batch['token_roles']
    elif change == 'position_shape':
        batch['token_positions'] = torch.tensor([[0], [1]])
    elif change == 'invalid_roles':
        batch['token_roles'][0][0] = 'unknown'
    else:
        batch['query_ids'] = ['train-1', 'train-2']
    torch.save(batch, target)
    manifest['geometry']['selection'][0].update(sha256=file_digest(target), view_sha256=batch_identity(batch))
    publish(path, manifest)
    with pytest.raises(ValueError):
        preparation.load_prepared(path)


def test_accepted_receipt_does_not_enable_production_execution(package):
    path, manifest, _ = package
    acceptance_path = path.parent / 'acceptance.json'
    receipt = {key: manifest[key] for key in ('source_identity', 'config_identity', 'recipe_identity', 'model_revision')}
    receipt['status'] = 'accepted'
    acceptance_path.write_text(json.dumps(receipt))
    manifest['gpu_acceptance'] = {'path': 'acceptance.json', 'sha256': file_digest(acceptance_path)}
    publish(path, manifest)
    loaded = preparation.load_prepared(path)
    loaded['synthetic_cpu'] = False
    with pytest.raises(ValueError, match='production execution disabled without controller authorization'):
        preparation.require_execution_ready(loaded)
    assert preparation.preview(loaded)['gpu_execution_ready'] is False
    receipt['status'] = 'pending'
    acceptance_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='checksum'):
        preparation.load_prepared(path)
    manifest['gpu_acceptance']['sha256'] = file_digest(acceptance_path)
    publish(path, manifest)
    with pytest.raises(ValueError, match='not bound'):
        preparation.load_prepared(path)


@pytest.fixture
def production_roles(tmp_path):
    """Small actual v2 producer views exercise roles without a 200-batch run."""
    from jscc.activation_replay import SEQUENCE_SCHEMA, sequence_view_v2

    root = tmp_path / 'role-package'
    root.mkdir()
    def ref(name):
        return {'path': name, 'sha256': file_digest(root / name)}
    content = b'# retained producer source\n'
    source_files = {name: {'content_hex': content.hex(), 'bytes': len(content),
                           'sha256': hashlib.sha256(content).hexdigest()}
                    for name in ('uv.lock', 'jscc/activation_replay.py')}
    role_rows = {'optimization': [1, 2], 'objective_validation': [10, 11],
                 'geometry_training': [3, 4], 'geometry_selection': [10, 12]}
    groups, producers = {}, {}
    for role, rows in role_rows.items():
        batch = {'input_ids': torch.tensor([[1, 2], [3, 4]]),
                 'attention_mask': torch.ones(2, 2, dtype=torch.long),
                 'labels': torch.tensor([[3, 4], [4, 5]]),
                 'decoder_input_ids': torch.tensor([[2, 3], [2, 4]]),
                 'decoder_attention_mask': torch.ones(2, 2, dtype=torch.bool),
                 'row_ids': torch.tensor(rows), 'batch_view_id': role,
                 'view_ids': [f'view-{row}' for row in rows],
                 'source_family_ids': [f'family-{row}' for row in rows],
                 'demo_ids': [[5], [5]], 'position_roles': [['demo', 'query']] * 2,
                 'source_views': [{'row_id': row, 'native_prefix': 'reference',
                                   'demo_source_family_ids': ['family-5']} for row in rows]}
        name = role + '.pt'
        torch.save(batch, root / name)
        groups[role] = [{**ref(name), 'view_sha256': batch_identity(batch)}]
        spec = dict(schema=SEQUENCE_SCHEMA, clean_bypass=True, data_role=role,
                    disjointness={'excluded_source_family_ids': ['never-used']},
                    backbone={'model_revision': 'pinned', 'tokenizer_revision': 'pinned',
                              'weights_sha256': 'weights', 'config_sha256': 'config'},
                    environment={'libraries': {'torch': torch.__version__},
                                 'backend': 'eager', 'activation_dtype': 'torch.float32'},
                    data={'task': 'hellaswag', 'fingerprints': {'train': 'data'},
                          'source_family_ids': [f'family-{row}' for row in
                                                (range(1, 6) if role == 'optimization' else rows)]},
                    policy={'native_prefix': 'reference'}, sites=[{'site_id': 'enc_fn'}],
                    views=[sequence_view_v2(batch)], source_files=copy.deepcopy(source_files))
        name = role + '.json'
        (root / name).write_text(json.dumps(spec))
        producers[role] = ref(name)
    template = {'input_ids': [100, 101], 'source_family_ids': ['dev-100', 'dev-101']}
    manifest = {'model_revision': 'pinned', 'producers': producers,
                'updates': groups['optimization'], 'validation': groups['objective_validation'],
                'geometry': {'train': groups['geometry_training'], 'selection': groups['geometry_selection']},
                'data_ids': {'train_rows': list(range(1, 6)), 'validation_rows': [10, 11, 12, 13],
                             'train_source_family_ids': [f'family-{row}' for row in range(1, 6)],
                             'validation_source_family_ids': [f'family-{row}' for row in (10, 11, 12, 13)],
                             'objective_rows': [10, 11], 'development': copy.deepcopy(template)}}
    return root, manifest, template


def test_production_role_contract_and_geometry_allowed_pool(production_roles):
    root, manifest, template = production_roles
    preparation._production_data_contract(root, manifest, template)
    # Distinct planned positions may reuse one exactly bound producer view.
    manifest['updates'] *= 2
    preparation._production_data_contract(root, manifest, template)


@pytest.mark.parametrize('failure', ['missing_producers', 'wrong_role', 'wrong_model', 'wrong_task',
    'source_bytes', 'changed_view', 'view_order', 'train_selection_rows', 'wrong_objective_rows',
    'wrong_development', 'optimization_family_leak', 'demo_family_leak', 'geometry_outside_pool'])
def test_production_role_binding_rejects_leaks_and_corruption(production_roles, failure):
    root, manifest, template = production_roles
    role = 'geometry_training' if failure == 'geometry_outside_pool' else 'optimization'
    spec_path = root / (role + '.json')
    spec = json.loads(spec_path.read_text())
    if failure == 'missing_producers':
        del manifest['producers']
    elif failure == 'wrong_role':
        spec['data_role'] = 'objective_validation'
    elif failure == 'wrong_model':
        spec['backbone']['model_revision'] = 'other'
    elif failure == 'wrong_task':
        spec['data']['task'] = 'other'
    elif failure == 'source_bytes':
        spec['source_files']['uv.lock']['content_hex'] = b'bad'.hex()
    elif failure == 'changed_view':
        spec['views'][0]['tensors']['input_ids']['sha256'] = 'bad'
    elif failure == 'view_order':
        extra = copy.deepcopy(spec['views'][0])
        extra['batch_view_id'] = 'unconsumed'
        spec['views'].insert(0, extra)
    elif failure == 'train_selection_rows':
        manifest['data_ids']['train_rows'].append(10)
    elif failure == 'wrong_objective_rows':
        manifest['data_ids']['objective_rows'] = [10, 12]
    elif failure == 'wrong_development':
        manifest['data_ids']['development']['source_family_ids'] = ['wrong', 'families']
    else:
        from jscc.activation_replay import sequence_view_v2
        batch_path = root / (role + '.pt')
        batch = torch.load(batch_path, weights_only=True)
        if failure == 'optimization_family_leak':
            batch['source_family_ids'][0] = 'dev-100'
            spec['data']['source_family_ids'].append('dev-100')
        elif failure == 'demo_family_leak':
            batch['source_views'][0]['demo_source_family_ids'] = ['family-10']
        else:
            batch['source_family_ids'][0] = 'outside-pool'
            spec['data']['source_family_ids'].append('outside-pool')
        torch.save(batch, batch_path)
        reference = manifest['geometry']['train'][0] if role == 'geometry_training' else manifest['updates'][0]
        reference.update(sha256=file_digest(batch_path), view_sha256=batch_identity(batch))
        spec['views'] = [sequence_view_v2(batch)]
    spec_path.write_text(json.dumps(spec))
    if 'producers' in manifest:
        manifest['producers'][role]['sha256'] = file_digest(spec_path)
    with pytest.raises(ValueError):
        preparation._production_data_contract(root, manifest, template)


@pytest.mark.parametrize('failure', ['unobserved_selection_family', 'demo_row_family_mismatch',
                                     'missing_catalog', 'incomplete_catalog', 'wrong_query_family',
                                     'wrong_optimization_catalog'])
def test_complete_pool_catalog_exclusions(production_roles, failure):
    from jscc.activation_replay import sequence_view_v2

    root, manifest, template = production_roles
    spec_path = root / 'optimization.json'
    spec = json.loads(spec_path.read_text())
    if failure == 'unobserved_selection_family':
        # Selection row13 is absent from both objective and geometry panels.
        manifest['data_ids']['validation_source_family_ids'][-1] = 'family-5'
    elif failure == 'missing_catalog':
        del manifest['data_ids']['train_source_family_ids']
    elif failure == 'incomplete_catalog':
        manifest['data_ids']['validation_source_family_ids'].pop()
    elif failure == 'wrong_query_family':
        manifest['data_ids']['train_source_family_ids'][0] = 'other-family'
    elif failure == 'wrong_optimization_catalog':
        spec['data']['source_family_ids'].append('unbound-family')
    else:
        batch_path = root / 'optimization.pt'
        batch = torch.load(batch_path, weights_only=True)
        batch['demo_ids'][0] = [4]  # Claimed family remains family5.
        torch.save(batch, batch_path)
        manifest['updates'][0].update(sha256=file_digest(batch_path), view_sha256=batch_identity(batch))
        spec['views'] = [sequence_view_v2(batch)]
    spec_path.write_text(json.dumps(spec))
    manifest['producers']['optimization']['sha256'] = file_digest(spec_path)
    with pytest.raises(ValueError):
        preparation._production_data_contract(root, manifest, template)
