"""Public request effects and actual tiny ordinary/shared consumer behavior."""

import jscc
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from jscc.experiment import execute, read_request, SCHEMA
from jscc.config import load_config
from request_helpers import invoke

ROOT = Path(jscc.__file__).resolve().parents[1]


def write(tmp_path, request, name='request.yaml'):
    filename = tmp_path/name
    filename.write_text(yaml.safe_dump(request, sort_keys=False))
    return filename


def request(operation, workflow='ordinary', **execution):
    return {'schema': SCHEMA, 'workflow': workflow, 'operation': operation, 'execution': execution}


@pytest.mark.parametrize('task', ['coco', 'hellaswag'])
def test_check_effective_config_and_declaring_paths(tmp_path, monkeypatch, task):
    filename = ROOT/f'configs/experiments/ordinary/{task}.yaml'
    before = set(tmp_path.iterdir())
    monkeypatch.chdir(tmp_path)
    assert execute(filename) == load_config(filename)
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize('operation', ['check', 'acquire', 'register', 'prepare', 'train', 'resume', 'evaluate', 'collect', 'preview', 'submit'])
def test_operation_inactive_field_rejects_before_output(tmp_path, operation):
    value = request(operation, root='uncreated', acquisition={'output': 'new'}, nonsense=True)
    filename = write(tmp_path, value)
    with pytest.raises(ValueError, match='execution'):
        execute(filename)
    assert list(tmp_path.iterdir()) == [filename]


@pytest.mark.parametrize('mutation', ['unknown', 'inactive_task', 'bool_count', 'duplicate_operation', 'ordinary_case'])
def test_authoring_rejects_unconsumed_controls(tmp_path, mutation):
    value = yaml.safe_load((ROOT/'configs/experiments/ordinary/coco.yaml').read_text())
    value['experiment']['model_config'] = str(ROOT/'configs/models/default.yaml')
    if mutation == 'unknown':
        value['experiment']['training']['typo_lr'] = .1
    elif mutation == 'inactive_task':
        value['experiment']['data']['max_length'] = 42
    elif mutation == 'bool_count':
        value['experiment']['training']['batch_size'] = True
    elif mutation == 'duplicate_operation':
        value['operation'] = ['train', 'evaluate']
    else:
        value['execution']['case'] = 'unconsumed'
    with pytest.raises((ValueError, TypeError)):
        execute(write(tmp_path, value))
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize('relation', ['same', 'inside', 'ancestor', 'used', 'dangling'])
def test_success_receipt_overlap_replay_rejects_before_engine(tmp_path, monkeypatch, relation):
    from jscc import ordinary_runs
    root = tmp_path/'new'/'root'
    link = root if relation == 'same' else root/'link' if relation == 'inside' else root.parent if relation == 'ancestor' else tmp_path/'link'
    if relation == 'used':
        link.write_text('existing')
    elif relation == 'dangling':
        link.symlink_to(tmp_path/'absent')
    value = request('train', root=str(root), attempt='train1', success_link=str(link))
    monkeypatch.setattr(ordinary_runs, 'train_task', lambda *a, **k: pytest.fail('training before link guard'))
    with pytest.raises((ValueError, FileExistsError)):
        execute(write(tmp_path, value))
    assert not root.exists()


@pytest.mark.parametrize('task', ['coco', 'hellaswag'])
def test_real_ordinary_train_prepare_evaluate_bound_science(tmp_path, monkeypatch, task):
    import torch
    torch.set_num_threads(1)
    from ordinary.test_ordinary_runs import fixture_task
    from jscc import ordinary_runs
    science, _ = fixture_task(tmp_path, monkeypatch, task)
    science['training'].pop('mixed_precision', None)
    science['model'].pop('asset_binding', None)
    root = tmp_path/'bound'
    invoke(tmp_path, 'ordinary', 'prepare', {'root': str(root), 'preparation': {}}, science)
    assert not (root/'training.json').exists()
    invoke(tmp_path, 'ordinary', 'train', {'root': str(root), 'attempt': 'tiny', 'success_link': str(tmp_path/'links'/'run.txt')})
    checkpoint = json.loads((root/'checkpoint.json').read_text())
    assert checkpoint['role'] == 'ordinary-best'
    assert json.loads((root/'attempts/training-tiny/completion.json').read_text())['optimizer_updates'] == 2
    result = invoke(tmp_path, 'ordinary', 'evaluate', {'root': str(root), 'attempt': 'score', 'evaluation': {}})
    assert json.loads((result/'results.json').read_text())['task'] == task
    mismatched = deepcopy(science)
    mismatched['training']['lr'] *= 2
    with pytest.raises(ValueError, match='authored science differs'):
        invoke(tmp_path, 'ordinary', 'train', {'root': str(root), 'attempt': 'different'}, mismatched)
    with pytest.raises(FileExistsError):
        ordinary_runs.train_task(science, root)


def test_study_preview_export_no_training(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('subprocess in preview'))
    # Provenance reads its own subprocess helper; only current engine calls are mocked above.
    from jscc import studies
    monkeypatch.setattr(studies, '_source_state', lambda: {'revision': 'test-source', 'dirty': False})
    value = request('preview', plan={'kind': 'study', 'path': str(ROOT/'configs/studies/baseline.yaml'), 'output': 'export'})
    manifest = execute(write(tmp_path, value))
    record = json.loads(manifest.read_text())
    for row in record['runs']:
        _, exported = read_request(manifest.parent/row['config'])
        assert exported['operation'] == 'train'
        assert exported['execution']['success_link']
        assert exported['experiment']['task'] in ('coco', 'hellaswag')
    assert not (manifest.parent/'runs').exists()


@pytest.mark.usefixtures('joined_assets')
def test_mock_source_admission_actual_shared_train_and_evaluate(joined_assets, monkeypatch):
    """Real24updates/scoring state; source admission explicitly mocked until clean105."""
    from shared.test_recipe_assets import registered
    from jscc import shared_runs, shared_evaluation, runtime, run_records
    from jscc.shared_config import read_recipe
    recipe, registrations, _ = registered(joined_assets)
    science = read_recipe(recipe)
    science.pop('schema')
    science['task_config'] = str(joined_assets['task'])
    root = joined_assets['root']/'entry-shared'
    invoke(root.parent, 'shared', 'prepare', {'root': str(root), 'preparation': {'registrations': str(registrations)}}, science)
    monkeypatch.setattr(shared_runs, 'verify_recipe_source', lambda *a: None)
    monkeypatch.setattr(shared_evaluation, 'verify_recipe_source', lambda *a: None)
    original = runtime.source_state
    def mocked_source():
        return {**original(), 'dirty': False}
    monkeypatch.setattr(runtime, 'source_state', mocked_source)
    monkeypatch.setattr(run_records, 'source_state', mocked_source)
    monkeypatch.setattr(shared_evaluation, 'source_state', mocked_source)
    invoke(root.parent, 'shared', 'train', {'root': str(root), 'attempt': 'tiny', 'case': 'width-8-ratio-50', 'timeout_seconds': 120})
    campaign = json.loads((root/'cases/width-8-ratio-50/attempts/training-tiny/engine/campaign.json').read_text())
    assert campaign['completed_updates'] == 24
    assert len(campaign['checkpoints']) >= 3
    output = invoke(root.parent, 'shared', 'evaluate', {'root': str(root), 'attempt': 'score', 'case': 'width-8-ratio-50', 'evaluation': {}})
    result = json.loads((output/'results.json').read_text())
    assert result['status'] == 'complete'
    assert result['counts']['cases']['complete'] == 1

pytest_plugins = ['shared.test_recipe_workflow']


@pytest.mark.parametrize('workflow', ['ordinary', 'shared'])
def test_collection_workflow_mismatch_rejects_without_copy_effects(tmp_path, workflow):
    root = tmp_path/'bound'
    root.mkdir()
    kind = 'shared-codec-recipe' if workflow == 'ordinary' else 'ordinary-task'
    (root/'experiment.json').write_text(json.dumps({'schema': 'experiment-run-v1', 'kind': kind}))
    with pytest.raises(ValueError, match='collection workflow'):
        invoke(tmp_path, workflow, 'collect', {'root': str(root), 'collection': {'snapshot': str(tmp_path/'snapshot'), 'destination': str(tmp_path/'copy')}})
    assert not (tmp_path/'snapshot').exists() and not (tmp_path/'copy').exists()


def test_shared_checkpoint_declaring_file_path_not_cwd(tmp_path, monkeypatch):
    request_dir = tmp_path/'requests'
    request_dir.mkdir()
    checkpoint = tmp_path/'bound'/'cases'/'case'/'attempts'/'training-parent'/'engine'/'step.pt'
    value = request('evaluate', 'shared', root='../bound', attempt='development',
                    evaluation={'scope': 'development', 'checkpoint': '../bound/cases/case/attempts/training-parent/engine/step.pt', 'expected_step': 1})
    filename = write(request_dir, value)
    monkeypatch.chdir(tmp_path.parent)
    _, normalized = read_request(filename)
    assert normalized['execution']['evaluation']['checkpoint'] == str(checkpoint)


@pytest.mark.parametrize('enabled', [False, True])
def test_hellaswag_rejects_coco_control_before_root_creation(tmp_path, enabled):
    value = yaml.safe_load((ROOT/'configs/experiments/ordinary/hellaswag.yaml').read_text())
    value['experiment']['model_config'] = str(ROOT/'configs/models/default.yaml')
    value['experiment']['training']['record_coco_presentations'] = enabled
    value.update(operation='train', execution={'root': str(tmp_path/'root'), 'attempt': 'fresh'})
    with pytest.raises(ValueError, match='record_coco_presentations'):
        execute(write(tmp_path, value))
    assert not (tmp_path/'root').exists()


@pytest.mark.parametrize('duplicate', ['operation', 'training'])
def test_literal_duplicate_yaml_keys_reject_before_effects(tmp_path, duplicate):
    value = yaml.safe_load((ROOT/'configs/experiments/ordinary/coco.yaml').read_text())
    value['experiment']['model_config'] = str(ROOT/'configs/models/default.yaml')
    value.update(operation='train', execution={'root': str(tmp_path/'root'), 'attempt': 'fresh'})
    text = yaml.safe_dump(value, sort_keys=False)
    if duplicate == 'operation':
        text = text.replace('operation: train', 'operation: check\noperation: train')
    else:
        import re
        text = re.sub(r'^(\s+)lr: (.+)$', r'\1lr: \2\n\1lr: 0.1', text, count=1, flags=re.MULTILINE)
    filename = tmp_path/'ambiguous.yaml'
    filename.write_text(text)
    with pytest.raises(ValueError, match='duplicate YAML'):
        execute(filename)
    assert not (tmp_path/'root').exists()


@pytest.mark.parametrize('value', [True, float('nan'), -1])
def test_invalid_numerical_controls_reject_before_root_creation(tmp_path, value):
    document = yaml.safe_load((ROOT/'configs/experiments/ordinary/coco.yaml').read_text())
    document['experiment']['model_config'] = str(ROOT/'configs/models/default.yaml')
    document['experiment']['training']['lr'] = value
    document.update(operation='train', execution={'root': str(tmp_path/'root'), 'attempt': 'new'})
    with pytest.raises(ValueError, match='finite nonnegative'):
        execute(write(tmp_path, document))
    assert not (tmp_path/'root').exists()


@pytest.mark.parametrize('field', ['vanilla', 'deterministic_algorithms'])
@pytest.mark.parametrize('operation', ['prepare', 'train'])
def test_quoted_evaluation_booleans_reject_before_binding(tmp_path, field, operation):
    value = yaml.safe_load((ROOT/'configs/experiments/ordinary/coco.yaml').read_text())
    value['experiment']['model_config'] = str(ROOT/'configs/models/default.yaml')
    value['experiment']['evaluation'][field] = 'false'
    execution: dict = {'root': str(tmp_path/'root')}
    execution.update({'preparation': {}} if operation == 'prepare' else {'attempt': 'new'})
    value.update(operation=operation, execution=execution)
    with pytest.raises(ValueError, match='evaluation.'+field):
        execute(write(tmp_path, value))
    assert not (tmp_path/'root').exists()


@pytest.mark.parametrize('operation', ['check', 'prepare', 'train', 'evaluate'])
def test_coco_rejects_unused_selection_seed_before_output(tmp_path, operation):
    value = yaml.safe_load((ROOT/'configs/experiments/ordinary/coco.yaml').read_text())
    value['experiment']['model_config'] = str(ROOT/'configs/models/default.yaml')
    value['experiment']['data']['selection_seed'] = 17
    root = tmp_path/'root'
    output = tmp_path/'resolved.json'
    execution: dict
    if operation == 'check':
        execution = {'resolution': {'output': str(output)}}
    elif operation == 'prepare':
        execution = {'root': str(root), 'preparation': {}}
    else:
        execution = {'root': str(root), 'attempt': 'fresh'}
        if operation == 'evaluate':
            execution['evaluation'] = {}
    value.update(operation=operation, execution=execution)
    filename = write(tmp_path, value)
    with pytest.raises(ValueError, match="inactive task data fields:.*selection_seed"):
        execute(filename)
    assert list(tmp_path.iterdir()) == [filename]
    assert not root.exists() and not output.exists()



def shared_plain_request(tmp_path, operation='check'):
    task = yaml.safe_load((ROOT/'configs/experiments/shared/base.yaml').read_text())
    task['model_config'] = str(ROOT/'configs/models/shared_direct_affine.yaml')
    task_path = write(tmp_path, task, 'base.yaml')
    model = tmp_path/'model'
    model.mkdir()
    metadata = model/'config.json'
    metadata.write_text(json.dumps({'encoder': {'num_hidden_layers': 26, 'hidden_size': 1152},
                                    'decoder': {'num_hidden_layers': 26, 'hidden_size': 1152}}))
    value = yaml.safe_load((ROOT/'configs/experiments/shared/shared_hellaswag_example.yaml').read_text())
    value['experiment']['task_config'] = str(task_path)
    value['operation'] = operation
    root = tmp_path/'uncreated-root'
    if operation == 'check':
        value['execution'] = {'resolution': {'model_config': str(metadata), 'output': str(tmp_path/'resolution.json')}}
    else:
        assets = tmp_path/'assets.json'
        assets.write_text(json.dumps({'model_path': str(model), 'model_binding': {'path': 'unread-binding', 'sha256': 'a'*64},
            'cache_manifest': 'unread-cache', 'cache_sha256': 'bad', 'initializations': {},
            'observation_template': {'path': 'unread-observation', 'sha256': 'a'*64},
            'terminal_template': {'path': 'unread-terminal', 'sha256': 'a'*64}, 'study_pairing_id': 'fixture', 'synthetic': True}))
        value['execution'] = {'root': str(root), 'preparation': {'model_config': str(metadata), 'assets': str(assets)}}
    return value, task, task_path, root


@pytest.mark.parametrize('operation', ['check', 'prepare'])
@pytest.mark.parametrize('section,field,value', [
    ('training', 'typo_lr', .1), ('training', 'max_minutes', 2),
    ('training', 'selection_steps', [1]), ('data', 'num_report', 3),
])
def test_shared_plain_base_rejects_unknown_and_inactive_before_output(tmp_path, operation, section, field, value):
    request_value, task, task_path, root = shared_plain_request(tmp_path, operation)
    task[section][field] = value
    task_path.write_text(yaml.safe_dump(task))
    with pytest.raises(ValueError, match='shared task.'+section):
        execute(write(tmp_path, request_value))
    assert not root.exists() and not (tmp_path/'resolution.json').exists()


def test_shared_plain_base_accepts_named_fixed_compatibility_values(tmp_path):
    value, _, _, _ = shared_plain_request(tmp_path)
    resolved = execute(write(tmp_path, value))
    training = resolved['trajectories'][0]['config']['training']
    assert training['streamed_backward'] is True and training['valid_only_kl'] is True
    assert training['low_sync_logging'] is False and training['expected_attention_backend'] == 'sdpa'


@pytest.mark.parametrize('workflow,step,role,valid', [
    ('shared', 0, 'initialization', True), ('shared', 1, 'reconstruction', True),
    ('shared', True, 'initialization', False), ('shared', False, 'initialization', False),
    ('shared', -1, 'initialization', False), ('shared', 0.0, 'initialization', False),
    ('shared', 0, 'terminal', False), ('ordinary', 0, 'ordinary-last', False),
    ('ordinary', 1, 'ordinary-last', True),
])
def test_continuation_step_parser_retains_workflow_semantics(tmp_path, workflow, step, role, valid):
    value = request('resume', workflow, root='root', attempt='next',
                    continuation={'path': 'checkpoint.pt', 'sha256': 'a'*64, 'size_bytes': 1,
                                  'role': role, 'step': step, 'parent_attempt': 'parent'})
    filename = write(tmp_path, value)
    if valid:
        assert read_request(filename)[1]['execution']['continuation']['step'] == step
    else:
        with pytest.raises(ValueError, match='step'):
            read_request(filename)


def published_initialization(tmp_path):
    from jscc.activation_replay import file_digest
    case = tmp_path/'root/cases/one'
    parent = case/'attempts/training-parent'
    engine = parent/'engine'
    engine.mkdir(parents=True)
    checkpoint = engine/'step_000000_initialization.pt'
    checkpoint.write_bytes(b'owned checkpoint fixture bytes; no tensor/model load')
    row = {'path': str(checkpoint), 'sha256': file_digest(checkpoint), 'size': checkpoint.stat().st_size,
           'step': 0, 'role': 'initialization'}
    (parent/'attempt.json').write_text(json.dumps({'case': str(case), 'status': 'partial'}))
    (parent/'checkpoints.json').write_text(json.dumps({'checkpoints': [row]}))
    return case, parent, checkpoint, row


@pytest.mark.parametrize('mutation', ['valid', 'unrecorded', 'role', 'bool_step', 'foreign_case', 'foreign_checkpoint', 'duplicate'])
def test_shared_resume_zero_requires_owned_published_initialization(tmp_path, mutation):
    from jscc.shared_runs import _resume_reference
    case, parent, checkpoint, row = published_initialization(tmp_path)
    rows = [row]
    if mutation == 'unrecorded':
        rows = []
    elif mutation == 'role':
        row['role'] = 'reconstruction'
    elif mutation == 'bool_step':
        row['step'] = False
    elif mutation == 'foreign_case':
        (parent/'attempt.json').write_text(json.dumps({'case': str(tmp_path/'foreign'), 'status': 'partial'}))
    elif mutation == 'foreign_checkpoint':
        checkpoint = tmp_path/'foreign-zero.pt'
        checkpoint.write_bytes(b'unowned zero-step bytes')
        row['path'] = str(checkpoint)
    elif mutation == 'duplicate':
        rows.append(deepcopy(row))
    (parent/'checkpoints.json').write_text(json.dumps({'checkpoints': rows}))
    if mutation == 'valid':
        reference = _resume_reference(case, checkpoint)
        assert reference.path == row['path'] and reference.sha256 == row['sha256'] and reference.size == row['size']
    else:
        with pytest.raises(ValueError, match='resume'):
            _resume_reference(case, checkpoint)


@pytest.mark.parametrize('mutation', ['valid', 'hash', 'size', 'role', 'bool_step'])
def test_public_shared_zero_resume_binds_exact_checkpoint_record(tmp_path, monkeypatch, mutation):
    from jscc import experiment, shared_runs
    case, parent, checkpoint, row = published_initialization(tmp_path)
    continuation = {'path': row['path'], 'sha256': row['sha256'], 'size_bytes': row['size'],
                    'step': 0, 'role': 'initialization', 'parent_attempt': str(parent)}
    if mutation == 'hash':
        row['sha256'] = 'b'*64
    elif mutation == 'size':
        row['size'] += 1
    elif mutation == 'role':
        row['role'] = 'reconstruction'
    elif mutation == 'bool_step':
        row['step'] = False
    (parent/'checkpoints.json').write_text(json.dumps({'checkpoints': [row]}))
    calls = []
    monkeypatch.setattr(experiment, '_bound', lambda *args: {})
    monkeypatch.setattr(shared_runs, 'recipe_run_record', lambda root: {'cases': [{'case_id': 'one'}]})
    monkeypatch.setattr(shared_runs, 'train_recipe_case', lambda *args, **kwargs: calls.append(kwargs))
    value = request('resume', 'shared', root=str(case.parents[1]), case='one', attempt='next', continuation=continuation)
    filename = write(tmp_path, value)
    if mutation == 'valid':
        execute(filename)
        assert len(calls) == 1 and calls[0]['resume'] == str(checkpoint)
    else:
        with pytest.raises(ValueError, match='continuation checkpoint role/step/attempt'):
            execute(filename)
        assert calls == []


@pytest.mark.parametrize('operation', ['check', 'prepare'])
@pytest.mark.parametrize('metadata', ['protocol_only', 'candidate_controls', 'budget_controls'])
def test_shared_coco_rejects_hellaswag_compatibility_before_output(tmp_path, operation, metadata):
    _, _, task_path, root = shared_plain_request(tmp_path, operation)
    task = yaml.safe_load((ROOT/'configs/experiments/shared/coco_base.yaml').read_text())
    task['model_config'] = str(ROOT/'configs/models/shared_direct_affine.yaml')
    task['protocol'] = 'sharing-dn-native16-long-v1'
    if metadata == 'candidate_controls':
        task['evaluation'].update(num_fewshot=5, scoring_policy='fp32-v1')
    elif metadata == 'budget_controls':
        task['training'].update(max_steps=6400, schedule_steps=6400)
    task_path.write_text(yaml.safe_dump(task))
    value = yaml.safe_load((ROOT/'configs/experiments/shared/shared_coco_example.yaml').read_text())
    value['experiment']['task_config'] = str(task_path)
    value['operation'] = operation
    model_config = tmp_path/'model/config.json'
    if operation == 'check':
        value['execution'] = {'resolution': {'model_config': str(model_config),
                                           'output': str(tmp_path/'resolution.json')}}
    else:
        value['execution'] = {'root': str(root), 'preparation': {
            'model_config': str(model_config), 'assets': str(tmp_path/'assets.json')}}
    with pytest.raises(ValueError, match='shared task historical protocol requires HellaSwag'):
        execute(write(tmp_path, value))
    assert not root.exists() and not (tmp_path/'resolution.json').exists()
