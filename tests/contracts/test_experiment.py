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
