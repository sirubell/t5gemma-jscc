"""Current request success receipts preserve overlap, race and output boundaries."""

import jscc
from pathlib import Path
import pytest
import yaml
from jscc import experiment
from request_helpers import invoke

ROOT = Path(jscc.__file__).resolve().parents[1]


@pytest.mark.parametrize('phase', ['train', 'evaluate'])
@pytest.mark.parametrize('workflow', ['ordinary', 'shared'])
def test_cli_creates_missing_link_parent(tmp_path, monkeypatch, phase, workflow):
    link = tmp_path/'omitted-directory'/'run.txt'
    root = tmp_path/'bound'
    root.mkdir()
    monkeypatch.setattr(experiment, '_bound', lambda *a: {'cases': [{'case_id': 'one'}]})
    def completed(*a, **k):
        assert link.parent.is_dir()
        return root
    monkeypatch.setattr('jscc.ordinary_runs.train_task', completed)
    monkeypatch.setattr('jscc.ordinary_runs.evaluate_run', completed)
    monkeypatch.setattr('jscc.shared_runs.recipe_run_record', lambda *a: {'cases': [{'case_id': 'one'}]})
    monkeypatch.setattr('jscc.shared_runs.train_recipe_case', completed)
    monkeypatch.setattr('jscc.shared_evaluation.evaluate_recipe_run', completed)
    ex: dict = {'root': str(root), 'attempt': 'new', 'success_link': str(link)}
    if phase == 'evaluate':
        ex['evaluation'] = {}
    invoke(tmp_path, workflow, phase, ex)
    assert link.read_text() == str(root)+'\n'


@pytest.mark.parametrize('workflow', ['ordinary', 'shared'])
@pytest.mark.parametrize('relation', ['same', 'inside', 'ancestor', 'alias', 'dangling', 'used'])
def test_training_link_rejects_overlap_or_replay_before_effects(tmp_path, monkeypatch, workflow, relation):
    root = tmp_path/'new-parent'/'root'
    link = tmp_path/'receipt'/'run.txt'
    if relation == 'same':
        link = root
    elif relation == 'inside':
        link = root/'receipt'/'run.txt'
    elif relation == 'ancestor':
        link = root.parent
    elif relation == 'alias':
        alias = tmp_path/'alias'
        alias.symlink_to(root, target_is_directory=True)
        link = alias/'run.txt'
    else:
        link.parent.mkdir()
        if relation == 'dangling':
            link.symlink_to(tmp_path/'absent')
        else:
            link.write_text('retained receipt')
    monkeypatch.setattr(experiment, '_ordinary', lambda *a: pytest.fail('config before rejection'))
    with pytest.raises((ValueError, FileExistsError)):
        invoke(tmp_path, workflow, 'train', {'root': str(root), 'attempt': 'new', 'success_link': str(link)})
    assert not root.parent.exists()


def test_training_missing_output_rejects_before_config_or_link_parent(tmp_path):
    with pytest.raises(ValueError, match='explicit root'):
        invoke(tmp_path, 'ordinary', 'train', {'attempt': 'new', 'success_link': str(tmp_path/'new'/'link')})
    assert not (tmp_path/'new').exists()


def test_training_check_is_output_free(tmp_path):
    result = experiment.execute(ROOT/'configs/experiments/ordinary/coco.yaml')
    assert result['task'] == 'coco'
    assert not list(tmp_path.iterdir())


def test_racing_success_link_does_not_overwrite_or_restart(tmp_path, monkeypatch):
    output, link = tmp_path/'root', tmp_path/'links'/'run.txt'
    calls = []
    def completed(*a, **k):
        calls.append(a)
        output.mkdir()
        link.write_text('racing receipt\n')
        return output
    monkeypatch.setattr('jscc.ordinary_runs.train_task', completed)
    science = yaml.safe_load((ROOT/'configs/experiments/ordinary/coco.yaml').read_text())['experiment']
    science['model_config'] = str(ROOT/'configs/models/default.yaml')
    with pytest.raises(FileExistsError):
        invoke(tmp_path, 'ordinary', 'train', {'root': str(output), 'attempt': 'new', 'success_link': str(link)}, science)
    assert len(calls) == 1 and output.is_dir()
    assert link.read_text() == 'racing receipt\n'
