import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from jscc import submission


def bundle(tmp_path, monkeypatch):
    out = tmp_path/'submission'
    (out/'tasks').mkdir(parents=True)
    (out/'requests').mkdir()
    case_path = tmp_path/'case'
    source = Path(submission.__file__).resolve().parents[1]
    request = {'execution': {'root': str(tmp_path/'run'), 'case': 'selected-third-case', 'attempt': 'fresh', 'evaluation': {'scope': 'initial', 'parallel': {'backend': 'local', 'workers': 2, 'devices': ['cpu', 'cpu']}}}}
    path = out/'requests/evaluation-2.yaml'
    path.write_text(yaml.safe_dump(request))
    command = ['python', str(source/'experiment.py'), '--config', str(path)]
    case = {'case_id': 'selected-third-case', 'path': str(case_path), 'commands': {'evaluation': command}, 'requests': {'evaluation': request, 'training': {'execution': {'attempt': 'exact'}}}}
    b = {'schema': submission.CASE_SCHEMA, 'identity': 'bound', 'root': str(tmp_path), 'source': {'path': str(source)}, 'python': 'python', 'output': str(out), 'run_output': str(tmp_path/'run'), 'resources': {'walltime_minutes': 10}, 'groups': [{'stage': 'evaluation', 'indices': [2], 'aftercorr': 0}], 'cases': [{}, {}, case]}
    monkeypatch.setenv('SLURM_JOB_ID', 'mock-allocation')
    monkeypatch.setenv('SLURM_ARRAY_TASK_ID', '2')
    monkeypatch.setattr(submission, 'validate_bundle', lambda _: None)
    monkeypatch.setattr(submission, 'submission_preflight', lambda _: None)
    monkeypatch.setattr(submission, 'local_path', lambda *_: source)
    monkeypatch.setattr(submission, 'source_consumers', lambda _: SimpleNamespace(record_state=lambda *a, **k: None))
    monkeypatch.setattr('jscc.shared_evaluation.recipe_evaluation_target', lambda *a: ({'checkpoint': {'training_attempt': str(case_path/'attempts/training-exact')}},))
    return b, path


def test_one_case_index_local_workers_and_exclusive_launch(tmp_path, monkeypatch):
    b, path = bundle(tmp_path, monkeypatch)
    seen = []
    def runner(command):
        seen.append(yaml.safe_load(Path(command[-1]).read_text())['execution']['evaluation']['parallel'])
        return subprocess.CompletedProcess(command, 0, '', '')
    submission._run_case_job(b, 'bound', 0, runner=runner)
    assert seen == [{'backend': 'local', 'workers': 2, 'devices': ['cpu', 'cpu']}]
    assert b['groups'][0]['indices'] == [2] and b['groups'][0]['aftercorr'] == 0
    with pytest.raises(FileExistsError):
        submission._run_case_job(b, 'bound', 0, runner=lambda _: pytest.fail('collision launched'))


def test_request_mismatch_fails_before_parallel_consumer(tmp_path, monkeypatch):
    b, path = bundle(tmp_path, monkeypatch)
    saved = yaml.safe_load(path.read_text())
    saved['execution']['evaluation']['parallel']['workers'] = 1
    path.write_text(yaml.safe_dump(saved))
    with pytest.raises(ValueError, match='generated request bytes differ'):
        submission._run_case_job(b, 'bound', 0, runner=lambda _: pytest.fail('changed identity launched'))
    outcome = json.loads((Path(b['output'])/'tasks/g0-i2.launch.json').read_text())
    assert outcome['status'] == 'failed' and outcome['commands'] == []


def admission_job(evaluation):
    return dict.fromkeys(('schema', 'recipe', 'registrations', 'run_output', 'root', 'source',
                          'python', 'cache', 'output', 'resources', 'cases', 'evaluation_cases',
                          'concurrency', 'combined', 'attempt'), None) | {
        'schema': submission.CASE_SCHEMA, 'evaluation': evaluation, 'root': '/tmp/bound',
        'combined': False, 'evaluation_cases': ['one']}


@pytest.mark.parametrize('field', ['checkpoint', 'expected_step', 'overrides'])
def test_unsupported_case_evaluation_rejects_before_source_binding(monkeypatch, field):
    job = admission_job({field: 'unsupported'})
    monkeypatch.setattr(submission, '_read_yaml', lambda _: job)
    monkeypatch.setattr(submission, 'capture_source_binding', lambda *a: pytest.fail('bound unsupported request'))
    with pytest.raises(ValueError, match='unsupported case evaluation fields'):
        submission.prepare_case_job('job', 'limits')


@pytest.mark.parametrize('combined,cases', [(True, ['one']), (False, ['one', 'two'])])
def test_retry_parent_requires_single_separate_case(monkeypatch, combined, cases):
    import sys
    job = admission_job({'parallel': {'backend': 'local', 'workers': 2, 'devices': ['cpu', 'cpu'],
                                     'parent_attempt': '/tmp/bound/previous'}})
    job.update(combined=combined, evaluation_cases=cases)
    monkeypatch.setattr(submission, '_read_yaml', lambda _: job)
    monkeypatch.setattr(submission, 'capture_source_binding', lambda *a: pytest.fail('bound ambiguous retry'))
    # Dependency validator is supplied by immutable core at assembly; this test isolates CASE admission.
    monkeypatch.setitem(sys.modules, 'jscc.panel_workers', SimpleNamespace(validate_parallel_controls=lambda _: None))
    with pytest.raises(ValueError, match='retry parent requires one separate evaluation case'):
        submission.prepare_case_job('job', 'limits')
