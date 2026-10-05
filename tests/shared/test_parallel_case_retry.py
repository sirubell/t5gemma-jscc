"""Real CASE retry/fence consumers; source/runtime and scheduler are CPU mocks."""
import json
from pathlib import Path
import subprocess

import pytest
import yaml

from jscc import submission
from jscc.activation_replay import canonical_digest, file_digest
from jscc.submission_reconciliation import authorize_retry_job, prepare_retry_bundle


def identify(value):
    value.pop('identity', None)
    value['identity'] = canonical_digest(value)
    return value


def approval(old, new):
    return {'approved': True, 'operation': 'authorize-recipe-retry', 'bundle_identity': old['identity'],
            'new_bundle_identity': new['identity'], 'source_commit': old['source']['commit'],
            'output': new['output'], 'user_authorization_reference': 'mock exact human authority'}


@pytest.fixture
def stopped(tmp_path, monkeypatch):
    source = Path(submission.__file__).resolve().parents[1]
    case_path = tmp_path/'case'
    case_path.mkdir()
    parent = case_path/'attempts/evaluation-prior/panels/worker-attempts/prior'
    parent.mkdir(parents=True)
    (parent/'final.json').write_text('{"immutable":"previous scientific attempt"}')
    output = tmp_path/'submission'
    parallel = {'backend': 'local', 'workers': 2, 'devices': ['cpu', 'cpu'], 'parent_attempt': str(parent)}
    request = {'execution': {'attempt': 'first-g0-i2', 'case': 'case-two', 'root': str(tmp_path/'run'),
                             'evaluation': {'scope': 'initial', 'parallel': parallel}}}
    command = ['python', str(source/'experiment.py'), '--config', str(output/'requests/evaluation-2.yaml')]
    case = {'index': 2, 'case_id': 'case-two', 'path': str(case_path), 'effective_config_sha256': 'e'*64,
            'commands': {'evaluation': command, 'training': ['retained-training-command']},
            'requests': {'evaluation': request, 'training': {'execution': {'attempt': 'exact-training'}}}}
    b = identify({'schema': submission.CASE_SCHEMA, 'root': str(tmp_path), 'output': str(output), 'source': {'path': str(source), 'commit': 'a'*40},
                  'source_binding': {'path': str(source), 'commit': 'a'*40}, 'python': 'python', 'cache': str(tmp_path/'cache'),
                  'run_output': str(tmp_path/'run'), 'run_identity': 'r', 'input_binding': {}, 'combined': False, 'attempt': 'first',
                  'resources': dict(partition='mock', account='mock', qos='mock', node='mock', gpus=1, cpus=2, memory_gb=1, walltime_minutes=10),
                  'groups': [{'stage': 'evaluation', 'indices': [2], 'concurrency': 1, 'aftercorr': None, 'accepted_dependency_job_id': '9101'}],
                  'cases': [{'commands': {}, 'requests': {}}, {'commands': {}, 'requests': {}}, case]})
    monkeypatch.setattr(submission, 'submission_preflight', lambda _: None)
    monkeypatch.setattr('jscc.source_lifetime.check_source_binding', lambda _: None)
    calls = []
    def rejected(command):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, '', 'mock rejection')
    with pytest.raises(ValueError, match='scheduler rejected'):
        submission.submit_job(b, b['output'], scheduler=rejected)
    folder = output/'reconciliation'
    folder.mkdir()
    (folder/'00000001.json').write_text(json.dumps({'sequence': 1, 'submission_sha256': file_digest(output/'SUBMISSION.json'),
         'groups': [{'group': 0, 'status': 'not-accepted'}], 'retry_eligible_groups': [0], 'conflicts': []}))
    return b, parent, calls


def test_fresh_retry_retains_exact_parent_request_index_and_holds(stopped, tmp_path, monkeypatch):
    old, parent, initial_calls = stopped
    prior = {p: p.read_bytes() for p in Path(old['output']).rglob('*') if p.is_file()}
    parent_bytes = (parent/'final.json').read_bytes()
    new = prepare_retry_bundle(old, old['output'], output=str(tmp_path/'retry'), attempt='second')
    assert new['groups'][0]['indices'] == [2]
    assert new['groups'][0]['accepted_dependency_job_id'] == '9101'
    case = new['cases'][2]
    assert case['requests']['evaluation']['execution']['attempt'] == 'second-g0-i2'
    assert case['requests']['evaluation']['execution']['evaluation'] == old['cases'][2]['requests']['evaluation']['execution']['evaluation']
    assert case['requests']['training'] == old['cases'][2]['requests']['training']
    result = authorize_retry_job(old, old['output'], new_bundle=new, authorization=approval(old, new))
    assert result['submission_performed'] is False and len(result['retired_fences']) == 1
    assert authorize_retry_job(old, old['output'], new_bundle=new, authorization=approval(old, new)) == result
    calls = []
    def scheduler(command):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, '9201', '')
    submission.submit_job(new, new['output'], scheduler=scheduler)
    assert len(initial_calls) == len(calls) == 1
    assert '--array=2%1' in calls[0] and '--dependency=aftercorr:9101' in calls[0]
    request_path = Path(new['output'])/'requests/evaluation-2.yaml'
    assert yaml.safe_load(request_path.read_text()) == case['requests']['evaluation']
    with pytest.raises(FileExistsError):
        submission.submit_job(new, new['output'], scheduler=lambda _: pytest.fail('duplicate submission'))
    with pytest.raises(ValueError, match='held'):
        submission.source_consumers(new).require_released(new['source']['path'])
    monkeypatch.setenv('SLURM_JOB_ID', '9201')
    monkeypatch.setenv('SLURM_ARRAY_TASK_ID', '2')
    monkeypatch.setattr(submission, 'local_path', lambda *_: Path(new['source']['path']))
    expected = str(Path(case['path'])/'attempts/training-exact-training')
    monkeypatch.setattr('jscc.shared_evaluation.recipe_evaluation_target', lambda *a: ({'checkpoint': {'training_attempt': expected}},))
    seen = []
    submission.run_job(new, new['identity'], 0, runner=lambda command: (seen.append(yaml.safe_load(Path(command[-1]).read_text())) or subprocess.CompletedProcess(command, 0, '', '')))
    assert seen == [case['requests']['evaluation']]
    with pytest.raises(FileExistsError):
        submission.run_job(new, new['identity'], 0, runner=lambda _: pytest.fail('duplicate launch'))
    assert all(p.read_bytes() == content for p, content in prior.items())
    assert (parent/'final.json').read_bytes() == parent_bytes


@pytest.mark.parametrize('kind', ['source', 'parallel', 'parent_attempt'])
def test_retry_mismatch_rejects_before_fence_retirement(stopped, tmp_path, kind):
    old, _, _ = stopped
    new = prepare_retry_bundle(old, old['output'], output=str(tmp_path/'retry'), attempt='second')
    if kind == 'source':
        new['source']['commit'] = 'b'*40
    else:
        parallel = new['cases'][2]['requests']['evaluation']['execution']['evaluation']['parallel']
        parallel['workers' if kind == 'parallel' else 'parent_attempt'] = 1 if kind == 'parallel' else str(tmp_path/'other')
    identify(new)
    fence = Path(old['cases'][2]['path'])/'scheduler-reservations/evaluation.json'
    before = fence.read_bytes()
    with pytest.raises(ValueError, match='retry changes'):
        authorize_retry_job(old, old['output'], new_bundle=new, authorization=approval(old, new))
    assert fence.read_bytes() == before


def test_retry_generated_request_mismatch_holds_source_and_fails_before_runner(stopped, tmp_path, monkeypatch):
    old, _, _ = stopped
    new = prepare_retry_bundle(old, old['output'], output=str(tmp_path/'retry'), attempt='second')
    authorize_retry_job(old, old['output'], new_bundle=new, authorization=approval(old, new))
    submission.submit_job(new, new['output'], scheduler=lambda command: subprocess.CompletedProcess(command, 0, '9201', ''))
    path = Path(new['output'])/'requests/evaluation-2.yaml'
    request = yaml.safe_load(path.read_text())
    request['execution']['evaluation']['parallel']['parent_attempt'] = str(tmp_path/'wrong-parent')
    path.write_text(yaml.safe_dump(request))
    monkeypatch.setenv('SLURM_JOB_ID', '9201')
    monkeypatch.setenv('SLURM_ARRAY_TASK_ID', '2')
    monkeypatch.setattr(submission, 'local_path', lambda *_: Path(new['source']['path']))
    with pytest.raises(ValueError, match='generated request bytes differ'):
        submission.run_job(new, new['identity'], 0, runner=lambda _: pytest.fail('changed request ran'))
    outcome = json.loads((Path(new['output'])/'tasks/g0-i2.launch.json').read_text())
    assert outcome['status'] == 'failed' and outcome['commands'] == []
    with pytest.raises(ValueError, match='held'):
        submission.source_consumers(new).require_released(new['source']['path'])
