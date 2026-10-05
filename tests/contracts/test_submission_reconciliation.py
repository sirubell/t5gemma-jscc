"""Recorded fake scheduler evidence: recovery never invokes an adapter."""

import jscc
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest
import yaml

from jscc.activation_replay import canonical_digest
from jscc import submission as submission
from jscc.submission_reconciliation import (authorize_retry_job, finalize_source_consumer,
                                           reconciliation_intents)
from compatibility.test_reviewed_case_arrays import case_job as case_job, graph_job as graph_job
from contracts.test_source_lifetime import source as source
from shared.test_recipe_submission import job_files as job_files, saved_submission


def stopped(graph_job, *, partial=False):
    bundle = submission.prepare_job(*graph_job[:2])
    calls = []
    def scheduler(command):
        calls.append(command)
        if partial and len(calls) == 1:
            return subprocess.CompletedProcess(command, 0, '9101\n', '')
        raise ConnectionError('fake accepted response lost')
    with pytest.raises(ConnectionError):
        submission.submit_job(bundle, bundle['output'], scheduler=scheduler)
    return bundle, calls


def evidence(bundle, *, result='accepted', query='complete'):
    intents = reconciliation_intents(bundle, bundle['output'])
    groups = []
    receipt = json.loads((Path(bundle['output'])/'SUBMISSION.json').read_text())
    retained = {row['group']: row['job_id'] for row in receipt['jobs']
                if row.get('job_id') and row['status'] in ('accepted', 'submitted')}
    for intent in intents:
        row = {'group': intent['group'], 'intent': intent, 'authoritative': True,
               'query_status': query, 'matches': []}
        match = {'job_id': retained.get(intent['group'], str(9200+intent['group'])), 'intent': intent}
        if result == 'accepted':
            row['matches'] = [match]
        elif result == 'ambiguous':
            row['matches'] = [match, dict(match, job_id='9999')]
        elif result == 'rejected' and intent['group'] in retained:
            row['matches'] = [match]
        elif result == 'rejected':
            row['nonacceptance'] = {'exhaustive': True, 'final': True, 'acceptance_window_closed': True}
        groups.append(row)
    return {'schema': 'scheduler-reconciliation-evidence-v1', 'bundle_identity': bundle['identity'],
            'reference': 'fake-scheduler-record-1', 'groups': groups, 'tasks': []}


def reconcile(bundle, value):
    return submission.reconcile_job(bundle, bundle['output'], evidence=value)


@pytest.mark.parametrize('result,query', [('delayed', 'complete'), ('accepted', 'failed'),
                                         ('ambiguous', 'complete')])
def test_unknown_blocks_retry_and_release(graph_job, result, query):
    bundle, calls = stopped(graph_job)
    record = reconcile(bundle, evidence(bundle, result=result, query=query))
    assert record['groups'][0]['status'] == 'unknown'
    assert 0 not in record['retry_eligible_groups']
    with pytest.raises(ValueError, match='terminal evidence'):
        finalize_source_consumer(bundle, bundle['output'], release=True)
    with pytest.raises(ValueError, match='fully reconciled'):
        authorize_retry_job(bundle, bundle['output'], new_bundle=bundle, authorization={})
    assert len(calls) == 1
    with pytest.raises(FileExistsError):
        submission.submit_job(bundle, bundle['output'], scheduler=lambda _: pytest.fail('duplicate submission'))


def test_lost_response_partial_monotonic_repeated_bytes(graph_job):
    bundle, calls = stopped(graph_job, partial=True)
    directory = Path(bundle['output'])
    original = (directory/'SUBMISSION.json').read_bytes()
    value = evidence(bundle)
    first = reconcile(bundle, value)
    assert [g['job_id'] for g in first['groups']] == ['9101', '9201']
    assert reconcile(bundle, value) == first
    assert len(list((directory/'reconciliation').glob('[0-9]*.json'))) == 1
    delayed = evidence(bundle, result='delayed', query='failed')
    later = reconcile(bundle, delayed)
    assert [g['job_id'] for g in later['groups']] == ['9101', '9201']
    assert later['previous_sha256'] == canonical_digest(first)
    assert (directory/'SUBMISSION.json').read_bytes() == original
    assert len(calls) == 2
    assert len(list(directory.glob('tasks/*.reservation.json'))) == 4


def test_all_bound_tasks_required_for_explicit_release(graph_job):
    bundle, _ = stopped(graph_job, partial=True)
    value = evidence(bundle)
    record = reconcile(bundle, value)
    for group in record['groups']:
        for index in bundle['groups'][group['group']]['indices']:
            value['tasks'].append({'group': group['group'], 'index': index, 'job_id': group['job_id'],
                'scheduler_task_id': group['job_id']+'_'+str(index), 'query_status': 'complete',
                'authoritative': True, 'status': 'completed', 'reference': 'fake-sacct-terminal'})
    missing = deepcopy(value)
    missing['tasks'].pop()
    assert not reconcile(bundle, missing)['all_tasks_terminal']
    registry = submission.source_consumers(bundle)
    with pytest.raises(ValueError, match='held'):
        registry.require_released(bundle['source']['path'])
    assert reconcile(bundle, value)['all_tasks_terminal']
    assert finalize_source_consumer(bundle, bundle['output'])['released'] is False
    assert finalize_source_consumer(bundle, bundle['output'], release=True)['released'] is True
    assert finalize_source_consumer(bundle, bundle['output'], release=True)['released'] is True
    registry.require_released(bundle['source']['path'])


def test_partial_rejected_and_unattempted_lifecycle(graph_job):
    # First group lost response, second never attempted; definitive absence for
    # the first plus durable stopped intent covers every reserved task.
    bundle, _ = stopped(graph_job)
    record = reconcile(bundle, evidence(bundle, result='rejected'))
    assert [g['status'] for g in record['groups']] == ['not-accepted', 'not-attempted']
    assert record['all_tasks_terminal']
    assert record['retry_eligible_groups'] == [0, 1]
    finalize_source_consumer(bundle, bundle['output'], release=True)


def fresh(bundle):
    new = deepcopy(bundle)
    new['output'] += '-retry'
    new['attempt'] = '0002'
    for position, group in enumerate(new['groups']):
        stages = ['training', 'evaluation'] if group['stage'] == 'combined' else [group['stage']]
        for index in group['indices']:
            case = new['cases'][index]
            for stage in stages:
                if stage in case['commands']:
                    case['requests'][stage]['execution']['attempt'] = f'0002-g{position}-i{index}'
                    case['commands'][stage] = [new['python'], new['source']['path']+'/experiment.py',
                        '--config', str(Path(new['output'])/'requests'/f'{stage}-{index}.yaml')]
    new.pop('identity')
    new['identity'] = canonical_digest(new)
    return new


def approval(bundle, new):
    return {'approved': True, 'operation': 'authorize-recipe-retry',
            'bundle_identity': bundle['identity'], 'new_bundle_identity': new['identity'],
            'source_commit': bundle['source']['commit'], 'output': new['output'],
            'user_authorization_reference': 'fake-human-exact-approval'}


def test_fresh_exact_authorization_fences_and_no_duplicate(graph_job):
    bundle, calls = stopped(graph_job)
    reconcile(bundle, evidence(bundle, result='rejected'))
    new = fresh(bundle)
    with pytest.raises(ValueError, match='authorization'):
        authorize_retry_job(bundle, bundle['output'], new_bundle=new, authorization={})
    result = authorize_retry_job(bundle, bundle['output'], new_bundle=new, authorization=approval(bundle, new))
    assert result['submission_performed'] is False
    assert len(result['retired_fences']) == 4
    assert all(Path(p).exists() for p in result['retired_fences'])
    assert authorize_retry_job(bundle, bundle['output'], new_bundle=new, authorization=approval(bundle, new)) == result
    accepted_calls = []
    def scheduler(command):
        accepted_calls.append(command)
        return subprocess.CompletedProcess(command, 0, str(9300+len(accepted_calls)), '')
    submission.submit_job(new, new['output'], scheduler=scheduler)
    assert len(accepted_calls) == 2 and len(calls) == 1
    requests = Path(new['output'])/'requests'
    assert {p.name for p in requests.iterdir()} == {
        'training-0.yaml', 'training-1.yaml', 'training-2.yaml', 'evaluation-1.yaml'}
    for case in new['cases']:
        for stage, request in case['requests'].items():
            assert yaml.safe_load((requests/f"{stage}-{case['index']}.yaml").read_text()) == request
    with pytest.raises(FileExistsError):
        submission.submit_job(new, new['output'], scheduler=lambda _: pytest.fail('no duplicates'))


def test_accepted_group_and_changed_budget_cannot_retry(graph_job):
    bundle, _ = stopped(graph_job, partial=True)
    reconcile(bundle, evidence(bundle, result='rejected'))
    new = fresh(bundle)
    with pytest.raises(ValueError, match='accepted groups'):
        authorize_retry_job(bundle, bundle['output'], new_bundle=new, authorization=approval(bundle, new))


@pytest.mark.parametrize('field', ['root', 'case', 'timeout_seconds', 'workflow', 'operation', 'schema',
                                 'scope', 'unchanged_attempt', 'wrong_attempt', 'config_path',
                                 'python', 'entry', 'argv', 'retained_request', 'retained_command'])
def test_retry_rejects_changed_request_or_command_before_retirement(graph_job, field):
    from jscc.submission_reconciliation import prepare_retry_bundle
    bundle, _ = stopped(graph_job, partial=True)
    reconcile(bundle, evidence(bundle, result='rejected'))
    new = prepare_retry_bundle(bundle, bundle['output'], output=bundle['output']+'-retry', attempt='0002')
    case = new['cases'][1]
    request = case['requests']['evaluation']
    command = case['commands']['evaluation']
    if field in ('root', 'case', 'timeout_seconds'):
        request['execution'][field] = 999 if field == 'timeout_seconds' else 'changed'
    elif field in ('workflow', 'operation', 'schema'):
        request[field] = 'changed'
    elif field == 'scope':
        request['execution']['evaluation']['scope'] = 'terminal'
    elif field in ('unchanged_attempt', 'wrong_attempt'):
        request['execution']['attempt'] = (bundle['cases'][1]['requests']['evaluation']['execution']['attempt']
                                           if field == 'unchanged_attempt' else '0002-g1-i1')
    elif field == 'config_path':
        command[3] = str(Path(new['output'])/'foreign.yaml')
    elif field in ('python', 'entry'):
        command[0 if field == 'python' else 1] += '-foreign'
    elif field == 'argv':
        command.append('--attempt')
    elif field == 'retained_request':
        case['requests']['training']['execution']['attempt'] = '0002-g0-i1'
    else:
        case['commands']['training'][3] = str(Path(new['output'])/'requests/training-1.yaml')
    new.pop('identity')
    new['identity'] = canonical_digest(new)
    fences = {p: p.read_bytes() for p in Path(bundle['run_output']).glob('cases/*/scheduler-reservations/*.json')}
    with pytest.raises(ValueError, match='retry (changes|requires)'):
        authorize_retry_job(bundle, bundle['output'], new_bundle=new, authorization=approval(bundle, new))
    assert all(p.read_bytes() == value for p, value in fences.items())
    assert not list(Path(bundle['run_output']).glob('cases/*/scheduler-reservations/*.retired-*.json'))
    record_path = sorted((Path(bundle['output'])/'reconciliation').glob('[0-9]*.json'))[-1]
    record = json.loads(record_path.read_text())
    assert not record.get('recovery_claims') and not record.get('recovery_approvals')


def test_atomic_snapshot_failure_preserves_original(graph_job, monkeypatch):
    bundle, _ = stopped(graph_job)
    directory = Path(bundle['output'])
    original = (directory/'SUBMISSION.json').read_bytes()
    writer = submission.write_json
    def failed(path, value):
        if Path(path).parent.name == 'reconciliation':
            raise OSError('fake disk failure before durable snapshot')
        return writer(path, value)
    monkeypatch.setattr(submission, 'write_json', failed)
    with pytest.raises(OSError):
        reconcile(bundle, evidence(bundle))
    assert not list((directory/'reconciliation').glob('[0-9]*.json'))
    assert (directory/'SUBMISSION.json').read_bytes() == original
    monkeypatch.setattr(submission, 'write_json', writer)
    assert reconcile(bundle, evidence(bundle))['sequence'] == 1


def test_partial_evaluation_only_retry_retains_training_and_dependency(graph_job, monkeypatch):
    from jscc.submission_reconciliation import prepare_retry_bundle
    bundle, _ = stopped(graph_job, partial=True)
    reconcile(bundle, evidence(bundle, result='rejected'))
    new = prepare_retry_bundle(bundle, bundle['output'], output=bundle['output']+'-retry', attempt='0002')
    assert len(new['groups']) == 1 and new['groups'][0]['stage'] == 'evaluation'
    assert new['groups'][0]['accepted_dependency_job_id'] == '9101'
    original_training = {c['index']: Path(c['commands']['training'][3]).read_bytes() for c in bundle['cases']}
    for old_case, case in zip(bundle['cases'], new['cases'], strict=True):
        assert case['commands']['training'] == old_case['commands']['training']
        assert case['requests']['training'] == old_case['requests']['training']
    case = new['cases'][1]
    assert case['requests']['evaluation']['execution']['attempt'] == '0002-g0-i1'
    assert case['commands']['evaluation'] == [new['python'], new['source']['path']+'/experiment.py',
        '--config', str(Path(new['output'])/'requests/evaluation-1.yaml')]
    authorized = authorize_retry_job(bundle, bundle['output'], new_bundle=new, authorization=approval(bundle, new))
    assert len(authorized['retired_fences']) == 1
    calls = []
    def scheduler(command):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, '9401', '')
    submission.submit_job(new, new['output'], scheduler=scheduler)
    assert len(calls) == 1 and '--dependency=aftercorr:9101' in calls[0]
    requests = Path(new['output'])/'requests'
    assert [p.name for p in requests.iterdir()] == ['evaluation-1.yaml']
    assert yaml.safe_load((requests/'evaluation-1.yaml').read_text()) == case['requests']['evaluation']
    from compatibility.test_reviewed_case_arrays import allocation
    allocation(monkeypatch, 1)
    expected = str(Path(case['path'])/'attempts'/('training-'+case['requests']['training']['execution']['attempt']))
    monkeypatch.setattr('jscc.shared_evaluation.recipe_evaluation_target',
                        lambda *args: ({'checkpoint': {'training_attempt': expected}},))
    executed = []
    def runner(command):
        executed.append(command)
        return subprocess.CompletedProcess(command, 0, 'evaluated retained checkpoint', '')
    assert submission.run_job(new, new['identity'], 0, runner=runner)['status'] == 'completed'
    assert executed == [case['commands']['evaluation']]
    for case in bundle['cases']:
        assert Path(case['commands']['training'][3]).read_bytes() == original_training[case['index']]
        fence = Path(case['path'])/'scheduler-reservations'/'training.json'
        assert json.loads(fence.read_text())['identity'] == bundle['identity']


def test_failed_reservation_write_before_scheduler_lifecycle(graph_job, monkeypatch):
    bundle = submission.prepare_job(*graph_job[:2])
    writer = submission.write_json
    def fail(path, value):
        if Path(path).name == 'training.json':
            raise OSError('fake failed reservation write')
        return writer(path, value)
    monkeypatch.setattr(submission, 'write_json', fail)
    with pytest.raises(OSError):
        submission.submit_job(bundle, bundle['output'], scheduler=lambda _: pytest.fail('must not submit'))
    monkeypatch.setattr(submission, 'write_json', writer)
    record = reconcile(bundle, evidence(bundle, result='delayed'))
    assert record['all_tasks_terminal']
    assert all(g['status'] == 'not-attempted' for g in record['groups'])
    original = (Path(bundle['output'])/'SUBMISSION.json').read_bytes()
    from jscc.submission_reconciliation import prepare_retry_bundle
    new = prepare_retry_bundle(bundle, bundle['output'], output=bundle['output']+'-retry', attempt='0002')
    result = authorize_retry_job(bundle, bundle['output'], new_bundle=new, authorization=approval(bundle, new))
    empty_archive = Path(result['retired_fences'][0])
    assert empty_archive.exists() and empty_archive.read_bytes() == b''
    submission.submit_job(new, new['output'], scheduler=lambda c: subprocess.CompletedProcess(c, 0, '9601', ''))
    assert (Path(bundle['output'])/'SUBMISSION.json').read_bytes() == original
    finalize_source_consumer(bundle, bundle['output'], release=True)


def test_retired_retry_conflicting_acceptance_fails(graph_job):
    bundle, _ = stopped(graph_job)
    reconcile(bundle, evidence(bundle, result='rejected'))
    new = fresh(bundle)
    authorize_retry_job(bundle, bundle['output'], new_bundle=new, authorization=approval(bundle, new))
    with pytest.raises(ValueError, match='contradict'):
        reconcile(bundle, evidence(bundle))


def test_exact_intent_time_required(graph_job):
    bundle, _ = stopped(graph_job)
    value = evidence(bundle)
    assert value['groups'][0]['intent']['attempted_at']
    value['groups'][0]['intent']['attempted_at'] = 'different-window'
    assert reconcile(bundle, value)['groups'][0]['status'] == 'unknown'


def test_cli_query_failure_and_repeated_reconciliation(graph_job):
    import runpy
    bundle, _ = stopped(graph_job)
    main = runpy.run_path(str(Path(jscc.__file__).resolve().parents[1]/'scripts/submit_recipe.py'))['main']
    # graph_job points __file__ at its fake source, so use the actual test root.
    directory = Path(bundle['output'])
    path = directory/'fake-evidence.json'
    path.write_text(json.dumps(evidence(bundle, query='failed')))
    argv = ['reconcile', '--bundle', str(directory/'bundle.json'), '--evidence', str(path)]
    result = main(argv)
    assert result['groups'][0]['status'] == 'unknown'
    assert main(argv) == result
    with pytest.raises(SystemExit):
        main(argv+['--live'])


def test_concurrent_idempotent_snapshot(graph_job):
    from concurrent.futures import ThreadPoolExecutor
    bundle, _ = stopped(graph_job)
    value = evidence(bundle)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: reconcile(bundle, value), range(2)))
    assert results[0] == results[1]
    assert len(list((Path(bundle['output'])/'reconciliation').glob('[0-9]*.json'))) == 1


def test_retry_changed_budget_blocked(graph_job):
    bundle, _ = stopped(graph_job)
    reconcile(bundle, evidence(bundle, result='rejected'))
    new = fresh(bundle)
    new['groups'][0]['concurrency'] += 1
    new.pop('identity')
    new['identity'] = canonical_digest(new)
    with pytest.raises(ValueError, match='budget'):
        authorize_retry_job(bundle, bundle['output'], new_bundle=new, authorization=approval(bundle, new))


def _complete_evidence(bundle):
    value = evidence(bundle)
    record = reconcile(bundle, value)
    for group in record['groups']:
        if group['status'] != 'accepted':
            continue
        for index in bundle['groups'][group['group']]['indices']:
            value['tasks'].append({'group': group['group'], 'index': index, 'job_id': group['job_id'],
                'scheduler_task_id': group['job_id']+'_'+str(index), 'query_status': 'complete',
                'authoritative': True, 'status': 'completed', 'reference': 'fake-terminal-record'})
    return value


@pytest.mark.parametrize('status', ['failed', 'running'])
def test_authoritative_task_conflict_durably_blocks_release(graph_job, status):
    bundle, _ = stopped(graph_job, partial=True)
    value = _complete_evidence(bundle)
    reconcile(bundle, value)
    conflicting = deepcopy(value)
    conflicting['tasks'][0]['status'] = status
    with pytest.raises(ValueError, match='conflict'):
        reconcile(bundle, conflicting)
    latest_path = sorted((Path(bundle['output'])/'reconciliation').glob('[0-9]*.json'))[-1]
    blocker = json.loads(latest_path.read_text())
    assert blocker['conflicts'] and not blocker['all_tasks_terminal']
    with pytest.raises(ValueError, match='terminal evidence'):
        finalize_source_consumer(bundle, bundle['output'], release=True)
    with pytest.raises(ValueError, match='conflict resolution'):
        reconcile(bundle, value)
    value['conflict_resolution'] = {'authoritative': True, 'reference': 'fake-final-corrected-accounting',
                                    'conflict_sha256': canonical_digest(blocker)}
    assert reconcile(bundle, value)['all_tasks_terminal']
    assert finalize_source_consumer(bundle, bundle['output'], release=True)['released']


def test_conflict_after_historical_release_reports_immutable_limit(graph_job):
    bundle, _ = stopped(graph_job, partial=True)
    value = _complete_evidence(bundle)
    reconcile(bundle, value)
    finalize_source_consumer(bundle, bundle['output'], release=True)
    value['tasks'][0]['status'] = 'running'
    with pytest.raises(ValueError, match='historical source release is immutable'):
        reconcile(bundle, value)


def test_reconcile_preserves_claim_and_exact_approval(graph_job):
    from jscc.submission_reconciliation import prepare_retry_bundle
    bundle, _ = stopped(graph_job)
    value = evidence(bundle, result='rejected')
    reconcile(bundle, value)
    new = prepare_retry_bundle(bundle, bundle['output'], output=bundle['output']+'-retry', attempt='0002')
    authorize_retry_job(bundle, bundle['output'], new_bundle=new, authorization=approval(bundle, new))
    value['reference'] = 'fake-second-consistent-query'
    record = reconcile(bundle, value)
    assert record['new_bundle_identity'] == new['identity']
    assert record['retry_authorization'] == approval(bundle, new)
    assert record['recovery_claims'] == {'0': new['identity'], '1': new['identity']}
    submission.submit_job(new, new['output'], scheduler=lambda c: subprocess.CompletedProcess(c, 0, '9701', ''))


def test_second_generation_evaluation_retry_retains_external_dependency(graph_job):
    from jscc.submission_reconciliation import prepare_retry_bundle
    bundle, _ = stopped(graph_job, partial=True)
    reconcile(bundle, evidence(bundle, result='rejected'))
    first = prepare_retry_bundle(bundle, bundle['output'], output=bundle['output']+'-A', attempt='0002')
    authorize_retry_job(bundle, bundle['output'], new_bundle=first, authorization=approval(bundle, first))
    with pytest.raises(ConnectionError):
        submission.submit_job(first, first['output'], scheduler=lambda _: (_ for _ in ()).throw(ConnectionError('lost')))
    reconcile(first, evidence(first, result='rejected'))
    second = prepare_retry_bundle(first, first['output'], output=bundle['output']+'-B', attempt='0003')
    authorize_retry_job(first, first['output'], new_bundle=second, authorization=approval(first, second))
    calls = []
    def scheduler(command):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, '9703', '')
    submission.submit_job(second, second['output'], scheduler=scheduler)
    assert len(calls) == 1 and '--dependency=aftercorr:9101' in calls[0]
    assert second['cases'][1]['commands']['training'] == bundle['cases'][1]['commands']['training']
    for generation, attempt in ((first, '0002'), (second, '0003')):
        for old_case, case in zip(bundle['cases'], generation['cases'], strict=True):
            assert case['commands']['training'] == old_case['commands']['training']
            assert case['requests']['training'] == old_case['requests']['training']
        case = generation['cases'][1]
        assert case['requests']['evaluation']['execution']['attempt'] == attempt+'-g0-i1'
        assert case['commands']['evaluation'][3] == str(Path(generation['output'])/'requests/evaluation-1.yaml')
        assert [p.name for p in (Path(generation['output'])/'requests').iterdir()] == ['evaluation-1.yaml']


def test_interrupted_fence_retirement_recovers_exact_claim(graph_job, monkeypatch):
    from jscc.submission_reconciliation import prepare_retry_bundle
    bundle, _ = stopped(graph_job)
    reconcile(bundle, evidence(bundle, result='rejected'))
    new = prepare_retry_bundle(bundle, bundle['output'], output=bundle['output']+'-retry', attempt='0002')
    rename = Path.rename
    count = 0
    def interrupted(path, target):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError('fake interruption between fence retirements')
        return rename(path, target)
    monkeypatch.setattr(Path, 'rename', interrupted)
    with pytest.raises(OSError):
        authorize_retry_job(bundle, bundle['output'], new_bundle=new, authorization=approval(bundle, new))
    assert len(list(Path(bundle['run_output']).glob('cases/*/scheduler-reservations/*.retired-*.json'))) == 1
    monkeypatch.setattr(Path, 'rename', rename)
    result = authorize_retry_job(bundle, bundle['output'], new_bundle=new, authorization=approval(bundle, new))
    assert all(Path(p).exists() for p in result['retired_fences'])


def test_foreign_empty_stage_fence_never_retired(graph_job):
    from jscc.submission_reconciliation import prepare_retry_bundle
    bundle = submission.prepare_job(*graph_job[:2])
    folder = Path(bundle['cases'][0]['path'])/'scheduler-reservations'
    folder.mkdir()
    foreign = folder/'training.json'
    foreign.write_bytes(b'')
    with pytest.raises(FileExistsError):
        submission.submit_job(bundle, bundle['output'], scheduler=lambda _: pytest.fail('no call'))
    reconcile(bundle, evidence(bundle, result='delayed'))
    new = prepare_retry_bundle(bundle, bundle['output'], output=bundle['output']+'-retry', attempt='0002')
    with pytest.raises(ValueError, match='ownership unproven'):
        authorize_retry_job(bundle, bundle['output'], new_bundle=new, authorization=approval(bundle, new))
    assert foreign.exists() and foreign.read_bytes() == b''


@pytest.mark.parametrize('basis', ['original-receipt', 'previous-reconciliation'])
@pytest.mark.parametrize('contradiction', ['definitive-absence', 'different-id'])
def test_authoritative_accepted_group_conflict_blocks_lifetime_and_retry(graph_job, basis, contradiction):
    from jscc.submission_reconciliation import prepare_retry_bundle
    bundle, _ = stopped(graph_job, partial=basis == 'original-receipt')
    value = evidence(bundle)
    if basis == 'previous-reconciliation':
        value = _complete_evidence(bundle)
        assert reconcile(bundle, value)['all_tasks_terminal']
    contradictory = deepcopy(value)
    row = contradictory['groups'][0]
    if contradiction == 'different-id':
        row['matches'][0]['job_id'] = '99999'
    else:
        row['matches'] = []
        row['nonacceptance'] = {'exhaustive': True, 'final': True, 'acceptance_window_closed': True}
    original = (Path(bundle['output'])/'SUBMISSION.json').read_bytes()
    with pytest.raises(ValueError, match='conflict'):
        reconcile(bundle, contradictory)
    path = sorted((Path(bundle['output'])/'reconciliation').glob('[0-9]*.json'))[-1]
    blocker = json.loads(path.read_text())
    expected_id = '9101' if basis == 'original-receipt' else '9200'
    assert blocker['groups'][0]['status'] == 'accepted'
    assert blocker['groups'][0]['job_id'] == expected_id
    assert blocker['conflicts'][0]['kind'] == 'group'
    assert not blocker['all_tasks_terminal']
    with pytest.raises(ValueError, match='terminal evidence'):
        finalize_source_consumer(bundle, bundle['output'], release=True)
    with pytest.raises(ValueError, match='fully reconciled'):
        prepare_retry_bundle(bundle, bundle['output'], output=bundle['output']+'-retry', attempt='0002')
    assert (Path(bundle['output'])/'SUBMISSION.json').read_bytes() == original


@pytest.mark.parametrize('missing_intent_fields', [False, True])
def test_saved_legacy_lost_response_reconciliation_and_terminal(job_files, missing_intent_fields):
    from jscc.submission_reconciliation import prepare_retry_bundle
    bundle = saved_submission(job_files, missing_intent_fields=missing_intent_fields)
    directory = Path(bundle['output'])
    original = {name: (directory/name).read_bytes() for name in ('bundle.json', 'job.sbatch', 'SUBMISSION.json')}
    intents = reconciliation_intents(bundle, directory)
    if missing_intent_fields:
        assert intents[1]['attempt_identity'] is None and intents[1]['attempted_at'] is None
        assert intents[1]['submission_sha256'] == submission.file_digest(directory/'SUBMISSION.json')
    first = reconcile(bundle, evidence(bundle, result='rejected'))
    assert [g['status'] for g in first['groups']] == ['accepted', 'not-accepted']
    assert first['groups'][0]['job_id'] == '9501'
    assert first['retry_eligible_groups'] == [1]
    assert reconcile(bundle, evidence(bundle, result='rejected')) == first
    assert not first['all_tasks_terminal']
    with pytest.raises(ValueError, match='terminal evidence'):
        finalize_source_consumer(bundle, directory, release=True)
    value = evidence(bundle, result='rejected')
    for index in bundle['groups'][0]['indices']:
        value['tasks'].append({'group': 0, 'index': index, 'job_id': '9501',
            'scheduler_task_id': '9501_'+str(index), 'query_status': 'complete',
            'authoritative': True, 'status': 'completed', 'reference': 'historical-terminal'})
    assert reconcile(bundle, value)['all_tasks_terminal']
    assert finalize_source_consumer(bundle, directory, release=True)['released']
    assert finalize_source_consumer(bundle, directory, release=True)['released']
    before = sorted(str(p) for p in directory.rglob('*'))
    for operation in (lambda: prepare_retry_bundle(bundle, directory, output=bundle['output']+'-retry', attempt='02'),
                      lambda: authorize_retry_job(bundle, directory, new_bundle=bundle, authorization={})):
        with pytest.raises(ValueError, match='execution retired'):
            operation()
    assert sorted(str(p) for p in directory.rglob('*')) == before
    assert {name: (directory/name).read_bytes() for name in original} == original


def test_saved_legacy_unknown_and_payload_integrity(job_files):
    bundle = saved_submission(job_files, accepted_first=False)
    directory = Path(bundle['output'])
    record = reconcile(bundle, evidence(bundle, result='delayed'))
    assert [g['status'] for g in record['groups']] == ['unknown', 'not-attempted']
    assert not record['all_tasks_terminal']
    with pytest.raises(ValueError, match='terminal evidence'):
        finalize_source_consumer(bundle, directory, release=True)
    (directory/'job.sbatch').write_text('changed historical payload')
    with pytest.raises(ValueError, match='payload'):
        reconciliation_intents(bundle, directory)


def test_saved_legacy_recovery_history_survives_new_reconciliation(job_files, monkeypatch):
    import shared.test_recipe_submission as fixtures
    saved_bundle = fixtures.bundle
    def inherited_bundle(files):
        value = saved_bundle(files)
        value['groups'][1].update(afterok=None, accepted_dependency_job_id='9400')
        value['identity'] = canonical_digest({k: v for k, v in value.items() if k != 'identity'})
        return value
    monkeypatch.setattr(fixtures, 'bundle', inherited_bundle)
    bundle = saved_submission(job_files)
    directory = Path(bundle['output'])
    original = {name: (directory/name).read_bytes() for name in ('bundle.json', 'job.sbatch', 'SUBMISSION.json')}
    first = reconcile(bundle, evidence(bundle, result='rejected'))
    # A saved historical approval/claim snapshot, never a new old-schema retry.
    history = deepcopy(first)
    authorization = {'approved': True, 'reference': 'saved-historical-owner-approval'}
    history.update(sequence=2, previous_sha256=canonical_digest(first),
                   retry_authorization=authorization, new_bundle_identity='a'*64,
                   recovery_claims={'1': 'a'*64}, recovery_approvals={'a'*64: authorization})
    path = directory/'reconciliation/00000002.json'
    submission.write_json(path, history)
    history_bytes = path.read_bytes()
    updated = evidence(bundle, result='delayed')
    updated['reference'] = 'later-incomplete-query'
    record = reconcile(bundle, updated)
    assert record['sequence'] == 3 and record['previous_sha256'] == canonical_digest(history)
    for key in ('retry_authorization', 'new_bundle_identity', 'recovery_claims', 'recovery_approvals'):
        assert record[key] == history[key]
    assert record['groups'][0]['job_id'] == '9501'
    assert record['groups'][1]['status'] == 'not-accepted'
    assert json.loads((directory/'bundle.json').read_text())['groups'][1]['accepted_dependency_job_id'] == '9400'
    assert path.read_bytes() == history_bytes
    assert {name: (directory/name).read_bytes() for name in original} == original
