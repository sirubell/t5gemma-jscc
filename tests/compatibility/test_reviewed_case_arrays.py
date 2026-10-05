"""Reviewed independent case graphs: local source, fake scheduler, no optimization."""
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import pytest
import yaml

from jscc.activation_replay import canonical_digest, file_digest
from jscc import submission as submission
from contracts.test_source_lifetime import source as source, git

pytest_plugins = ['shared.test_recipe_workflow']


def read(path):
    return json.loads(Path(path).read_text())


def write_yaml(path, value):
    Path(path).write_text(yaml.safe_dump(value))


@pytest.fixture
def case_job(source, tmp_path):
    python = tmp_path/'python'
    python.symlink_to(sys.executable)
    resources = dict(partition='h200q', account='research', qos='h200q_1g', node='hgx-cn01',
                     gpus=1, cpus=16, memory_gb=170, walltime_minutes=120)
    job = dict(schema=submission.CASE_SCHEMA, recipe=str(tmp_path/'recipe.yaml'),
               registrations=str(tmp_path/'registrations'), run_output=str(tmp_path/'run'),
               root=str(tmp_path), source={'path': str(source), 'commit': git(source, 'rev-parse', 'HEAD')},
               python=str(python), cache=str(tmp_path/'cache'), output=str(tmp_path/'submission'),
               resources=resources, cases=None, evaluation_cases=['width-8-ratio-50'],
               concurrency=2, combined=False, attempt='0001')
    limits = dict(root=str(tmp_path), resources=resources, max_concurrency=3, max_array_tasks=9)
    job_path, limits_path = tmp_path/'job.yaml', tmp_path/'limits.yaml'
    write_yaml(job_path, job)
    write_yaml(limits_path, limits)
    return job_path, limits_path, job


def test_real_tiny_assets_prepare_only(joined_assets, monkeypatch, tmp_path):
    """Real clean implementation source and released preparation, no learner."""
    from shared.test_recipe_assets import registered
    from jscc.shared_runs import recipe_run_record
    from time import perf_counter
    source_root = Path(submission.__file__).resolve().parents[1]
    recipe, registrations, _ = registered(joined_assets)
    resources = dict(partition='h200q', account='research', qos='h200q_1g', node='hgx-cn01',
                     gpus=1, cpus=16, memory_gb=170, walltime_minutes=120)
    root = source_root.parent
    run_output = source_root/'runs'/('ticket56-'+uuid4().hex)
    run_output.parent.mkdir(exist_ok=True)
    job = dict(schema=submission.CASE_SCHEMA, recipe=str(recipe), registrations=str(registrations),
               run_output=str(run_output), root=str(root),
               source={'path': str(source_root), 'commit': git(source_root, 'rev-parse', 'HEAD')},
               python=sys.executable, cache=str(source_root/'runs/cache'),
               output=str(run_output)+'-submission', resources=resources, cases=None,
               evaluation_cases=['width-8-ratio-50'], concurrency=1, combined=False, attempt='0001')
    path, limits = tmp_path/'real-job.yaml', tmp_path/'real-limits.yaml'
    write_yaml(path, job)
    write_yaml(limits, dict(root=str(root), resources=resources, max_concurrency=3, max_array_tasks=9))
    monkeypatch.setattr('transformers.AutoModelForSeq2SeqLM.from_pretrained',
                        lambda *a, **k: pytest.fail('preparation must not load a backbone'))
    monkeypatch.setattr('jscc.shared_training.run_recipe_training',
                        lambda *a, **k: pytest.fail('preparation must not optimize'))
    started = perf_counter()
    bundle = submission.prepare_case_job(path, limits)
    assert submission.prepare_case_job(path, limits) == bundle
    record = recipe_run_record(job['run_output'])
    assert record['cases'][0]['case_id'] == 'width-8-ratio-50'
    assert record['cases'][0]['counts']['selected_rows'] == 12
    assert record['cases'][0]['counts']['optimizer_updates'] == 24
    preview = submission.preview_job(bundle, str(job['output'])+'/bundle.json')
    assert preview['task_count'] == 2
    assert preview['scientific_totals']['optimizer_updates'] == 24
    assert preview['scientific_totals']['selected_rows'] == 12
    assert preview['scheduler_invoked'] is False
    assert not list(run_output.glob('cases/*/attempts/*'))
    if os.environ.get('TICKET56_EVIDENCE_DIR'):
        evidence = Path(os.environ['TICKET56_EVIDENCE_DIR'])
        evidence.mkdir(parents=True, exist_ok=True)
        (evidence/'REAL-PREPARE-COUNTS.json').write_text(json.dumps({
            'source_commit': bundle['source']['commit'], 'source_path': bundle['source']['path'],
            'seconds': perf_counter()-started, 'run_output': str(run_output),
            'selected_counts': record['cases'][0]['counts'],
            'task_count': preview['task_count'], 'scientific_totals': preview['scientific_totals'],
            'model_optimization': False, 'scheduler_invoked': False}, indent=2)+'\n')


def test_changed_generated_request_rejects_before_runner_and_holds_source(graph_job, monkeypatch):
    bundle = prepare_graph(graph_job)
    accepted(bundle)
    case = bundle['cases'][0]
    path = Path(case['commands']['training'][3])
    request = yaml.safe_load(path.read_text())
    request['execution']['case'] = 'different-case'
    write_yaml(path, request)
    allocation(monkeypatch, 0)
    with pytest.raises(ValueError, match='generated request bytes differ'):
        submission.run_job(bundle, bundle['identity'], 0,
                           runner=lambda _: pytest.fail('changed request reached runner'))
    outcome = read(Path(bundle['output'])/'tasks/g0-i0.launch.json')
    assert outcome['status'] == 'failed' and outcome['commands'] == []
    with pytest.raises(ValueError, match='held'):
        submission.source_consumers(bundle).require_released(bundle['source']['path'])


@pytest.fixture
def graph_job(case_job, monkeypatch):
    """Stub only released preparation, retaining actual Git/runtime/scheduler seams."""
    path, limits, job = case_job
    Path(job['recipe']).write_text('fixture recipe\n')
    Path(job['registrations']).mkdir()
    (Path(job['registrations'])/'assets.json').write_text('{}')
    monkeypatch.setattr(submission, '_declared_case_ids',
                        lambda *args: [f'width-8-ratio-{ratio}' for ratio in (25, 50, 75)])
    monkeypatch.setattr(submission, '__file__', job['source']['path']+'/jscc/recipe_submission.py')

    def prepare(recipe, output, *, registrations):
        root = Path(output)
        root.mkdir()
        prepared = root/'prepared.json'
        prepared.write_text(json.dumps({'source_commit': job['source']['commit']}))
        rows = []
        for ratio in (25, 50, 75):
            case = root/'cases'/f'width-8-ratio-{ratio}'
            case.mkdir(parents=True)
            config = {'data': {'num_train': 12}, 'training': {'lr': .001}, 'ratio': ratio}
            (case/'recipe.json').write_text(json.dumps({'config': config, 'locations': ['dec_l3', 'enc_l2', 'enc_emb'],
                                                     'updates_per_location': 8}))
            (case/'config.yaml').write_text(yaml.safe_dump(config))
            rows.append({'case_id': case.name, 'path': str(case), 'width': 8, 'ratio': ratio,
                         'counts': {'selected_rows': 12, 'optimizer_updates': 24,
                                    'base_stream_presentations': 128, 'processed_presentations': 384}})
        record = {'schema': 'recipe-run-v1', 'root': str(root),
                  'prepared': {'path': str(prepared), 'sha256': file_digest(prepared)}, 'cases': rows}
        record['identity'] = canonical_digest(record)
        (root/'recipe-run.json').write_text(json.dumps(record))
        return root

    monkeypatch.setattr('jscc.shared_runs.prepare_recipe_run', prepare)
    monkeypatch.setattr('jscc.shared_runs.recipe_run_record', lambda root: read(Path(root)/'recipe-run.json'))
    return path, limits, job


def prepare_graph(graph_job, **changes):
    path, limits, job = graph_job
    job.update(changes)
    write_yaml(path, job)
    return submission.prepare_case_job(path, limits)


def accepted(bundle):
    calls = []
    def scheduler(command):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, str(100+len(calls))+';fixture\n', '')
    receipt = submission.submit_job(bundle, bundle['output'], scheduler=scheduler)
    return receipt, calls


def allocation(monkeypatch, index):
    monkeypatch.setenv('SLURM_JOB_ID', '101')
    monkeypatch.setenv('SLURM_ARRAY_TASK_ID', str(index))


@pytest.mark.parametrize('selection,evaluations,indices,eval_indices', [
    (['width-8-ratio-75', 'width-8-ratio-25'], ['width-8-ratio-75'], [0, 2], [2]),
    (['width-8-ratio-50'], ['width-8-ratio-50'], [1], [1]),
    (['width-8-ratio-75'], [], [2], []),
])
def test_stable_subset_singleton_own_task_pairs(graph_job, selection, evaluations, indices, eval_indices):
    b = prepare_graph(graph_job, cases=selection, evaluation_cases=evaluations)
    assert b['groups'][0]['indices'] == indices
    assert [row['case_id'] for row in b['cases']] == [f'width-8-ratio-{r}' for r in (25, 50, 75)]
    preview = submission.preview_job(b, b['output']+'/bundle.json')
    assert preview['task_count'] == len(indices)+len(eval_indices)
    assert preview['scientific_totals']['optimizer_updates'] == 24*len(indices)
    assert preview['scientific_totals']['selected_rows'] == 12*len(indices)
    assert preview['resources_per_task'] == b['resources']
    for case in preview['cases']:
        assert case['effective_config']['training']['lr'] == .001
        assert case['locations'] == ['dec_l3', 'enc_l2', 'enc_emb']
        assert case['updates_per_location'] == 8
        assert case['selected_rows_request']['num_train'] == 12
    if eval_indices:
        assert b['groups'][1]['indices'] == eval_indices
        assert b['groups'][1]['aftercorr'] == 0
        command = submission.scheduler_command(b, b['output']+'/job.sbatch', 1, '123')
        assert '--dependency=aftercorr:123' in command
        assert '--array='+','.join(map(str, eval_indices))+'%2' in command
    else:
        assert len(b['groups']) == 1
    for index, row in enumerate(b['cases']):
        assert ('training' in row['commands']) == (index in indices)
        assert ('evaluation' in row['commands']) == (index in eval_indices)


def test_exact_payload_intent_and_all_task_reservations_precede_scheduler(graph_job):
    b = prepare_graph(graph_job, evaluation_cases=['width-8-ratio-25', 'width-8-ratio-75'])
    output = Path(b['output'])
    preview = submission.preview_job(b, output/'bundle.json')
    calls = []
    def scheduler(command):
        receipt = read(output/'SUBMISSION.json')
        assert receipt['jobs'][-1]['status'] == 'attempting'
        assert receipt['jobs'][-1]['arguments'] == command
        assert (output/'job.sbatch').read_text() == preview['payload']
        assert receipt['payload_sha256'] == file_digest(output/'job.sbatch')
        assert len(list((output/'tasks').glob('*.reservation.json'))) == 5
        with pytest.raises(ValueError, match='held'):
            submission.source_consumers(b).require_released(b['source']['path'])
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, str(100+len(calls))+'\n', '')
    receipt = submission.submit_job(b, output, scheduler=scheduler)
    assert receipt['status'] == 'submitted'
    assert [j['status'] for j in receipt['jobs']] == ['accepted', 'accepted']
    assert '--dependency=aftercorr:101' in calls[1]
    assert receipt['bundle_sha256'] == file_digest(output/'bundle.json')
    assert preview['payload'] == submission.render_payload(b, output/'bundle.json')


@pytest.mark.parametrize('failure,expected', [('rejected', 'rejected'), ('ambiguous', 'unknown'),
                                            ('id-with-error', 'unknown'), ('exception', 'unknown')])
def test_partial_acceptance_unknown_rejected_blocks_retry_and_holds_source(graph_job, failure, expected):
    b = prepare_graph(graph_job)
    calls = []
    def scheduler(command):
        calls.append(command)
        if len(calls) == 1:
            return subprocess.CompletedProcess(command, 0, '123\n', '')
        if failure == 'exception':
            raise OSError('connection lost after possible acceptance')
        return subprocess.CompletedProcess(command, 0 if failure == 'ambiguous' else 1,
            'no receipt' if failure in ('ambiguous', 'rejected') else '124\n', 'fixture diagnostic')
    with pytest.raises((ValueError, OSError)):
        submission.submit_job(b, b['output'], scheduler=scheduler)
    receipt = read(Path(b['output'])/'SUBMISSION.json')
    assert receipt['status'] == 'stopped-no-retry'
    assert receipt['jobs'][0]['job_id'] == '123'
    assert receipt['jobs'][1]['status'] == expected
    assert receipt['jobs'][1]['arguments'] == calls[1]
    assert len(list((Path(b['output'])/'tasks').glob('*.reservation.json'))) == 4
    with pytest.raises(FileExistsError):
        submission.submit_job(b, b['output'], scheduler=scheduler)
    assert len(calls) == 2
    registry = submission.source_consumers(b)
    with pytest.raises(ValueError, match='cannot be released'):
        registry.release(b['source']['path'], b['identity'], owner=b['output'])
    with pytest.raises(ValueError, match='held'):
        registry.require_released(b['source']['path'])


@pytest.mark.parametrize('field,value', [('gpus', 2), ('cpus', 17), ('memory_gb', float('inf')),
                                        ('walltime_minutes', 0)])
def test_finite_resource_limits_before_preparation(graph_job, monkeypatch, field, value):
    path, limits, job = graph_job
    job['resources'][field] = value
    write_yaml(path, job)
    monkeypatch.setattr('jscc.shared_runs.prepare_recipe_run', lambda *a, **k: pytest.fail('invalid resources prepared'))
    with pytest.raises(ValueError):
        submission.prepare_case_job(path, limits)
    assert not Path(job['run_output']).exists()


def test_wrong_queued_source_rejected_before_public_callback(graph_job, monkeypatch):
    b = prepare_graph(graph_job)
    accepted(b)
    allocation(monkeypatch, 0)
    (Path(b['source']['path'])/'worker.py').write_text('print("changed queued source")')
    with pytest.raises(ValueError, match='tracked changes'):
        submission.run_job(b, b['identity'], 0,
            runner=lambda args: pytest.fail('wrong source reached public command'))
    outcome = read(Path(b['output'])/'tasks/g0-i0.launch.json')
    assert outcome['preflight'] == 'failed'
    assert outcome['commands'] == []
    with pytest.raises(ValueError, match='held'):
        submission.source_consumers(b).require_released(b['source']['path'])


def test_own_case_terminal_gate_precedes_evaluation_callback(graph_job, monkeypatch):
    b = prepare_graph(graph_job, cases=['width-8-ratio-50'])
    accepted(b)
    allocation(monkeypatch, 1)
    gates = []
    def gate(root, case):
        gates.append((root, case))
        raise ValueError('own terminal is unavailable; another case completed')
    monkeypatch.setattr('jscc.shared_evaluation.recipe_evaluation_target', gate)
    with pytest.raises(ValueError, match='own terminal'):
        submission.run_job(b, b['identity'], 1,
            runner=lambda args: pytest.fail('ineligible own terminal reached evaluation'))
    assert gates == [(b['run_output'], 'width-8-ratio-50')]
    outcome = read(Path(b['output'])/'tasks/g1-i1.launch.json')
    assert outcome['commands'][0]['stage'] == 'evaluation'
    assert outcome['commands'][0]['status'] == 'failed'


@pytest.mark.parametrize('evaluation_success', [False, True])
def test_combined_stages_separate_outcomes_intents_and_source_hold(graph_job, monkeypatch, evaluation_success):
    b = prepare_graph(graph_job, cases=['width-8-ratio-50'], combined=True)
    accepted(b)
    assert len(b['groups']) == 1
    assert submission.preview_job(b, b['output']+'/bundle.json')['task_count'] == 1
    allocation(monkeypatch, 1)
    gate_calls, calls = [], []
    marker = Path(b['output'])/'tasks/g0-i1.launch.json'
    def gate(root, case):
        gate_calls.append(case)
        assert calls == ['training']
        assert read(marker)['commands'][0]['status'] == 'completed'
        attempt = b['cases'][1]['requests']['training']['execution']['attempt']
        return ({'checkpoint': {'training_attempt': str(Path(b['cases'][1]['path'])/'attempts'/('training-'+attempt))}},)
    monkeypatch.setattr('jscc.shared_evaluation.recipe_evaluation_target', gate)
    def runner(args):
        outcome = read(marker)
        assert outcome['commands'][-1]['arguments'] == args
        assert outcome['commands'][-1]['status'] == 'running'
        stage = outcome['commands'][-1]['stage']
        calls.append(stage)
        assert yaml.safe_load(Path(args[3]).read_text())['execution']['case'] == 'width-8-ratio-50'
        return subprocess.CompletedProcess(args, 0 if stage == 'training' or evaluation_success else 3,
                                           stage+' stdout', stage+' stderr')
    if evaluation_success:
        assert submission.run_job(b, b['identity'], 0, runner=runner)['status'] == 'completed'
    else:
        with pytest.raises(ValueError, match='evaluation command failed'):
            submission.run_job(b, b['identity'], 0, runner=runner)
    outcome = read(marker)
    assert [r['status'] for r in outcome['commands']] == ['completed', 'completed' if evaluation_success else 'failed']
    assert outcome['commands'][0]['returncode'] == 0
    assert outcome['commands'][0]['stdout'] == 'training stdout'
    assert gate_calls == ['width-8-ratio-50']
    with pytest.raises(FileExistsError):
        submission.run_job(b, b['identity'], 0, runner=runner)
    assert calls == ['training', 'evaluation']
    # A task finishing never infers all other scheduler outcomes or releases source.
    with pytest.raises(ValueError, match='held'):
        submission.source_consumers(b).require_released(b['source']['path'])


def test_second_output_cannot_queue_same_case_attempt(graph_job):
    b = prepare_graph(graph_job)
    accepted(b)
    new_job = prepare_graph(graph_job, output=b['output']+'-other', attempt='0002')
    with pytest.raises(FileExistsError):
        submission.submit_job(new_job, new_job['output'],
            scheduler=lambda args: pytest.fail('duplicate case attempt reached scheduler'))
    assert read(Path(new_job['output'])/'SUBMISSION.json')['jobs'] == []


def test_same_case_wrong_training_attempt_terminal_is_rejected(graph_job, monkeypatch):
    b = prepare_graph(graph_job, cases=['width-8-ratio-50'])
    accepted(b)
    allocation(monkeypatch, 1)
    wrong = str(Path(b['cases'][1]['path'])/'attempts/training-old-completed')
    monkeypatch.setattr('jscc.shared_evaluation.recipe_evaluation_target',
                        lambda *a, **k: ({'checkpoint': {'training_attempt': wrong}},))
    with pytest.raises(ValueError, match='not this graph training attempt'):
        submission.run_job(b, b['identity'], 1,
            runner=lambda args: pytest.fail('different training attempt reached evaluation'))
    assert read(Path(b['output'])/'tasks/g1-i1.launch.json')['commands'][0]['status'] == 'failed'


def test_unknown_acceptance_cannot_escape_hold_with_new_output_attempt(graph_job):
    b = prepare_graph(graph_job)
    with pytest.raises(ValueError, match='unknown'):
        submission.submit_job(b, b['output'], scheduler=lambda args:
            subprocess.CompletedProcess(args, 0, 'accepted without job receipt', ''))
    other = prepare_graph(graph_job, output=b['output']+'-other', attempt='brand-new')
    with pytest.raises(FileExistsError):
        submission.submit_job(other, other['output'],
            scheduler=lambda args: pytest.fail('unknown acceptance escaped into a retry'))
    assert read(Path(other['output'])/'SUBMISSION.json')['jobs'] == []
    with pytest.raises(ValueError, match='held'):
        submission.source_consumers(b).require_released(b['source']['path'])


def test_frozen_effective_preview_ignores_later_authored_yaml(graph_job):
    b = prepare_graph(graph_job, cases=['width-8-ratio-50'])
    Path(graph_job[2]['recipe']).write_text('later author changed YAML: [')
    preview = submission.preview_job(b, b['output']+'/bundle.json')
    assert len(preview['cases']) == 1
    assert preview['cases'][0]['effective_config']['training']['lr'] == .001
    assert preview['cases'][0]['updates_per_location'] == 8


def test_failed_durable_case_fence_write_cannot_reach_scheduler(graph_job, monkeypatch):
    b = prepare_graph(graph_job)
    original = submission.write_json
    def fail_reservation(path, value):
        if Path(path).name == 'training.json' and Path(path).parent.name == 'scheduler-reservations':
            raise OSError('fixture durable fence write failed')
        return original(path, value)
    monkeypatch.setattr(submission, 'write_json', fail_reservation)
    with pytest.raises(OSError, match='durable fence'):
        submission.submit_job(b, b['output'], scheduler=lambda args: pytest.fail('undurable fence submitted'))
    receipt = read(Path(b['output'])/'SUBMISSION.json')
    assert receipt['jobs'] == [] and receipt['status'] == 'stopped-no-retry'
    assert (Path(b['cases'][0]['path'])/'scheduler-reservations/training.json').exists()
    with pytest.raises(ValueError, match='held'):
        submission.source_consumers(b).require_released(b['source']['path'])


def test_yaml_entry_submission_preview_and_mocked_submit_requests(graph_job, monkeypatch):
    from request_helpers import invoke
    path, limits, job = graph_job
    directory = path.parent
    exported = directory/'preview-bundle.json'
    calls = []
    def scheduler(command):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, str(900+len(calls))+'\n', '')
    monkeypatch.setattr(submission, '_slurm_submit', scheduler)
    preview = invoke(directory, 'shared', 'preview', {'plan': {'kind': 'submission', 'path': str(path)},
                     'submission': {'limits': str(limits), 'bundle_output': str(exported)}})
    assert preview['scheduler_invoked'] is False and not calls
    bundle = read(exported)
    authorization = {'approved': True, 'operation': 'submit-recipe', 'bundle_identity': bundle['identity'],
                     'source_commit': job['source']['commit'], 'output': job['output'],
                     'user_authorization_reference': 'explicit-fixture-mock-authority'}
    receipt = invoke(directory, 'shared', 'submit', {'submission': {'plan': str(path), 'limits': str(limits),
                     'bundle_output': job['output']+'/bundle.json', 'receipt_dir': job['output'], 'authorization': authorization}})
    assert len(calls) == len(bundle['groups']) and receipt['jobs']
    for case in bundle['cases']:
        for stage, payload in case['requests'].items():
            command = case['commands'][stage]
            assert Path(command[1]).name == 'experiment.py' and command[2] == '--config'
            saved = yaml.safe_load(Path(command[3]).read_text())
            assert saved == payload
            assert saved['execution']['root'] == bundle['run_output']
            assert saved['execution']['case'] == case['case_id']


@pytest.mark.parametrize("changes", [
    {"cases": []}, {"cases": ["unknown"]},
    {"cases": ["width-8-ratio-25", "width-8-ratio-25"]},
    {"evaluation_cases": ["unknown"]},
    {"cases": ["width-8-ratio-25"], "evaluation_cases": ["width-8-ratio-50"]},
    {"evaluation_cases": ["width-8-ratio-50", "width-8-ratio-50"]},
])
def test_invalid_declared_selectors_precede_preparation(graph_job, monkeypatch, changes):
    path, limits, job = graph_job
    job.update(changes)
    write_yaml(path, job)
    monkeypatch.setattr('jscc.shared_runs.prepare_recipe_run',
                        lambda *a, **k: pytest.fail('invalid selectors prepared root'))
    with pytest.raises(ValueError):
        submission.prepare_case_job(path, limits)
    assert not Path(job['run_output']).exists()
    assert not Path(job['output']).exists()


def test_array_limit_precedes_preparation(graph_job, monkeypatch):
    path, limits, job = graph_job
    policy = yaml.safe_load(limits.read_text())
    policy['max_array_tasks'] = 2
    write_yaml(limits, policy)
    monkeypatch.setattr('jscc.shared_runs.prepare_recipe_run',
                        lambda *a, **k: pytest.fail('over-limit selectors prepared root'))
    with pytest.raises(ValueError, match='invalid array size'):
        submission.prepare_case_job(path, limits)
    assert not Path(job['run_output']).exists()
    assert not Path(job['output']).exists()


def test_generated_requests_synced_before_scheduler(graph_job, monkeypatch):
    bundle = prepare_graph(graph_job)
    output = Path(bundle['output'])
    synced = set()
    real_fsync = submission.os.fsync

    def fsync(fd):
        stat = os.fstat(fd)
        requests = output / 'requests'
        if requests.exists():
            for path in [requests, *requests.glob('*.yaml')]:
                if path.stat().st_ino == stat.st_ino and path.stat().st_dev == stat.st_dev:
                    if path.is_file():
                        assert yaml.safe_load(path.read_text()) == bundle['cases'][int(path.stem.split('-')[-1])]['requests'][path.stem.rsplit('-', 1)[0]]
                    synced.add(path.name)
        real_fsync(fd)

    monkeypatch.setattr(submission.os, 'fsync', fsync)
    expected = {'requests', 'training-0.yaml', 'training-1.yaml', 'training-2.yaml', 'evaluation-1.yaml'}

    def scheduler(command):
        assert expected <= synced
        return subprocess.CompletedProcess(command, 0, '101;cluster\n', '')

    submission.submit_job(bundle, output, scheduler=scheduler)
    assert expected <= synced
