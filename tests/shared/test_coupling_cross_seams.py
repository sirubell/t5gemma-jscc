"""CPU/mock checks on the assembled draft's public and durable evidence seams."""
import hashlib
import json
from pathlib import Path
import yaml
from jscc import panel_workers as panels
from jscc import result_collection, experiment, shared_evaluation


def load_runtime(plan, device):
    return {'device': device}


def score_panel(runtime, plan, panel, output):
    return {'item_ids': plan['item_ids'], 'scores': {'mock_cider': 3.5},
            'noise_digest': panels.canonical_digest(plan['noise'])}


def score_one_failed(runtime, plan, panel, output):
    if panel['index'] == 0:
        raise panels.PanelScoringError('mock scorer error')
    return score_panel(runtime, plan, panel, output)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_actual_worker_retry_records_survive_collection(tmp_path):
    root = tmp_path/'run'
    write_json(root/'experiment.json', {'schema': 'experiment-run-v1', 'kind': 'shared-codec-recipe'})
    (root/'config.yaml').write_text('task: coco\n')
    write_json(root/'case.json', {'case_id': 'one', 'config_sha256': hashlib.sha256((root/'config.yaml').read_bytes()).hexdigest()})
    checkpoint = root/'frozen-checkpoint.bin'
    checkpoint.write_bytes(b'mock immutable checkpoint')
    plan = panels.make_plan(case_id='one', checkpoint={**panels.file_reference(checkpoint), 'role': 'terminal', 'step': 16},
        scope='initial', sites=['enc-0', 'dec-0'], conditions=['no_noise'], item_ids=[11, 12],
        identities={'source': 'mock-source', 'runtime': 'cpu-mock', 'input': 'two-image-whole-corpus',
                    'scorer': 'mock-whole-corpus', 'training_observation_eligible': True},
        noise={'seed': 42, 'namespace': 'mock-paired'})
    outputs = []
    for name, parent, scorer in [('first', None, score_one_failed), ('retry', 'first', score_panel)]:
        owned = root/f'attempts/evaluation-{name}'
        write_json(owned/'attempt.json', {'stage': 'evaluation', 'attempt_id': name, 'status': 'completed' if parent else 'partial'})
        previous = outputs[0] if parent else None
        attempt = panels.reserve_attempt(owned/'panels/worker-attempts', name, plan, parent=previous)
        inherited = {r['panel_id'] for r in json.loads((attempt/'attempt.json').read_text())['inherited']}
        assigned = panels.assignments_for(plan, devices=['cpu'], workers=1,
            panel_ids=[p['id'] for p in plan['panels'] if p['id'] not in inherited])
        states = panels.run_workers(plan, attempt, assigned, runtime_factory=load_runtime, score_panel=scorer)
        aggregate = panels.collect_results(attempt, plan, worker_states=states, final=True)
        assert aggregate['retry_count'] == (1 if parent else 0)
        assert aggregate['counts']['completed'] == (2 if parent else 1)
        outputs.append(attempt)
    parent_bytes = {str(p): p.read_bytes() for p in outputs[0].rglob('*.json')}
    snapshot = tmp_path/'snapshot'
    manifest = result_collection.prepare_snapshot(root, snapshot)
    copied = {a['source']: a for a in manifest['artifacts']}
    for attempt in outputs:
        assert all(str(p) in copied for p in attempt.rglob('*.json'))
        for p in attempt.rglob('*.json'):
            assert copied[str(p)]['sha256'] == hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = result_collection.collect_snapshot(snapshot, tmp_path/'collected')
    assert receipt['counts']['missing']['count'] == 0
    assert parent_bytes == {str(p): p.read_bytes() for p in outputs[0].rglob('*.json')}
    assert json.loads((outputs[1]/'aggregate.json').read_text())['rows'][0]['receipt']['scores'] == {'mock_cider': 3.5}


def test_public_yaml_execute_forwards_parallel_and_exact_retry(tmp_path, monkeypatch):
    controls = {'backend': 'local', 'workers': 2, 'devices': ['cpu', 'cpu'], 'parent_attempt': './prior/panels/worker-attempts/first'}
    request = {'schema': experiment.SCHEMA, 'workflow': 'shared', 'operation': 'evaluate',
        'execution': {'root': './recorded-run', 'case': 'one', 'attempt': 'retry',
                      'evaluation': {'scope': 'initial', 'parallel': controls}}}
    filename = tmp_path/'evaluation.yaml'
    filename.write_text(yaml.safe_dump(request))
    monkeypatch.setattr(experiment, '_bound', lambda *args: {})
    seen = []
    def evaluate(root, **kwargs):
        seen.append((root, kwargs))
        return tmp_path/'result'
    monkeypatch.setattr(shared_evaluation, 'evaluate_recipe_run', evaluate)
    experiment.execute(filename)
    assert len(seen) == 1
    assert seen[0][1]['selected'] == 'one'
    assert seen[0][1]['attempt_id'] == 'retry'
    assert seen[0][1]['parallel'] == {**controls, 'parent_attempt': str((tmp_path/controls['parent_attempt']).resolve())}


def test_generated_case_request_reaches_public_execution(tmp_path, monkeypatch):
    from jscc import submission, shared_runs
    source = Path(submission.__file__).resolve().parents[1]
    run = tmp_path/'bound-run'
    run.mkdir()
    commit = 'a'*40
    binding = {'mock': 'source binding admission'}
    rows = []
    for index in range(3):
        case = run/f'cases/case-{index}'
        write_json(case/'recipe.json', {'config': {'task': 'coco', 'data': {}},
            'locations': ['enc-0'], 'updates_per_location': 16})
        rows.append({'case_id': f'case-{index}', 'path': str(case)})
    prepared = tmp_path/'prepared.json'
    write_json(prepared, {'source_commit': commit})
    record = {'identity': 'mock-run-identity', 'prepared': {'path': str(prepared)}, 'cases': rows}
    source_ref = {'commit': commit, 'path': str(source)}
    write_json(run/'submission-input.json', {'source': source_ref, 'source_binding': binding})
    resources = {'partition': 'mock', 'account': 'mock', 'qos': 'mock', 'node': 'mock',
        'gpus': 1, 'cpus': 2, 'memory_gb': 2, 'walltime_minutes': 10}
    parallel = {'backend': 'local', 'workers': 2, 'devices': ['cpu', 'cpu']}
    job = {'schema': submission.CASE_SCHEMA, 'recipe': str(tmp_path/'recipe.yaml'),
        'registrations': [], 'run_output': str(run), 'root': '/', 'source': source_ref,
        'python': str(tmp_path/'python'), 'cache': str(tmp_path/'cache'),
        'output': str(tmp_path/'submission'), 'resources': resources,
        'cases': ['case-2'], 'evaluation_cases': ['case-2'], 'concurrency': 1,
        'combined': False, 'attempt': 'fresh', 'evaluation': {'scope': 'initial', 'parallel': parallel}}
    limits = {'root': '/', 'resources': resources, 'max_concurrency': 1, 'max_array_tasks': 3}
    filename, limits_file = tmp_path/'job.yaml', tmp_path/'limits.yaml'
    filename.write_text(yaml.safe_dump(job))
    limits_file.write_text(yaml.safe_dump(limits))
    monkeypatch.setattr(submission, 'capture_source_binding', lambda *args: binding)
    monkeypatch.setattr(submission, 'check_source_binding', lambda *args: None)
    monkeypatch.setattr(submission, '_case_inputs', lambda *args: (tmp_path/'recipe.yaml', [], {}))
    monkeypatch.setattr(shared_runs, 'recipe_run_record', lambda *args: record)
    bundle = submission.prepare_case_job(filename, limits_file)
    assert bundle['groups'] == [{'stage': 'training', 'indices': [2], 'concurrency': 1, 'aftercorr': None},
        {'stage': 'evaluation', 'indices': [2], 'concurrency': 1, 'aftercorr': 0}]
    request = bundle['cases'][2]['requests']['evaluation']
    request_file = tmp_path/'generated-evaluation.yaml'
    request_file.write_text(yaml.safe_dump(request))
    monkeypatch.setattr(experiment, '_bound', lambda *args: record)
    seen = []
    monkeypatch.setattr(shared_evaluation, 'evaluate_recipe_run', lambda *args, **kwargs: seen.append(kwargs) or tmp_path/'result')
    experiment.execute(request_file)
    assert seen[0]['selected'] == 'case-2' and seen[0]['parallel'] == parallel
    assert seen[0]['attempt_id'] == 'fresh-g1-i2'
