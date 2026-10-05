from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from jscc import experiment, panel_workers as core, panel_evaluation as coupling


@dataclass
class MockRuntimeFactory:
    request: dict

    def __call__(self, plan, device):
        # These injections execute inside each spawned process, at the real adapter seam.
        @contextmanager
        def restore(model, state):
            before = model['route']
            try:
                yield
            finally:
                model['route'] = before
        def score(model, processor, **kwargs):
            assert kwargs['request']['input_ids'] == [1, 2]
            assert 'worker' not in kwargs['request']['noise']['namespace']
            model['route'] = kwargs['site_id']
            Path(kwargs['output'], 'task.json').write_text('whole corpus')
            return {'completed_items': 2, 'denominator': 2,
                    'metrics': {'cider': 0.75, 'finite_scores': True, 'metadata': 'retained'},
                    'payload': {'valid': {'image': 2}}}
        coupling._panel_runtime = restore
        coupling._score_panel = score
        return {'request': self.request, 'state': None, 'model': {'route': 'idle'},
                'processor': None, 'data': 'whole corpus', 'guard': lambda: None}


def test_spawn_real_adapter_bridge_and_retry(tmp_path, monkeypatch):
    cp = tmp_path/'checkpoint'
    cp.write_text('exact bytes')
    request = {'task': 'coco', 'sites': ['site-a', 'site-b'], 'conditions': ['no_noise'],
               'input_ids': [1, 2], 'expected_items': 2,
               'noise': {'seed': 7, 'namespace': 'paired'}}
    plan = core.make_plan(case_id='case-a', checkpoint={**core.file_reference(cp), 'role': 'terminal', 'step': 3},
        scope='initial', sites=request['sites'], conditions=request['conditions'], item_ids=[1, 2],
        identities={'source': 'pinned', 'runtime': 'cpu', 'input': 'images', 'scorer': 'mock',
                    'training_observation_eligible': True}, noise=request['noise'])
    attempt = core.reserve_attempt(tmp_path/'attempts', 'first', plan)
    workers = core.run_workers(plan, attempt, core.assignments_for(plan, devices=['cpu', 'cpu'], workers=2),
        runtime_factory=MockRuntimeFactory(request), score_panel=coupling.score_recipe_panel, timeout_seconds=30)
    aggregate = core.collect_results(attempt, plan, worker_states=workers, final=True)
    assert aggregate['counts']['complete'] == 2
    assert aggregate['observed_items'] == 4
    pids = [json.loads(p.read_text())['pid'] for p in (attempt/'workers').glob('*/worker-start.json')]
    assert len(set(pids)) == 2
    results = [json.loads(p.read_text()) for p in (attempt/'workers').glob('*/results/*.json')]
    assert {r['noise_digest'] for r in results} == {core.canonical_digest(request['noise'])}
    assert all(r['scores'] == {'cider': 0.75} for r in results)
    from jscc.experiment_state import CheckpointRef
    state = SimpleNamespace(reference=CheckpointRef(str(cp), core.file_reference(cp)['sha256'], cp.stat().st_size),
                            payload={'metadata': {'plan_identity': 'recipe-plan'}})
    monkeypatch.setattr(coupling, '_verify_recipe_result', lambda *a, **k: None)
    receipt = coupling.bridge_receipt(aggregate, plan, request, state, attempt, tmp_path)
    assert receipt['status'] == 'complete'
    assert receipt['panels'][0]['metrics']['metadata'] == 'retained'
    retry = core.reserve_attempt(tmp_path/'attempts', 'retry', plan, parent=attempt)
    assert len(json.loads((retry/'attempt.json').read_text())['inherited']) == 2
    with pytest.raises(FileExistsError):
        core.reserve_attempt(tmp_path/'attempts', 'first', plan)


def test_public_yaml_forwarding(tmp_path, monkeypatch):
    from jscc import shared_evaluation
    path = tmp_path/'request.yaml'
    request = {'schema': experiment.SCHEMA, 'workflow': 'shared', 'operation': 'evaluate',
               'execution': {'root': 'run', 'attempt': 'new', 'case': 'case-a',
                 'evaluation': {'scope': 'initial', 'parallel': {'backend': 'local', 'workers': 2,
                    'devices': ['cpu', 'cpu'], 'parent_attempt': 'previous'}}}}
    path.write_text(yaml.safe_dump(request))
    _, parsed = experiment.read_request(path)
    assert parsed['execution']['evaluation']['parallel']['parent_attempt'] == str(tmp_path/'previous')
    monkeypatch.setattr(experiment, '_bound', lambda *a: {})
    received = {}
    def evaluate(root, **kwargs):
        received.update(kwargs)
        return tmp_path
    monkeypatch.setattr(shared_evaluation, 'evaluate_recipe_run', evaluate)
    experiment.execute(path)
    assert received['parallel'] == parsed['execution']['evaluation']['parallel']
    assert received['selected'] == 'case-a'
    request['execution']['cases'] = ['case-a']
    del request['execution']['case']
    path.write_text(yaml.safe_dump(request))
    with pytest.raises(ValueError, match='one shared case'):
        experiment.read_request(path)


def test_integrity_failure_restores_and_is_not_scoring_failure(tmp_path, monkeypatch):
    model = {'route': 'idle'}
    @contextmanager
    def restore(model, state):
        try:
            yield
        finally:
            model['route'] = 'idle'
    def score(*a, **k):
        model['route'] = 'active'
        raise ValueError('checkpoint changed')
    monkeypatch.setattr(coupling, '_panel_runtime', restore)
    monkeypatch.setattr(coupling, '_score_panel', score)
    runtime = {'request': {'task': 'hellaswag', 'conditions': ['no_noise']}, 'state': None,
               'model': model, 'processor': None, 'guard': lambda: None}
    with pytest.raises(ValueError, match='checkpoint changed'):
        coupling.score_recipe_panel(runtime, {'expected_items': 2}, {'site': 'a', 'condition': 'no_noise'}, tmp_path)
    assert model['route'] == 'idle'


def test_runtime_initializes_once_and_loads_full_coco(tmp_path, monkeypatch):
    from jscc import shared_evaluation, shared_preparation, shared_runs, shared_inputs
    request = {'task': 'coco', 'input_ids': [1, 2]}
    state = SimpleNamespace(payload={'metadata': {'recipe_plan': {'description': {'locations': ['a', 'b']}}}})
    checkpoint = tmp_path/'exact-checkpoint'
    checkpoint.write_bytes(b'exact')
    target = {'path': str(checkpoint), 'sha256': core.file_reference(checkpoint)['sha256'], 'size': 5,
              'role': 'terminal', 'step': 3, 'training_attempt': str(tmp_path/'training-a'),
              'engine_path': str(tmp_path/'training-a/engine'), 'policy': 'recipe-terminal'}
    calls = []
    monkeypatch.setattr(shared_evaluation, 'recipe_evaluation_target', lambda *a, **k:
        ({'checkpoint': target}, request, state, {'model': {'device': 'cpu'}}, 'initial', {'synthetic_cpu': True}))
    monkeypatch.setattr(coupling, '_validate_recipe_request', lambda *a, **k: None)
    monkeypatch.setattr(shared_runs, '_guard', lambda *a, **k: lambda: None)
    class Base:
        def parameters(self):
            return iter([SimpleNamespace(device='cpu')])
    def initialize(*a, **k):
        calls.append('initialize')
        return 'processor', SimpleNamespace(base=Base())
    monkeypatch.setattr(shared_preparation, 'initialize_recipe', initialize)
    monkeypatch.setattr(shared_inputs, 'load_coco_panel_data', lambda request, processor:
                        calls.append(('corpus', request['input_ids'])) or 'whole corpus')
    runtime = coupling.RecipeRuntimeFactory(str(tmp_path), 'a', None, None, 'initial', request, 30)(
        {'checkpoint': coupling.worker_checkpoint_reference(target)}, 'cpu')
    assert runtime['data'] == 'whole corpus'
    assert calls == ['initialize', ('corpus', [1, 2])]


def test_pending_observations_not_final_eligible():
    assert coupling.verified_observations({'observations_status_recorded': True,
        'observations_status': 'pending', 'observations': [{'status': 'pending'}]}, None) is False


def test_shared_case_incomplete_outcome_mapping(tmp_path, monkeypatch):
    from jscc import shared_evaluation as shared
    request = {'sites': ['a'], 'conditions': ['no_noise'], 'expected_items': 2}
    monkeypatch.setattr(shared, 'write_json_atomic', lambda p, value: p.parent.mkdir(parents=True, exist_ok=True) or p.write_text(json.dumps(value)))
    monkeypatch.setattr(shared, 'validate_reserved_attempt', lambda *a: None)
    monkeypatch.setattr(shared.engine, '_reopen_recipe', lambda *a: None)
    monkeypatch.setattr(shared, '_guard', lambda *a, **k: lambda: None)
    monkeypatch.setattr(coupling, 'evaluate_parallel_recipe', lambda **k:
        {'status': 'incomplete', 'final_eligible': False})
    finished = []
    monkeypatch.setattr(shared, 'finish_attempt', lambda path, status, **k: finished.append(status))
    shared._case_evaluation(tmp_path, {'case_id': 'a', 'scope': 'initial'}, request, None, {}, None,
        {'synthetic_cpu': True}, timeout_seconds=30, device_uuid=None,
        parallel={'backend': 'local', 'workers': 1, 'devices': ['cpu']})
    saved = json.loads((tmp_path/'result.json').read_text())
    assert saved['status'] == 'incomplete' and saved['final_eligible'] is False
    assert finished == ['partial']


def test_actual_recipe_target_checkpoint_boundary(tmp_path):
    checkpoint = tmp_path/'checkpoint'
    checkpoint.write_bytes(b'exact checkpoint')
    actual = core.file_reference(checkpoint)
    target = {'path': actual['path'], 'sha256': actual['sha256'], 'size': actual['bytes'],
              'role': 'terminal', 'step': 9, 'training_attempt': str(tmp_path/'training-a'),
              'engine_path': str(tmp_path/'training-a/engine'), 'policy': 'recipe-terminal'}
    translated = coupling.worker_checkpoint_reference(target)
    assert translated == {**target, 'bytes': actual['bytes']}
    plan = core.make_plan(case_id='case-a', checkpoint=translated, scope='initial', sites=['a'],
        conditions=['no_noise'], item_ids=[1], identities={'source': 's', 'runtime': 'r',
            'input': 'i', 'scorer': 'c'}, noise={'seed': 1, 'namespace': 'paired'})
    assert plan['checkpoint'] == translated
    assert 'bytes' not in target
    with pytest.raises(ValueError, match='bytes/hash differ'):
        coupling.worker_checkpoint_reference({**target, 'size': actual['bytes']+1})
    with pytest.raises(ValueError, match='size/bytes differ'):
        coupling.worker_checkpoint_reference({**target, 'bytes': actual['bytes']+1})
    with pytest.raises(ValueError, match='size must'):
        coupling.worker_checkpoint_reference({**target, 'size': True})


def test_parent_interrupt_finalizes_bridge_and_retry(tmp_path, monkeypatch):
    from jscc.experiment_state import CheckpointRef
    cp = tmp_path/'checkpoint'
    cp.write_bytes(b'exact')
    ref = core.file_reference(cp)
    target = {'path': ref['path'], 'sha256': ref['sha256'], 'size': ref['bytes'],
              'role': 'terminal', 'step': 1, 'engine_path': str(tmp_path/'engine')}
    eligibility = {'case_id': 'a', 'checkpoint': target, 'scope': 'initial',
                   'source': 's', 'runtime': 'r', 'observations_status_recorded': False}
    request = {'task': 'hellaswag', 'sites': ['a', 'b'], 'conditions': ['no_noise'],
               'input_ids': [1, 2], 'expected_items': 2, 'noise': {'seed': 1, 'namespace': 'paired'}}
    state = SimpleNamespace(reference=CheckpointRef(ref['path'], ref['sha256'], ref['bytes']),
                            payload={'metadata': {'plan_identity': 'plan'}})
    captured = {}
    def interrupt(plan, attempt, assignments, **kwargs):
        captured.update(plan=plan, attempt=attempt)
        worker = attempt/'workers'/'worker-0000'
        worker.mkdir(parents=True)
        (worker/'receipts').mkdir()
        pid = plan['panels'][0]['id']
        (worker/'starts').mkdir()
        core.exclusive_json(worker/'starts'/f'{pid}.json', {
            'schema': 'panel-worker-start-v1', 'plan_digest': plan['digest'],
            'panel_id': pid, 'worker_id': 'worker-0000'})
        core.exclusive_json(worker/'receipts'/f'{pid}.json', {
            'schema': 'panel-worker-receipt-v1', 'plan_digest': plan['digest'],
            'panel_id': pid, 'worker_id': 'worker-0000', 'status': 'interrupted',
            'item_ids': [], 'expected_items': 2, 'observed_items': 0, 'scores': {}, 'noise_digest': None})
        core.exclusive_json(worker/'worker-terminal.json', {'worker_id': 'worker-0000', 'status': 'interrupted'})
        raise KeyboardInterrupt('parent stop')
    monkeypatch.setattr(core, 'run_workers', interrupt)
    monkeypatch.setattr(coupling, '_verify_recipe_result', lambda *a, **k: None)
    output = tmp_path/'panels'
    with pytest.raises(KeyboardInterrupt, match='parent stop'):
        coupling.evaluate_parallel_recipe(root=tmp_path, eligibility=eligibility, request=request,
            state=state, output=output, controls={'backend': 'local', 'workers': 1, 'devices': ['cpu']},
            attempt_id='first', checkpoint=None, expected_step=None, timeout_seconds=30)
    assert (captured['attempt']/'final.json').is_file()
    aggregate = json.loads((captured['attempt']/'aggregate.json').read_text())
    assert [row['status'] for row in aggregate['rows']] == ['interrupted', 'pending']
    receipt = json.loads((output/'result.json').read_text())
    assert receipt['status'] == 'incomplete' and receipt['final_eligible'] is False
    assert [row['worker_status'] for row in receipt['panels']] == ['interrupted', 'pending']
    retry = core.reserve_attempt(tmp_path/'retries', 'retry', captured['plan'], parent=captured['attempt'])
    assert json.loads((retry/'attempt.json').read_text())['retry_count'] == 1


def test_shared_parent_interrupt_retains_attempt_status(tmp_path, monkeypatch):
    from jscc import shared_evaluation as shared
    monkeypatch.setattr(shared, 'validate_reserved_attempt', lambda *a: None)
    monkeypatch.setattr(shared.engine, '_reopen_recipe', lambda *a: None)
    monkeypatch.setattr(shared, '_guard', lambda *a, **k: lambda: None)
    def interrupt(**kwargs):
        raise KeyboardInterrupt('parent stop')
    monkeypatch.setattr(coupling, 'evaluate_parallel_recipe', interrupt)
    finished = []
    monkeypatch.setattr(shared, 'finish_attempt', lambda path, status, **kwargs: finished.append(status))
    with pytest.raises(KeyboardInterrupt, match='parent stop'):
        shared._case_evaluation(tmp_path, {'case_id': 'a', 'scope': 'initial'},
            {'sites': ['a'], 'conditions': ['no_noise'], 'expected_items': 2}, None, {}, None,
            {'synthetic_cpu': True}, timeout_seconds=30, device_uuid=None, parallel={})
    assert json.loads((tmp_path/'result.json').read_text())['status'] == 'interrupted'
    assert finished == ['interrupted']


@pytest.mark.parametrize('parallel', [[], [['backend', 'local'], ['workers', 1], ['devices', ['cpu']]], 'local'])
def test_parallel_mapping_rejects_nonmapping_before_effects(tmp_path, parallel):
    request = {'schema': experiment.SCHEMA, 'workflow': 'shared', 'operation': 'evaluate',
        'execution': {'root': 'run', 'attempt': 'new', 'case': 'one', 'evaluation': {'parallel': parallel}}}
    filename = tmp_path/'bad.yaml'
    filename.write_text(yaml.safe_dump(request))
    with pytest.raises(ValueError, match='evaluation.parallel requires'):
        experiment.read_request(filename)


def test_bridge_rejects_scientific_receipt_outside_owned_panel(tmp_path):
    result = tmp_path/'worker-result.json'
    external = tmp_path/'external.json'
    external.write_text('{}')
    result.write_text(json.dumps({'scientific_receipt': core.file_reference(external)}))
    aggregate = {'rows': [{'status': 'complete', 'owner': str(tmp_path/'worker-0'), 'panel_id': 'panel-000000',
                           'receipt': {'result': {'path': str(result)}}}]}
    with pytest.raises(ValueError, match='reference outside owned directory'):
        coupling.bridge_receipt(aggregate, {}, {}, None, tmp_path, tmp_path)


def test_parallel_accounting_keeps_exclusive_missing_interrupted_pending():
    from jscc.shared_evaluation import _counts
    counts = {'selected': 4, 'started': 3, 'completed': 1, 'complete': 1, 'failed': 0,
              'missing': 1, 'pending': 1, 'interrupted': 1, 'running': 0}
    receipt = {'panels': [{'status': 'complete', 'completed_items': 2}], 'worker_counts': counts}
    actual = _counts({'sites': ['a', 'b'], 'conditions': [0, 6], 'expected_items': 2}, receipt)
    assert sum(actual[key] for key in ('completed', 'failed', 'missing', 'pending', 'interrupted', 'running')) == actual['requested'] == 4
    assert actual['requested_items'] == 8 and actual['observed_items'] == 2
