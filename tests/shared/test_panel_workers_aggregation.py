from pathlib import Path
import pytest
from jscc import panel_workers as m


def plan(tmp_path, eligible=False):
    checkpoint = tmp_path / 'checkpoint.bin'
    checkpoint.write_bytes(b'actual checkpoint fixture')
    return m.make_plan(case_id='case', checkpoint=dict(m.file_reference(checkpoint), role='terminal', step=3), scope='initial', sites=[0, 1], conditions=['clean', 'noise'], item_ids=['a', 'b'], identities={'source': 'sha', 'runtime': 'cpu', 'input': 'corpussha', 'scorer': 'scorersha', 'training_observation_eligible': eligible}, noise={'seed': 12, 'namespace': 'case'})


def receipt(attempt, p, index=0, worker='worker-0', status='complete'):
    panel_id = p['panels'][index]['id']
    owner = attempt / 'workers' / worker
    m.exclusive_json(owner / 'starts' / (panel_id + '.json'), {'schema': 'panel-worker-start-v1', 'plan_digest': p['digest'], 'panel_id': panel_id, 'worker_id': worker, 'pid': 123})
    data = {'schema': 'panel-worker-receipt-v1', 'plan_digest': p['digest'], 'panel_id': panel_id, 'worker_id': worker, 'status': status, 'item_ids': p['item_ids'], 'scores': {'acc': 0.2, 'acc_norm': 0.4}, 'expected_items': 2, 'observed_items': 2, 'noise_digest': 'deterministic-noise'}
    if status == 'complete':
        result = owner / 'results' / (panel_id + '.json')
        m.exclusive_json(result, {k: data[k] for k in ('plan_digest', 'panel_id', 'item_ids', 'scores', 'noise_digest')})
        data['result'] = m.file_reference(result)
    else:
        data['error'] = 'mock failure'
    m.exclusive_json(owner / 'receipts' / (panel_id + '.json'), data)
    terminal = owner / 'worker-terminal.json'
    if not terminal.exists():
        m.exclusive_json(terminal, {'worker_id': worker, 'status': 'exited'})
    return data


def test_order_partial_retry_parent_preserved(tmp_path):
    p = plan(tmp_path)
    assert [(x['condition'], x['site']) for x in p['panels']] == [('clean', 0), ('clean', 1), ('noise', 0), ('noise', 1)]
    a = m.reserve_attempt(tmp_path / 'runs', 'first', p)
    receipt(a, p, 1)
    receipt(a, p, 0, status='failed')
    initial = m.collect_results(a, p, final=True)
    assert initial['counts']['completed'] == 1
    assert initial['counts']['failed'] == 1
    assert initial['counts']['pending'] == 2
    assert initial['observed_items'] == 2 and initial['expected_items'] == 8
    parent_bytes = (a / 'aggregate.json').read_bytes()
    b = m.reserve_attempt(tmp_path / 'runs', 'retry', p, parent=a)
    for index in (3, 2, 0):
        receipt(b, p, index)
    result = m.collect_results(b, p, final=True)
    assert result['counts']['completed'] == 4 and result['retry_count'] == 1
    assert not result['incomplete'] and not result['final_eligible']
    assert (a / 'aggregate.json').read_bytes() == parent_bytes
    assert m.collect_results(b, p, final=True) == result


def test_collisions_and_competing_success(tmp_path):
    p = plan(tmp_path)
    a = m.reserve_attempt(tmp_path / 'runs', 'first', p)
    with pytest.raises(FileExistsError):
        m.reserve_attempt(tmp_path / 'runs', 'first', p)
    with pytest.raises(ValueError):
        m.reserve_attempt(tmp_path / 'runs', '../escape', p)
    receipt(a, p)
    receipt(a, p, worker='worker-1')
    with pytest.raises(ValueError, match='competing'):
        m.collect_results(a, p)


def test_corruption_and_terminal_retry_gate(tmp_path):
    p = plan(tmp_path)
    a = m.reserve_attempt(tmp_path / 'runs', 'first', p)
    with pytest.raises(FileNotFoundError):
        m.reserve_attempt(tmp_path / 'runs', 'retry', p, parent=a)
    r = receipt(a, p)
    Path(r['result']['path']).write_text('{}')
    with pytest.raises(ValueError, match='hash'):
        m.collect_results(a, p)


def test_incomplete_eligibility_gate(tmp_path):
    p = plan(tmp_path, eligible=True)
    a = m.reserve_attempt(tmp_path / 'runs', 'first', p)
    receipt(a, p)
    result = m.collect_results(a, p)
    assert not result['final_eligible']
    for i in (3, 2, 1):
        receipt(a, p, i)
    assert m.collect_results(a, p, final=True)['final_eligible']


def test_missing_interrupted_running_and_denominators(tmp_path):
    p = plan(tmp_path)
    a = m.reserve_attempt(tmp_path / 'runs', 'first', p)
    for i, state in enumerate(('exited', 'timed_out', 'running')):
        worker = a / 'workers' / f'w{i}'
        m.exclusive_json(worker / 'starts' / (p['panels'][i]['id'] + '.json'), {'schema': 'panel-worker-start-v1', 'plan_digest': p['digest'], 'panel_id': p['panels'][i]['id'], 'worker_id': worker.name, 'pid': 123})
        m.exclusive_json(worker / 'worker-terminal.json', {'status': state})
    result = m.collect_results(a, p)
    assert [r['status'] for r in result['rows']] == ['missing', 'interrupted', 'running', 'pending']
    assert result['counts']['started'] == 3 and result['observed_items'] == 0


def test_terminal_corruption_and_wrong_membership(tmp_path):
    p = plan(tmp_path)
    a = m.reserve_attempt(tmp_path / 'runs', 'first', p)
    r = receipt(a, p)
    m.collect_results(a, p, final=True)
    Path(r['result']['path']).write_text('{}')
    with pytest.raises(ValueError, match='hash'):
        m.collect_results(a, p, final=True)
    with pytest.raises(ValueError, match='hash'):
        m.reserve_attempt(tmp_path / 'runs', 'retry', p, parent=a)


def test_nonfinite_and_changed_checkpoint(tmp_path):
    p = plan(tmp_path)
    with pytest.raises(ValueError):
        m.make_plan(case_id='x', checkpoint=p['checkpoint'], scope='x', sites=[True], conditions=['x'], item_ids=['a'], identities=p['identities'], noise=p['noise'])
    Path(p['checkpoint']['path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash'):
        m.reserve_attempt(tmp_path / 'runs', 'first', p)


def test_scope_and_numeric_condition_boundaries(tmp_path):
    p = plan(tmp_path, eligible=True)
    args = {k: p[k] for k in ('case_id', 'checkpoint', 'scope', 'item_ids', 'identities', 'noise')}
    args.update(sites=[0], conditions=[0, 0.0])
    with pytest.raises(ValueError, match='duplicate'):
        m.make_plan(**args)
    args.update(conditions=['clean'], scope='unknown')
    with pytest.raises(ValueError, match='scope'):
        m.make_plan(**args)
    for scope in ('supplemental', 'development'):
        args['scope'] = scope
        scoped = m.make_plan(**args)
        a = m.reserve_attempt(tmp_path / 'runs', scope, scoped)
        receipt(a, scoped)
        result = m.collect_results(a, scoped, final=True)
        assert not result['incomplete'] and not result['final_eligible']


def test_exclusive_publication_and_active_final_guard(tmp_path):
    path = tmp_path / 'receipt.json'
    m.exclusive_json(path, {'complete': True})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        m.exclusive_json(path, {'complete': False})
    assert path.read_bytes() == before
    assert not list(tmp_path.glob('.exclusive-*'))
    p = plan(tmp_path)
    a = m.reserve_attempt(tmp_path / 'runs', 'first', p)
    with pytest.raises(ValueError, match='running'):
        m.collect_results(a, p, worker_states={'worker-0': {'status': 'running'}}, final=True)
    assert not (a / 'final.json').exists()


@pytest.mark.parametrize('replacement', [[True], [1.0]])
def test_typed_membership_and_integer_denominators(tmp_path, replacement):
    p = plan(tmp_path)
    p = m.make_plan(case_id='case', checkpoint=p['checkpoint'], scope='initial',
                    sites=[0], conditions=['no_noise'], item_ids=[1],
                    identities=p['identities'], noise=p['noise'])
    a = m.reserve_attempt(tmp_path / 'runs', 'typed', p)
    r = receipt(a, p)
    result_path = Path(r['result']['path'])
    data = m._read_json(result_path)
    data['item_ids'] = replacement
    m.atomic_json(result_path, data)
    r.update(item_ids=replacement, expected_items=1, observed_items=1, result=m.file_reference(result_path))
    m.atomic_json(a / 'workers/worker-0/receipts' / (p['panels'][0]['id'] + '.json'), r)
    with pytest.raises(ValueError, match='membership'):
        m.collect_results(a, p, final=True)


@pytest.mark.parametrize('persisted', [None, 'running'])
def test_final_requires_persisted_terminal_lifecycle(tmp_path, persisted):
    p = plan(tmp_path)
    a = m.reserve_attempt(tmp_path / 'runs', 'active', p)
    worker = a / 'workers/worker-0'
    m.exclusive_json(worker / 'starts' / (p['panels'][0]['id'] + '.json'),
                     {'schema': 'panel-worker-start-v1', 'plan_digest': p['digest'],
                      'panel_id': p['panels'][0]['id'], 'worker_id': worker.name})
    if persisted is not None:
        m.exclusive_json(worker / 'worker-terminal.json', {'status': persisted})
    for supplied in (None, {}, {'worker-0': {'status': 'exited'}}):
        with pytest.raises(ValueError, match='terminal'):
            m.collect_results(a, p, worker_states=supplied, final=True)
    assert not (a / 'final.json').exists()


@pytest.mark.parametrize('late', ['unrelated', 'competing-success'])
def test_retry_rejects_late_terminal_evidence(tmp_path, late):
    p = plan(tmp_path)
    a = m.reserve_attempt(tmp_path / 'runs', 'parent', p)
    receipt(a, p)
    m.collect_results(a, p, final=True)
    if late == 'unrelated':
        m.exclusive_json(a / 'workers/worker-0/late.json', {'late': True})
    else:
        receipt(a, p, worker='worker-1')
    with pytest.raises(ValueError, match='evidence changed'):
        m.reserve_attempt(tmp_path / 'runs', 'retry', p, parent=a)
    assert not (tmp_path / 'runs/retry').exists()


@pytest.mark.parametrize('supplied', ['exited', 'failed'])
def test_conflicting_terminal_states_cannot_change_outcome(tmp_path, supplied):
    p = plan(tmp_path)
    a = m.reserve_attempt(tmp_path / 'runs', 'conflict', p)
    worker = a / 'workers/worker-0'
    m.exclusive_json(worker / 'starts' / (p['panels'][0]['id'] + '.json'),
        {'schema': 'panel-worker-start-v1', 'plan_digest': p['digest'],
         'panel_id': p['panels'][0]['id'], 'worker_id': worker.name})
    m.exclusive_json(worker / 'worker-terminal.json', {'worker_id': worker.name, 'status': 'timed_out'})
    with pytest.raises(ValueError, match='state conflict'):
        m.collect_results(a, p, worker_states={worker.name: {'status': supplied}}, final=True)
    with pytest.raises(ValueError, match='state conflict'):
        m.collect_results(a, p, worker_states={worker.name: {'status': supplied}}, final=False)
    assert not (a / 'final.json').exists()
    result = m.collect_results(a, p, final=True)
    assert result['rows'][0]['status'] == 'interrupted'


def test_foreign_terminal_worker_identity_rejects(tmp_path):
    p = plan(tmp_path)
    a = m.reserve_attempt(tmp_path / 'runs', 'foreign', p)
    with pytest.raises(ValueError, match='identity'):
        m.collect_results(a, p, worker_states={'foreign': {'status': 'exited'}}, final=True)
