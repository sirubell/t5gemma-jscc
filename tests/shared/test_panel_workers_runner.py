import json
import os
from pathlib import Path
import time
import pytest
from jscc import panel_workers as workers


def factory(plan, device):
    if 'load_log' in plan['identities']:
        with open(Path(plan['identities']['load_log']) / (str(os.getpid()) + '.log'), 'a') as stream:
            stream.write('load\n')
    return {'loads': 1, 'device': device}


def score(runtime, plan, panel, output):
    time.sleep((3-panel['index']) * 0.01)
    assert runtime['loads'] == 1
    return {'item_ids': plan['item_ids'], 'scores': {'acc': float(panel['index']), 'acc_norm': float(panel['index'])},
            'noise_digest': 'noise-' + panel['id']}


def bad_score(runtime, plan, panel, output):
    if panel['index'] == 0:
        raise workers.PanelScoringError('ordinary failure')
    return score(runtime, plan, panel, output)


def corrupt_score(runtime, plan, panel, output):
    return {'item_ids': ['wrong'], 'scores': {'acc': 1.0}, 'noise_digest': 'x'}


def slow_score(runtime, plan, panel, output):
    time.sleep(5)
    return score(runtime, plan, panel, output)


@pytest.fixture
def plan(tmp_path):
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.write_bytes(b'mock')
    reference = dict(workers.file_reference(checkpoint), role='mock', step=0)
    return workers.make_plan(case_id='mock', checkpoint=reference, scope='initial',
                             sites=[0, 1], conditions=['no_noise', 0], item_ids=['item'],
                             identities={'source': 'mock', 'runtime': 'mock', 'input': 'mock', 'scorer': 'mock'},
                             noise={'seed': 1, 'namespace': 'mock'})


def run(plan, attempt, assignments, **kwargs):
    plan['digest'] = workers.canonical_digest({k: v for k, v in plan.items() if k != 'digest'})
    (attempt / 'plan.json').write_text(json.dumps(plan))
    return workers.run_workers(plan, attempt, assignments, **kwargs)


def results(root):
    return {p.stem: json.loads(p.read_text()) for p in root.glob('workers/worker-*/results/*.json')}


def test_serial_parallel_cpu(plan, tmp_path):
    loads = tmp_path / 'loads'
    loads.mkdir()
    plan['identities']['load_log'] = str(loads)
    serial = tmp_path / 'serial'
    serial.mkdir()
    parallel = tmp_path / 'parallel'
    parallel.mkdir()
    run(plan, serial, workers.assignments_for(plan, devices=['cpu'], workers=1), runtime_factory=factory, score_panel=score)
    states = run(plan, parallel, workers.assignments_for(plan, devices=['cpu', 'cpu'], workers=2), runtime_factory=factory, score_panel=score)
    assert results(serial) == results(parallel)
    assert len({s['pid'] for s in states.values()}) == 2
    assert all(s['status'] == 'exited' for s in states.values())
    assert len(list(loads.glob('*.log'))) == 3
    assert all(p.read_text() == 'load\n' for p in loads.glob('*.log'))


def test_ordinary_failure_continues(plan, tmp_path):
    run(plan, tmp_path, workers.assignments_for(plan, devices=['cpu'], workers=1), runtime_factory=factory, score_panel=bad_score)
    assert set(results(tmp_path)) == {'panel-000001', 'panel-000002', 'panel-000003'}
    assert json.loads((tmp_path / 'workers/worker-0000/receipts/panel-000000.json').read_text())['status'] == 'failed'


def test_corruption_aborts(plan, tmp_path):
    states = run(plan, tmp_path, workers.assignments_for(plan, devices=['cpu'], workers=1), runtime_factory=factory, score_panel=corrupt_score)
    assert states['worker-0000']['status'] == 'failed'
    assert not (tmp_path / 'workers/worker-0000/starts/panel-000001.json').exists()


def test_timeout_cleanup(plan, tmp_path):
    states = run(plan, tmp_path, workers.assignments_for(plan, devices=['cpu'], workers=1), runtime_factory=factory, score_panel=slow_score, timeout_seconds=.4)
    assert states['worker-0000']['status'] == 'timed_out'
    assert (tmp_path / 'workers/worker-0000/worker-terminal.json').exists()


def test_collision_and_partition(plan, tmp_path):
    (tmp_path / 'workers/worker-0000').mkdir(parents=True)
    with pytest.raises(FileExistsError):
        run(plan, tmp_path, workers.assignments_for(plan, devices=['cpu'], workers=1), runtime_factory=factory, score_panel=score)
    with pytest.raises(ValueError):
        workers.assignments_for(plan, devices=['cuda:01', 'cuda:1'], workers=2)
    with pytest.raises(ValueError):
        workers.assignments_for(plan, devices=['cpu'], workers=1, panel_ids=['panel-000000', 'panel-000000'])


@pytest.mark.parametrize('failure', [RuntimeError('launch failed'), KeyboardInterrupt()])
def test_launch_failure_and_interrupt_cleanup(plan, tmp_path, monkeypatch, failure):
    context = workers.multiprocessing.get_context('spawn')
    original = type(context.Process()).start
    calls = []
    def start(process):
        calls.append(process)
        if len(calls) == 2:
            raise failure
        return original(process)
    monkeypatch.setattr(type(context.Process()), 'start', start)
    with pytest.raises(type(failure)):
        run(plan, tmp_path, workers.assignments_for(plan, devices=['cpu', 'cpu'], workers=2), runtime_factory=factory, score_panel=slow_score)
    assert not calls[0].is_alive()
    assert len(list(tmp_path.glob('workers/worker-*/worker-terminal.json'))) == 2


def conflicting_score(runtime, plan, panel, output):
    return dict(score(runtime, plan, panel, output), plan_digest='conflicting-plan')


def nonfinite_score(runtime, plan, panel, output):
    return dict(score(runtime, plan, panel, output), scores={'cider': float('nan')})


@pytest.mark.parametrize('callback', [conflicting_score, nonfinite_score])
def test_metric_and_identity_integrity(plan, tmp_path, callback):
    states = run(plan, tmp_path, workers.assignments_for(plan, devices=['cpu'], workers=1), runtime_factory=factory, score_panel=callback)
    assert states['worker-0000']['status'] == 'failed'
    assert not results(tmp_path)
    assert not (tmp_path / 'workers/worker-0000/starts/panel-000001.json').exists()


@pytest.mark.parametrize('changed', ['plan', 'checkpoint'])
def test_changed_trust_boundary_before_load(plan, tmp_path, changed):
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    if changed == 'plan':
        (tmp_path / 'plan.json').write_text('{}')
    else:
        Path(plan['checkpoint']['path']).write_bytes(b'changed')
    with pytest.raises(ValueError):
        workers.run_workers(plan, tmp_path, workers.assignments_for(plan, devices=['cpu'], workers=1), runtime_factory=factory, score_panel=score)
    assert not (tmp_path / 'workers').exists()


def fatal_score(runtime, plan, panel, output):
    raise ValueError('denominator/source/restoration failure')


def test_callback_untyped_exception_aborts(plan, tmp_path):
    states = run(plan, tmp_path, workers.assignments_for(plan, devices=['cpu'], workers=1), runtime_factory=factory, score_panel=fatal_score)
    assert states['worker-0000']['status'] == 'failed'
    assert not results(tmp_path)
    assert not (tmp_path / 'workers/worker-0000/starts/panel-000001.json').exists()


def typed_wrong_score(runtime, plan, panel, output):
    return dict(score(runtime, plan, panel, output), item_ids=[True])


def test_typed_callback_membership_aborts_worker(plan, tmp_path):
    plan['item_ids'] = [1]
    states = run(plan, tmp_path, workers.assignments_for(plan, devices=['cpu'], workers=1),
                 runtime_factory=factory, score_panel=typed_wrong_score)
    assert states['worker-0000']['status'] == 'failed'
    assert not list((tmp_path / 'workers').rglob('results/*.json'))
