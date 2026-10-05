"""Isolated whole-panel process work and one-writer verified partial aggregation."""

from pathlib import Path
import hashlib
import json
import math
import multiprocessing
import os
import re
import tempfile
import time


def _primitive(value):
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, allow_nan=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value) or value in ('.', '..'):
        raise ValueError('unsafe ID')
    return value


def file_reference(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('reference is not a regular nonsymlink file')
    data = path.read_bytes()
    return {'path': str(path.absolute()), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def _verify_reference(ref, owner=None):
    if not isinstance(ref, dict) or not isinstance(ref.get('path'), str) or not Path(ref['path']).is_absolute():
        raise ValueError('invalid reference')
    if type(ref.get('bytes')) is not int or ref['bytes'] < 0:
        raise ValueError('reference byte count must be integer')
    path = Path(ref['path'])
    if owner is not None:
        try:
            path.resolve().relative_to(Path(owner).resolve())
        except ValueError:
            raise ValueError('reference outside owned directory') from None
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('symlink reference')
    actual = file_reference(path)
    if any(ref.get(k) != actual[k] for k in ('path', 'sha256', 'bytes')):
        raise ValueError('reference hash or size mismatch')
    return path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, allow_nan=False, sort_keys=True, indent=2).encode() + b'\n'
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.atomic-')
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def exclusive_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, allow_nan=False, sort_keys=True, indent=2).encode() + b'\n'
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.exclusive-')
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # complete bytes become visible with exclusive no-clobber semantics
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        os.unlink(temporary)


def _read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def _validate_plan(plan):
    if plan.get('schema') != 'panel-workers-v1' or plan.get('digest') != canonical_digest({k: v for k, v in plan.items() if k != 'digest'}):
        raise ValueError('plan digest mismatch')
    rebuilt = make_plan(case_id=plan['case_id'], checkpoint=plan['checkpoint'], scope=plan['scope'], sites=list(dict.fromkeys(p['site'] for p in plan['panels'])), conditions=list(dict.fromkeys(p['condition'] for p in plan['panels'])), item_ids=plan['item_ids'], identities=plan['identities'], noise=plan['noise'])
    if rebuilt != plan:
        raise ValueError('plan semantic mismatch')
    return plan


def make_plan(*, case_id, checkpoint, scope, sites, conditions, item_ids, identities, noise):
    _safe_id(case_id)
    if scope not in ('initial', 'supplemental', 'development'):
        raise ValueError('unsupported scope')
    checkpoint, identities, noise = _primitive(checkpoint), _primitive(identities), _primitive(noise)
    _verify_reference(checkpoint)
    if not isinstance(checkpoint.get('role'), str) or not checkpoint['role'] or type(checkpoint.get('step')) is not int or checkpoint['step'] < 0:
        raise ValueError('checkpoint role and step required')
    if type(noise.get('seed')) is not int or not isinstance(noise.get('namespace'), str) or not noise['namespace']:
        raise ValueError('noise seed and namespace required')
    if not all(identities.get(k) for k in ('source', 'runtime', 'input', 'scorer')):
        raise ValueError('source/runtime/input/scorer identity required')
    if 'training_observation_eligible' in identities and type(identities['training_observation_eligible']) is not bool:
        raise ValueError('final eligibility must be boolean')
    sites, conditions, item_ids = _primitive(list(sites)), _primitive(list(conditions)), _primitive(list(item_ids))
    for values in (sites, conditions, item_ids):
        if not values or len({canonical_digest(v) for v in values}) != len(values):
            raise ValueError('empty or duplicate membership')
        if any(type(v) not in (str, int, float) or isinstance(v, str) and not v for v in values):
            raise ValueError('membership IDs must be typed scalar values')
        if len(set(values)) != len(values):
            raise ValueError('numerically equivalent duplicate membership')
    panels = [{'id': f'panel-{i:06d}', 'index': i, 'site': site, 'condition': condition} for i, (condition, site) in enumerate((c, s) for c in conditions for s in sites)]
    plan = {'schema': 'panel-workers-v1', 'case_id': case_id, 'scope': _primitive(scope), 'checkpoint': checkpoint, 'panels': panels, 'item_ids': item_ids, 'expected_items': len(item_ids), 'identities': identities, 'noise': noise}
    plan['digest'] = canonical_digest(plan)
    return plan


def _verify_snapshot_evidence(snapshot):
    for ref in snapshot.get('evidence', []):
        _verify_reference(ref)
    for row in snapshot['rows']:
        if row['status'] == 'complete':
            _verify_reference(row['receipt']['result'], row['owner'])


def _verify_terminal_inventory(attempt, snapshot):
    """A terminal snapshot binds all worker evidence, including failed receipts."""
    _verify_snapshot_evidence(snapshot)
    attempt = Path(attempt)
    current = sorted(str(p.absolute()) for p in (attempt / 'workers').rglob('*.json')) if (attempt / 'workers').exists() else []
    if current != sorted(ref['path'] for ref in snapshot['evidence']):
        raise ValueError('terminal attempt evidence changed')


def _require_terminal_workers(attempt, worker_states):
    terminal_states = {'exited', 'failed', 'interrupted', 'timed_out'}
    for supplied in (worker_states or {}).values():
        state = supplied.get('status', supplied.get('state', '')) if isinstance(supplied, dict) else supplied
        if state not in terminal_states:
            raise ValueError('cannot finalize while worker state is not terminal')
    workers = list((Path(attempt) / 'workers').glob('*'))
    if set(worker_states or {}) - {worker.name for worker in workers}:
        raise ValueError('supplied worker identity has no persisted evidence')
    for worker in workers:
        terminal = worker / 'worker-terminal.json'
        if worker.is_symlink() or not worker.is_dir() or not terminal.is_file():
            raise ValueError('cannot finalize without terminal worker evidence')
        persisted = _read_json(terminal)
        if persisted.get('status') not in terminal_states:
            raise ValueError('cannot finalize while persisted worker state is not terminal')
        if persisted.get('worker_id', worker.name) != worker.name:
            raise ValueError('terminal worker identity conflict')
        supplied = (worker_states or {}).get(worker.name)
        if supplied is not None:
            state = supplied.get('status', supplied.get('state', '')) if isinstance(supplied, dict) else supplied
            if state != persisted['status'] or (isinstance(supplied, dict) and supplied.get('worker_id', worker.name) != worker.name):
                raise ValueError('terminal worker state conflict')


def reserve_attempt(root, attempt_id, plan, *, parent=None):
    _validate_plan(plan)
    root = Path(root).absolute()
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError('symlink attempt root')
    attempt = root / _safe_id(attempt_id)
    inherited, parent_ref, retry_count = [], None, 0
    if parent is not None:
        if isinstance(parent, dict):
            parent_ref = _primitive(parent)
        else:
            parent_ref = _read_json(Path(parent) / 'final.json')
        snapshot_path = _verify_reference(parent_ref)
        snapshot = _read_json(snapshot_path)
        _verify_terminal_inventory(snapshot_path.parent.parent, snapshot)
        if snapshot.get('plan_digest') != plan['digest'] or snapshot.get('terminal') is not True:
            raise ValueError('retry requires same plan terminal immutable snapshot')
        if snapshot_path.parent.name != 'snapshots' or _read_json(snapshot_path.parent.parent / 'final.json') != parent_ref:
            raise ValueError('retry snapshot is not registered terminal snapshot')
        for row in snapshot['rows']:
            if row['status'] == 'complete':
                _verify_complete(row['receipt'], plan, row['panel_id'], row['owner'])
                inherited.append(row)
        retry_count = snapshot['retry_count'] + 1
    root.mkdir(parents=True, exist_ok=True)
    attempt.mkdir()  # occupied attempt IDs are deliberately never reused
    exclusive_json(attempt / 'plan.json', plan)
    exclusive_json(attempt / 'attempt.json', {'plan_digest': plan['digest'], 'parent': parent_ref, 'inherited': inherited, 'retry_count': retry_count})
    return attempt


def _verify_complete(receipt, plan, panel_id, owner):
    if receipt.get('schema') != 'panel-worker-receipt-v1' or receipt.get('plan_digest') != plan['digest'] or receipt.get('panel_id') != panel_id or receipt.get('status') != 'complete':
        raise ValueError('receipt identity mismatch')
    result = _read_json(_verify_reference(receipt['result'], owner))
    for data in (receipt, result):
        if canonical_digest(data.get('item_ids')) != canonical_digest(plan['item_ids']):
            raise ValueError('incomplete whole panel membership')
        scores = data.get('scores')
        if not isinstance(scores, dict) or not scores or any(not isinstance(k, str) or not k for k in scores) or any(type(x) not in (int, float) or not math.isfinite(x) for x in scores.values()):
            raise ValueError('nonfinite score')
        for key, expected in (('plan_digest', plan['digest']), ('panel_id', panel_id), ('identities', plan['identities'])):
            if key in data and canonical_digest(data[key]) != canonical_digest(expected):
                raise ValueError('result identity mismatch')
    if any(type(receipt.get(key)) is not int or receipt[key] != plan['expected_items']
           for key in ('expected_items', 'observed_items')):
        raise ValueError('receipt denominator mismatch')
    if not isinstance(receipt.get('noise_digest'), str) or not receipt['noise_digest'] or any(result.get(k) != receipt[k] for k in ('item_ids', 'scores', 'noise_digest')):
        raise ValueError('result/receipt mismatch')
    return receipt


def collect_results(attempt, plan, *, worker_states=None, final=False):
    _validate_plan(plan)
    if final and any((state.get('status', state.get('state', '')) if isinstance(state, dict) else state) in ('running', 'started') for state in (worker_states or {}).values()):
        raise ValueError('cannot finalize while workers are running')
    attempt = Path(attempt).absolute()
    if any(p.is_symlink() for p in (attempt, *attempt.parents)) or (attempt / 'workers').is_symlink():
        raise ValueError('symlink attempt or worker root')
    if final:
        _require_terminal_workers(attempt, worker_states)
    if _read_json(attempt / 'plan.json') != plan:
        raise ValueError('attempt plan mismatch')
    lock = attempt / '.aggregate-lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        if (attempt / 'final.json').exists():
            previous = _read_json(_verify_reference(_read_json(attempt / 'final.json'), attempt))
            _verify_terminal_inventory(attempt, previous)
            return previous
        metadata = _read_json(attempt / 'attempt.json')
        if metadata.get('plan_digest') != plan['digest']:
            raise ValueError('attempt metadata plan mismatch')
        if metadata.get('parent') is not None:
            parent_path = _verify_reference(metadata['parent'])
            parent_snapshot = _read_json(parent_path)
            _verify_terminal_inventory(parent_path.parent.parent, parent_snapshot)
            if parent_snapshot.get('terminal') is not True or parent_snapshot.get('plan_digest') != plan['digest'] or metadata['inherited'] != [r for r in parent_snapshot['rows'] if r['status'] == 'complete']:
                raise ValueError('inherited snapshot mismatch')
        inherited = {r['panel_id']: r for r in metadata['inherited']}
        workers = list((attempt / 'workers').glob('*')) if (attempt / 'workers').exists() else []
        rows = []
        known = {p['id'] for p in plan['panels']}
        for worker in workers:
            if worker.is_symlink() or not worker.is_dir():
                raise ValueError('invalid worker directory')
            for sub in ('starts', 'receipts'):
                for path in (worker / sub).glob('*.json'):
                    if path.stem not in known or path.is_symlink():
                        raise ValueError('unknown panel or symlink receipt')
        for panel in plan['panels']:
            pid = panel['id']
            successes = []
            started, failures, interrupted, active = False, [], False, False
            if pid in inherited:
                row = inherited[pid]
                _verify_complete(row['receipt'], plan, pid, row['owner'])
                successes.append(row)
                started = True
            for worker in workers:
                start_path, receipt_path = worker / 'starts' / (pid + '.json'), worker / 'receipts' / (pid + '.json')
                state = (worker_states or {}).get(worker.name, {})
                state = state.get('status', state.get('state', '')) if isinstance(state, dict) else state
                terminal_path = worker / 'worker-terminal.json'
                if terminal_path.exists():
                    persisted_state = _read_json(terminal_path).get('status', '')
                    if state and state not in ('running', 'started') and state != persisted_state:
                        raise ValueError('terminal worker state conflict')
                    state = persisted_state
                if start_path.exists():
                    start = _read_json(start_path)
                    if start.get('schema') != 'panel-worker-start-v1' or start.get('plan_digest') != plan['digest'] or start.get('panel_id') != pid or start.get('worker_id') != worker.name:
                        raise ValueError('start identity mismatch')
                    started = True
                    interrupted |= state in ('interrupted', 'timed_out')
                    active |= state in ('running', 'started', '') and not final
                if receipt_path.exists():
                    receipt = _read_json(receipt_path)
                    if not start_path.exists() or receipt.get('schema') != 'panel-worker-receipt-v1' or receipt.get('plan_digest') != plan['digest'] or receipt.get('panel_id') != pid or receipt.get('worker_id') != worker.name:
                        raise ValueError('receipt identity or start mismatch')
                    status = receipt.get('status')
                    if status == 'complete':
                        _verify_complete(receipt, plan, pid, worker)
                        successes.append({'panel_id': pid, 'status': 'complete', 'receipt': receipt, 'owner': str(worker)})
                    elif status == 'failed':
                        failures.append(receipt)
                    elif status == 'interrupted':
                        interrupted = True
                    else:
                        raise ValueError('unknown receipt status')
            if len(successes) > 1:
                raise ValueError('competing panel successes')
            row = successes[0] if successes else {'panel_id': pid, 'status': 'failed' if failures else 'interrupted' if interrupted else 'running' if active else 'missing' if started else 'pending', 'failures': failures}
            row = dict(row, index=panel['index'], site=panel['site'], condition=panel['condition'], started=started)
            rows.append(row)
        counts = {key: sum(r['status'] == key for r in rows) for key in ('complete', 'failed', 'missing', 'pending', 'interrupted', 'running')}
        counts.update(selected=len(rows), started=sum(r['started'] for r in rows), completed=counts['complete'])
        value = {'schema': 'panel-worker-aggregate-v1', 'plan_digest': plan['digest'], 'rows': rows, 'counts': counts, 'expected_items': len(rows) * plan['expected_items'], 'observed_items': counts['complete'] * plan['expected_items'], 'retry_count': metadata['retry_count'], 'incomplete': counts['complete'] != len(rows), 'final_eligible': plan['scope'] == 'initial' and counts['complete'] == len(rows) and plan['identities'].get('training_observation_eligible', False) is True, 'terminal': bool(final)}
        value['evidence'] = [file_reference(path) for path in sorted((attempt / 'workers').rglob('*.json'))] if (attempt / 'workers').exists() else []
        snapshots = attempt / 'snapshots'
        snapshots.mkdir(exist_ok=True)
        snapshot = snapshots / f'{len(list(snapshots.glob("*.json"))) + 1:06d}.json'
        exclusive_json(snapshot, value)
        atomic_json(attempt / 'aggregate.json', value)
        if final:
            exclusive_json(attempt / 'final.json', file_reference(snapshot))
        return value
    finally:
        os.close(fd)
        lock.unlink()


class PanelScoringError(RuntimeError):
    """A callback-declared recoverable scoring/output failure for one panel."""


def validate_parallel_controls(value):
    if not isinstance(value, dict) or set(value) != {'backend', 'workers', 'devices'}:
        raise ValueError('parallel controls require backend, workers, devices')
    if value['backend'] != 'local':
        raise ValueError('parallel backend must be local')
    count = value['workers']
    devices = value['devices']
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('workers must be positive integer')
    if not isinstance(devices, list) or len(devices) != count:
        raise ValueError('explicit devices must match workers')
    normalized = []
    for device in devices:
        if device == 'cpu':
            normalized.append(device)
        elif isinstance(device, str) and re.fullmatch(r'cuda:\d+', device):
            normalized.append('cuda:' + str(int(device.split(':')[1])))
        else:
            raise ValueError('device must be cpu or cuda:index')
    gpu = [d for d in normalized if d != 'cpu']
    if len(set(gpu)) != len(gpu):
        raise ValueError('CUDA devices must be disjoint')
    return {'backend': 'local', 'workers': count, 'devices': normalized}


def assignments_for(plan, *, devices, workers, panel_ids=None):
    normalized = validate_parallel_controls({'backend': 'local', 'workers': workers, 'devices': devices})['devices']
    known = [p['id'] for p in plan['panels']]
    selected = known if panel_ids is None else list(panel_ids)
    if len(set(selected)) != len(selected) or any(p not in known for p in selected):
        raise ValueError('invalid panel partition')
    return [{'worker_id': f'worker-{i:04d}', 'device': device,
             'panel_ids': selected[i::workers]} for i, device in enumerate(normalized)]


def _runner_child(plan, assignment, directory, runtime_factory, score_panel):
    directory = Path(directory)
    worker = assignment['worker_id']
    current = None
    terminal = {'worker_id': worker, 'pid': os.getpid(), 'status': 'exited'}
    try:
        exclusive_json(directory / 'worker-start.json', {'worker_id': worker, 'pid': os.getpid(),
                     'device': assignment['device'], 'plan_digest': plan['digest']})
        _validate_plan(plan)
        if json.loads((directory.parent.parent / 'plan.json').read_text()) != plan:
            raise ValueError('attempt plan differs from worker plan')
        runtime = runtime_factory(plan, assignment['device'])
        panels = {p['id']: p for p in plan['panels']}
        for panel_id in assignment['panel_ids']:
            current = panel_id
            common = {'plan_digest': plan['digest'], 'panel_id': panel_id, 'worker_id': worker}
            exclusive_json(directory / 'starts' / (panel_id + '.json'),
                         dict(common, schema='panel-worker-start-v1', pid=os.getpid()))
            output = directory / 'outputs' / panel_id
            output.mkdir()
            try:
                result = score_panel(runtime, plan, panels[panel_id], str(output))
            except PanelScoringError as error:
                exclusive_json(directory / 'receipts' / (panel_id + '.json'),
                             dict(common, schema='panel-worker-receipt-v1', status='failed',
                                  failure=f'{type(error).__name__}: {error}', item_ids=[],
                                  expected_items=plan['expected_items'], observed_items=0,
                                  scores={}, noise_digest=None))
                current = None
                continue
            # Invalid scorer output is an integrity failure, never a complete row.
            if not isinstance(result, dict) or canonical_digest(result.get('item_ids')) != canonical_digest(plan['item_ids']):
                raise ValueError('scorer item identities differ from plan')
            scores = result.get('scores')
            if (not isinstance(scores, dict) or not scores or
                any(not isinstance(key, str) or not key for key in scores) or
                any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in scores.values())):
                raise ValueError('scorer task metrics invalid')
            if len(result['item_ids']) != plan['expected_items']:
                raise ValueError('scorer item count differs from plan')
            for key, expected in {'plan_digest': plan['digest'], 'panel_id': panel_id,
                                  'identities': plan.get('identities', {})}.items():
                if key in result and canonical_digest(result[key]) != canonical_digest(expected):
                    raise ValueError('scorer result identity differs from plan: ' + key)
            if not isinstance(result.get('noise_digest'), str) or not result['noise_digest']:
                raise ValueError('scorer noise identity absent')
            result = dict(result, plan_digest=plan['digest'], panel_id=panel_id, identities=plan.get('identities', {}))
            result_path = directory / 'results' / (panel_id + '.json')
            exclusive_json(result_path, result)
            exclusive_json(directory / 'receipts' / (panel_id + '.json'),
                         dict(common, schema='panel-worker-receipt-v1', status='complete',
                              result=file_reference(result_path), item_ids=result['item_ids'],
                              expected_items=plan['expected_items'], observed_items=len(result['item_ids']),
                              scores=scores, noise_digest=result['noise_digest']))
            current = None
    except BaseException as error:
        terminal.update(status='interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed',
                        failure=f'{type(error).__name__}: {error}')
        if current is not None:
            receipt = directory / 'receipts' / (current + '.json')
            if not receipt.exists():
                exclusive_json(receipt, {'schema': 'panel-worker-receipt-v1', 'plan_digest': plan['digest'],
                             'panel_id': current, 'worker_id': worker, 'status': 'interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed',
                             'failure': terminal['failure'], 'item_ids': [], 'expected_items': plan['expected_items'],
                             'observed_items': 0, 'scores': {}, 'noise_digest': None})
    finally:
        exclusive_json(directory / 'worker-terminal.json', terminal)


def run_workers(plan, attempt, assignments, *, runtime_factory, score_panel, timeout_seconds=300, backend='spawn', publish_partial=False):
    if type(publish_partial) is not bool:
        raise ValueError('publish_partial must be boolean')
    if backend != 'spawn':
        raise ValueError('only isolated spawn backend supported')
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError('timeout_seconds must be positive finite')
    validate_parallel_controls({'backend': 'local', 'workers': len(assignments), 'devices': [a['device'] for a in assignments]})
    attempt = Path(attempt).resolve(strict=True)
    _validate_plan(plan)
    if json.loads((attempt / 'plan.json').read_text()) != plan:
        raise ValueError('attempt plan differs from requested plan')
    workers_root = attempt / 'workers'
    workers_root.mkdir(exist_ok=True)
    if workers_root.is_symlink():
        raise ValueError('workers directory cannot be symlink')
    worker_ids = [a['worker_id'] for a in assignments]
    selected = [p for a in assignments for p in a['panel_ids']]
    known = {p['id'] for p in plan['panels']}
    if len(set(worker_ids)) != len(worker_ids) or len(set(selected)) != len(selected) or any(p not in known for p in selected):
        raise ValueError('assignments overlap or reference unknown panels')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', x) for x in worker_ids + selected):
        raise ValueError('unsafe worker or panel ID')
    context = multiprocessing.get_context('spawn')
    processes = []
    states = {}
    try:
        for assignment in assignments:
            directory = workers_root / assignment['worker_id']
            directory.mkdir()  # All output collisions fail before launching any child.
            for name in ('starts', 'results', 'receipts', 'outputs'):
                (directory / name).mkdir()
            processes.append((assignment, directory, context.Process(target=_runner_child,
                             args=(plan, assignment, str(directory), runtime_factory, score_panel))))
        deadline = time.monotonic() + timeout_seconds
        for assignment, directory, process in processes:
            process.start()
        published_inventory = None
        while any(p.is_alive() for _, _, p in processes):
            if publish_partial:
                inventory = tuple(sorted(str(path) for path in workers_root.glob('*/receipts/*.json'))) + tuple(sorted(str(path) for path in workers_root.glob('*/worker-terminal.json')))
                if inventory != published_inventory:
                    collect_results(attempt, plan, worker_states=states, final=False)
                    published_inventory = inventory
            if time.monotonic() >= deadline:
                for assignment, _, process in processes:
                    if process.is_alive():
                        states[assignment['worker_id']] = {'status': 'timed_out', 'pid': process.pid}
                break
            time.sleep(min(0.02, max(0, deadline - time.monotonic())))
    except BaseException:
        for assignment, _, process in processes:
            states[assignment['worker_id']] = {'status': 'interrupted', 'pid': process.pid}
        raise
    finally:
        for assignment, directory, process in processes:
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=1)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1)
            worker_id = assignment['worker_id']
            terminal_path = directory / 'worker-terminal.json'
            if worker_id in states:
                if not terminal_path.exists():
                    exclusive_json(terminal_path, dict(states[worker_id], worker_id=worker_id))
                else:
                    event = states[worker_id]
                    states[worker_id] = json.loads(terminal_path.read_text())
                    if event['status'] != states[worker_id]['status']:
                        states[worker_id]['parent_event'] = event
            elif terminal_path.exists():
                states[worker_id] = json.loads(terminal_path.read_text())
            else:
                states[worker_id] = {'status': 'failed', 'pid': process.pid, 'exitcode': process.exitcode}
                exclusive_json(terminal_path, dict(states[worker_id], worker_id=worker_id))
    if publish_partial:
        collect_results(attempt, plan, worker_states=states, final=False)
    return states
