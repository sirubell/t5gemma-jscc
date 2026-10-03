"""One owner-reserved, externally bounded sharing execution; never edit ledgers."""
from __future__ import annotations

import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from .activation_replay import canonical_digest, file_digest


def binding_identity(manifest):
    """Qualification and reservation must match these exact scientific inputs."""
    return {key: manifest[key] for key in (
        'prepared_identity', 'cell', 'source_identity', 'config_identity',
        'model_revision', 'recipe_identity', 'architecture_decision', 'protocol_id'
    )} | {
        'bottleneck_dim': manifest['resolved_config']['codec']['bottleneck_dim'],
        'partition': manifest.get('execution_partition'),
        'input_identity': canonical_digest({key: manifest[key] for key in (
            'updates', 'validation', 'geometry', 'task_request', 'data_ids')}),
        'initialization_identity': manifest['initialization']['state_identity'],
        'model_identity': canonical_digest(manifest['resolved_config']['model']),
    }


def _read_ref(reference):
    path = Path(reference['path'])
    if not path.is_absolute() or file_digest(path) != reference['sha256']:
        raise ValueError('absolute immutable controller reference required')
    return json.loads(path.read_text())


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f'positive integer {name} required')
    return value


def validate_contract(path, manifest=None):
    from .sharing_preparation import load_prepared
    path = Path(path).resolve()
    contract = json.loads(path.read_text())
    if contract.get('schema') != 'sharing-production-controller-v1':
        raise ValueError('production controller contract required')
    _read_ref(contract['prepared'])
    loaded = load_prepared(contract['prepared']['path'])
    if manifest is not None and binding_identity(manifest) != binding_identity(loaded):
        raise ValueError('worker preparation differs from controller')
    manifest = loaded
    if contract.get('binding') != binding_identity(manifest):
        raise ValueError('controller exact binding mismatch')
    cap = _positive(contract['hard_cap_seconds'], 'hard cap')
    if cap < 20 or cap > manifest['hard_cap_seconds']:
        raise ValueError('controller hard cap outside prepared bound')
    _positive(contract['output_byte_cap'], 'output byte cap')
    run_id = contract.get('run_id')
    if not isinstance(run_id, str) or not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in run_id):
        raise ValueError('safe unique run ID required')
    root = Path(contract['allocation_root'])
    if not root.is_absolute() or root.resolve() != path.parent:
        raise ValueError('allocation root must be contract directory')
    if contract.get('automatic_retry') is not False:
        raise ValueError('automatic retry forbidden')
    if contract.get('entry_route') not in (None, 'h200-target-startup-v1', '5090-q400-target-startup-v1'):
        raise ValueError('unknown production entry route')
    if manifest['synthetic_cpu'] and contract.get('entry_route') in ('h200-target-startup-v1', '5090-q400-target-startup-v1'):
        from .sharing_startup import validate_request
        validate_request(contract, manifest)
    if not manifest['synthetic_cpu']:
        _positive(contract.get('finish_before_epoch'), 'absolute finish boundary')
        if contract.get('entry_route') in ('h200-target-startup-v1', '5090-q400-target-startup-v1'):
            from .sharing_startup import validate_request
            validate_request(contract, manifest)
        else:
            qualification = _read_ref(contract['qualification'])
            if (qualification.get('schema') != 'sharing-production-qualification-v1'
                    or qualification.get('status') != 'qualified_for_production'
                    or qualification.get('scope') != 'full_sharing_lifecycle'
                    or qualification.get('full_lifecycle_ready') is not True
                    or qualification.get('binding') != contract['binding']):
                raise ValueError('exact production qualification required; diagnostics cannot qualify')
            measurement = _read_ref(qualification['measurement'])
            if (measurement.get('schema') != 'sharing-production-measurement-v1'
                    or measurement.get('status') != 'qualified_scope_fit'
                    or measurement.get('binding') != contract['binding']):
                raise ValueError('exact production measurement evidence required')
            evidence = measurement.get('measured_evidence')
            if not isinstance(evidence, list) or not evidence:
                raise ValueError('nonempty immutable measured evidence required')
            for reference in evidence:
                _read_ref(reference)
            measured = measurement.get('estimated_full_lifecycle_seconds')
            reserve = measurement.get('reserve_seconds')
            if (any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                    for value in (measured, reserve)) or measured + reserve > cap):
                raise ValueError('estimated full lifecycle plus reserve exceeds allocation cap')
        ledger = json.loads(Path(contract['ledger_path']).read_text())
        owner = contract.get('sole_owner')
        if owner != 'task-7':
            raise ValueError('sole reservation owner must be task-7')
        if not Path(contract['device_lease']).is_absolute():
            raise ValueError('absolute existing device lease required')
        idle_device_query(contract)
        proposal = _read_ref(contract['proposal'])
        if contract.get('entry_route') in ('h200-target-startup-v1', '5090-q400-target-startup-v1'):
            if ledger.get('schema') != 'bounded-overnight-campaign-ledger-v1':
                raise ValueError('startup route requires exact overnight owner ledger')
            from .sharing_startup import validate_owner_route
            validate_owner_route(contract, proposal)
        elif (proposal.get('schema') != 'sharing-production-proposal-v1'
                or proposal.get('binding') != contract['binding']
                or proposal.get('qualification_sha256') != contract['qualification']['sha256']
                or proposal.get('run_id') != run_id
                or proposal.get('hard_cap_seconds') != cap
                or proposal.get('finish_before_epoch') != contract['finish_before_epoch']
                or proposal.get('automatic_retry') is not False):
            raise ValueError('production proposal exact binding required')
        if ledger.get('schema') == 'bounded-overnight-campaign-ledger-v1':
            from .sharing_admission import validate_overnight_admission
            validate_overnight_admission(contract, diagnostic_only=False)
            return contract, manifest
        reservation = _read_ref(contract['reservation'])
        holds = ledger.get('pending_reservations', [])
        matching = [row for row in holds if row.get('run_id') == run_id]
        if len(matching) != 1:
            raise ValueError('one existing owner reservation required')
        hold = matching[0]
        if (reservation.get('status') != 'reserved_once_not_launched'
                or reservation.get('run_id') != run_id or reservation.get('account') != 'sharing'
                or reservation.get('reserved_device_seconds') != cap
                or reservation.get('cap_device_seconds') != ledger.get('cap_device_seconds')
                or reservation.get('baseline_and_COCO_unchanged') is not True
                or reservation.get('ledger') != contract.get('ledger_origin_path', contract['ledger_path'])
                or ledger.get('account') != 'sharing' or ledger.get('ledger_owner') != owner
                or hold.get('ledger_owner') != owner
                or hold.get('state') != 'approved_reserved_not_launched_pending_package_review'
                or hold.get('max_device_seconds') != cap or type(hold.get('devices')) is not int
                or hold.get('devices') != 1 or hold.get('automatic_retry') is not False
                or hold.get('proposal_sha256') != contract['proposal']['sha256']
                or any(row.get('run_id') == run_id for row in ledger.get('allocations', []))):
            raise ValueError('active exact sole-owner reservation required; no retry')
        charged = ledger.get('charged_device_seconds')
        aggregate = _positive(ledger.get('cap_device_seconds'), 'account cap')
        if type(charged) is not int or charged < 0 or charged + sum(
                _positive(row.get('max_device_seconds'), 'reservation cap') for row in holds) > aggregate:
            raise ValueError('aggregate sharing budget exceeded')
    return contract, manifest


def idle_device_query(contract):
    """Scope H200 parallel-job admission to the verified assigned GPU."""
    argv = ['nvidia-smi']
    if contract.get('device_uuid') is not None:
        device = contract['device_uuid']
        if not isinstance(device, str) or not device.startswith('GPU-'):
            raise ValueError('exact assigned device UUID required')
        uuid.UUID(device[4:])
        argv.extend(['-i', device])
    return [*argv, '--query-compute-apps=pid', '--format=csv,noheader,nounits']


def external_command(path):
    contract, _ = validate_contract(path)
    entry = Path(__file__).resolve().parents[1] / 'scripts/sharing_production.py'
    return ['/usr/bin/timeout', '--signal=TERM', '--kill-after=2s',
            str(contract['hard_cap_seconds'] - 2), sys.executable, '-B', str(entry),
            '--controller', str(Path(path).resolve())]


def _watchdog(contract):
    parent = os.getppid()
    cmdline = Path(f'/proc/{parent}/cmdline')
    expected = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=2s',
                str(contract['hard_cap_seconds'] - 2)]
    if (not cmdline.is_file() or cmdline.read_bytes().decode().split('\0')[:4] != expected
            or os.getpgrp() != parent or os.getpgid(parent) != parent):
        raise ValueError('exact external group watchdog required')


def output_bytes(root):
    total = 0
    for path in Path(root).rglob('*'):
        if path.is_symlink():
            raise ValueError('output symlinks forbidden')
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            pass
    return total


def require_worker_authorization(manifest):
    path = os.environ.get('SHARING_CONTROLLER_CONTRACT')
    if not path:
        raise ValueError('production execution disabled without controller authorization')
    if file_digest(Path(path)) != os.environ.get('SHARING_CONTROLLER_CONTRACT_SHA256'):
        raise ValueError('controller contract bytes changed')
    contract, _ = validate_contract(path, manifest)
    if str(os.getppid()) != os.environ.get('SHARING_CONTROLLER_PID'):
        raise ValueError('worker must remain direct controller child')
    claim = json.loads((Path(contract['allocation_root']) / 'attempt.json').read_text())
    if (claim.get('controller_pid') != os.getppid() or claim.get('binding') != contract['binding']
            or claim.get('contract_sha256') != file_digest(Path(path))):
        raise ValueError('live controller claim mismatch')
    if not manifest['synthetic_cpu']:
        parent = Path(f'/proc/{os.getppid()}/stat').read_text().split(') ', 1)[1].split()
        watchdog = int(parent[1])
        argv = Path(f'/proc/{watchdog}/cmdline').read_bytes().decode().split('\0')
        expected = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=2s',
                    str(contract['hard_cap_seconds'] - 2)]
        if argv[:4] != expected or os.getpgrp() != watchdog:
            raise ValueError('worker external watchdog mismatch')
    check_worker_resources(manifest)
    return contract


def check_worker_resources(manifest):
    if manifest['synthetic_cpu'] and 'SHARING_CONTROLLER_CONTRACT' not in os.environ:
        return
    pid = int(os.environ['SHARING_CONTROLLER_PID'])
    if os.getppid() != pid:
        raise RuntimeError('controller no longer owns worker')
    os.kill(pid, 0)
    if time.monotonic() >= float(os.environ['SHARING_WORK_DEADLINE_MONOTONIC']):
        raise TimeoutError('sharing controller deadline')
    if time.time() >= float(os.environ.get('SHARING_FINISH_BEFORE_EPOCH', 'inf')):
        raise TimeoutError('sharing absolute finish boundary')
    if output_bytes(os.environ['SHARING_ALLOCATION_ROOT']) > int(os.environ['SHARING_OUTPUT_BYTE_CAP']):
        raise RuntimeError('sharing output byte cap exceeded')


def check_target_device(manifest):
    """Verify actual scheduler-visible CUDA UUID before loading model weights."""
    if manifest['synthetic_cpu']:
        return
    path = os.environ.get('SHARING_CONTROLLER_CONTRACT')
    if not path or file_digest(Path(path)) != os.environ.get('SHARING_CONTROLLER_CONTRACT_SHA256'):
        raise ValueError('immutable controller required before target device check')
    contract = json.loads(Path(path).read_text())
    expected = contract.get('device_uuid')
    if expected is None:
        return  # Retained single-device legacy contracts have separate target acceptance.
    model_config = manifest['resolved_config']['model']
    if model_config['device'] != 'cuda' or model_config['dtype'] != 'bfloat16':
        raise ValueError('CUDA BF16 model configuration required')
    import torch
    if torch.cuda.device_count() != 1:
        raise ValueError('one actual scheduler-visible CUDA device required')
    properties = torch.cuda.get_device_properties(0)
    actual = str(getattr(properties, 'uuid', ''))
    if actual and not actual.startswith('GPU-'):
        actual = 'GPU-' + actual
    if actual.lower() != expected.lower():
        raise ValueError('visible CUDA UUID differs from sealed assigned device')
    if not torch.cuda.is_bf16_supported():
        raise ValueError('actual CUDA BF16 target required')
    if contract.get('entry_route') == '5090-q400-target-startup-v1':
        if str(properties.name) != 'NVIDIA GeForce RTX 5090':
            raise ValueError('q400 startup requires actual RTX 5090')
    if contract.get('device_binding_receipt') is not None:
        receipt = _read_ref(contract['device_binding_receipt'])
        if receipt.get('device_uuid') != expected or not str(properties.name).startswith('NVIDIA H200'):
            raise ValueError('sealed H200 allocation differs from actual CUDA device')


def _put(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _cleanup_group():
    """Finish descendants before releasing lease, as in reviewed controller."""
    ignored = {os.getpid(), os.getppid()}
    deadline = time.monotonic() + 1
    while True:
        probe = subprocess.Popen(['ps', '-eo', 'pid=,pgid=,stat='], stdout=subprocess.PIPE, text=True)
        try:
            rows, _ = probe.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            probe.kill()
            probe.wait(timeout=1)
            raise RuntimeError('descendant inspection timed out') from None
        if probe.returncode:
            raise RuntimeError('descendant inspection failed')
        live = [int(pid) for line in rows.splitlines() for pid, group, state in [line.split()]
                if int(group) == os.getpgrp() and int(pid) not in ignored | {probe.pid}
                and not state.startswith('Z')]
        if not live:
            return
        for pid in live:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if time.monotonic() >= deadline:
            raise RuntimeError('allocation descendants remain alive')
        time.sleep(.02)


def run_controller(path):
    started = time.monotonic()
    contract, manifest = validate_contract(path)
    if contract.get('entry_route') in ('h200-target-startup-v1', '5090-q400-target-startup-v1'):
        from .sharing_startup import require_preflight
        require_preflight(path, contract, manifest)
    if not manifest['synthetic_cpu']:
        _watchdog(contract)
    if time.time() + contract['hard_cap_seconds'] > contract.get('finish_before_epoch', float('inf')):
        raise ValueError('allocation cannot finish before absolute boundary')
    root = Path(contract['allocation_root'])
    lease = None
    claim_root = root if manifest['synthetic_cpu'] else Path(contract['device_lease']).parent
    claim_path = claim_root / (contract['run_id'] + '.attempt.json')
    claim = {'binding': contract['binding'], 'controller_pid': os.getpid(),
             'contract_sha256': file_digest(Path(path)), 'automatic_retry': False}
    with claim_path.open('x') as stream:
        json.dump(claim, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(claim_path, root / 'attempt.json')
    report = {'run_id': contract['run_id'], 'status': 'starting', 'automatic_retry': False,
              'synthetic_cpu': manifest['synthetic_cpu'], 'binding': contract['binding'],
              'settlement_owner': contract.get('sole_owner'), 'ledger_mutated': False}
    child = None
    stopped = False
    previous = {}

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, stop)
    try:
        if not manifest['synthetic_cpu']:
            lease = Path(contract['device_lease']).open('r+')
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if subprocess.check_output(idle_device_query(contract), timeout=3).strip():
                raise RuntimeError('device busy; no work launched')
        if (root / 'outputs').exists():
            raise ValueError('fresh output directory required')
        env = os.environ.copy()
        env.update(SHARING_CONTROLLER_CONTRACT=str(Path(path).resolve()),
                   SHARING_CONTROLLER_CONTRACT_SHA256=file_digest(Path(path)),
                   SHARING_CONTROLLER_PID=str(os.getpid()), SHARING_ALLOCATION_ROOT=str(root),
                   SHARING_OUTPUT_BYTE_CAP=str(contract['output_byte_cap']),
                   SHARING_FINISH_BEFORE_EPOCH=str(contract.get('finish_before_epoch', float('inf'))),
                   SHARING_WORK_DEADLINE_MONOTONIC=str(started + contract['hard_cap_seconds'] - 10),
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1')
        if manifest['synthetic_cpu']:
            env['CUDA_VISIBLE_DEVICES'] = ''
        entry = Path(__file__).resolve().parents[1] / 'scripts/shared_codec.py'
        argv = [sys.executable, '-B', str(entry), '--execute', contract['prepared']['path'],
                '--output', str(root / 'outputs')]
        with (root / 'worker.log').open('x') as log:
            child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env,
                                     stdin=subprocess.DEVNULL, start_new_session=manifest['synthetic_cpu'])
            while child.poll() is None:
                if (stopped or time.monotonic() - started >= contract['hard_cap_seconds'] - 10
                        or time.time() >= contract.get('finish_before_epoch', float('inf'))
                        or output_bytes(root) > contract['output_byte_cap']):
                    report['stop_reason'] = 'signal_deadline_or_output_cap'
                    child.terminate()
                    try:
                        child.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=2)
                    break
                time.sleep(.05)
        code = child.returncode
        if output_bytes(root) > contract['output_byte_cap']:
            report['stop_reason'] = 'output_byte_cap'
        if report.get('stop_reason') and code == 0:
            code = 124
        if code == 0:
            campaign = json.loads((root / 'outputs/campaign.json').read_text())
            if (campaign.get('status') != 'complete'
                    or campaign.get('completed_updates') != 6 * manifest['q']
                    or campaign.get('completed_checkpoints') != 17
                    or set(campaign.get('completed_obligations', [])) != set(campaign.get('obligations', []))):
                raise ValueError('worker exited without complete sharing lifecycle')
        report.update(status='terminal', exit_code=code)
        return code
    except BaseException as error:
        report.update(status='controller_failed', error=str(error))
        raise
    finally:
        cleanup_errors = []
        try:
            try:
                if child is not None:
                    if manifest['synthetic_cpu']:
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    elif child.poll() is None:
                        child.kill()
                    child.wait(timeout=2)
            except BaseException as error:
                cleanup_errors.append(error)
            # Group cleanup must still run if waiting for the direct child failed.
            try:
                if not manifest['synthetic_cpu']:
                    _cleanup_group()
                    report['descendants_live'] = []
                report['cleanup_verified'] = not cleanup_errors
            except BaseException as error:
                cleanup_errors.append(error)
            if cleanup_errors:
                report.update(status='cleanup_failed', cleanup_verified=False,
                              cleanup_error='; '.join(str(error) for error in cleanup_errors))
        finally:
            elapsed = time.monotonic() - started
            report.update(elapsed_seconds=elapsed,
                          charged_device_seconds=0 if manifest['synthetic_cpu'] else math.ceil(elapsed))
            try:
                _put(root / 'allocation.json', report)
            finally:
                try:
                    if lease is not None:
                        lease.close()
                finally:
                    for signum, handler in previous.items():
                        signal.signal(signum, handler)
        if cleanup_errors:
            raise RuntimeError('allocation cleanup failed; inspect durable receipt') from cleanup_errors[0]
