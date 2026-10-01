"""CPU subprocess controller lifecycle and fail-closed binding checks."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import pytest
import torch
import yaml

from jscc.activation_replay import canonical_digest, file_digest
from jscc.baseline_protocol import batch_identity
from jscc.config import load_config
from jscc import sharing_controller as controller
from jscc.sharing_preparation import load_prepared, source_inventory, state_dict_identity, write_prepared
from sharing_fixtures import sharing_batches, sharing_model
from test_shared_lifecycle import geometry_batches, task_template

SOURCE = Path(__file__).resolve().parents[1]


def ref(path):
    return {'path': str(path.resolve()), 'sha256': file_digest(path)}


@pytest.fixture
def controlled(tmp_path):
    root = tmp_path / 'package'
    root.mkdir()
    model = sharing_model()
    config = load_config(SOURCE / 'configs/tasks/hellaswag.yaml')
    config['model'].update(device='cpu', dtype='float32', revision='tiny-cpu', numerical_policy='native')
    config['codec'] = dict(model.codec.config)
    config['split'] = dict(model.split)
    config['training'].update(batch_size=2, gradient_accumulation=1, lr=2e-4, weight_decay=.01,
                              grad_clip=1, temperature=1, kl_weight=1, mse_weight=.1)
    config['seed'] = 0
    (root / 'config.yaml').write_text(yaml.safe_dump(config))
    config = load_config(root / 'config.yaml')
    def relative(name):
        return {'path': name, 'sha256': file_digest(root / name)}
    def batch(name, value):
        torch.save(value, root / name)
        return {**relative(name), 'view_sha256': batch_identity(value)}
    updates = [batch(f'update{i}.pt', value) for i, value in enumerate(sharing_batches())]
    validation = [batch('validation.pt', sharing_batches()[0])]
    geometry = {name: [batch(f'geometry-{name}.pt', values[0])]
                for name, values in geometry_batches().items()}
    template = task_template()
    template['backend'] = model.base.config.decoder._attn_implementation
    (root / 'task.json').write_text(json.dumps(template))
    torch.save(model.codec.state_dict(), root / 'initial.pt')
    inventory = source_inventory()
    with zipfile.ZipFile(root / 'source.zip', 'w') as bundle:
        for name in inventory:
            bundle.write(SOURCE / name, name)
    prepared = write_prepared(root / 'prepared.json', dict(
        schema='sharing-prepared-v1', cell='D-LN', synthetic_cpu=True, q=4, batch_size=2,
        selected_bottleneck_dim=8, protocol_id='sharing-synthetic-test-v1', model_revision='tiny-cpu',
        study_pairing_id='cpu-controller', recipe_identity='fixed-K+.1R', architecture_decision='D-LN-fixture',
        hard_cap_seconds=120, config=relative('config.yaml'), config_identity=canonical_digest(config),
        updates=updates, validation=validation, geometry=geometry, task_request=relative('task.json'),
        source_inventory=inventory, source_identity=canonical_digest(inventory), source_archive=relative('source.zip'),
        initialization={**relative('initial.pt'), 'state_identity': state_dict_identity(model.codec.state_dict())},
        data_ids={'synthetic': True}))
    allocation = tmp_path / 'allocation'
    allocation.mkdir()
    contract = dict(schema='sharing-production-controller-v1', prepared=ref(prepared),
                    binding=controller.binding_identity(load_prepared(prepared)),
                    run_id='cpu-controller-test', hard_cap_seconds=120, output_byte_cap=100_000_000,
                    automatic_retry=False, allocation_root=str(allocation))
    path = allocation / 'contract.json'
    path.write_text(json.dumps(contract))
    return path, contract


def test_actual_cpu_controller_subprocess_lifecycle(controlled, tmp_path):
    path, _ = controlled
    # Subprocess-only tiny model/scorer adapters; the package, CLI, controller,
    # optimizer, checkpoint reload, freeze, heldout and report code are actual.
    hooks = tmp_path / 'hooks'
    hooks.mkdir()
    (hooks / 'sitecustomize.py').write_text(
        'import torch, pytest\n'
        'from jscc import sharing_run\n'
        'from sharing_fixtures import sharing_model\n'
        'from test_shared_lifecycle import install_adapter\n'
        'torch.set_num_threads(1)\n'
        'sharing_run.build_model = lambda config: (object(), sharing_model())\n'
        'patch = pytest.MonkeyPatch()\n'
        'install_adapter(patch)\n')
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(hooks), str(SOURCE), str(SOURCE / 'tests')])
    result = subprocess.run([sys.executable, str(SOURCE / 'scripts/sharing_production.py'),
                             '--controller', str(path)], env=env, capture_output=True, text=True, timeout=140)
    log = (path.parent / 'worker.log').read_text() if (path.parent / 'worker.log').exists() else ''
    assert result.returncode == 0, result.stderr + log
    campaign = json.loads((path.parent / 'outputs/campaign.json').read_text())
    assert (campaign['status'], campaign['completed_updates'], campaign['completed_checkpoints']) == ('complete', 24, 17)
    observations = json.loads((path.parent / 'outputs/observations.json').read_text())
    assert sum(len(o['conditions']) for o in observations if o['request']['purpose'] == 'task') == 108
    assert sum(len(o['conditions']) for o in observations if o['request']['purpose'] == 'objective') == 192
    receipt = json.loads((path.parent / 'allocation.json').read_text())
    assert receipt['charged_device_seconds'] == 0 and receipt['ledger_mutated'] is False
    retry = subprocess.run([sys.executable, str(SOURCE / 'scripts/sharing_production.py'),
                            '--controller', str(path)], env=env, capture_output=True, timeout=30)
    assert retry.returncode != 0 and b'FileExistsError' in retry.stderr


@pytest.mark.parametrize('field,value', [('cell', 'R-LN'), ('bottleneck_dim', 512),
    ('source_identity', 'wrong'), ('partition', {'microbatch_size': 16}),
    ('initialization_identity', 'wrong'), ('model_identity', 'wrong'), ('input_identity', 'wrong')])
def test_exact_binding_rejects_mutation(controlled, field, value):
    path, contract = controlled
    contract['binding'][field] = value
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match='exact binding'):
        controller.validate_contract(path)


def test_no_direct_production_worker_authorization(monkeypatch):
    monkeypatch.delenv('SHARING_CONTROLLER_CONTRACT', raising=False)
    with pytest.raises(ValueError, match='production execution disabled'):
        controller.require_worker_authorization({'synthetic_cpu': False})


def test_qualification_and_owner_receipt_binding(controlled, monkeypatch):
    from jscc import sharing_preparation
    path, contract = controlled
    manifest = load_prepared(contract['prepared']['path'])
    manifest['synthetic_cpu'] = False
    contract['finish_before_epoch'] = int(time.time()) + 600
    monkeypatch.setattr(sharing_preparation, 'load_prepared', lambda path: manifest)
    qualification: dict = dict(schema='sharing-production-qualification-v1', status='qualified_for_production',
                         scope='full_sharing_lifecycle', full_lifecycle_ready=True, binding=contract['binding'])
    measurement_path = path.parent / 'measurement.json'
    evidence_path = path.parent / 'measured-evidence.json'
    evidence_path.write_text(json.dumps({'synthetic_test_fixture': True}))
    measurement = dict(schema='sharing-production-measurement-v1', status='qualified_scope_fit',
                       binding=contract['binding'], estimated_full_lifecycle_seconds=90, reserve_seconds=20,
                       measured_evidence=[ref(evidence_path)])
    measurement_path.write_text(json.dumps(measurement))
    qualification['measurement'] = ref(measurement_path)
    qualification_path = path.parent / 'qualification.json'
    qualification_path.write_text(json.dumps(qualification))
    contract['qualification'] = ref(qualification_path)
    proposal = dict(schema='sharing-production-proposal-v1', binding=contract['binding'],
                    qualification_sha256=contract['qualification']['sha256'],
                    run_id=contract['run_id'], hard_cap_seconds=120, automatic_retry=False,
                    finish_before_epoch=contract['finish_before_epoch'])
    proposal_path = path.parent / 'proposal.json'
    proposal_path.write_text(json.dumps(proposal))
    contract['proposal'] = ref(proposal_path)
    ledger_path = path.parent / 'ledger.json'
    reservation = dict(run_id=contract['run_id'], status='reserved_once_not_launched',
                       reserved_device_seconds=120, cap_device_seconds=120, account='sharing',
                       baseline_and_COCO_unchanged=True, ledger=str(ledger_path))
    hold = dict(run_id=contract['run_id'], ledger_owner='task-7',
                state='approved_reserved_not_launched_pending_package_review', max_device_seconds=120,
                devices=1, automatic_retry=False, proposal_sha256=contract['proposal']['sha256'])
    reservation_path = path.parent / 'reservation.json'
    reservation_path.write_text(json.dumps(reservation))
    ledger = dict(account='sharing', ledger_owner='task-7', pending_reservations=[hold],
                  allocations=[], charged_device_seconds=0, cap_device_seconds=120)
    ledger_path.write_text(json.dumps(ledger))
    contract.update(reservation=ref(reservation_path), ledger_path=str(ledger_path),
                    sole_owner='task-7', device_lease=str(path.parent / 'device.lease'))
    path.write_text(json.dumps(contract))
    controller.validate_contract(path)
    # Overnight admission must not bypass the existing production proposal gate.
    from jscc import sharing_admission
    with monkeypatch.context() as patch:
        patch.setattr(sharing_admission, 'validate_overnight_admission',
                      lambda *args, **kwargs: pytest.fail('helper reached before proposal validation'))
        ledger['schema'] = 'bounded-overnight-campaign-ledger-v1'
        ledger_path.write_text(json.dumps(ledger))
        for field, value in [('qualification_sha256', 'wrong'), ('finish_before_epoch', 1)]:
            changed = {**proposal, field: value}
            proposal_path.write_text(json.dumps(changed))
            contract['proposal'] = ref(proposal_path)
            path.write_text(json.dumps(contract))
            with pytest.raises(ValueError, match='production proposal exact binding'):
                controller.validate_contract(path)
        ledger.pop('schema')
        ledger_path.write_text(json.dumps(ledger))
        proposal_path.write_text(json.dumps(proposal))
        contract['proposal'] = ref(proposal_path)
        path.write_text(json.dumps(contract))
    contract['sole_owner'] = ledger['ledger_owner'] = hold['ledger_owner'] = 'wrong-owner'
    path.write_text(json.dumps(contract))
    ledger_path.write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match='must be task-7'):
        controller.validate_contract(path)
    contract['sole_owner'] = ledger['ledger_owner'] = hold['ledger_owner'] = 'task-7'
    path.write_text(json.dumps(contract))
    ledger_path.write_text(json.dumps(ledger))
    original = copy.deepcopy(qualification)
    for key, value in [('scope', 'diagnostic'), ('status', 'accepted'), ('binding', {})]:
        changed = {**original, key: value}
        qualification_path.write_text(json.dumps(changed))
        contract['qualification'] = ref(qualification_path)
        path.write_text(json.dumps(contract))
        with pytest.raises(ValueError, match='qualification'):
            controller.validate_contract(path)
    qualification_path.write_text(json.dumps(original))
    contract['qualification'] = ref(qualification_path)
    path.write_text(json.dumps(contract))
    ledger['charged_device_seconds'] = 1
    ledger_path.write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match='budget'):
        controller.validate_contract(path)


def test_output_limit_includes_fast_exit(controlled, monkeypatch):
    path, contract = controlled
    contract['output_byte_cap'] = 1
    path.write_text(json.dumps(contract))
    assert controller.run_controller(path) != 0
    receipt = json.loads((path.parent / 'allocation.json').read_text())
    assert receipt['stop_reason'] == 'output_byte_cap'


def test_late_allocation_rejected_before_claim(controlled):
    path, contract = controlled
    contract['finish_before_epoch'] = int(time.time()) + 60
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match='cannot finish before absolute boundary'):
        controller.run_controller(path)
    assert not (path.parent / 'attempt.json').exists()
    assert not (path.parent / 'outputs').exists()


def test_parallel_host_idle_query_scopes_assigned_device():
    device = 'GPU-324fae60-a5b1-ebc1-146f-c5a94d90dbb7'
    assert controller.idle_device_query({'device_uuid': device}) == [
        'nvidia-smi', '-i', device, '--query-compute-apps=pid', '--format=csv,noheader,nounits']
    with pytest.raises(ValueError, match='UUID'):
        controller.idle_device_query({'device_uuid': 'all'})


@pytest.mark.parametrize('visible', [0, 2])
def test_target_device_rejects_non_single_cuda_visibility(tmp_path, monkeypatch, visible):
    path = tmp_path / 'contract.json'
    path.write_text(json.dumps({'device_uuid': 'GPU-324fae60-a5b1-ebc1-146f-c5a94d90dbb7'}))
    monkeypatch.setenv('SHARING_CONTROLLER_CONTRACT', str(path))
    monkeypatch.setenv('SHARING_CONTROLLER_CONTRACT_SHA256', file_digest(path))
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: visible)
    with pytest.raises(ValueError, match='one actual scheduler-visible'):
        controller.check_target_device({'synthetic_cpu': False, 'resolved_config': {'model': {'device': 'cuda', 'dtype': 'bfloat16'}}})


def test_target_device_verifies_actual_uuid_and_bf16(tmp_path, monkeypatch):
    from types import SimpleNamespace
    expected = 'GPU-324fae60-a5b1-ebc1-146f-c5a94d90dbb7'
    path = tmp_path / 'contract.json'
    path.write_text(json.dumps({'device_uuid': expected}))
    monkeypatch.setenv('SHARING_CONTROLLER_CONTRACT', str(path))
    monkeypatch.setenv('SHARING_CONTROLLER_CONTRACT_SHA256', file_digest(path))
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    properties = SimpleNamespace(uuid='GPU-00000000-0000-0000-0000-000000000000', name='NVIDIA H200')
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda index: properties)
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda: True)
    manifest = {'synthetic_cpu': False, 'resolved_config': {'model': {'device': 'cuda', 'dtype': 'bfloat16'}}}
    with pytest.raises(ValueError, match='visible CUDA UUID'):
        controller.check_target_device(manifest)
    properties.uuid = expected.removeprefix('GPU-')
    controller.check_target_device(manifest)
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda: False)
    with pytest.raises(ValueError, match='BF16'):
        controller.check_target_device(manifest)
    assert not torch.cuda.is_initialized()


def test_production_requires_external_watchdog(controlled, monkeypatch):
    path, contract = controlled
    manifest = {'synthetic_cpu': False}
    monkeypatch.setattr(controller, 'validate_contract', lambda path: (contract, manifest))
    with pytest.raises(ValueError, match='external group watchdog'):
        controller.run_controller(path)
    assert not (path.parent / 'attempt.json').exists()


def test_zero_exit_without_campaign_is_not_success(controlled, monkeypatch):
    path, _ = controlled
    popen = subprocess.Popen
    monkeypatch.setattr(controller.subprocess, 'Popen',
                        lambda argv, **kwargs: popen([sys.executable, '-c', 'pass'], **kwargs))
    with pytest.raises(FileNotFoundError):
        controller.run_controller(path)
    receipt = json.loads((path.parent / 'allocation.json').read_text())
    assert receipt['status'] == 'controller_failed'


def test_deadline_stops_worker_and_disallows_retry(controlled, monkeypatch):
    path, contract = controlled
    contract['hard_cap_seconds'] = 20
    path.write_text(json.dumps(contract))
    popen = subprocess.Popen
    monkeypatch.setattr(controller.subprocess, 'Popen',
                        lambda argv, **kwargs: popen([sys.executable, '-c', 'import time; time.sleep(60)'], **kwargs))
    assert controller.run_controller(path) != 0
    receipt = json.loads((path.parent / 'allocation.json').read_text())
    assert receipt['stop_reason'] == 'signal_deadline_or_output_cap'
    assert receipt['elapsed_seconds'] < 15
    with pytest.raises(FileExistsError):
        controller.run_controller(path)


@pytest.mark.parametrize('failure', ['group', 'child_wait'])
def test_cleanup_failure_retains_accounting_and_releases_resources(controlled, monkeypatch, failure):
    """Exercise production cleanup branches with fake processes and no GPU calls."""
    import fcntl
    import math
    import signal

    path, contract = controlled
    lease_path = path.parent / 'device.lease'
    lease_path.touch()
    contract['device_lease'] = str(lease_path)
    manifest = {'synthetic_cpu': False}
    monkeypatch.setattr(controller, 'validate_contract', lambda path: (contract, manifest))
    monkeypatch.setattr(controller, '_watchdog', lambda contract: None)
    monkeypatch.setattr(controller.subprocess, 'check_output', lambda *args, **kwargs: b'')
    calls = []

    class Child:
        returncode = 1

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            if failure == 'child_wait':
                raise RuntimeError('injected child wait failure')
            return self.returncode

    def cleanup_group():
        calls.append('group_cleanup')
        if failure == 'group':
            raise RuntimeError('injected group cleanup failure')

    monkeypatch.setattr(controller.subprocess, 'Popen', lambda *args, **kwargs: Child())
    monkeypatch.setattr(controller, '_cleanup_group', cleanup_group)
    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    with pytest.raises(RuntimeError, match='allocation cleanup failed'):
        controller.run_controller(path)
    report = json.loads((path.parent / 'allocation.json').read_text())
    assert report['status'] == 'cleanup_failed' and report['cleanup_verified'] is False
    assert report['elapsed_seconds'] > 0
    assert report['charged_device_seconds'] == math.ceil(report['elapsed_seconds'])
    assert 'injected' in report['cleanup_error']
    assert calls == ['group_cleanup']
    assert all(signal.getsignal(signum) == handler for signum, handler in previous.items())
    with lease_path.open('r+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with pytest.raises(FileExistsError):
        controller.run_controller(path)
