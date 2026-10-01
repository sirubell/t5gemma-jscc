"""One H200 allocation: target guard, conservative fit gate, then fresh science.

Startup evidence is diagnostic evidence, never a full-lifecycle qualification.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import socket
import time

from .activation_replay import file_digest
from .sharing_controller import _put, _read_ref, binding_identity
from .sharing_qualification import GUARD_SHA256, _assets, run_diagnostic, validate_result
from .sharing_admission import DIAGNOSTIC_WORKLOAD, PRODUCTION_WORKLOAD, CONTINUATION_WORKLOAD
from .sharing_preparation import state_dict_identity

ROUTE = 'h200-target-startup-v1'
Q400_ROUTE = '5090-q400-target-startup-v1'
ROUTES = (ROUTE, Q400_ROUTE)

def workload(contract):
    return CONTINUATION_WORKLOAD if contract.get('entry_route') == Q400_ROUTE else PRODUCTION_WORKLOAD

def target_class(contract):
    return 'RTX5090' if contract.get('entry_route') == Q400_ROUTE else 'H200'

def schema(contract, suffix):
    prefix = 'sharing-5090-q400' if contract.get('entry_route') == Q400_ROUTE else 'sharing-h200'
    return prefix + suffix
UNMEASURED = {'checkpoint_freeze', 'geometry', 'report_collection', 'vanilla'}


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_request(contract, manifest):
    require('qualification' not in contract, 'startup route cannot supply production qualification')
    request = _read_ref(contract['target_startup_request'])
    require(request.get('schema') == schema(contract, '-target-startup-request-v1')
            and request.get('entry_route') == contract.get('entry_route') in ROUTES
            and request.get('run_id') == contract['run_id']
            and request.get('prepared') == contract['prepared']
            and request.get('binding') == binding_identity(manifest) == contract['binding']
            and request.get('initialization_identity') == manifest['initialization']['state_identity']
            and request.get('guard_sha256') == GUARD_SHA256
            and file_digest(Path(__file__).with_name('sharing_qualification_guard.py')) == GUARD_SHA256
            and request.get('guard_workload') == DIAGNOSTIC_WORKLOAD
            and request.get('production_workload') == workload(contract)
            and request.get('hard_cap_seconds') == contract['hard_cap_seconds']
            and request.get('finish_before_epoch') == contract.get('finish_before_epoch')
            and request.get('automatic_retry') is False,
            'exact reviewed H200 startup request required')
    for key in ('target_assets', 'runtime_versions', 'expected_panel'):
        require(request.get(key) == contract.get(key), 'startup target reference mismatch: ' + key)
    _read_ref(request['cpu_lifecycle_proof'])
    _read_ref(request['expected_panel'])
    policy = request.get('fit_policy', {})
    require(set(policy) == {'safety_multiplier', 'unmeasured_seconds', 'cleanup_reserve_seconds'}
            and type(policy.get('safety_multiplier')) in (int, float)
            and math.isfinite(policy['safety_multiplier']) and policy['safety_multiplier'] >= 1,
            'explicit conservative fit multiplier required')
    allowance = policy['unmeasured_seconds']
    require(isinstance(allowance, dict) and set(allowance) == UNMEASURED
            and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in allowance.values())
            and type(policy['cleanup_reserve_seconds']) in (int, float)
            and math.isfinite(policy['cleanup_reserve_seconds']) and policy['cleanup_reserve_seconds'] >= 10,
            'explicit unmeasured work allowances and cleanup reserve required')
    if contract.get('entry_route') == Q400_ROUTE:
        require(contract['hard_cap_seconds'] <= 14400, 'q400 startup exceeds approved per-run ceiling')
        if not manifest['synthetic_cpu']:
            require(manifest['protocol_id'] == 'sharing-dn-q400-effective64-v1'
                    and manifest['q'] == 400 and contract['binding']['cell'] == 'D-N'
                    and contract['binding']['bottleneck_dim'] == 512, 'exact q400 D-N/B512 startup required')
            require(request.get('device_binding_policy') == {
                'policy': 'single-rtx5090-runtime-uuid-v1', 'gpu_class': 'RTX5090',
                'device_uuid': contract.get('device_uuid'), 'device_lease': contract.get('device_lease')}
                and contract.get('device_binding_receipt') is None,
                'exact q400 owner-bound UUID/shared lease policy required')
    if not manifest['synthetic_cpu']:
        require(request.get('target_class') == target_class(contract)
                and (contract.get('entry_route') == Q400_ROUTE or contract.get('device_binding_receipt') is not None)
                and contract.get('device_uuid') is not None,
                'H200 startup requires sealed assigned device')
        model = manifest['resolved_config']['model']
        require(model.get('device') == 'cuda' and model.get('dtype') == 'bfloat16'
                and model.get('numerical_policy') == 'native'
                and model.get('attn_implementation') == 'sdpa'
                and model.get('local_files_only') is True,
                'H200 startup requires pinned local CUDA/BF16/native/SDPA')
    return request


def validate_owner_route(contract, proposal):
    payload = _read_ref(contract['payload_receipt'])
    request = _read_ref(contract['target_startup_request'])
    require(proposal.get('schema') == 'sharing-production-proposal-v1'
            and 'qualification_sha256' not in proposal
            and all(proposal.get(key) == contract.get(key) for key in
                    ('entry_route', 'target_startup_request', 'binding', 'run_id',
                     'hard_cap_seconds', 'finish_before_epoch', 'automatic_retry'))
            and proposal.get('prepared') == contract['prepared']
            and proposal.get('workload') == workload(contract),
            'startup proposal must bind route/request/full workload')
    require(payload.get('entry_route') == contract.get('entry_route') in ROUTES
            and payload.get('target_startup_request') == contract['target_startup_request']
            and payload.get('device_binding') == request.get('device_binding_policy'),
            'owner payload must approve exact startup route and device policy')


def preflight(path):
    """Actual target CPU entry. Caller hides CUDA before importing torch."""
    from .sharing_controller import validate_contract
    import torch
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'startup preflight must hide CUDA')
    contract, manifest = validate_contract(path)
    require(contract.get('entry_route') in ROUTES, 'startup preflight requires startup route')
    assets = {} if manifest['synthetic_cpu'] else _assets(contract, manifest)
    result = {'schema': schema(contract, '-startup-preflight-v1'), 'status': 'cpu_verified',
              'contract_sha256': file_digest(Path(path)), 'binding': contract['binding'],
              'hostname': socket.gethostname(), 'target_startup_request': contract['target_startup_request'],
              'assets': assets, 'cuda_initialized': torch.cuda.is_initialized(),
              'full_model_constructed': False}
    require(result['cuda_initialized'] is False, 'preflight initialized CUDA')
    output = Path(contract['allocation_root']) / 'target-startup-preflight.json'
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
    return result


def require_preflight(path, contract, manifest):
    receipt = json.loads((Path(contract['allocation_root']) / 'target-startup-preflight.json').read_text())
    require(receipt.get('schema') == schema(contract, '-startup-preflight-v1')
            and receipt.get('status') == 'cpu_verified'
            and receipt.get('contract_sha256') == file_digest(Path(path))
            and receipt.get('binding') == contract['binding']
            and receipt.get('target_startup_request') == contract['target_startup_request']
            and receipt.get('hostname') == socket.gethostname()
            and receipt.get('cuda_initialized') is False
            and receipt.get('full_model_constructed') is False,
            'actual target sealed startup preflight required')
    if not manifest['synthetic_cpu']:
        require(receipt.get('assets') == _assets(contract, manifest), 'startup assets/runtime changed')


def estimate_remaining(result, policy, *, q=200):
    """Estimate every full q200 obligation; heldout cost uses worst trained site."""
    sites = list(result['sites'].values())
    worst_updates = [max(row['seconds'] for row in site['stress_timings']) for site in sites]
    task = max(row['seconds'] for site in sites for row in site['task'])
    objective = max(site['objective']['seconds'] / 6 for site in sites)
    measured = {'specialist_updates': q * sum(worst_updates),
                'shared_updates': 3 * q * max(worst_updates),
                'task_panels_including_heldout': 108 * task,
                'objective_panels_including_heldout': 192 * objective}
    require(all(math.isfinite(v) and v > 0 for v in measured.values()), 'invalid target timing evidence')
    return {'measured_extrapolations_seconds': measured,
            'estimated_unmeasured_seconds': policy['unmeasured_seconds'],
            'safety_multiplier': policy['safety_multiplier'],
            'estimated_remaining_seconds': sum(measured.values()) * policy['safety_multiplier']
                + sum(policy['unmeasured_seconds'].values()),
            'cleanup_reserve_seconds': policy['cleanup_reserve_seconds'],
            'coverage': {'scientific_updates': 6 * q, 'states': 17, 'task_panels': 108,
                         'objective_panels': 192, 'heldout_measured_before_freeze': False}}


def run_startup_if_requested(manifest, package_root, processor, model, resource_guard):
    path = os.environ.get('SHARING_CONTROLLER_CONTRACT')
    if not path:
        return
    contract = json.loads(Path(path).read_text())
    if contract.get('entry_route') not in ROUTES:
        return
    from .sharing_controller import require_worker_authorization
    require_worker_authorization(manifest)
    require_preflight(path, contract, manifest)
    request = validate_request(contract, manifest)
    import torch
    from .experiment_state import isolated_rng
    output = Path(contract['allocation_root']) / 'startup'
    output.mkdir(exist_ok=False)
    report = {'schema': schema(contract, '-target-startup-result-v1'), 'status': 'incomplete',
              'diagnostic_only': True, 'production_qualified': False,
              'binding': contract['binding'], 'target_startup_request': contract['target_startup_request']}
    begun = time.monotonic()
    try:
        # Each diagnostic learner restores its complete zero optimizer/scheduler/RNG.
        # Restore caller RNG and initialization again before creating fresh science learners.
        with isolated_rng(0):
            result = run_diagnostic(manifest, package_root, output, resource_guard,
                expected_panel=_read_ref(request['expected_panel']), model_bundle=(processor, model),
                target_class=target_class(contract))
        if not manifest['synthetic_cpu']:
            validate_result(result, contract)
        else:
            require(result.get('status') == 'diagnostic_completed' and result.get('completed_updates') == 24,
                    'tiny startup guard incomplete')
        model.codec.load_state_dict(manifest['initialization_state'], strict=True)
        require(state_dict_identity(model.codec.state_dict()) == manifest['initialization']['state_identity'],
                'startup did not restore exact fresh initialization')
        for parameter in model.codec.parameters():
            parameter.grad = None
        if not manifest['synthetic_cpu']:
            torch.cuda.synchronize()
        resource_guard()
        fit = estimate_remaining(result, request['fit_policy'],
                                 q=400 if contract.get('entry_route') == Q400_ROUTE else 200)
        remaining = min(float(os.environ['SHARING_WORK_DEADLINE_MONOTONIC']) - time.monotonic(),
                        contract['finish_before_epoch'] - time.time())
        fit.update(remaining_allocation_seconds=remaining, startup_seconds=time.monotonic()-begun)
        report['fit'] = fit
        require(fit['estimated_remaining_seconds'] + fit['cleanup_reserve_seconds'] <= remaining,
                'H200 target full lifecycle does not fit remaining allocation')
        report.update(status='startup_guard_passed', fresh_initialization_restored=True,
                      rng_restored=True, science_learner_states_reused=False)
        return report
    except BaseException as error:
        report.update(status='startup_failed_no_science', error=str(error))
        raise
    finally:
        _put(output / 'startup-receipt.json', report)
