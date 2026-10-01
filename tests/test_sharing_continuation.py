"""Continuation admission cannot spend the primary H200 allocation budget."""
import copy
import json
from pathlib import Path

import pytest

from jscc import sharing_admission as admission, sharing_preparation
from jscc.sharing_controller import binding_identity
from test_sharing_admission import approved_fixture  # noqa: F401
from test_sharing_qualification import contract_fixture  # noqa: F401


@pytest.fixture
def continuation_fixture(approved_fixture):  # noqa: F811
    contract, ledger, payload, old_hold, write = approved_fixture
    manifest = sharing_preparation.load_prepared(contract['prepared']['path'])
    manifest.update(q=400, protocol_id='sharing-dn-q400-effective64-v1',
                    recipe_identity='K+0.1R-native-effective64-q400',
                    initialization_policy={'seed': 0, 'fresh_cpu_codec': True,
                                           'baseline_state_reused': False})
    manifest['resolved_config']['seed'] = 0
    config = write('config.json', manifest['resolved_config'])
    manifest['config'] = {**config, 'path': 'config.json'}
    payload['config'] = config
    binding = binding_identity(manifest)
    hold = copy.deepcopy(old_hold)
    contract.update(run_id='q400-one-run', hard_cap_seconds=14400, binding=binding)
    hold.update(run_id=contract['run_id'], component='5090-shared-q400-B512',
                max_device_seconds=14400, seed=0, q=400, diagnostic_only=False,
                payload_binding=binding)
    ledger['continuation_accounting'] = {
        'schema': 'overnight-q400-continuation-accounting-v1',
        'campaign_id': admission.CAMPAIGN, 'ledger_owner': 'task-7',
        'cap_device_seconds': 14400, 'protected_h200_device_seconds': 32400,
        'authorized_scope': {'cell': 'D-N', 'bottleneck_dim': 512, 'seed': 0,
                            'q': 400, 'maximum_distinct_runs': 1,
                            'automatic_retry': False,
                            'finish_before_epoch': admission.DEADLINE},
        'charged_device_seconds': 0, 'allocations': [],
        'pending_reservations': [hold],
    }
    budget = {'schema': 'overnight-q400-continuation-admission-v1',
              'template_only': False, 'status': 'APPROVED_BUDGET_SLOT',
              'campaign_id': admission.CAMPAIGN, 'ledger_path': contract['ledger_path'],
              'accounting_key': 'continuation_accounting', 'reservation_owner': 'task-7',
              'sole_launcher': admission.QUALIFIER, 'run_id': contract['run_id'],
              'component': hold['component'], 'host': '5090B', 'devices': 1,
              'bottleneck_dim': 512, 'seed': 0, 'q': 400, 'cap_device_seconds': 14400,
              'aggregate_cap_device_seconds': 54000, 'protected_h200_device_seconds': 32400,
              'automatic_retry': False, 'finish_before_epoch': admission.DEADLINE,
              'scope': admission.CONTINUATION_WORKLOAD}
    contract['admission'] = write('continuation-budget.json', budget)
    contract['proposal'] = write('proposal.json', {
        'run_id': contract['run_id'], 'diagnostic_only': False,
        'hard_cap_seconds': 14400, 'binding': binding,
        'workload': admission.CONTINUATION_WORKLOAD})
    payload.update(schema='overnight-q400-payload-approval-v1',
                   accounting_key='continuation_accounting', run_id=contract['run_id'],
                   component=hold['component'], cap_device_seconds=14400,
                   diagnostic_only=False, binding=binding,
                   budget_admission=contract['admission'], proposal=contract['proposal'],
                   workload=admission.CONTINUATION_WORKLOAD)
    payload.pop('scientific_promotion', None)
    payload['owner_handoff'] = write('q400-handoff.json', {
        'schema': 'explicit-q400-resource-handoff-v1',
        'status': 'released_resource_handoff_granted',
        'sole_execution_owner': admission.QUALIFIER,
        'device_lease': contract['device_lease'],
        'workload': admission.CONTINUATION_WORKLOAD})
    contract['payload_receipt'] = write('payload.json', payload)
    hold.update(admission_path=contract['admission']['path'],
                admission_sha256=contract['admission']['sha256'],
                payload_receipt_path=contract['payload_receipt']['path'],
                payload_receipt_sha256=contract['payload_receipt']['sha256'])
    Path(contract['ledger_path']).write_text(json.dumps(ledger))
    return contract, ledger, payload, hold, write


def test_continuation_exact_binding(continuation_fixture):
    contract, ledger, payload, hold, write = continuation_fixture
    assert admission.validate_overnight_admission(contract, False) == payload
    assert len(admission._budget_rows(ledger)[0]) == 3


@pytest.mark.parametrize('change', ['primary_pressure', 'duplicate_primary', 'second_run',
                                    'wrong_q', 'wrong_owner', 'gated', 'already_settled',
                                    'wrong_workload', 'old_protocol', 'relabelled_account',
                                    'promotion', 'boolean_seed'])
def test_continuation_fails_closed(continuation_fixture, change):
    contract, ledger, payload, hold, write = continuation_fixture
    extension = ledger['continuation_accounting']
    if change == 'primary_pressure':
        ledger['pending_reservations'].append({
            'run_id': 'spark-extra', 'component': 'spark-clean-full-evaluation',
            'max_device_seconds': 10000})
    elif change == 'duplicate_primary':
        ledger['pending_reservations'][0]['run_id'] = contract['run_id']
    elif change == 'second_run':
        extension['pending_reservations'].append({**hold, 'run_id': 'another-run'})
    elif change == 'wrong_q':
        hold['q'] = 200
    elif change == 'wrong_owner':
        hold['execution_owner'] = 'someone-else'
    elif change == 'gated':
        hold['state'] = 'HELD_DISPATCH_GATED'
    elif change == 'already_settled':
        extension['pending_reservations'] = []
        extension['allocations'] = [{**hold, 'state': 'SETTLED', 'charged_device_seconds': 1}]
        extension['charged_device_seconds'] = 1
    elif change == 'wrong_workload':
        payload['workload'] = admission.PRODUCTION_WORKLOAD
    elif change == 'old_protocol':
        contract['binding']['protocol_id'] = 'sharing-dn-q200-effective64-v1'
    elif change == 'promotion':
        payload['scientific_promotion'] = True
    elif change == 'boolean_seed':
        hold['seed'] = False
    else:
        payload['accounting_key'] = 'primary'
    contract['payload_receipt'] = write('payload.json', payload)
    hold['payload_receipt_sha256'] = contract['payload_receipt']['sha256']
    Path(contract['ledger_path']).write_text(json.dumps(ledger))
    with pytest.raises(ValueError):
        admission.validate_overnight_admission(contract, False)
