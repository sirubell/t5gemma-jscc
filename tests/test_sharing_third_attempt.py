"""The owner's third and final named continuation slot, never automatic retry."""

import copy
import json
from pathlib import Path

import pytest

from jscc import sharing_admission as admission
from jscc.activation_replay import file_digest
from test_sharing_corrected_attempt import corrected_fixture  # noqa: F401
from test_sharing_continuation import continuation_fixture  # noqa: F401
from test_sharing_admission import approved_fixture  # noqa: F401
from test_sharing_qualification import contract_fixture  # noqa: F401


@pytest.fixture
def third_fixture(corrected_fixture):  # noqa: F811
    contract, ledger, payload, hold, budget, seal = corrected_fixture
    ext = ledger["continuation_accounting"]
    second = {
        **copy.deepcopy(hold),
        "state": "SETTLED",
        "charged_device_seconds": 871,
        "scientific_training_updates": 0,
        "cleanup_verified": True,
    }
    ext["allocations"].append(second)
    ext["charged_device_seconds"] = 1154
    ext["authorized_scope"].update(
        maximum_distinct_runs=3,
        explicitly_allowed_run_ids=[
            admission.CORRECTED_PRIOR_RUN,
            admission.CORRECTED_NEW_RUN,
            admission.THIRD_RUN,
        ],
    )
    authority = dict(
        schema="explicit-q400-third-attempt-authorization-v1",
        status="OWNER_APPROVED_ONE_NEW_DISTINCT_ATTEMPT",
        new_run_id=admission.THIRD_RUN,
        new_cap_device_seconds=13246,
        maximum_total_attempts=3,
        combined_q400_cap_device_seconds=14400,
        prior_charge_device_seconds=1154,
        automatic_retry=False,
        finish_before_epoch=admission.DEADLINE,
        sole_execution_owner=admission.QUALIFIER,
        authorization_source="Explicit parent approval",
        corrected_source_handoff={
            "path": "/reviewed/vanilla/HANDOFF.json",
            "sha256": "aa10b308590b2a93825f0c55a973a19134076aee964e8a47ad4a51bfb718fff1",
        },
        prior_consumed_runs=[
            dict(
                run_id=run,
                charged_device_seconds=charge,
                state="SETTLED",
                scientific_training_updates=0,
                cleanup_verified=True,
                settlement_receipt={
                    "path": f"/reviewed/{charge}/settlement.json",
                    "sha256": digest,
                },
            )
            for run, charge, digest in [
                (
                    admission.CORRECTED_PRIOR_RUN,
                    283,
                    admission.CORRECTED_SETTLEMENT_SHA,
                ),
                (admission.CORRECTED_NEW_RUN, 871, admission.SECOND_SETTLEMENT_SHA),
            ]
        ],
    )
    ref = {
        "path": "/reviewed/ATTEMPT03-AUTHORIZATION.json",
        "sha256": admission.THIRD_AUTHORITY_SHA,
    }
    ext.update(
        explicit_third_attempt_authorization=authority,
        explicit_third_attempt_authorization_receipt=ref,
    )
    contract.update(
        run_id=admission.THIRD_RUN,
        hard_cap_seconds=13246,
        device_uuid="GPU-324fae60-a5b1-ebc1-146f-c5a94d90dbb7",
    )
    hold.update(
        run_id=admission.THIRD_RUN,
        max_device_seconds=13246,
        prior_consumed_run_ids=[
            admission.CORRECTED_PRIOR_RUN,
            admission.CORRECTED_NEW_RUN,
        ],
    )
    budget.update(
        run_id=admission.THIRD_RUN,
        cap_device_seconds=13246,
        explicit_third_attempt=copy.deepcopy(authority),
        explicit_third_attempt_authorization=copy.deepcopy(ref),
        prior_consumed_runs=copy.deepcopy(authority["prior_consumed_runs"]),
    )
    proposal_path = Path(contract["proposal"]["path"])
    proposal = json.loads(proposal_path.read_text())
    proposal.update(run_id=admission.THIRD_RUN, hard_cap_seconds=13246)
    proposal_path.write_text(json.dumps(proposal))
    contract["proposal"]["sha256"] = file_digest(proposal_path)
    payload.update(
        schema="overnight-run-payload-approval-v1",
        run_id=admission.THIRD_RUN,
        cap_device_seconds=13246,
        proposal=contract["proposal"],
    )
    handoff_path = Path(payload["owner_handoff"]["path"])
    handoff = json.loads(handoff_path.read_text())
    handoff.update(
        run_id=admission.THIRD_RUN,
        campaign_id=admission.CAMPAIGN,
        device_uuid=contract["device_uuid"],
        automatic_retry=False,
        finish_before_epoch=admission.DEADLINE,
    )
    handoff_path.write_text(json.dumps(handoff))
    payload["owner_handoff"]["sha256"] = file_digest(handoff_path)
    seal()
    return contract, ledger, payload, hold, budget, seal


def test_exact_third_attempt(third_fixture):
    contract, ledger, payload, *_ = third_fixture
    assert admission.validate_overnight_admission(contract, False) == payload
    pending, settled = admission._continuation_rows(ledger)
    assert (
        sum(r["charged_device_seconds"] for r in settled)
        + pending[0]["max_device_seconds"]
        == 14400
    )


@pytest.mark.parametrize(
    "fault",
    [
        "fourth",
        "charge01",
        "charge02",
        "science",
        "cleanup",
        "missing_prior",
        "missing_authority",
        "authority_ref",
        "settlement_ref",
        "admission_authority",
        "admission_ref",
        "old_payload",
        "gated",
        "cap",
        "cross_account",
        "extra",
        "wrong_owner",
        "auto_retry",
        "missing_allowed",
    ],
)
def test_third_attempt_rejects(third_fixture, fault):
    contract, ledger, payload, hold, budget, seal = third_fixture
    ext = ledger["continuation_accounting"]
    if fault == "fourth":
        hold["run_id"] = admission.THIRD_RUN[:-2] + "04"
    elif fault == "charge01":
        ext["allocations"][0]["charged_device_seconds"] = 282
    elif fault == "charge02":
        ext["allocations"][1]["charged_device_seconds"] = 870
    elif fault == "science":
        ext["allocations"][1]["scientific_training_updates"] = 1
    elif fault == "cleanup":
        ext["allocations"][0]["cleanup_verified"] = False
    elif fault == "missing_prior":
        ext["allocations"].pop()
    elif fault == "missing_authority":
        ext.pop("explicit_third_attempt_authorization")
    elif fault == "authority_ref":
        ext["explicit_third_attempt_authorization_receipt"]["sha256"] = "0" * 64
    elif fault == "settlement_ref":
        ext["explicit_third_attempt_authorization"]["prior_consumed_runs"][1][
            "settlement_receipt"
        ]["sha256"] = "0" * 64
    elif fault == "admission_authority":
        budget.pop("explicit_third_attempt")
    elif fault == "admission_ref":
        budget["explicit_third_attempt_authorization"]["sha256"] = "0" * 64
    elif fault == "old_payload":
        payload["schema"] = "overnight-q400-payload-approval-v1"
    elif fault == "gated":
        hold["state"] = "HELD_DISPATCH_GATED"
    elif fault == "cap":
        hold["max_device_seconds"] = 13247
    elif fault == "cross_account":
        ledger["pending_reservations"][0]["run_id"] = admission.THIRD_RUN
    elif fault == "extra":
        ext["pending_reservations"].append({**hold, "run_id": "extra"})
    elif fault == "wrong_owner":
        ext["explicit_third_attempt_authorization"]["sole_execution_owner"] = "other"
    elif fault == "auto_retry":
        ext["explicit_third_attempt_authorization"]["automatic_retry"] = True
    elif fault == "missing_allowed":
        ext["authorized_scope"].pop("explicitly_allowed_run_ids")
    seal()
    with pytest.raises(ValueError):
        admission.validate_overnight_admission(contract, False)
