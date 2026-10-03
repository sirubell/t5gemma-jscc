"""Exactly one named correction after the consumed q400 startup attempt."""

import copy
import json
from pathlib import Path

import pytest

from jscc import sharing_admission as admission
from test_sharing_continuation import continuation_fixture  # noqa: F401
from test_sharing_admission import approved_fixture  # noqa: F401
from test_sharing_qualification import contract_fixture  # noqa: F401


def authorization():
    return dict(
        schema="explicit-q400-corrected-attempt-authorization-v1",
        status="OWNER_APPROVED_ONE_NEW_DISTINCT_ATTEMPT",
        prior_run_id=admission.CORRECTED_PRIOR_RUN,
        prior_charge_device_seconds=283,
        new_run_id=admission.CORRECTED_NEW_RUN,
        new_cap_device_seconds=14117,
        maximum_total_attempts=2,
        combined_q400_cap_device_seconds=14400,
        automatic_retry=False,
        authorization_source="explicit parent correction authorization",
        corrected_source_handoff={
            "path": "/reviewed/correction/HANDOFF.json",
            "sha256": "844d2d622dc4be2d46cc03a45fb843ef29f88be82d7630dd9b1abde317dbef97",
        },
    )


@pytest.fixture
def corrected_fixture(continuation_fixture):  # noqa: F811
    contract, ledger, payload, hold, write = continuation_fixture
    extension = ledger["continuation_accounting"]
    prior = {
        **copy.deepcopy(hold),
        "run_id": admission.CORRECTED_PRIOR_RUN,
        "state": "SETTLED",
        "charged_device_seconds": 283,
        "scientific_training_updates": 0,
        "cleanup_verified": True,
        "compute_status": "FAILED_STARTUP_NO_SCIENCE",
    }
    extension.update(
        charged_device_seconds=283,
        allocations=[prior],
        explicit_corrected_attempt_authorization=authorization(),
    )
    extension["authorized_scope"]["maximum_distinct_runs"] = 2
    contract.update(run_id=admission.CORRECTED_NEW_RUN, hard_cap_seconds=14117)
    hold.update(
        run_id=contract["run_id"],
        max_device_seconds=14117,
        prior_consumed_run_id=admission.CORRECTED_PRIOR_RUN,
    )
    budget = json.loads(Path(contract["admission"]["path"]).read_text())
    budget.update(
        run_id=contract["run_id"],
        cap_device_seconds=14117,
        explicit_corrected_attempt=authorization(),
        prior_consumed_run={
            "run_id": admission.CORRECTED_PRIOR_RUN,
            "charged_device_seconds": 283,
            "scientific_training_updates": 0,
            "settlement_receipt": {
                "path": "/reviewed/SETTLEMENT-RECEIPT.json",
                "sha256": admission.CORRECTED_SETTLEMENT_SHA,
            },
        },
    )
    proposal = json.loads(Path(contract["proposal"]["path"]).read_text())
    proposal.update(run_id=contract["run_id"], hard_cap_seconds=14117)
    contract["proposal"] = write("proposal.json", proposal)
    payload.update(
        run_id=contract["run_id"],
        cap_device_seconds=14117,
        proposal=contract["proposal"],
    )

    def seal():
        contract["admission"] = write("continuation-budget.json", budget)
        payload["budget_admission"] = contract["admission"]
        contract["payload_receipt"] = write("payload.json", payload)
        hold.update(
            admission_sha256=contract["admission"]["sha256"],
            payload_receipt_sha256=contract["payload_receipt"]["sha256"],
        )
        Path(contract["ledger_path"]).write_text(json.dumps(ledger))

    seal()
    return contract, ledger, payload, hold, budget, seal


def test_exact_corrected_attempt_route(corrected_fixture):
    contract, ledger, payload, _, _, _ = corrected_fixture
    assert admission.validate_overnight_admission(contract, False) == payload
    pending, settled = admission._continuation_rows(ledger)
    assert (
        pending[0]["max_device_seconds"] + settled[0]["charged_device_seconds"] == 14400
    )
    assert len(admission._budget_rows(ledger)[0]) == 3


@pytest.mark.parametrize(
    "fault",
    [
        "third",
        "extra_pending",
        "missing_settlement",
        "wrong_prior",
        "wrong_charge",
        "science_started",
        "unclean",
        "wrong_new",
        "excess_cap",
        "missing_authority",
        "automatic_retry",
        "wrong_handoff",
        "missing_admission_authority",
        "wrong_settlement_hash",
        "wrong_admission_prior",
        "gated",
        "cross_account",
        "max_three",
        "unsettled_prior",
    ],
)
def test_corrected_attempt_fails_closed(corrected_fixture, fault):
    contract, ledger, payload, hold, budget, seal = corrected_fixture
    extension = ledger["continuation_accounting"]
    prior = extension["allocations"][0]
    if fault == "third":
        hold["run_id"] = admission.CORRECTED_NEW_RUN[:-2] + "03"
    elif fault == "extra_pending":
        extension["pending_reservations"].append({**hold, "run_id": "extra"})
    elif fault == "missing_settlement":
        extension["allocations"] = []
    elif fault == "wrong_prior":
        prior["run_id"] = "different-prior"
    elif fault == "wrong_charge":
        prior["charged_device_seconds"] = 282
        extension["charged_device_seconds"] = 282
    elif fault == "science_started":
        prior["scientific_training_updates"] = 1
    elif fault == "unclean":
        prior["cleanup_verified"] = False
    elif fault == "wrong_new":
        hold["prior_consumed_run_id"] = "another"
    elif fault == "excess_cap":
        hold["max_device_seconds"] = 14118
    elif fault == "missing_authority":
        extension.pop("explicit_corrected_attempt_authorization")
    elif fault == "automatic_retry":
        extension["explicit_corrected_attempt_authorization"]["automatic_retry"] = True
    elif fault == "wrong_handoff":
        extension["explicit_corrected_attempt_authorization"][
            "corrected_source_handoff"
        ]["sha256"] = "0" * 64
    elif fault == "missing_admission_authority":
        budget.pop("explicit_corrected_attempt")
    elif fault == "wrong_settlement_hash":
        budget["prior_consumed_run"]["settlement_receipt"]["sha256"] = "0" * 64
    elif fault == "wrong_admission_prior":
        budget["prior_consumed_run"]["run_id"] = "another"
    elif fault == "gated":
        hold["state"] = "HELD_DISPATCH_GATED"
    elif fault == "cross_account":
        ledger["pending_reservations"][0]["run_id"] = hold["run_id"]
    elif fault == "max_three":
        extension["authorized_scope"]["maximum_distinct_runs"] = 3
    elif fault == "unsettled_prior":
        prior["state"] = "HELD_PAYLOAD_APPROVED"
    seal()
    with pytest.raises(ValueError):
        admission.validate_overnight_admission(contract, False)
