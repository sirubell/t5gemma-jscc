"""Actual owner record shapes, copied to temporary files; no ledger mutation."""

import copy
import json
from pathlib import Path

import pytest

from jscc import sharing_admission as admission
from jscc import sharing_preparation
from test_sharing_qualification import contract_fixture, ref  # noqa: F401

OWNER_ROOT = Path(
    "/Users/tim_c_wang/Lab/mtk/t5gemma-jscc/docs/local/research/execution-preparation/overnight-shared-eval-20261001-01"
)


@pytest.fixture
def approved_fixture(contract_fixture, monkeypatch):  # noqa: F811
    path, contract, manifest = contract_fixture
    manifest.update(q=200, batch_size=64)
    manifest["resolved_config"]["training"] = {
        "batch_size": 16,
        "gradient_accumulation": 4,
    }
    if not (OWNER_ROOT / "payload-receipt-template.json").exists():
        pytest.skip("actual owner schema fixtures unavailable")
    ledger = json.loads((OWNER_ROOT / "allocation-ledger.json").read_text())
    budget = json.loads((OWNER_ROOT / "5090-dn-three-width-admission.json").read_text())
    payload = json.loads((OWNER_ROOT / "payload-receipt-template.json").read_text())
    ledger["allocations"] = []
    ledger["charged_device_seconds"] = 0
    ledger["pending_reservations"] = [
        {
            "run_id": row["run_id"],
            "component": "5090-qualification",
            "host": "5090B",
            "devices": 1,
            "bottleneck_dim": row["bottleneck_dim"],
            "max_device_seconds": row["cap_device_seconds"],
            "execution_owner": admission.QUALIFIER,
            "no_auto_retry": True,
            "finish_before_epoch": admission.DEADLINE,
        }
        for row in budget["runs"]
    ]
    ledger_path = path.parent / "ledger.json"
    run = next(row for row in budget["runs"] if row["bottleneck_dim"] == 512)
    contract.update(
        run_id=run["run_id"], hard_cap_seconds=1200, ledger_path=str(ledger_path)
    )
    budget["ledger_path"] = str(ledger_path)

    def write(name, value):
        p = path.parent / name
        p.write_text(json.dumps(value))
        return ref(p)

    contract["admission"] = write("budget.json", budget)
    contract["proposal"] = write(
        "proposal.json",
        {
            "run_id": contract["run_id"],
            "diagnostic_only": True,
            "hard_cap_seconds": 1200,
            "binding": contract["binding"],
            "workload": admission.DIAGNOSTIC_WORKLOAD,
        },
    )
    contract["target_assets"] = {
        "test-asset": write("asset.json", {"cpu_fixture": True})
    }
    source = write("source.zip", {"cpu_fixture": True})
    config = write("config.json", manifest["resolved_config"])
    initialization = write("initial.json", {"cpu_fixture": True})
    manifest["source_archive"] = {**source, "path": "source.zip"}
    manifest["config"] = {**config, "path": "config.json"}
    manifest["initialization"] = {
        **initialization,
        "path": "initial.json",
        "state_identity": "initial",
    }
    payload.update(
        template_only=False,
        status="APPROVED_FOR_BOUND_PAYLOAD",
        run_id=contract["run_id"],
        component="5090-qualification",
        host="5090B",
        devices=1,
        cap_device_seconds=1200,
        sole_execution_owner=admission.QUALIFIER,
        diagnostic_only=True,
        ledger_path=str(ledger_path),
        budget_admission=contract["admission"],
        proposal=contract["proposal"],
        prepared=contract["prepared"],
        binding=contract["binding"],
        source_archive=source,
        source_inventory=manifest["source_inventory"],
        config=config,
        initialization={**initialization, "state_identity": "initial"},
        target_assets=contract["target_assets"],
        expected_panel=contract["expected_panel"],
        review=write(
            "review.json",
            {
                "status": "cpu-test-review",
                "source_inventory": manifest["source_inventory"],
            },
        ),
        owner_handoff=write("handoff.json", {"status": "cpu-test-handoff"}),
        workload=admission.DIAGNOSTIC_WORKLOAD,
    )
    contract["device_lease"] = str(
        Path(contract["ledger_path"]).parent / "device-lease.lock"
    )
    handoff = {
        "schema": "explicit-resource-handoff-task3-v1",
        "status": "released_resource_handoff_granted_no_launch_here",
        "from_owner_thread": "01a0f30f-58de-719f-97ac-b870486db0fc",
        "sole_5090_qualification_owner_thread": admission.QUALIFIER,
        "lease": {"path": contract["device_lease"]},
    }
    payload["owner_handoff"] = write("handoff.json", handoff)
    contract["payload_receipt"] = write("payload.json", payload)
    hold = next(
        row
        for row in ledger["pending_reservations"]
        if row["run_id"] == contract["run_id"]
    )
    hold.update(
        state="HELD_PAYLOAD_APPROVED",
        admission_path=contract["admission"]["path"],
        admission_sha256=contract["admission"]["sha256"],
        payload_receipt_path=contract["payload_receipt"]["path"],
        payload_receipt_sha256=contract["payload_receipt"]["sha256"],
        payload_binding=contract["binding"],
        diagnostic_only=True,
    )
    ledger_path.write_text(json.dumps(ledger))
    monkeypatch.setattr(sharing_preparation, "load_prepared", lambda path: manifest)
    return contract, ledger, payload, hold, write


def test_approved_exact_actual_schema_roundtrip(approved_fixture):
    contract, ledger, payload, hold, write = approved_fixture
    assert admission.validate_overnight_admission(contract, True) == payload


@pytest.mark.parametrize(
    "change",
    [
        "gated",
        "wrong_owner",
        "wrong_width",
        "wrong_run",
        "duplicate",
        "settled",
        "deadline",
        "payload_hash",
        "budget_hash",
        "component_budget",
        "total_budget",
    ],
)
def test_live_owner_hold_fails_closed(approved_fixture, change):
    contract, ledger, payload, hold, write = approved_fixture
    if change == "gated":
        hold["state"] = "HELD_DISPATCH_GATED"
    elif change == "wrong_owner":
        ledger["ledger_owner"] = "someone-else"
    elif change == "wrong_width":
        hold["bottleneck_dim"] = 1152
    elif change == "wrong_run":
        contract["run_id"] = "unreserved"
    elif change == "duplicate":
        ledger["pending_reservations"].append(copy.deepcopy(hold))
    elif change == "settled":
        ledger["pending_reservations"].remove(hold)
        ledger["allocations"].append({**hold, "charged_device_seconds": 1})
        ledger["charged_device_seconds"] = 1
    elif change == "deadline":
        contract["finish_before_epoch"] += 1
    elif change == "payload_hash":
        hold["payload_receipt_sha256"] = "0" * 64
    elif change == "budget_hash":
        hold["admission_sha256"] = "0" * 64
    elif change == "component_budget":
        ledger["pending_reservations"].append(
            {
                "run_id": "extra",
                "component": "5090-qualification",
                "max_device_seconds": 1,
            }
        )
    else:
        ledger["pending_reservations"].append(
            {
                "run_id": "too-large",
                "component": "spark-clean-full-evaluation",
                "max_device_seconds": 54000,
            }
        )
    Path(contract["ledger_path"]).write_text(json.dumps(ledger))
    with pytest.raises(ValueError):
        admission.validate_overnight_admission(contract, True)


@pytest.mark.parametrize(
    "change",
    [
        "template",
        "binding",
        "proposal",
        "inventory",
        "init",
        "workload",
        "admission_link",
        "review_bytes",
    ],
)
def test_payload_must_be_exact_owner_approval(approved_fixture, change):
    contract, ledger, payload, hold, write = approved_fixture
    if change == "template":
        payload["template_only"] = True
    elif change == "binding":
        payload["binding"] = {}
    elif change == "proposal":
        payload["proposal"] = {"path": "other", "sha256": "0" * 64}
    elif change == "inventory":
        payload["source_inventory"] = {}
    elif change == "init":
        payload["initialization"]["state_identity"] = "other"
    elif change == "workload":
        payload["workload"] = {}
    elif change == "admission_link":
        payload["budget_admission"]["sha256"] = "0" * 64
    else:
        Path(payload["review"]["path"]).write_text("changed")
    contract["payload_receipt"] = write("payload.json", payload)
    hold["payload_receipt_sha256"] = contract["payload_receipt"]["sha256"]
    Path(contract["ledger_path"]).write_text(json.dumps(ledger))
    with pytest.raises(ValueError):
        admission.validate_overnight_admission(contract, True)


def test_diagnostic_budget_cannot_unlock_h200(approved_fixture):
    contract, ledger, payload, hold, write = approved_fixture
    with pytest.raises(ValueError):
        admission.validate_overnight_admission(contract, False)


@pytest.fixture
def h200_fixture(approved_fixture, monkeypatch):
    contract, ledger, payload, hold, write = approved_fixture
    budget = json.loads(
        (OWNER_ROOT / "h200-budget-admission-template.json").read_text()
    )
    owner = "01a0f19d-7f2e-739e-8aaf-b4153af2b0d8"
    budget.update(
        template_only=False,
        status="APPROVED_BUDGET_SLOTS",
        sole_launcher=owner,
        ledger_path=contract["ledger_path"],
    )
    for row in budget["runs"]:
        row["run_id"] = f"test-h200-{row['bottleneck_dim']}"
    contract.update(run_id="test-h200-512", hard_cap_seconds=10800)
    contract["admission"] = write("h200-budget.json", budget)
    proposal = json.loads(Path(contract["proposal"]["path"]).read_text())
    proposal.update(
        run_id=contract["run_id"],
        diagnostic_only=False,
        hard_cap_seconds=10800,
        workload=admission.PRODUCTION_WORKLOAD,
    )
    contract["proposal"] = write("h200-proposal.json", proposal)
    payload.update(
        run_id=contract["run_id"],
        component="h200-sharing-B512",
        host="H200",
        cap_device_seconds=10800,
        sole_execution_owner=owner,
        diagnostic_only=False,
        budget_admission=contract["admission"],
        proposal=contract["proposal"],
        workload=admission.PRODUCTION_WORKLOAD,
        device_binding={
            "policy": "slurm-one-h200-runtime-uuid-v1",
            "gpu_class": "H200",
            "device_count": 1,
            "lease_root": str(Path(contract["ledger_path"]).parent),
            "lease_filename_rule": "device-{GPU_UUID}.lock",
            "runtime_receipt_required": True,
        },
    )
    contract["payload_receipt"] = write("h200-payload.json", payload)
    hold.update(
        run_id=contract["run_id"],
        component=payload["component"],
        host="H200",
        max_device_seconds=10800,
        execution_owner=owner,
        diagnostic_only=False,
        admission_path=contract["admission"]["path"],
        admission_sha256=contract["admission"]["sha256"],
        payload_receipt_path=contract["payload_receipt"]["path"],
        payload_receipt_sha256=contract["payload_receipt"]["sha256"],
    )
    ledger["pending_reservations"] = [hold]
    Path(contract["ledger_path"]).write_text(json.dumps(ledger))
    uuid = "GPU-12345678-1234-1234-1234-123456789abc"
    lease = str(Path(payload["device_binding"]["lease_root"]) / f"device-{uuid}.lock")
    contract.update(device_uuid=uuid, device_lease=lease)
    runtime = dict(
        schema="slurm-h200-device-binding-v1",
        status="BOUND_FROM_VERIFIED_SLURM_ALLOCATION",
        run_id=contract["run_id"],
        campaign_id=admission.CAMPAIGN,
        component=payload["component"],
        execution_owner=owner,
        payload_receipt=contract["payload_receipt"],
        budget_admission=contract["admission"],
        slurm_job_id="123",
        slurm_array_task_id="0",
        node="test-node",
        device_uuid=uuid,
        device_name="NVIDIA H200",
        device_lease=lease,
        allocation_evidence={"test": True},
        visibility_evidence={"test": True},
        finish_before_epoch=admission.DEADLINE,
    )
    contract["device_binding_receipt"] = write("runtime.json", runtime)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    monkeypatch.setenv("SLURMD_NODENAME", "test-node")
    return contract, runtime, write


def test_h200_exact_owner_schema_and_runtime(h200_fixture):
    contract, _, _ = h200_fixture
    assert (
        admission.validate_overnight_admission(contract, False)["diagnostic_only"]
        is False
    )


@pytest.mark.parametrize(
    "field",
    [
        "device_uuid",
        "device_lease",
        "slurm_job_id",
        "node",
        "payload_receipt",
        "allocation_evidence",
    ],
)
def test_h200_runtime_mismatch_rejected(h200_fixture, field):
    contract, runtime, write = h200_fixture
    runtime[field] = (
        {} if field in ("payload_receipt", "allocation_evidence") else "wrong"
    )
    contract["device_binding_receipt"] = write("runtime.json", runtime)
    with pytest.raises(ValueError):
        admission.validate_overnight_admission(contract, False)


@pytest.mark.parametrize(
    "change",
    [
        "partition",
        "task_batch",
        "norm",
        "architecture",
        "q",
        "batch",
        "width",
        "training",
    ],
)
def test_h200_actual_package_scope_cannot_be_declared_away(
    h200_fixture, monkeypatch, change
):
    from jscc.sharing_controller import binding_identity
    from jscc.sharing_accumulation import partition_policy

    contract, runtime, write = h200_fixture
    manifest = sharing_preparation.load_prepared(None)
    if change == "partition":
        manifest["execution_partition"] = partition_policy(32)
    elif change == "task_batch":
        manifest["task_template"]["settings"]["batch_size"] = 32
    elif change == "norm":
        manifest["resolved_config"]["codec"]["layernorm"] = "both"
    elif change == "architecture":
        manifest["resolved_config"]["codec"]["architecture"] = "two_linear_gelu"
    elif change == "q":
        manifest["q"] = 400
    elif change == "batch":
        manifest["batch_size"] = 32
    elif change == "width":
        manifest["selected_bottleneck_dim"] = 1152
    else:
        manifest["resolved_config"]["training"]["batch_size"] = 32
    contract["binding"] = binding_identity(manifest)
    payload = admission.read_ref(contract["payload_receipt"])
    proposal = admission.read_ref(contract["proposal"])
    proposal["binding"] = contract["binding"]
    contract["proposal"] = write("h200-proposal.json", proposal)
    payload.update(binding=contract["binding"], proposal=contract["proposal"])
    contract["payload_receipt"] = write("h200-payload.json", payload)
    runtime["payload_receipt"] = contract["payload_receipt"]
    contract["device_binding_receipt"] = write("runtime.json", runtime)
    ledger = json.loads(Path(contract["ledger_path"]).read_text())
    hold = ledger["pending_reservations"][0]
    hold.update(
        payload_binding=contract["binding"],
        payload_receipt_sha256=contract["payload_receipt"]["sha256"],
    )
    Path(contract["ledger_path"]).write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="actual prepared package"):
        admission.validate_overnight_admission(contract, False)
