"""Read-only verifier for task-7's single bounded overnight campaign.

Budget holds alone are never executable. The owner's exact payload approval and
its live-ledger link are mandatory; this module neither reserves nor settles.
"""

from pathlib import Path
import json
import os
import re

from .activation_replay import file_digest
from .sharing_accumulation import partition_policy

CAMPAIGN = "overnight-shared-eval-20261001-01"
DEADLINE = 1790935200
QUALIFIER = "01a0f5bf-3a37-7011-81af-ed5ff67f5098"
WIDTHS = (512, 1152, 2304)
CONDITIONS = ["no_noise", -6, 0, 6, 12, 18]
DIAGNOSTIC_WORKLOAD = {
    "physical_optimizer_calls": 24,
    "scientific_training_updates": 0,
    "trained_sites": ["enc_l9", "enc_l19", "enc_fn"],
    "objective": {
        "panels": 18,
        "sequences_per_panel": 128,
        "conditions": CONDITIONS,
        "failed_allowed": 0,
    },
    "task": {
        "panels": 18,
        "queries_per_panel": 256,
        "candidates_per_query": 4,
        "candidate_requests_per_panel": 1024,
        "conditions": CONDITIONS,
        "failed_allowed": 0,
    },
    "l14_access": "FORBIDDEN",
    "guard_state_restored_before_any_science": True,
}
PRODUCTION_WORKLOAD = {
    "specialists": {
        "sites": ["enc_l9", "enc_l19", "enc_fn"],
        "fresh": True,
        "updates_per_site": 200,
    },
    "shared": {"fresh": True, "updates": 600},
    "scientific_training_updates": 1200,
    "effective_batch": 64,
    "physical_batch": 16,
    "accumulation": 4,
    "objective_sequences": 128,
    "task_queries": 256,
    "conditions": CONDITIONS,
    "l14": "only after this width pilot’s three specialists and shared learner are complete and frozen; no adaptation or selection across widths using l14.",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def verify_ref(ref):
    require(
        isinstance(ref, dict)
        and isinstance(ref.get("path"), str)
        and Path(ref["path"]).is_absolute(),
        "absolute immutable receipt reference required",
    )
    digest = ref.get("sha256")
    require(
        isinstance(digest, str)
        and len(digest) == 64
        and all(c in "0123456789abcdef" for c in digest)
        and file_digest(Path(ref["path"])) == digest,
        "receipt file checksum mismatch",
    )
    return Path(ref["path"])


def read_ref(ref):
    return json.loads(verify_ref(ref).read_text())


def _budget_rows(ledger):
    pending = ledger.get("pending_reservations", [])
    settled = ledger.get("allocations", [])
    ids = [r.get("run_id") for r in pending + settled]
    require(
        all(isinstance(run, str) and run for run in ids) and len(ids) == len(set(ids)),
        "duplicate or missing campaign run identity",
    )
    caps = [r.get("max_device_seconds") for r in pending]
    charges = [r.get("charged_device_seconds") for r in settled]
    charged = ledger.get("charged_device_seconds")
    require(
        type(charged) is int
        and charged >= 0
        and all(type(n) is int and n > 0 for n in caps)
        and all(type(n) is int and n >= 0 for n in charges)
        and sum(charges) == charged
        and charged + sum(caps) <= 54000,
        "overnight aggregate accounting exceeds or disagrees with54000 cap",
    )
    totals = {}
    for row in pending:
        totals[row.get("component")] = (
            totals.get(row.get("component"), 0) + row["max_device_seconds"]
        )
    for row in settled:
        totals[row.get("component")] = (
            totals.get(row.get("component"), 0) + row["charged_device_seconds"]
        )
    limits = {
        "5090-qualification": 3600,
        "spark-clean-full-evaluation": 18000,
        **{f"h200-sharing-B{width}": 10800 for width in WIDTHS},
    }
    require(
        set(totals) <= set(limits)
        and all(amount <= limits[key] for key, amount in totals.items())
        and sum(
            amount
            for key, amount in totals.items()
            if str(key).startswith("h200-sharing-B")
        )
        <= 32400,
        "component device-second cap exceeded",
    )
    components = ledger.get("components", [])
    require(
        len(components) == len(limits)
        and {r.get("component"): r.get("cap_device_seconds") for r in components}
        == limits,
        "authorized component ceilings changed",
    )
    return pending, settled


def validate_overnight_admission(contract, diagnostic_only):
    """Validate actual owner schemas; caller retains target/lease/qualification gates."""
    from .sharing_preparation import load_prepared
    from .sharing_controller import binding_identity

    require(type(diagnostic_only) is bool, "explicit diagnostic boolean required")
    ledger = json.loads(Path(contract["ledger_path"]).read_text())
    require(
        ledger.get("schema") == "bounded-overnight-campaign-ledger-v1"
        and ledger.get("campaign_id") == CAMPAIGN
        and ledger.get("ledger_owner") == "task-7"
        and ledger.get("cap_device_seconds") == 54000
        and contract.get("sole_owner") == "task-7",
        "exact task-7 overnight campaign ledger required",
    )
    pending, settled = _budget_rows(ledger)
    run_id = contract["run_id"]
    binding = contract["binding"]
    width = binding.get("bottleneck_dim")
    component = "5090-qualification" if diagnostic_only else f"h200-sharing-B{width}"
    host = "5090B" if diagnostic_only else "H200"
    cap = contract["hard_cap_seconds"]
    require(
        width in WIDTHS
        and binding.get("cell") == "D-N"
        and type(cap) is int
        and 0 < cap <= (1200 if diagnostic_only else 10800)
        and contract.get("automatic_retry") is False
        and contract.get("finish_before_epoch") == DEADLINE,
        "exact no-norm width/cap/no-retry/deadline required",
    )
    matching = [r for r in pending if r["run_id"] == run_id]
    require(
        len(matching) == 1 and not any(r["run_id"] == run_id for r in settled),
        "one pending never-allocated run required",
    )
    hold = matching[0]
    require(
        hold.get("state") == "HELD_PAYLOAD_APPROVED",
        "budget held but exact payload approval still gated",
    )
    require(
        hold.get("component") == component
        and hold.get("host") == host
        and hold.get("devices") == 1
        and hold.get("bottleneck_dim") == width
        and hold.get("max_device_seconds") == cap
        and hold.get("no_auto_retry") is True
        and hold.get("finish_before_epoch") == DEADLINE
        and hold.get("diagnostic_only") is diagnostic_only
        and hold.get("payload_binding") == binding,
        "live hold exact scope/binding mismatch",
    )
    admission = read_ref(contract["admission"])
    admission_origin = contract.get(
        "admission_origin_path", contract["admission"]["path"]
    )
    payload_origin = contract.get(
        "payload_receipt_origin_path", contract["payload_receipt"]["path"]
    )
    ledger_origin = contract.get("ledger_origin_path", contract["ledger_path"])
    require(
        hold.get("admission_path") == admission_origin
        and hold.get("admission_sha256") == contract["admission"]["sha256"]
        and hold.get("payload_receipt_path") == payload_origin
        and hold.get("payload_receipt_sha256") == contract["payload_receipt"]["sha256"],
        "live hold immutable admission/payload links mismatch",
    )
    payload = read_ref(contract["payload_receipt"])
    require(
        payload.get("schema") == "overnight-run-payload-approval-v1"
        and payload.get("template_only") is False
        and payload.get("status") == "APPROVED_FOR_BOUND_PAYLOAD"
        and payload.get("campaign_id") == CAMPAIGN
        and payload.get("ledger_owner") == "task-7"
        and payload.get("scientific_promotion") is False,
        "actual task-7 bound payload approval required",
    )
    require(
        all(
            payload.get(k) == v
            for k, v in {
                "run_id": run_id,
                "component": component,
                "host": host,
                "devices": 1,
                "cap_device_seconds": cap,
                "sole_execution_owner": hold.get("execution_owner"),
                "diagnostic_only": diagnostic_only,
                "automatic_retry": False,
                "finish_before_epoch": DEADLINE,
                "ledger_path": ledger_origin,
                "binding": binding,
                "prepared": contract["prepared"],
                "proposal": contract["proposal"],
                "target_assets": contract["target_assets"],
                "expected_panel": contract["expected_panel"],
            }.items()
        ),
        "approved payload differs from exact controller binding",
    )
    require(
        payload.get("budget_admission")
        == {"path": admission_origin, "sha256": contract["admission"]["sha256"]},
        "payload budget admission link mismatch",
    )
    if diagnostic_only:
        handoff = read_ref(payload["owner_handoff"])
        require(
            handoff.get("schema") == "explicit-resource-handoff-task3-v1"
            and handoff.get("status")
            == "released_resource_handoff_granted_no_launch_here"
            and handoff.get("from_owner_thread")
            == "01a0f30f-58de-719f-97ac-b870486db0fc"
            and handoff.get("sole_5090_qualification_owner_thread") == QUALIFIER
            and handoff.get("lease", {}).get("path") == contract.get("device_lease")
            and isinstance(contract.get("device_lease"), str)
            and Path(contract["device_lease"]).is_absolute(),
            "exact released 5090 owner and shared lease required",
        )
        require(
            admission.get("schema") == "three-width-diagnostic-budget-admission-v1"
            and admission.get("campaign_id") == CAMPAIGN
            and admission.get("ledger_path") == ledger_origin
            and admission.get("reservation_owner") == "task-7"
            and admission.get("sole_launcher") == QUALIFIER
            and hold.get("execution_owner") == QUALIFIER
            and admission.get("host") == "5090B"
            and admission.get("total_cap_device_seconds") == 3600
            and admission.get("finish_before_epoch") == DEADLINE
            and admission.get("qualification_is_not_promotion") is True,
            "exact three-width diagnostic budget admission required",
        )
        runs = admission.get("runs", [])
        require(
            len(runs) == 3
            and {r.get("bottleneck_dim") for r in runs} == set(WIDTHS)
            and len({r.get("run_id") for r in runs}) == 3
            and all(
                type(r.get("cap_device_seconds")) is int
                and 0 < r["cap_device_seconds"] <= 1200
                and r.get("devices") == 1
                for r in runs
            )
            and sum(r["cap_device_seconds"] for r in runs) <= 3600,
            "exact three unique width slots within3600 required",
        )
        allowed = {r["run_id"]: r for r in runs}
        require(
            run_id in allowed
            and allowed[run_id]["bottleneck_dim"] == width
            and allowed[run_id]["cap_device_seconds"] == cap
            and all(
                r["run_id"] in allowed
                for r in pending + settled
                if r.get("component") == component
            ),
            "diagnostic run outside immutable three-width slots",
        )
    else:
        require(
            admission.get("schema") == "three-width-h200-sharing-budget-admission-v1"
            and admission.get("template_only") is False
            and admission.get("status") == "APPROVED_BUDGET_SLOTS"
            and admission.get("campaign_id") == CAMPAIGN
            and admission.get("ledger_path") == ledger_origin
            and admission.get("reservation_owner") == "task-7"
            and admission.get("sole_launcher") == "01a0f19d-7f2e-739e-8aaf-b4153af2b0d8"
            and hold.get("execution_owner") == admission.get("sole_launcher")
            and admission.get("host") == "H200"
            and admission.get("total_cap_device_seconds") == 32400
            and admission.get("finish_before_epoch") == DEADLINE
            and admission.get("automatic_retry") is False
            and cap == 10800,
            "exact approved H200 owner budget required",
        )
        runs = admission.get("runs", [])
        require(
            len(runs) == 3
            and {r.get("bottleneck_dim") for r in runs} == set(WIDTHS)
            and len({r.get("run_id") for r in runs}) == 3
            and all(
                isinstance(r.get("run_id"), str)
                and r["run_id"]
                and r.get("component") == f"h200-sharing-B{r['bottleneck_dim']}"
                and r.get("devices") == 1
                and r.get("cap_device_seconds") == 10800
                for r in runs
            ),
            "exact H200 three width slots required",
        )
        require(
            any(r["run_id"] == run_id and r["bottleneck_dim"] == width for r in runs),
            "H200 run outside approved slots",
        )
        allowed_ids = {r["run_id"] for r in runs}
        require(
            all(
                r["run_id"] in allowed_ids
                for r in pending + settled
                if str(r.get("component", "")).startswith("h200-sharing-B")
            ),
            "H200 prior or pending run outside immutable three slots",
        )
        scope = {
            k: v
            for k, v in PRODUCTION_WORKLOAD.items()
            if k not in ("scientific_training_updates", "l14")
        }
        scope.update(
            cell="D-N",
            architecture="direct_affine",
            outer_normalization="none",
            l14="only after trained-study freeze",
        )
        require(
            admission.get("scope") == scope, "H200 budget scientific scope mismatch"
        )
        _runtime_device(contract, payload, admission_origin, payload_origin)
    require(
        payload.get("workload")
        == (DIAGNOSTIC_WORKLOAD if diagnostic_only else PRODUCTION_WORKLOAD),
        "approved workload differs from fixed science",
    )
    proposal = read_ref(contract["proposal"])
    require(
        all(
            proposal.get(k) == v
            for k, v in {
                "run_id": run_id,
                "diagnostic_only": diagnostic_only,
                "hard_cap_seconds": cap,
                "binding": binding,
                "workload": payload["workload"],
            }.items()
        ),
        "immutable reviewed proposal scope mismatch",
    )
    manifest = load_prepared(verify_ref(contract["prepared"]))
    require(
        binding_identity(manifest) == binding
        and payload.get("source_inventory") == manifest["source_inventory"],
        "executed prepared/source binding mismatch",
    )
    config = manifest["resolved_config"]
    require(
        manifest.get("synthetic_cpu") is False
        and manifest.get("cell") == "D-N"
        and manifest.get("selected_bottleneck_dim") == width
        and manifest.get("q") == 200
        and manifest.get("batch_size") == 64
        and manifest.get("execution_partition") == partition_policy(16)
        and config.get("codec", {}).get("architecture") == "direct_affine"
        and config.get("codec", {}).get("layernorm") == "none"
        and config.get("codec", {}).get("bottleneck_dim") == width
        and config.get("training", {}).get("batch_size") == 16
        and config.get("training", {}).get("gradient_accumulation") == 4
        and manifest.get("task_template", {}).get("settings", {}).get("batch_size")
        == 16,
        "actual prepared package must be D-N q200 effective64 physical16x4 with task16",
    )
    package = Path(contract["prepared"]["path"]).parent
    for field in ("source_archive", "config", "initialization"):
        expected = {
            **manifest[field],
            "path": str((package / manifest[field]["path"]).resolve()),
        }
        require(
            payload.get(field) == expected,
            "payload " + field + " differs from prepared reference",
        )
        verify_ref(payload[field])
    for field in ("review", "owner_handoff", "expected_panel"):
        read_ref(payload[field])
    for reference in payload["target_assets"].values():
        verify_ref(reference)
    return payload


def _runtime_device(contract, payload, admission_origin, payload_origin):
    policy = payload.get("device_binding", {})
    lease_root = policy.get("lease_root")
    require(
        isinstance(lease_root, str)
        and Path(lease_root).is_absolute()
        and policy
        == {
            "policy": "slurm-one-h200-runtime-uuid-v1",
            "gpu_class": "H200",
            "device_count": 1,
            "lease_root": lease_root,
            "lease_filename_rule": "device-{GPU_UUID}.lock",
            "runtime_receipt_required": True,
        },
        "exact H200 device policy required",
    )
    runtime = read_ref(contract["device_binding_receipt"])
    uuid = contract.get("device_uuid", "")
    require(
        bool(
            re.fullmatch(
                r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", uuid
            )
        ),
        "actual full GPU UUID required",
    )
    lease = str(Path(lease_root) / f"device-{uuid}.lock")
    expected = {
        "schema": "slurm-h200-device-binding-v1",
        "status": "BOUND_FROM_VERIFIED_SLURM_ALLOCATION",
        "run_id": contract["run_id"],
        "campaign_id": CAMPAIGN,
        "component": payload["component"],
        "execution_owner": payload["sole_execution_owner"],
        "payload_receipt": {
            "path": payload_origin,
            "sha256": contract["payload_receipt"]["sha256"],
        },
        "budget_admission": {
            "path": admission_origin,
            "sha256": contract["admission"]["sha256"],
        },
        "device_uuid": uuid,
        "device_lease": lease,
        "finish_before_epoch": DEADLINE,
    }
    require(
        all(runtime.get(k) == v for k, v in expected.items())
        and contract.get("device_lease") == lease
        and runtime.get("device_name") in ("NVIDIA H200", "NVIDIA H200 NVL")
        and isinstance(runtime.get("allocation_evidence"), dict)
        and bool(runtime["allocation_evidence"])
        and isinstance(runtime.get("visibility_evidence"), dict)
        and bool(runtime["visibility_evidence"]),
        "sealed H200 runtime binding mismatch or missing evidence",
    )
    require(
        bool(os.environ.get("SLURM_JOB_ID"))
        and runtime.get("slurm_job_id") == os.environ.get("SLURM_JOB_ID")
        and runtime.get("slurm_array_task_id") == os.environ.get("SLURM_ARRAY_TASK_ID")
        and bool(os.environ.get("SLURMD_NODENAME"))
        and runtime.get("node") == os.environ.get("SLURMD_NODENAME"),
        "runtime receipt does not belong to current Slurm allocation",
    )
