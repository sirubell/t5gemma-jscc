"""Direct current numerical state guard; no historical target qualification launch."""

import torch
from jscc.qualification import snapshot, restore, state_exact
from shared.test_recipe_training import setup_recipe
import copy
import json
import shutil
from pathlib import Path
import pytest
from jscc import checkpoint_readers as _body_checkpoint_readers
from jscc.activation_replay import file_digest
from jscc.batch_partition import partition_policy
from sharing_fixtures import sharing_model
from jscc.qualification import controller_binding_identity



def test_current_native16_guard_restores_exact_state_and_next_update(tmp_path):
    learner, batch, events = setup_recipe(tmp_path, locations=['enc_fn'], q=8)
    learner.update([batch])
    zero = snapshot(learner)
    learner.update([batch])
    expected = snapshot(learner)
    expected_event = events[-1]['payload']
    with torch.no_grad():
        for parameter in learner.parameters:
            parameter.add_(3)
        for state in learner.optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    value.add_(2)
    learner.scheduler.last_epoch += 5
    torch.manual_seed(999)
    restore(learner, zero)
    assert state_exact(zero, snapshot(learner))
    event = learner.update([batch])['payload']
    actual = snapshot(learner)
    # Event receipts include wall time, while exact numerical continuation does not.
    actual['exposure'].pop('receipts')
    expected['exposure'].pop('receipts')
    assert state_exact(actual, expected)
    for key in ('objective', 'lr_used', 'lr_next', 'snr', 'site_id', 'site_step'):
        assert state_exact(event[key], expected_event[key])
    assert all(parameter.grad is None for parameter in learner.model.base.parameters())
    assert learner.active_site is learner.pending_update is None


def ref(path):
    return {"path": str(path.resolve()), "sha256": file_digest(path)}


@pytest.fixture
def contract_fixture(tmp_path, monkeypatch):
    source_root = tmp_path / "source-tree"
    original = Path(__file__).resolve().parents[1] / "fixtures/saved-qualification-source"
    for name, fixture in (("jscc/sharing_qualification_guard.py", "sharing_qualification_guard.py.txt"),
                          ("scripts/sharing_qualification.py", "sharing_qualification.py.txt")):
        target = source_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / fixture, target)
    model = sharing_model("D-none")
    config = {
        "codec": {**model.codec.config, "bottleneck_dim": 512},
        "model": {
            "name": str(tmp_path / "model"),
            "revision": "fixed",
            "device": "cuda",
            "dtype": "bfloat16",
            "numerical_policy": "native",
            "attn_implementation": "sdpa",
            "local_files_only": True,
        },
    }
    manifest = dict(
        synthetic_cpu=False,
        cell="D-N",
        selected_bottleneck_dim=512,
        execution_partition=partition_policy(16),
        resolved_config=config,
        prepared_identity="prepared",
        source_identity="source",
        config_identity="config",
        model_revision="fixed",
        recipe_identity="recipe",
        architecture_decision="test-decision",
        protocol_id="protocol",
        updates=[],
        validation=[],
        geometry={},
        task_request={"path": "task.json"},
        data_ids={},
        initialization={"state_identity": "initial"},
        hard_cap_seconds=120,
        source_inventory={
            "scripts/sharing_qualification.py": file_digest(
                source_root / "scripts/sharing_qualification.py"
            )
        },
        task_template={
            "settings": {"batch_size": 16},
            "conditions": _body_checkpoint_readers.SHARING_QUALIFICATION_CONDITIONS,
        },
    )
    prepared = tmp_path / "prepared.json"
    prepared.write_text("{}")
    expected = tmp_path / "expected.json"
    expected.write_text("{}")
    contract = dict(
        schema="sharing-no-norm-qualification-v1",
        prepared=ref(prepared),
        expected_panel=ref(expected),
        binding=controller_binding_identity(manifest),
        diagnostic_only=True,
        automatic_retry=False,
        sole_owner="task-7",
        hard_cap_seconds=120,
        output_byte_cap=100000000,
        allocation_root=str(tmp_path),
        run_id="cpu-gates",
        hf_home=str(tmp_path / "hf"),
        finish_before_epoch=_body_checkpoint_readers.SHARING_QUALIFICATION_FINISH_BEFORE_EPOCH,
    )
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    monkeypatch.setattr(_body_checkpoint_readers, "load_prepared", lambda path, *, source_root: manifest)
    return path, contract, manifest


@pytest.mark.parametrize(
    "field,value",
    [
        ("sole_owner", "other"),
        ("hard_cap_seconds", None),
        ("automatic_retry", True),
        ("diagnostic_only", False),
        ("binding", {}),
    ],
)
def test_contract_fail_closed(contract_fixture, field, value):
    path, contract, _ = contract_fixture
    contract[field] = value
    path.write_text(json.dumps(contract))
    with pytest.raises((ValueError, TypeError)):
        _body_checkpoint_readers.validate_saved_qualification_contract(path, source_root=path.parent / "source-tree")


def test_wrong_width_layout_or_norm_rejected(contract_fixture):
    path, contract, manifest = contract_fixture
    for change in ("width", "partition", "norm"):
        changed = copy.deepcopy(manifest)
        if change == "width":
            manifest["selected_bottleneck_dim"] = 64
        elif change == "partition":
            manifest["execution_partition"] = partition_policy(32)
        else:
            manifest["resolved_config"]["codec"]["layernorm"] = "both"
        with pytest.raises(ValueError, match="D-N"):
            _body_checkpoint_readers.validate_saved_qualification_contract(path, source_root=path.parent / "source-tree")
        manifest.clear()
        manifest.update(changed)


def test_target_cpu_config_cannot_supply_gpu_timings(contract_fixture):
    path, _, manifest = contract_fixture
    manifest["resolved_config"]["model"]["device"] = "cpu"
    with pytest.raises(ValueError, match="CUDA/BF16"):
        _body_checkpoint_readers.validate_saved_qualification_contract(path, source_root=path.parent / "source-tree")


def test_complete_receipt_requires_all_panels_and_reloads():
    record = {
        "status": "diagnostic_completed",
        "guard": {
            "status": "passed",
            "physical_updates": 5,
            "checkpoint": {"exact": True},
            "restored_second_update_exact": True,
        },
        "stress_timings": [{}, {}, {}],
        "objective": {
            "sequences": 128,
            "rows": [
                {"condition": c, "completed": 128, "failed": 0}
                for c in _body_checkpoint_readers.SHARING_QUALIFICATION_CONDITIONS
            ],
        },
        "task": [
            {"condition": c, "candidate_forward_requests": 1024}
            for c in _body_checkpoint_readers.SHARING_QUALIFICATION_CONDITIONS
        ],
    }
    result = {
        "status": "diagnostic_completed",
        "completed_updates": 24,
        "diagnostic_only": True,
        "production_qualified": False,
        "binding": {},
        "sites": {site: copy.deepcopy(record) for site in ("enc_l9", "enc_l19", "enc_fn")},
    }
    _body_checkpoint_readers.validate_saved_qualification_result(result, {"binding": {}})
    result["sites"]["enc_l9"]["task"].pop()
    with pytest.raises(ValueError, match="full256task"):
        _body_checkpoint_readers.validate_saved_qualification_result(result, {"binding": {}})


def test_one_owner_campaign_bounds_all_three_widths(contract_fixture):
    path, contract, _ = contract_fixture

    def record(name, value):
        target = path.parent / name
        target.write_text(json.dumps(value))
        return ref(target)

    admission = {
        "owner": "task-7",
        "run_id": contract["run_id"],
        "diagnostic_only": True,
        "binding": contract["binding"],
        "reserved_device_seconds": 120,
        "status": "reserved_once_not_launched",
        "automatic_retry": False,
    }
    contract["admission"] = record("admission.json", admission)
    campaign = {
        "owner": "task-7",
        "diagnostic_only": True,
        "cap_device_seconds": 3600,
        "width_allocations": [
            {
                "width": width,
                "run_id": contract["run_id"] if width == 512 else "width-" + str(width),
                "cap_seconds": 120,
            }
            for width in (512, 1152, 2304)
        ],
    }
    contract["campaign_admission"] = record("campaign.json", campaign)
    proposal = {
        "binding": contract["binding"],
        "diagnostic_only": True,
        "run_id": contract["run_id"],
        "hard_cap_seconds": 120,
    }
    contract["proposal"] = record("proposal.json", proposal)
    ledger = {
        "account": "sharing",
        "ledger_owner": "task-7",
        "charged_device_seconds": 0,
        "cap_device_seconds": 3600,
        "pending_reservations": [
            {
                "run_id": contract["run_id"],
                "max_device_seconds": 120,
                "devices": 1,
                "automatic_retry": False,
                "ledger_owner": "task-7",
                "state": "approved_reserved_not_launched_pending_package_review",
                "proposal_sha256": contract["proposal"]["sha256"],
            }
        ],
        "allocations": [],
    }
    ledger_path = path.parent / "ledger.json"
    ledger_path.write_text(json.dumps(ledger))
    contract["ledger_path"] = str(ledger_path)
    with pytest.raises(ValueError, match="exact task-7 overnight"):
        _body_checkpoint_readers.saved_qualification_admission(contract)
    campaign["width_allocations"][1]["cap_seconds"] = 1201
    contract["campaign_admission"] = record("campaign.json", campaign)
    with pytest.raises(ValueError, match="exact task-7 overnight"):
        _body_checkpoint_readers.saved_qualification_admission(contract)
    campaign["width_allocations"][1]["cap_seconds"] = 120
    campaign["width_allocations"][1]["run_id"] = contract["run_id"]
    contract["campaign_admission"] = record("campaign.json", campaign)
    with pytest.raises(ValueError, match="exact task-7 overnight"):
        _body_checkpoint_readers.saved_qualification_admission(contract)




@pytest.mark.parametrize("defect", ["missing_guard", "wrong_guard", "missing_script", "wrong_script", "current_layout"])
def test_saved_contract_requires_explicit_exact_retained_source(contract_fixture, defect):
    path, _, _ = contract_fixture
    root = path.parent / "source-tree"
    if defect == "current_layout":
        root = path.parent / "empty-current-layout"
        root.mkdir()
    else:
        target = root / ("jscc/sharing_qualification_guard.py" if "guard" in defect else "scripts/sharing_qualification.py")
        if defect.startswith("missing"):
            target.unlink()
        else:
            target.write_text("changed fixture bytes")
    with pytest.raises((ValueError, FileNotFoundError)):
        _body_checkpoint_readers.validate_saved_qualification_contract(path, source_root=root)


def test_saved_contract_has_no_implicit_current_source(contract_fixture):
    path, _, _ = contract_fixture
    with pytest.raises(TypeError, match="source_root"):
        getattr(_body_checkpoint_readers, 'validate_saved_qualification_contract')(path)


def test_saved_contract_accepts_explicit_frozen_reviewed_source_bytes(contract_fixture):
    path, contract, manifest = contract_fixture
    saved, loaded = _body_checkpoint_readers.validate_saved_qualification_contract(
        path, source_root=path.parent/'source-tree')
    assert saved == contract
    assert loaded == manifest


def test_unreviewed_alias_guard_is_rejected_against_original_review_digest(contract_fixture):
    path, _, _ = contract_fixture
    fixture = Path(__file__).resolve().parents[1] / 'fixtures/saved-qualification-source/unreviewed_alias_guard.py.txt'
    target = path.parent / 'source-tree/jscc/sharing_qualification_guard.py'
    target.write_bytes(fixture.read_bytes())
    with pytest.raises(ValueError, match="reviewed numerical guard bytes changed"):
        _body_checkpoint_readers.validate_saved_qualification_contract(path, source_root=path.parent/'source-tree')
