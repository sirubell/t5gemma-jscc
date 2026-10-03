"""CPU-only admission gates and real no-norm diagnostic numerical guard."""

import copy
import json
from pathlib import Path
import socket
import sys

import pytest
import torch

from jscc import sharing_qualification as qualification
from jscc.activation_replay import file_digest
from jscc.baseline_protocol import batch_identity
from jscc.sharing_accumulation import partition_policy
from jscc.sharing_preparation import state_dict_identity
from sharing_fixtures import sharing_model
from test_sharing_accumulation import parent64
from test_shared_lifecycle import task_template


def ref(path):
    return {"path": str(path.resolve()), "sha256": file_digest(path)}


@pytest.fixture
def contract_fixture(tmp_path, monkeypatch):
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
                Path(__file__).resolve().parents[1] / "scripts/sharing_qualification.py"
            )
        },
        task_template={
            "settings": {"batch_size": 16},
            "conditions": qualification.CONDITIONS,
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
        binding=qualification.binding_identity(manifest),
        diagnostic_only=True,
        automatic_retry=False,
        sole_owner="task-7",
        hard_cap_seconds=120,
        output_byte_cap=100000000,
        allocation_root=str(tmp_path),
        run_id="cpu-gates",
        hf_home=str(tmp_path / "hf"),
        finish_before_epoch=qualification.FINISH_BEFORE_EPOCH,
    )
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    monkeypatch.setattr(qualification, "load_prepared", lambda path: manifest)
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
        qualification.validate_contract(path)


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
            qualification.validate_contract(path)
        manifest.clear()
        manifest.update(changed)


def test_exact_guard_bytes():
    assert (
        file_digest(
            Path(qualification.__file__).with_name("sharing_qualification_guard.py")
        )
        == qualification.GUARD_SHA256
    )


def test_preflight_requires_hidden_cuda(contract_fixture, monkeypatch):
    path, _, _ = contract_fixture
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError, match="CUDA hidden"):
        qualification.preflight(path)
    assert not torch.cuda.is_initialized()


def test_target_preflight_missing_before_admission_or_model(
    contract_fixture, monkeypatch
):
    from jscc.models import split_model

    path, _, _ = contract_fixture
    calls = []
    monkeypatch.setattr(split_model, "build_model", lambda *_: calls.append("model"))
    monkeypatch.setattr(
        qualification, "_admission", lambda *_: calls.append("admission")
    )
    with pytest.raises(FileNotFoundError):
        qualification.worker(path)
    assert calls == []


def test_external_watchdog_before_gpu_admission(contract_fixture, monkeypatch):
    path, _, _ = contract_fixture
    calls = []
    monkeypatch.setattr(
        qualification, "_target_receipt", lambda *_: calls.append("target")
    )
    monkeypatch.setattr(
        qualification, "_admission", lambda *_: calls.append("admission")
    )
    with pytest.raises(ValueError, match="watchdog"):
        qualification.controller(path)
    assert calls == []
    assert not (path.parent / "attempt.json").exists()


def test_actual_host_receipt_binds_contract_and_runtime(contract_fixture, monkeypatch):
    path, contract, manifest = contract_fixture
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(
        qualification,
        "_assets",
        lambda *_: {"assets_identity": "assets", "runtime_versions": {"fixture": "1"}},
    )
    result = qualification.preflight(path)
    assert result["hostname"] == socket.gethostname() and result["python"] == str(
        Path(sys.executable).resolve()
    )
    qualification._target_receipt(path, contract, manifest)
    receipt = path.parent / "target-preflight.json"
    result["hostname"] = "other-host"
    receipt.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="actual target"):
        qualification._target_receipt(path, contract, manifest)


def test_missing_actual_admission_before_model(contract_fixture, monkeypatch):
    from jscc.models import split_model

    path, _, _ = contract_fixture
    calls = []
    monkeypatch.setattr(qualification, "_target_receipt", lambda *_: None)
    monkeypatch.setattr(split_model, "build_model", lambda *_: calls.append("model"))
    with pytest.raises(KeyError, match="ledger_path|admission"):
        qualification.worker(path)
    assert calls == []


def test_missing_live_controller_before_model(contract_fixture, monkeypatch):
    from jscc.models import split_model

    path, _, _ = contract_fixture
    calls = []
    monkeypatch.setattr(qualification, "_target_receipt", lambda *_: None)
    monkeypatch.setattr(qualification, "_admission", lambda *_: None)
    monkeypatch.setattr(split_model, "build_model", lambda *_: calls.append("model"))
    with pytest.raises(FileNotFoundError):
        qualification.worker(path)
    assert calls == []


def stress_parent(index):
    parent: dict = parent64(index)
    if index == 1:
        parent["input_ids"] = torch.cat(
            [parent["input_ids"], torch.full((64, 2), 3)], dim=1
        )
        parent["attention_mask"] = torch.cat(
            [parent["attention_mask"], torch.ones((64, 2), dtype=torch.long)], dim=1
        )
        parent["position_roles"] = [
            row + ["query", "query"] for row in parent["position_roles"]
        ]
        parent["token_roles"] = copy.deepcopy(parent["position_roles"])
        parent["token_positions"] = torch.arange(7).repeat(64, 1)
        parent["input_ids"].fill_(3)
        parent["attention_mask"].fill_(1)
        parent["position_roles"] = [["demo"] + ["query"] * 6 for _ in range(64)]
        parent["token_roles"] = copy.deepcopy(parent["position_roles"])
    if index == 2:
        parent["labels"] = torch.cat([parent["labels"], torch.full((64, 2), 4)], dim=1)
        parent["decoder_input_ids"] = torch.cat(
            [
                torch.full((64, 1), 2),
                parent["labels"][:, :-1].masked_fill(
                    parent["labels"][:, :-1] == -100, 0
                ),
            ],
            dim=1,
        )
        parent["labels"].fill_(4)
        parent["decoder_input_ids"].fill_(4)
        parent["decoder_attention_mask"] = parent["labels"] != -100
    return parent


def test_tiny_actual_guard_worker_three_sites(tmp_path, monkeypatch):
    """Only pretrained construction and task scorer replaced; numerical guard is exact copied code."""
    from jscc.models import split_model
    from jscc import evaluation

    torch.set_num_threads(1)
    model = sharing_model("D-none")
    config = {
        "codec": dict(model.codec.config),
        "model": {"device": "cpu", "dtype": "float32", "revision": "tiny-cpu"},
        "training": {"deterministic_algorithms": True},
    }
    template = task_template()
    template["settings"]["batch_size"] = 16
    refs = []
    for index in range(4):
        batch = stress_parent(index)
        target = tmp_path / f"parent{index}.pt"
        torch.save(batch, target)
        refs.append(
            {
                "path": target.name,
                "sha256": file_digest(target),
                "view_sha256": batch_identity(batch),
            }
        )
    initial = copy.deepcopy(model.codec.state_dict())
    manifest = dict(
        synthetic_cpu=True,
        cell="D-N",
        selected_bottleneck_dim=8,
        execution_partition=partition_policy(16),
        resolved_config=config,
        prepared_identity="tiny",
        source_identity="source",
        config_identity="config",
        model_revision="tiny-cpu",
        recipe_identity="recipe",
        architecture_decision="cpu-test-only",
        protocol_id="synthetic",
        updates=refs,
        validation=[refs[0]],
        geometry={},
        task_request={},
        data_ids={},
        initialization_state=initial,
        initialization={"state_identity": state_dict_identity(initial)},
        task_template=template,
        study_pairing_id="cpu-diag",
    )
    monkeypatch.setattr(split_model, "build_model", lambda config: (object(), model))
    panel = {
        "documents": [{"sample_id": 0}, {"sample_id": 1}],
        "prompts": [{"prompt_id": "a"}, {"prompt_id": "b"}],
        "fewshots": [{"fewshot_id": "demo"}],
    }

    def scorer(model, processor, settings, output, condition, data_settings):
        for key, rows in panel.items():
            (output / (key + ".jsonl")).write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
        return {"candidate_forward_requests": 8}

    monkeypatch.setattr(evaluation, "evaluate_hellaswag", scorer)
    result = qualification.run_diagnostic(
        manifest, tmp_path, tmp_path, lambda: None, expected_panel=panel
    )
    assert (
        result["status"] == "diagnostic_completed" and result["completed_updates"] == 24
    )
    assert result["diagnostic_only"] is True and result["production_qualified"] is False
    assert result["stress_production_indices"] == [0, 1, 2]
    assert set(result["sites"]) == {"enc_l9", "enc_l19", "enc_fn"}
    for record in result["sites"].values():
        assert record["guard"]["status"] == "passed"
        assert record["guard"]["checkpoint"]["exact"]
        assert record["guard"]["restored_second_update_exact"]
        assert (
            len(record["stress_timings"]) == 3
            and len(record["objective"]["rows"]) == len(record["task"]) == 6
        )
    assert not torch.cuda.is_initialized()


def test_target_cpu_config_cannot_supply_gpu_timings(contract_fixture):
    path, _, manifest = contract_fixture
    manifest["resolved_config"]["model"]["device"] = "cpu"
    with pytest.raises(ValueError, match="CUDA/BF16"):
        qualification.validate_contract(path)


def test_fast_exit_output_cap_is_terminal_failure(contract_fixture, monkeypatch):
    path, contract, _ = contract_fixture
    lease = path.parent / "lease"
    lease.touch()
    contract["device_lease"] = str(lease)
    contract["output_byte_cap"] = 10000
    path.write_text(json.dumps(contract))
    monkeypatch.setattr(qualification, "_watchdog", lambda *_: None)
    monkeypatch.setattr(qualification, "_target_receipt", lambda *_: None)
    monkeypatch.setattr(qualification, "_admission", lambda *_: None)
    monkeypatch.setattr(qualification, "_cleanup_group", lambda: None)
    monkeypatch.setattr(
        qualification.time, "time", lambda: qualification.FINISH_BEFORE_EPOCH - 2000
    )
    monkeypatch.setattr(
        qualification.subprocess, "check_output", lambda *args, **kwargs: b""
    )

    class Child:
        returncode = 0

        def poll(self):
            (path.parent / "too-large").write_bytes(b"x" * 20000)
            return 0

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(
        qualification.subprocess, "Popen", lambda *args, **kwargs: Child()
    )
    with pytest.raises((RuntimeError, ValueError)):
        qualification.controller(path)
    receipt = json.loads((path.parent / "allocation.json").read_text())
    assert receipt["status"] != "terminal"
    assert receipt["charged_device_seconds"] >= 1
    assert "output byte cap" in receipt["error"]
    assert receipt["cleanup_verified"] is False


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
                for c in qualification.CONDITIONS
            ],
        },
        "task": [
            {"condition": c, "candidate_forward_requests": 1024}
            for c in qualification.CONDITIONS
        ],
    }
    result = {
        "status": "diagnostic_completed",
        "completed_updates": 24,
        "diagnostic_only": True,
        "production_qualified": False,
        "binding": {},
        "sites": {site: copy.deepcopy(record) for site in qualification.SITES},
    }
    qualification.validate_result(result, {"binding": {}})
    result["sites"]["enc_l9"]["task"].pop()
    with pytest.raises(ValueError, match="full256task"):
        qualification.validate_result(result, {"binding": {}})


def test_target_assets_are_hashed_before_preflight_receipt(
    contract_fixture, monkeypatch
):
    import importlib.metadata

    path, contract, manifest = contract_fixture
    model_root = Path(manifest["resolved_config"]["model"]["name"])
    model_root.mkdir()
    names = [
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "hellaswag-train.arrow",
        "hellaswag-validation.arrow",
    ]
    contract["target_assets"] = {}
    for name in names:
        target = model_root / name
        target.write_text("cpu identity fixture")
        contract["target_assets"][name] = ref(target)
    contract["runtime_versions"] = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "datasets", "lm-eval")
    }
    path.write_text(json.dumps(contract))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    receipt = qualification.preflight(path)
    assert (
        receipt["status"] == "cpu_verified"
        and receipt["full_model_constructed"] is False
    )
    (model_root / "model.safetensors").write_text("changed")
    with pytest.raises(ValueError, match="target asset mismatch"):
        qualification._target_receipt(path, contract, manifest)


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
        qualification._admission(contract)
    campaign["width_allocations"][1]["cap_seconds"] = 1201
    contract["campaign_admission"] = record("campaign.json", campaign)
    with pytest.raises(ValueError, match="exact task-7 overnight"):
        qualification._admission(contract)
    campaign["width_allocations"][1]["cap_seconds"] = 120
    campaign["width_allocations"][1]["run_id"] = contract["run_id"]
    contract["campaign_admission"] = record("campaign.json", campaign)
    with pytest.raises(ValueError, match="exact task-7 overnight"):
        qualification._admission(contract)


def test_snapshot_symlink_assets_keep_expected_canonical_file(contract_fixture):
    import importlib.metadata

    _, contract, manifest = contract_fixture
    root = Path(manifest["resolved_config"]["model"]["name"])
    root.mkdir()
    blobs = root.parent / "blobs"
    blobs.mkdir()
    names = [
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "hellaswag-train.arrow",
        "hellaswag-validation.arrow",
    ]
    contract["target_assets"] = {}
    for name in names:
        blob = blobs / name
        blob.write_text("CPU symlink fixture " + name)
        member = root / name
        member.symlink_to(blob)
        contract["target_assets"][name] = {
            "path": str(member),
            "sha256": file_digest(blob),
        }
    contract["runtime_versions"] = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "datasets", "lm-eval")
    }
    assert (
        qualification._assets(contract, manifest)["runtime_versions"]
        == contract["runtime_versions"]
    )
    alternate = root.parent / "other-snapshot"
    alternate.mkdir()
    (alternate / "model.safetensors").write_bytes(
        (blobs / "model.safetensors").read_bytes()
    )
    contract["target_assets"]["model.safetensors"]["path"] = str(
        alternate / "model.safetensors"
    )
    with pytest.raises(ValueError, match="outside exact configured snapshot"):
        qualification._assets(contract, manifest)
