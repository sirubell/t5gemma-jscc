"""Prepared CLI gate and allocation-boundary tests with retained source bytes."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
import torch

from jscc.activation_replay import canonical_digest, file_digest
from jscc.config import load_config, save_config
from jscc.models.codec import Codec
from scripts import encfn_baseline
from test_core import toy_model


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared(tmp_path):
    config = load_config(ROOT / "configs/tasks/hellaswag.yaml")
    config["split"] = {"stack": "enc", "where": "after_final_norm"}
    config["model"].update(device="cpu", dtype="float32")
    config["codec"].update(
        architecture="direct_affine", n_res_blocks=0, layernorm="none"
    )
    config["codec"].pop("memory", None)
    config["training"].update(max_steps=400, schedule_steps=400)
    config_path = tmp_path / "config.yaml"
    save_config(config, config_path)
    paths = [
        *sorted((ROOT / "jscc").rglob("*.py")),
        ROOT / "scripts/encfn_baseline.py",
        ROOT / "uv.lock",
        ROOT / "pyproject.toml",
    ]
    inventory = {str(path.relative_to(ROOT)): file_digest(path) for path in paths}
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in paths:
            bundle.write(path, str(path.relative_to(ROOT)))
    data = {"development": [f"row-{i}" for i in range(256)]}
    updates = [
        [{"path": f"batch-{i}.pt", "sha256": f"prepared-{i}"}] for i in range(400)
    ]
    manifest = {
        "schema": "prepared-encfn-baseline-v2",
        "config": {"path": config_path.name, "sha256": file_digest(config_path)},
        "cell": "D-none",
        "strategy": "both",
        "source_inventory": inventory,
        "source_archive": {"path": archive.name, "sha256": file_digest(archive)},
        "updates": updates,
        "data_ids": data,
        "run_id": "prepared-cpu-gates",
        "pairing_id": "paired-fixture",
        "state_metadata": {
            "source_identity": canonical_digest(inventory),
            "config_identity": canonical_digest(config),
            "stream_identity": canonical_digest(updates),
            "data_identity": canonical_digest(data),
            "parent_identity": None,
            "initialization_identity": "pending",
            "protocol_identity": "baseline-v2",
            "lineage": [],
        },
        "task_request": {"comparison": {}, "expected_items": 256},
    }
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps(manifest))
    return path, manifest, config


def write_manifest(path, manifest):
    path.write_text(json.dumps(manifest))


def rewrite_config(path, manifest, config):
    config_path = path.parent / manifest["config"]["path"]
    save_config(config, config_path)
    manifest["config"]["sha256"] = file_digest(config_path)
    manifest["state_metadata"]["config_identity"] = canonical_digest(
        load_config(config_path)
    )
    write_manifest(path, manifest)


def forbid_construction(monkeypatch):
    def forbidden(_config):
        pytest.fail("rejected preparation must not construct/download a model")

    monkeypatch.setattr("jscc.models.split_model.build_model", forbidden)


@pytest.mark.parametrize(
    "field,value",
    [
        ("lr", 1e-4),
        ("weight_decay", 0),
        ("grad_clip", 0.5),
        ("batch_size", 32),
        ("gradient_accumulation", 2),
    ],
)
def test_bound_wrong_optimizer_or_effective_batch_fails_before_model(
    prepared, monkeypatch, field, value
):
    path, manifest, config = prepared
    config["training"][field] = value
    rewrite_config(path, manifest, config)
    forbid_construction(monkeypatch)
    with pytest.raises(ValueError, match="optimizer/effective batch"):
        encfn_baseline.execute(path, path.parent / "run")


def test_cell_topology_fails_before_model(prepared, monkeypatch):
    path, manifest, _ = prepared
    manifest["cell"] = "R-none"
    write_manifest(path, manifest)
    forbid_construction(monkeypatch)
    with pytest.raises(ValueError, match="cell topology"):
        encfn_baseline.execute(path, path.parent / "run")


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing_source", "inventory must cover"),
        ("source_bytes", "executed source differs"),
        ("source_identity", "source identity"),
        ("config_identity", "config identity"),
        ("archive_checksum", "archive checksum"),
        ("archive_bytes", "retained source bytes"),
        ("data_identity", "data identity"),
        ("stream_identity", "stream identity"),
    ],
)
def test_retained_source_config_data_binding_fails_before_model(
    prepared, monkeypatch, mutation, match
):
    path, manifest, _ = prepared
    if mutation == "missing_source":
        del manifest["source_inventory"]["jscc/config.py"]
    elif mutation == "source_bytes":
        manifest["source_inventory"]["jscc/config.py"] = "bad"
    elif mutation == "archive_checksum":
        (path.parent / "source.zip").write_bytes(b"truncated")
    elif mutation == "archive_bytes":
        archive = path.parent / "source.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            for name in manifest["source_inventory"]:
                bundle.writestr(
                    name,
                    b"changed"
                    if name == "jscc/config.py"
                    else (ROOT / name).read_bytes(),
                )
        manifest["source_archive"]["sha256"] = file_digest(archive)
    else:
        manifest["state_metadata"][mutation] = "wrong"
    write_manifest(path, manifest)
    forbid_construction(monkeypatch)
    with pytest.raises(ValueError, match=match):
        encfn_baseline.execute(path, path.parent / "run")


def test_actual_tiny_model_rejects_wrong_comparison_controls(prepared, monkeypatch):
    path, manifest, config = prepared
    model = toy_model(where="after_final_norm")
    model.split.pop("index")
    model.base.enc.config = SimpleNamespace(hidden_size=8)
    model.codec = Codec(8, copy.deepcopy(config["codec"]))
    initial = path.parent / "initialization.pt"
    torch.save(model.codec.state_dict(), initial)
    manifest["initialization"] = {"path": initial.name, "sha256": file_digest(initial)}
    manifest["state_metadata"]["initialization_identity"] = file_digest(initial)
    manifest["task_request"]["comparison"] = {"architecture": "wrong-control"}
    write_manifest(path, manifest)
    constructions = []

    def build(actual_config):
        constructions.append(actual_config)
        return None, model

    monkeypatch.setattr("jscc.models.split_model.build_model", build)
    with pytest.raises(ValueError, match="task comparison"):
        encfn_baseline.execute(path, path.parent / "run")
    assert len(constructions) == 1
    assert not (path.parent / "run").exists()


def add_allocation(path, manifest, config):
    config["model"]["device"] = "cuda"
    rewrite_config(path, manifest, config)
    metadata = manifest["state_metadata"]
    measured = {
        "campaign_journal": str((path.parent / "campaign.json").resolve()),
        "max_duration_seconds": 60,
        "mandatory_reserve_device_seconds": 120,
        "devices": 2,
        "identities": {
            "source": metadata["source_identity"],
            "config": metadata["config_identity"],
            "input": metadata["stream_identity"],
        },
    }
    measurement = path.parent / "measurement.json"
    measurement.write_text(json.dumps(measured))
    manifest["allocation"] = {
        **measured,
        "campaign_id": "cpu-fixture-only",
        "command_id": "attempt-1",
        "stage": "setup-through-save",
        "cap_device_seconds": 7200,
        "measurement": {"path": measurement.name, "sha256": file_digest(measurement)},
    }
    write_manifest(path, manifest)


@pytest.mark.parametrize("failed", [False, True])
def test_allocation_starts_before_command_and_charges_terminal_outcome(
    prepared, monkeypatch, failed
):
    path, manifest, config = prepared
    add_allocation(path, manifest, config)
    output = path.parent / "run"
    ledger_path = Path(str(output) + ".allocation.json")
    calls = []

    def command(actual_path, actual_output, resource_guard):
        assert actual_path == path and actual_output == output
        state = json.loads(ledger_path.read_text())
        assert state["allocations"][0]["status"] == "allocated"
        assert state["allocations"][0]["devices"] == 2
        calls.append("command")
        resource_guard()
        if failed:
            raise RuntimeError("construction failed")
        # Boundary stub deliberately produces no scientific evidence.
        actual_output.mkdir()
        (actual_output / "metrics.jsonl").write_text("")
        return {"boundary_only": True}

    monkeypatch.setattr(encfn_baseline, "_execute", command)
    if failed:
        with pytest.raises(RuntimeError, match="construction failed"):
            encfn_baseline.execute(path, output)
    else:
        # This assertion isolates resource wrapping; completion is tested below.
        monkeypatch.setattr(
            encfn_baseline, "bind_allocation_result", lambda *_: {"status": "complete"}
        )
        assert encfn_baseline.execute(path, output) == {
            "boundary_only": True,
            "status": "complete",
        }
    state = json.loads(ledger_path.read_text())
    allocation = state["allocations"][0]
    assert allocation["status"] == ("failed" if failed else "completed")
    assert allocation["charged_device_seconds"] > 0
    assert allocation["charged_device_seconds"] == 2 * allocation["elapsed_seconds"]
    with pytest.raises((ValueError, FileExistsError)):
        encfn_baseline.execute(path, output)
    assert calls == ["command"]
    if failed:
        journal = Path(manifest["allocation"]["campaign_journal"])
        before = journal.read_bytes()
        with pytest.raises(ValueError, match="exact current prior journal"):
            encfn_baseline.execute(path, path.parent / "unchanged-manifest-new-output")
        assert journal.read_bytes() == before
        manifest["allocation"]["prior_ledger"] = {
            "path": journal.name,
            "sha256": file_digest(journal),
        }
        write_manifest(path, manifest)
        with pytest.raises(ValueError, match="cannot automatically continue"):
            encfn_baseline.execute(path, path.parent / "replacement")
        assert calls == ["command"]


def test_command_success_without_records_is_still_incomplete(prepared, monkeypatch):
    path, manifest, config = prepared
    add_allocation(path, manifest, config)
    output = path.parent / "empty-command"

    def command(_path, actual_output, resource_guard):
        resource_guard()
        actual_output.mkdir()
        return {"status": "complete"}

    monkeypatch.setattr(encfn_baseline, "_execute", command)
    with pytest.raises(
        RuntimeError, match="allocation-bound records remain incomplete"
    ):
        encfn_baseline.execute(path, output)
    assert (
        json.loads((output / "completion.json").read_text())["status"] == "incomplete"
    )
    assert (
        json.loads((output / "allocation.json").read_text())["allocations"][0]["status"]
        == "completed"
    )


@pytest.mark.parametrize("inherited", [False, True])
def test_duplicate_prior_campaign_command_rejected_before_new_allocation(
    prepared, monkeypatch, inherited
):
    from jscc.experiment_schedule import AllocationLedger

    path, manifest, config = prepared
    add_allocation(path, manifest, config)
    declaration = manifest["allocation"]
    prior_path = path.parent / "prior.json"
    prior = AllocationLedger(
        prior_path,
        declaration["campaign_id"],
        7200,
        prior_command_ids=["attempt-1"] if inherited else [],
    )
    prior.start(
        "intermediate-command" if inherited else "attempt-1",
        stage="setup",
        devices=declaration["devices"],
        max_duration_seconds=60,
        mandatory_reserve_device_seconds=120,
        measurement_receipt=declaration["measurement"]["sha256"],
        identities=declaration["identities"],
    )
    prior.stop("intermediate-command" if inherited else "attempt-1")
    journal = Path(declaration["campaign_journal"])
    journal.write_text(
        json.dumps(
            {
                "schema": "baseline-campaign-journal-v1",
                "status": "complete",
                "campaign_id": declaration["campaign_id"],
                "ledger": prior.snapshot(),
            }
        )
    )
    declaration["prior_ledger"] = {
        "path": journal.name,
        "sha256": file_digest(journal),
    }
    write_manifest(path, manifest)

    def forbidden(*_):
        pytest.fail("duplicate campaign command must not execute")

    monkeypatch.setattr(encfn_baseline, "_execute", forbidden)
    output = path.parent / "new-output"
    with pytest.raises(ValueError, match="duplicate campaign command"):
        encfn_baseline.execute(path, output)
    assert not Path(str(output) + ".allocation.json").exists()


@pytest.mark.parametrize("rejection", ["omitted", "stale", "claim"])
def test_stable_campaign_journal_blocks_unproven_continuation(
    prepared, monkeypatch, rejection
):
    path, manifest, config = prepared
    add_allocation(path, manifest, config)
    declaration = manifest["allocation"]
    journal = Path(declaration["campaign_journal"])
    journal.write_text(
        json.dumps(
            {
                "schema": "baseline-campaign-journal-v1",
                "campaign_id": declaration["campaign_id"],
                "status": "complete",
            }
        )
    )
    original = journal.read_bytes()
    if rejection == "stale":
        declaration["prior_ledger"] = {"path": journal.name, "sha256": "0" * 64}
    if rejection == "claim":
        Path(str(journal) + ".claim").mkdir()
    write_manifest(path, manifest)

    def forbidden(*_):
        pytest.fail("unproven campaign continuation must not execute")

    monkeypatch.setattr(encfn_baseline, "_execute", forbidden)
    output = path.parent / "new-output"
    error, match = (
        (FileExistsError, None)
        if rejection == "claim"
        else (
            ValueError,
            "stale" if rejection == "stale" else "exact current prior journal",
        )
    )
    with pytest.raises(error, match=match):
        encfn_baseline.execute(path, output)
    assert journal.read_bytes() == original
    assert not Path(str(output) + ".allocation.json").exists()
    assert Path(str(journal) + ".claim").exists() == (rejection == "claim")


def test_exact_prior_journal_carries_charge_and_old_commands(prepared, monkeypatch):
    path, manifest, config = prepared
    add_allocation(path, manifest, config)
    declaration = manifest["allocation"]
    journal = Path(declaration["campaign_journal"])
    starts = []

    def command(_path, output, resource_guard):
        resource_guard()
        starts.append(json.loads(journal.read_text())["ledger"])
        return {"boundary_only": True}

    monkeypatch.setattr(encfn_baseline, "_execute", command)
    monkeypatch.setattr(
        encfn_baseline, "bind_allocation_result", lambda *_: {"status": "complete"}
    )
    encfn_baseline.execute(path, path.parent / "first")
    first = json.loads(journal.read_text())
    previous_charge = first["ledger"]["charged_device_seconds"]
    assert previous_charge > 0
    for number in (2, 3):
        previous_hash = file_digest(journal)
        declaration["prior_ledger"] = {"path": journal.name, "sha256": previous_hash}
        declaration["command_id"] = f"attempt-{number}"
        write_manifest(path, manifest)
        encfn_baseline.execute(path, path.parent / f"run-{number}")
        current = json.loads(journal.read_text())
        assert (
            current["status"] == "complete" and current["prior_sha256"] == previous_hash
        )
        assert current["ledger"]["prior_device_seconds"] == previous_charge
        assert current["ledger"]["prior_command_ids"] == [
            f"attempt-{i}" for i in range(1, number)
        ]
        assert starts[-1]["prior_device_seconds"] == previous_charge
        assert current["ledger"]["charged_device_seconds"] > previous_charge
        assert current["ledger"]["remaining_device_seconds"] == pytest.approx(
            7200 - current["ledger"]["charged_device_seconds"]
        )
        previous_charge = current["ledger"]["charged_device_seconds"]
    assert len(starts) == 3


def test_real_core_stays_incomplete_when_allocation_binding_fails(
    prepared, monkeypatch
):
    from jscc.baseline_protocol import run_baseline
    from jscc.experiment_records import completion_status, read_events
    from test_baseline_pipeline import setup_pipeline

    path, manifest, config = prepared
    add_allocation(path, manifest, config)
    learner, metadata, request, batch, assess = setup_pipeline(monkeypatch)
    output = path.parent / "real-core"
    calls = []

    def command(_path, actual_output, resource_guard):
        calls.append("command")
        return run_baseline(
            learner,
            output=actual_output,
            metadata=metadata,
            update_batches=lambda *_: [batch],
            validation_batches=lambda _: [batch],
            assess=assess,
            task_request=request,
            resource_guard=resource_guard,
            allocation_required=True,
            on_output_created=getattr(resource_guard, "on_output_created", None),
        )

    def broken_binding(*_):
        raise OSError("injected allocation binding failure")

    monkeypatch.setattr(encfn_baseline, "_execute", command)
    monkeypatch.setattr(encfn_baseline, "bind_allocation_result", broken_binding)
    with pytest.raises(OSError, match="allocation binding failure"):
        encfn_baseline.execute(path, output)
    saved = json.loads((output / "completion.json").read_text())
    actual_manifest = json.loads((output / "manifest.json").read_text())
    assert saved["status"] == "incomplete"
    assert "allocation.json" in actual_manifest["expected_artifacts"]
    verified = completion_status(
        actual_manifest,
        read_events(output / "metrics.jsonl"),
        json.loads((output / "inventory.json").read_text()),
        output,
    )
    assert verified["status"] == "incomplete" and verified["missing_artifacts"] == [
        "allocation.json"
    ]
    assert learner.completed == 4
    journal = Path(manifest["allocation"]["campaign_journal"])
    retained = json.loads(journal.read_text())
    assert retained["status"] == "failed"
    assert retained["ledger"]["charged_device_seconds"] > 0
    with pytest.raises(ValueError, match="exact current prior journal"):
        encfn_baseline.execute(path, path.parent / "fresh-output")
    assert calls == ["command"]


def test_existing_output_remains_byte_identical_before_any_allocation(prepared, monkeypatch):
    path, manifest, config = prepared
    add_allocation(path, manifest, config)
    allocation = manifest["allocation"]
    output = path.parent / "existing"
    (output / "nested").mkdir(parents=True)
    (output / "completion.json").write_text('{"status":"complete","prior":true}')
    (output / "nested/evidence.bin").write_bytes(b"prior immutable evidence")
    before = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    monkeypatch.setattr(encfn_baseline, "_execute", lambda *_: pytest.fail("must reject before execution"))
    with pytest.raises(FileExistsError, match="fresh"):
        encfn_baseline.execute(path, output)
    assert before == {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    assert not Path(allocation["campaign_journal"]).exists()
    assert not Path(str(output) + ".allocation.json").exists()


def test_output_created_by_other_actor_during_preparation_is_untouched(prepared, monkeypatch):
    path, manifest, config = prepared
    add_allocation(path, manifest, config)
    allocation = manifest["allocation"]
    output = path.parent / "raced-output"
    prior = b'{"status":"complete","other_actor":true}'
    def race(_path, destination, guard):
        destination.mkdir()
        (destination / "completion.json").write_bytes(prior)
        raise ValueError("source validation rejected after another actor created output")
    monkeypatch.setattr(encfn_baseline, "_execute", race)
    with pytest.raises(ValueError, match="source validation"):
        encfn_baseline.execute(path, output)
    assert list(output.iterdir()) == [output / "completion.json"]
    assert (output / "completion.json").read_bytes() == prior
    assert json.loads(Path(allocation["campaign_journal"]).read_text())["status"] == "failed"
