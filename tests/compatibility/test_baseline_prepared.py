"""Prepared CLI gate and allocation-boundary tests with retained source bytes."""

import jscc

import json
from pathlib import Path
import zipfile

import pytest

from jscc.activation_replay import canonical_digest, file_digest
from jscc.config import load_config, save_config
from jscc.checkpoint_readers import validate_saved_baseline_prepared


ROOT = Path(jscc.__file__).resolve().parents[1]


@pytest.fixture
def prepared(tmp_path):
    config = load_config(ROOT / "configs/experiments/ordinary/hellaswag.yaml")
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

        ROOT / "uv.lock",
        ROOT / "pyproject.toml",
    ]
    source = tmp_path/'retained-source'
    source.mkdir()
    import shutil
    for path in paths:
        target = source/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    historical_runner = source/'scripts/encfn_baseline.py'
    historical_runner.parent.mkdir()
    historical_runner.write_text('synthetic historical source fixture; no executable runner')
    paths = [source/path.relative_to(ROOT) for path in paths] + [historical_runner]
    inventory = {str(path.relative_to(source)): file_digest(path) for path in paths}
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in paths:
            bundle.write(path, str(path.relative_to(source)))
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
        validate_saved_baseline_prepared(path, source_root=path.parent/"retained-source")


def test_cell_topology_fails_before_model(prepared, monkeypatch):
    path, manifest, _ = prepared
    manifest["cell"] = "R-none"
    write_manifest(path, manifest)
    forbid_construction(monkeypatch)
    with pytest.raises(ValueError, match="cell topology"):
        validate_saved_baseline_prepared(path, source_root=path.parent/"retained-source")


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
                    else (path.parent / "retained-source" / name).read_bytes(),
                )
        manifest["source_archive"]["sha256"] = file_digest(archive)
    else:
        manifest["state_metadata"][mutation] = "wrong"
    write_manifest(path, manifest)
    forbid_construction(monkeypatch)
    with pytest.raises(ValueError, match=match):
        validate_saved_baseline_prepared(path, source_root=path.parent/"retained-source")
