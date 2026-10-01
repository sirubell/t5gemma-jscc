"""Verified, immutable input package for the finite sharing lifecycle.

Preparation does not construct models, fetch data, select an architecture, or
approve a GPU run. Runtime construction belongs to the connected executor.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import zipfile

import torch
import yaml

from .activation_replay import canonical_digest, file_digest
from .baseline_protocol import read_prepared_batch
from .config import load_config
from .presentation import tensor_digest
from .sharing_protocol import batch_view
from .sharing_schedule import SharingPlan
from .sharing_accumulation import partition_policy

SCHEMA = "sharing-prepared-v1"
CELLS = {"D-N": ("direct_affine", "none"), "D-LN": ("direct_outer_ln", "both"), "R-LN": ("residual_mlp", "both"),
         "T-LN": ("two_linear_gelu", "both")}


def state_dict_identity(state):
    if not isinstance(state, dict) or not state or any(
        not isinstance(key, str) or not torch.is_tensor(value)
        for key, value in state.items()
    ):
        raise ValueError("initialization must contain a nonempty codec tensor state")
    return canonical_digest({key: tensor_digest(value) for key, value in state.items()})


def source_inventory(source_root=None):
    root = Path(source_root or Path(__file__).resolve().parents[1]).resolve()
    names = {str(path.relative_to(root)) for path in (root / "jscc").rglob("*.py")}
    names |= {"scripts/shared_codec.py", "scripts/sharing_production.py", "scripts/sharing_qualification.py",
              "scripts/render_experiment_report.py",
              "uv.lock", "pyproject.toml"}
    return {name: file_digest(root / name) for name in sorted(names)}


def _reference(root, reference):
    if not isinstance(reference, dict) or not {"path", "sha256"} <= reference.keys():
        raise ValueError("immutable file reference required")
    name = reference["path"]
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ValueError("prepared references must be relative")
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("prepared reference escapes package or is missing")
    if file_digest(path) != reference["sha256"]:
        raise ValueError(f"prepared byte checksum mismatch: {name}")
    return path


def write_prepared(path, manifest):
    """Exclusively publish a complete supplied package manifest with its digest.

    All referenced inputs must already exist; verification precedes publication.
    No missing scientific identity or candidate is inferred by this function.
    """
    path = Path(path).resolve()
    value = dict(manifest)
    value.pop("manifest_identity", None)
    value["manifest_identity"] = canonical_digest(value)
    # Source root is always the code actually running this writer.
    _verify(value, path.parent)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path


def load_prepared(path, *, source_root=None):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    result = _verify(manifest, path.parent, source_root=source_root)
    result["prepared_path"] = str(path)
    return result


def _verify(manifest, root, *, source_root=None):
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ValueError("sharing-prepared-v1 manifest required")
    original = {key: value for key, value in manifest.items() if key != "manifest_identity"}
    if manifest.get("manifest_identity") != canonical_digest(original):
        raise ValueError("manifest identity mismatch")
    required = {"cell", "synthetic_cpu", "q", "batch_size", "protocol_id", "model_revision",
                "study_pairing_id", "config", "config_identity", "updates", "validation",
                "task_request", "geometry", "source_inventory", "source_identity", "source_archive",
                "initialization", "data_ids", "recipe_identity", "architecture_decision", "hard_cap_seconds"}
    if not required <= manifest.keys():
        raise ValueError("incomplete sharing preparation")
    cell, synthetic = manifest["cell"], manifest["synthetic_cpu"]
    if cell not in CELLS or type(synthetic) is not bool:
        raise ValueError("one explicit retained candidate and synthetic marker required")
    q, size = manifest["q"], manifest["batch_size"]
    partition = manifest.get("execution_partition")
    if partition is not None and (not isinstance(partition, dict) or partition != partition_policy(partition.get("microbatch_size"))):
        raise ValueError("explicit supported execution partition required")
    expected_size = 64 if partition is not None or not synthetic else 2
    if type(q) is not int or type(size) is not int or (q, size) != (4 if synthetic else 200, expected_size):
        raise ValueError("production requires q200/native64; synthetic CPU requires q4/batch2")
    for key in ("protocol_id", "model_revision", "study_pairing_id", "recipe_identity", "architecture_decision"):
        if not isinstance(manifest[key], str) or not manifest[key].strip():
            raise ValueError(f"explicit {key} required")
    cap = manifest["hard_cap_seconds"]
    if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap) or cap <= 0:
        raise ValueError("finite positive hard cap required")
    inventory = source_inventory(source_root)
    if manifest["source_inventory"] != inventory or manifest["source_identity"] != canonical_digest(inventory):
        raise ValueError("executed source inventory/identity differs from preparation")
    archive = _reference(root, manifest["source_archive"])
    with zipfile.ZipFile(archive) as bundle:
        if len(bundle.namelist()) != len(inventory) or set(bundle.namelist()) != set(inventory):
            raise ValueError("source archive must contain exact declared inventory")
        for name, digest in inventory.items():
            if hashlib.sha256(bundle.read(name)).hexdigest() != digest:
                raise ValueError("retained source archive bytes differ")
    config_path = _reference(root, manifest["config"])
    # A resolved snapshot cannot hide an unbound model-design include.
    if "model_config" in yaml.safe_load(config_path.read_text()):
        raise ValueError("prepared config must be a complete resolved snapshot")
    config = load_config(config_path)
    if canonical_digest(config) != manifest["config_identity"]:
        raise ValueError("resolved config identity mismatch")
    _config_contract(config, manifest)
    updates = manifest["updates"]
    if not isinstance(updates, list) or len(updates) != q:
        raise ValueError("complete ordered update stream required")
    views = []
    for reference in updates:
        _reference(root, reference)
        view = batch_view(read_prepared_batch(reference, root))
        if view.sequences != size:
            raise ValueError("update differs from declared native batch size")
        views.append(view)
    SharingPlan(tuple(views), manifest["study_pairing_id"], synthetic=synthetic)
    validation = _batch_group(root, manifest["validation"])
    if sum(view.sequences for view in validation) != (2 if synthetic else 128):
        raise ValueError("objective validation panel count mismatch")
    geometry = manifest["geometry"]
    if not isinstance(geometry, dict) or set(geometry) != {"train", "selection"}:
        raise ValueError("geometry requires train and selection references; no heldout capture")
    for partition in geometry.values():
        group = _batch_group(root, partition)
        if sum(view.sequences for view in group) != (2 if synthetic else 128):
            raise ValueError("geometry panel count mismatch")
    _geometry_contract(root, geometry)
    task_template = json.loads(_reference(root, manifest["task_request"]).read_text())
    validate_task_template(task_template, config, synthetic=synthetic)
    if not synthetic:
        _production_data_contract(root, manifest, task_template)
    initialization = manifest["initialization"]
    state = torch.load(_reference(root, initialization), map_location="cpu", weights_only=True)
    if state_dict_identity(state) != initialization.get("state_identity"):
        raise ValueError("initial codec state identity mismatch")
    acceptance = manifest.get("gpu_acceptance")
    if acceptance is not None:
        _acceptance_receipt(root, manifest)
    return {**manifest, "resolved_config": config, "task_template": task_template,
            "initialization_state": state, "prepared_identity": manifest["manifest_identity"]}


def _batch_group(root, references):
    if not isinstance(references, list) or not references:
        raise ValueError("nonempty batch reference panel required")
    result = []
    for reference in references:
        _reference(root, reference)
        result.append(batch_view(read_prepared_batch(reference, root)))
    return result


def _config_contract(config, manifest):
    synthetic = manifest["synthetic_cpu"]
    codec, model, training = config["codec"], config["model"], config["training"]
    if (codec["architecture"], codec["layernorm"]) != CELLS[manifest["cell"]]:
        raise ValueError("prepared candidate/config topology mismatch")
    if (config["task"] != "hellaswag" or config["seed"] != 0
            or config["split"] != {"stack": "enc", "where": "after_final_norm"}
            or model["revision"] != manifest["model_revision"]
            or model.get("numerical_policy", "native") != "native"):
        raise ValueError("config violates sharing task/seed/site/model contract")
    if synthetic and (model["device"] != "cpu" or model["dtype"] != "float32"):
        raise ValueError("synthetic fixture must use CPU/float32")
    partition = manifest.get("execution_partition")
    width = codec["bottleneck_dim"]
    if type(width) is not int or width <= 0:
        raise ValueError("positive integer bottleneck required")
    if not synthetic:
        if (codec.get("hidden_dim") != 1152 or codec.get("activation") != "gelu"
                or codec.get("n_res_blocks") != (2 if manifest["cell"] == "R-LN" else 0)):
            raise ValueError("production codec differs from reviewed D-N/D-LN/R-LN/T-LN topology")
        if manifest.get("selected_bottleneck_dim") != width or width not in (512, 1152, 2304):
            raise ValueError("production requires explicit reviewed bottleneck binding")
        if partition is None:
            raise ValueError("production requires explicit qualified execution partition")
    if partition is None:
        microbatch, accumulation = manifest["batch_size"], 1
    else:
        if partition != partition_policy(partition.get("microbatch_size")) or manifest["batch_size"] != 64:
            raise ValueError("partition requires parent effective64")
        if manifest.get("selected_bottleneck_dim") != codec["bottleneck_dim"]:
            raise ValueError("explicit selected bottleneck binding required")
        microbatch, accumulation = partition["microbatch_size"], partition["gradient_accumulation"]
    if (training["batch_size"] != microbatch or training["gradient_accumulation"] != accumulation
            or training["lr"] != 2e-4 or training["weight_decay"] != .01
            or training["grad_clip"] != 1 or training["temperature"] != 1
            or training["kl_weight"] != 1 or training["mse_weight"] != .1):
        raise ValueError("config differs from accepted native-batch optimizer/objective")
    if (codec["snr_film"] or codec["dropout"] != 0 or config["channel"]["type"] != "awgn"
            or not config["channel"]["normalize_power"]):
        raise ValueError("sharing requires dropout0/FiLMoff/masked normalized AWGN")


def _acceptance_receipt(root, manifest):
    receipt = json.loads(_reference(root, manifest["gpu_acceptance"]).read_text())
    if not isinstance(receipt, dict) or receipt.get("status") != "accepted" or any(
        receipt.get(key) != manifest[key]
        for key in ("source_identity", "config_identity", "recipe_identity", "model_revision")
    ):
        raise ValueError("runtime acceptance is not bound to this preparation")
    return receipt


def require_execution_ready(manifest):
    if not manifest["synthetic_cpu"] or os.environ.get("SHARING_CONTROLLER_CONTRACT"):
        from .sharing_controller import require_worker_authorization
        require_worker_authorization(manifest)


def validate_task_template(template, config, *, synthetic):
    required = {"task", "expected_items", "conditions", "input_ids", "source_family_ids",
                "prompt_policy", "noise", "layout", "backend", "precision", "scorer",
                "reference_corpus", "settings", "data_settings"}
    if not isinstance(template, dict) or not required <= template.keys():
        raise ValueError("task template requires complete static observation fields")
    count = template["expected_items"]
    if type(count) is not int or count != (2 if synthetic else 256) or template["task"] != "hellaswag":
        raise ValueError("task request must bind the fixed HellaSwag panel")
    if template["conditions"] != ["no_noise", -6, 0, 6, 12, 18]:
        raise ValueError("task request must retain all six conditions")
    ids, families = template["input_ids"], template["source_family_ids"]
    if (not isinstance(ids, list) or not isinstance(families, list)
            or len(ids) != count or len(families) != count
            or any(type(value) not in (str, int) or value == "" for value in [*ids, *families])
            or len(set(ids)) != count):
        raise ValueError("task requires unique input IDs and actual source-family IDs")
    for field in ("prompt_policy", "layout", "backend", "precision", "scorer", "reference_corpus"):
        if not template[field]:
            raise ValueError(f"task template identity requires {field}")
    noise = template["noise"]
    if (not isinstance(noise, dict) or type(noise.get("seed")) is not int
            or not isinstance(noise.get("namespace"), str) or not noise["namespace"].strip()):
        raise ValueError("task requires explicit noise seed and namespace")
    settings = template["settings"]
    if (not isinstance(settings, dict) or type(settings.get("num_samples")) is not int
            or settings["num_samples"] != count or settings.get("numerical_policy", "native") != "native"
            or not isinstance(template["data_settings"], dict)):
        raise ValueError("task settings require exact sample count and native numerical policy")
    dtype = config["model"]["dtype"]
    alias = {"float32": "fp32", "bfloat16": "bf16", "float16": "fp16"}.get(dtype)
    if template["precision"] not in (alias, f"torch.{dtype}"):
        raise ValueError("task precision differs from configured backbone")
    backend = config["model"].get("attn_implementation")
    if backend is not None and template["backend"] != backend:
        raise ValueError("task backend differs from configured backbone")
    if settings.get("attention_backend", template["backend"]) != template["backend"]:
        raise ValueError("task settings and declared backend differ")


def _geometry_contract(root, geometry):
    from .sharing_geometry_capture import _batch_metadata

    queries = {}
    for partition, references in geometry.items():
        seen = set()
        queries[partition] = set()
        for reference in references:
            batch = read_prepared_batch(reference, root)
            required = {"query_ids", "view_ids", "token_positions", "token_roles"}
            if not required <= batch.keys():
                raise ValueError("geometry batch requires query/view/token positions and roles")
            _batch_metadata(batch)
            pairs = set(zip(batch["query_ids"], batch["view_ids"]))
            if seen & pairs:
                raise ValueError("duplicate geometry query/view across batches")
            seen |= pairs
            queries[partition].update(batch["query_ids"])
    if queries["train"] & queries["selection"]:
        raise ValueError("geometry training and selection queries must be disjoint")


def preview(manifest):
    return {"schema": SCHEMA, "cell": manifest["cell"], "synthetic_cpu": manifest["synthetic_cpu"],
            "physical_updates": manifest["q"] * 6, "learned_saves": 16, "initializations": 1,
            "task_panels": 108, "vanilla_panels": 1, "objective_panels": 192,
            "gpu_execution_ready": False,
            "prepared_identity": manifest["prepared_identity"]}


def _production_data_contract(root, manifest, task_template):
    """Bind production roles to retained v2 producers and actual native views.

    Ordered manifest references may repeat a producer view. The producer's
    unique view inventory follows their first-occurrence order. Dataset row IDs
    are scoped by their existing train/validation/development partitions.
    """
    from .activation_replay import _validate_producer_v2, sequence_view_v2

    role_references = {
        "optimization": manifest["updates"],
        "objective_validation": manifest["validation"],
        "geometry_training": manifest["geometry"]["train"],
        "geometry_selection": manifest["geometry"]["selection"],
    }
    producers = manifest.get("producers")
    if not isinstance(producers, dict) or set(producers) != set(role_references):
        raise ValueError("production requires all four role-bound v2 producer references")
    ids = manifest["data_ids"]
    required_ids = ("train_rows", "validation_rows", "objective_rows")
    if not isinstance(ids, dict) or any(key not in ids for key in required_ids):
        raise ValueError("production data_ids require explicit train/validation/objective/development rows")
    for key in required_ids:
        rows = ids[key]
        if (not isinstance(rows, list) or not rows or any(type(row) is not int or row < 0 for row in rows)
                or len(set(rows)) != len(rows)):
            raise ValueError("data role memberships must contain unique nonnegative row IDs")
    catalogs = {}
    for partition in ("train", "validation"):
        families = ids.get(f"{partition}_source_family_ids")
        rows = ids[f"{partition}_rows"]
        if (not isinstance(families, list) or len(families) != len(rows)
                or any(not isinstance(family, str) or not family for family in families)):
            raise ValueError("complete row-aligned training/selection source-family catalogs required")
        catalogs[partition] = dict(zip(rows, families))
    if set(ids["train_rows"]) & set(ids["validation_rows"]):
        raise ValueError("optimization and selection row pools overlap")
    if not set(ids["objective_rows"]) <= set(ids["validation_rows"]):
        raise ValueError("objective rows must belong to the selection pool")
    development = ids.get("development")
    if (not isinstance(development, dict)
            or development.get("input_ids") != task_template["input_ids"]
            or development.get("source_family_ids") != task_template["source_family_ids"]):
        raise ValueError("development membership differs from actual task panel")
    role_rows, role_families, role_demos, specs = {}, {}, {}, {}
    for role, references in role_references.items():
        spec = json.loads(_reference(root, producers[role]).read_text())
        _validate_producer_v2(spec)
        if (spec["data_role"] != role or spec["data"]["task"] != "hellaswag"
                or spec["backbone"]["model_revision"] != manifest["model_revision"]):
            raise ValueError("producer task/model/data role differs from preparation")
        by_view = {view["batch_view_id"]: view for view in spec["views"]}
        ordered_views, rows, families, demos = [], [], set(), set()
        for reference in references:
            _reference(root, reference)
            batch = read_prepared_batch(reference, root)
            try:
                view = sequence_view_v2(batch)
            except (KeyError, AttributeError, TypeError) as error:
                raise ValueError("production batch lacks complete native v2 view provenance") from error
            view_id = view["batch_view_id"]
            if view != by_view.get(view_id):
                raise ValueError("actual ordered batch differs from role-bound producer view")
            if view_id not in ordered_views:
                ordered_views.append(view_id)
            batch_rows = batch["row_ids"].tolist()
            count = len(batch["input_ids"])
            if (not isinstance(batch_rows, list) or len(batch_rows) != count
                    or any(type(row) is not int or row < 0 for row in batch_rows)
                    or any(len(view[key]) != count for key in
                           ("view_ids", "source_family_ids", "demo_ids", "source_views"))):
                raise ValueError("producer provenance must bind every native sequence")
            if any(not isinstance(family, str) or not family for family in view["source_family_ids"]):
                raise ValueError("actual source families are required")
            catalog = catalogs["train" if role in ("optimization", "geometry_training") else "validation"]
            if any(catalog.get(row) != family for row, family in zip(batch_rows, view["source_family_ids"])):
                raise ValueError("actual query row/source family differs from complete role catalog")
            for row, source_view, demo_ids in zip(batch_rows, view["source_views"], view["demo_ids"]):
                demo_families = source_view.get("demo_source_family_ids")
                if (source_view.get("row_id") != row or source_view.get("native_prefix") != "reference"
                        or not isinstance(demo_families, list) or not isinstance(demo_ids, list)
                        or len(demo_families) != len(demo_ids)
                        or any(not isinstance(family, str) or not family for family in demo_families)):
                    raise ValueError("native source-view row/demo provenance mismatch")
                if any(type(demo_id) is not int or catalogs["train"].get(demo_id) != family
                       for demo_id, family in zip(demo_ids, demo_families)):
                    raise ValueError("demonstration row/source family differs from training catalog")
                demos.update(demo_families)
            rows.extend(batch_rows)
            families.update(view["source_family_ids"])
        if ordered_views != list(by_view):
            raise ValueError("producer view inventory differs from ordered prepared stream")
        role_rows[role], role_families[role], role_demos[role] = rows, families, demos
        specs[role] = spec
    if (not set(role_rows["optimization"]) <= set(ids["train_rows"])
            or role_rows["objective_validation"] != ids["objective_rows"]
            or not set(role_rows["geometry_training"]) <= set(ids["train_rows"])
            or not set(role_rows["geometry_selection"]) <= set(ids["validation_rows"])):
        raise ValueError("actual batch rows violate declared optimization/selection roles")
    allowed = set(specs["optimization"]["data"]["source_family_ids"])
    if allowed != set(catalogs["train"].values()):
        raise ValueError("optimization producer pool differs from complete training family catalog")
    forbidden = set(catalogs["validation"].values()) | set(task_template["source_family_ids"])
    if allowed & forbidden or role_families["optimization"] & forbidden:
        raise ValueError("optimization pool overlaps selection/development source families")
    if not role_families["geometry_training"] <= allowed:
        raise ValueError("geometry training families must belong to optimization pool")
    if any(demos & forbidden or not demos <= allowed for demos in role_demos.values()):
        raise ValueError("demonstrations must use the optimization pool and exclude heldout families")
