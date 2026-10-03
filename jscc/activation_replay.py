"""Complete encoder-boundary microbatch capture for local reconstruction.

The store deliberately keeps every padded presentation microbatch. Repacking
tokens or deduplicating views would change BF16 execution and AWGN draw shapes.
"""

from __future__ import annotations

import hashlib
import json
import copy
import time
from pathlib import Path
from typing import Any

import torch

from .models.split_model import stack_module
from .presentation import tensor_digest, training_source_digest


SCHEMA = "enc-boundary-microbatches-v1"


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      default=str).encode()).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(config: dict, data_ids: dict, prompt_evidence: dict | None = None) -> dict:
    """Bind model/source/data/view policy while allowing an explicit L-step budget."""
    settings = config["training"]
    split = config["split"]
    if config["task"] != "hellaswag" or split != {"stack": "enc", "where": "after_layer", "index": 9}:
        raise ValueError("local reconstruction is restricted to HellaSwag enc_l9")
    if settings["batch_size"] < 1 or settings["gradient_accumulation"] < 1:
        raise ValueError("invalid microbatch policy")
    stream = settings.get("presentation_stream")
    if not stream or stream.get("policy") != "epoch-permutations-v1":
        raise ValueError("local reconstruction requires an explicit fixed presentation stream")
    if stream["total_presentations"] != settings["max_steps"] * settings["batch_size"] * settings["gradient_accumulation"]:
        raise ValueError("presentation budget differs from update budget")
    if set(data_ids["train_rows"]) & set(data_ids["validation_rows"]) and data_ids.get("validation_split") == "train":
        raise ValueError("training and selection rows overlap")
    if config["data"].get("prompt_policy") is not None and not prompt_evidence:
        raise ValueError("prompt-policy cache requires per-row prompt evidence")
    codec = copy.deepcopy(config["codec"])
    codec.pop("input_dim", None)  # Derived by build_model from the pinned backbone.
    return {
        "schema": SCHEMA,
        "task": config["task"], "model": config["model"], "split": split,
        "codec": codec, "channel": config["channel"],
        "data": config["data"], "data_ids_sha256": canonical_digest(data_ids),
        "prompt_evidence_sha256": canonical_digest(prompt_evidence),
        "training_source_sha256": training_source_digest(),
        "microbatch_size": settings["batch_size"],
        "gradient_accumulation": settings["gradient_accumulation"],
        "presentation_stream": stream,
        "paired_randomness": settings.get("paired_randomness"),
        "tensor_schema": {"activation": "[B,T,D] native backbone dtype",
                          "attention_mask": "[B,T] binary", "input_ids": "[B,T] int",
                          "labels": "[B,L] int", "row_ids": "[B] int"},
    }


class _BoundaryReached(Exception):
    pass


@torch.no_grad()
def capture_encoder_microbatch(model, batch: dict) -> torch.Tensor:
    """Stop the encoder at its split layer, before the installed codec hook."""
    if model.split != {"stack": "enc", "where": "after_layer", "index": 9}:
        raise ValueError("capture requires enc_l9")
    layer = stack_module(model.base, "enc").layers[9]
    captured: list[torch.Tensor] = []

    def stop(_module, _args, output):
        value = output[0] if isinstance(output, tuple) else output
        captured.append(value.detach().clone())
        raise _BoundaryReached

    handle = layer.register_forward_hook(stop, prepend=True)
    device = next(model.base.parameters()).device
    try:
        try:
            with model.attention_context():
                model.base.get_encoder()(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    return_dict=True,
                )
        except _BoundaryReached:
            pass
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("encoder boundary was not reached exactly once")
    activation = captured[0]
    if activation.ndim != 3 or activation.shape[:2] != batch["attention_mask"].shape:
        raise RuntimeError("captured activation does not match full sequence mask")
    return activation


def write_cache(directory: str | Path, config: dict, data_ids: dict,
                batches, model, *, prompt_evidence: dict | None = None,
                deadline: float | None = None,
                execution_started: float | None = None,
                max_cache_bytes: int | None = None) -> dict:
    """Capture exactly the declared stream and publish a manifest only on completion."""
    directory = Path(directory)
    if max_cache_bytes is not None and max_cache_bytes < 1:
        raise ValueError("max_cache_bytes must be positive")
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("cache directory must be empty")
    directory.mkdir(parents=True, exist_ok=True)
    capture_started = time.monotonic()
    basis = identity(config, data_ids, prompt_evidence)
    steps = config["training"]["max_steps"]
    accumulation = config["training"]["gradient_accumulation"]
    expected_ids = None
    sampler = getattr(batches, "sampler", None)
    if sampler is not None:
        expected_ids = getattr(sampler, "actual_ids", None)
    shards = []
    cache_bytes = 0
    prompt_rows = ({row["prompt_row_id"]: row for row in prompt_evidence["train"]["rows"]}
                   if prompt_evidence is not None else {})
    train_rows = set(data_ids["train_rows"])
    selection_rows = set(data_ids["validation_rows"]) if data_ids.get("validation_split") == "train" else set()
    iterator = iter(batches)
    offset = 0
    try:
        for index in range(steps * accumulation):
          if deadline is not None and time.monotonic() >= deadline:
              raise TimeoutError("capture wall deadline reached before next microbatch")
          try:
              batch = next(iterator)
          except StopIteration as exc:
              raise RuntimeError("presentation stream ended before capture budget") from exc
          row_ids = batch.get("row_ids")
          if row_ids is None or len(row_ids) != basis["microbatch_size"]:
              raise ValueError("capture requires full microbatches with row IDs")
          if expected_ids is not None and not torch.equal(row_ids, expected_ids[offset:offset + len(row_ids)]):
              raise ValueError("actual presentation IDs differ from declared stream")
          if not set(row_ids.tolist()).issubset(train_rows):
              raise ValueError("selection or foreign row entered training cache")
          views = []
          for position, row_id in enumerate(row_ids.tolist()):
              metadata = prompt_rows.get(row_id)
              input_tokens = batch["input_ids"][position][batch["attention_mask"][position].bool()].tolist()
              target_tokens = batch["labels"][position][batch["labels"][position] != -100].tolist()
              if prompt_evidence is not None:
                  if metadata is None or metadata["input_hash"] != canonical_digest(input_tokens) or metadata["target_hash"] != canonical_digest(target_tokens):
                      raise ValueError("prompt view tokens differ from recorded source")
                  if not set(metadata["demo_ids"]).issubset(train_rows) or set(metadata["demo_ids"]) & selection_rows:
                      raise ValueError("prompt demonstration leaks selection rows")
              views.append(canonical_digest({"row_id": row_id, "prompt": metadata,
                                              "input_tokens": input_tokens, "target_tokens": target_tokens}))
          activation = capture_encoder_microbatch(model, batch).cpu()
          saved = {key: batch[key].detach().cpu().clone()
                   for key in ("input_ids", "attention_mask", "labels", "row_ids")}
          saved["activation"] = activation
          saved["step"] = index // accumulation + 1
          saved["microbatch"] = index % accumulation
          saved["presentation_offsets"] = torch.arange(offset, offset + len(row_ids))
          saved["view_ids"] = views
          estimated = sum(value.numel() * value.element_size() for value in saved.values()
                          if torch.is_tensor(value))
          if max_cache_bytes is not None and cache_bytes + estimated > max_cache_bytes:
              raise RuntimeError("cache byte cap would be exceeded before shard serialization")
          path = directory / f"microbatch_{index:08d}.pt"
          temporary = path.with_suffix(".partial")
          torch.save(saved, temporary)
          shard_bytes = temporary.stat().st_size
          if max_cache_bytes is not None and cache_bytes + shard_bytes > max_cache_bytes:
              temporary.unlink()
              raise RuntimeError("cache byte cap exceeded by serialized shard")
          temporary.rename(path)
          cache_bytes += shard_bytes
          shards.append({"file": path.name, "sha256": file_digest(path),
                         "step": saved["step"], "microbatch": saved["microbatch"],
                         "row_ids": row_ids.tolist(), "presentation_start": offset,
                         "view_ids": views,
                         "input_ids_sha256": tensor_digest(saved["input_ids"]),
                         "attention_mask_sha256": tensor_digest(saved["attention_mask"]),
                         "activation_sha256": tensor_digest(activation),
                         "valid_vectors": int(saved["attention_mask"].sum()),
                       "allocated_vectors": saved["attention_mask"].numel()})
          shards[-1]["bytes"] = shard_bytes
          offset += len(row_ids)
        if offset != basis["presentation_stream"]["total_presentations"]:
            raise RuntimeError("captured presentation count differs from budget")
        manifest = {"identity": basis, "identity_sha256": canonical_digest(basis),
                    "data_ids": data_ids, "prompt_evidence": prompt_evidence,
                    "shards": shards, "presentations": offset,
                    "valid_vectors": sum(s["valid_vectors"] for s in shards),
                    "allocated_vectors": sum(s["allocated_vectors"] for s in shards),
                    "cache_bytes": cache_bytes, "max_cache_bytes": max_cache_bytes,
                    "elapsed_seconds": time.monotonic() - (execution_started or capture_started),
                    "setup_seconds": capture_started - execution_started if execution_started is not None else None,
                    "capture_seconds": time.monotonic() - capture_started,
                    "status": "complete"}
        temporary_manifest = directory / "manifest.tmp"
        temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("capture wall deadline reached before manifest publication")
        temporary_manifest.replace(directory / "manifest.json")
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("capture wall deadline exceeded during manifest publication")
        return manifest
    except Exception as exc:
        for unfinished in directory.glob("microbatch_*.partial"):
            unfinished.unlink()
        (directory / "manifest.tmp").unlink(missing_ok=True)
        (directory / "manifest.json").unlink(missing_ok=True)
        (directory / "partial.json").write_text(json.dumps({"status": "partial", "reason": str(exc),
             "captured_microbatches": len(shards), "captured_presentations": offset,
             "cache_bytes": cache_bytes, "max_cache_bytes": max_cache_bytes,
             "elapsed_seconds": time.monotonic() - (execution_started or capture_started),
             "setup_seconds": capture_started - execution_started if execution_started is not None else None,
             "identity_sha256": canonical_digest(basis), "shards": shards}, indent=2) + "\n")
        raise


def load_manifest(directory: str | Path, config: dict | None = None) -> dict:
    path = Path(directory) / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "complete" or manifest.get("identity", {}).get("schema") != SCHEMA:
        raise ValueError("cache is incomplete or has an unsupported schema")
    if canonical_digest(manifest["identity"]) != manifest["identity_sha256"]:
        raise ValueError("cache identity digest mismatch")
    if canonical_digest(manifest["data_ids"]) != manifest["identity"]["data_ids_sha256"]:
        raise ValueError("cache data IDs changed")
    if canonical_digest(manifest.get("prompt_evidence")) != manifest["identity"]["prompt_evidence_sha256"]:
        raise ValueError("cache prompt evidence changed")
    if config is not None:
        if identity(config, manifest["data_ids"], manifest.get("prompt_evidence")) != manifest["identity"]:
            raise ValueError("stale cache identity for current recipe or source")
    basis = manifest["identity"]
    expected_count = basis["presentation_stream"]["total_presentations"] // basis["microbatch_size"]
    if len(manifest["shards"]) != expected_count or manifest["presentations"] != basis["presentation_stream"]["total_presentations"]:
        raise ValueError("cache exposure count mismatch")
    if manifest["cache_bytes"] != sum(shard["bytes"] for shard in manifest["shards"]):
        raise ValueError("cache byte accounting mismatch")
    train = set(manifest["data_ids"]["train_rows"])
    selection = set(manifest["data_ids"]["validation_rows"]) if manifest["data_ids"].get("validation_split") == "train" else set()
    for index, shard in enumerate(manifest["shards"]):
        if shard["file"] != f"microbatch_{index:08d}.pt" or shard["presentation_start"] != index * basis["microbatch_size"]:
            raise ValueError("cache shard order or presentation mapping changed")
        rows = shard["row_ids"]
        if len(rows) != basis["microbatch_size"] or not set(rows).issubset(train) or set(rows) & selection:
            raise ValueError("cache contains selection or foreign rows")
        if len(shard["view_ids"]) != len(rows):
            raise ValueError("cache presentation/view count mismatch")
    return manifest


def load_shard(directory: str | Path, manifest: dict, index: int) -> dict:
    shard = manifest["shards"][index]
    path = Path(directory) / shard["file"]
    if file_digest(path) != shard["sha256"]:
        raise ValueError(f"cache shard checksum mismatch: {path.name}")
    if path.stat().st_size != shard["bytes"]:
        raise ValueError(f"cache shard byte count mismatch: {path.name}")
    data = torch.load(path, map_location="cpu", weights_only=True)
    mask, activation = data["attention_mask"], data["activation"]
    if mask.ndim != 2 or mask.shape != activation.shape[:2] or activation.ndim != 3:
        raise ValueError("cached full-sequence activation/mask shape mismatch")
    if (data["input_ids"].shape != mask.shape or data["labels"].ndim != 2 or
            data["row_ids"].shape != (mask.shape[0],) or
            data["presentation_offsets"].tolist() != list(range(
                shard["presentation_start"], shard["presentation_start"] + mask.shape[0])) or
            data["step"] != shard["step"] or data["microbatch"] != shard["microbatch"]):
        raise ValueError("cached presentation, grouping or token schema mismatch")
    if not bool(torch.all((mask == 0) | (mask == 1))) or not bool(mask.any(dim=1).all()):
        raise ValueError("invalid or empty sequence mask")
    if not bool(torch.isfinite(activation).all()):
        raise ValueError("nonfinite cached activation")
    if (data["row_ids"].tolist() != shard["row_ids"] or
            data["view_ids"] != shard["view_ids"] or
            tensor_digest(data["input_ids"]) != shard["input_ids_sha256"] or
            tensor_digest(mask) != shard["attention_mask_sha256"] or
            tensor_digest(activation) != shard["activation_sha256"]):
        raise ValueError("cache presentation/view tensor identity mismatch")
    return data


SEQUENCE_SCHEMA = "encoder-sequences-v2"
_SEQUENCE_TENSORS = ("input_ids", "attention_mask", "labels", "decoder_input_ids",
                     "decoder_attention_mask", "row_ids", "activation")
_SEQUENCE_METADATA = ("batch_view_id", "view_ids", "source_family_ids", "demo_ids",
                      "position_roles", "source_views")
_DATA_ROLES = {"optimization", "objective_validation", "geometry_training",
               "geometry_selection", "heldout_after_freeze"}


def capture_source_inventory(source_root: str | Path) -> dict:
    """Retain a conservative clean-capture dependency closure, including bytes.

    All model, data, config and preprocessing modules are roots; local imports
    are followed recursively. Learner-only modules are excluded only when no
    capture dependency imports them. Configuration bytes are conservative roots.
    """
    import ast
    import importlib.util

    root = Path(source_root).resolve()
    roots = {"jscc/activation_replay.py", "jscc/presentation.py", "jscc/runtime.py",
             "jscc/config.py", "jscc/__init__.py"}
    for folder in ("jscc/models", "jscc/data"):
        roots.update(str(p.relative_to(root)) for p in (root / folder).rglob("*.py"))
    roots.update(str(p.relative_to(root)) for p in (root / "configs").rglob("*.yaml"))
    roots.update({"uv.lock", "pyproject.toml"})
    pending, files = list(sorted(roots)), {}
    while pending:
        name = pending.pop()
        if name in files:
            continue
        path = root / name
        content = path.read_bytes()
        files[name] = {"sha256": hashlib.sha256(content).hexdigest(),
                       "bytes": len(content), "content_hex": content.hex()}
        if path.suffix != ".py":
            continue
        package = ".".join(Path(name).with_suffix("").parts[:-1])
        for node in ast.walk(ast.parse(content)):
            modules = []
            if isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    module = importlib.util.resolve_name("." * node.level + module, package)
                modules = [module, *(f"{module}.{item.name}" for item in node.names)]
            for module in modules:
                if not module.startswith("jscc"):
                    continue
                base = module.replace(".", "/")
                for candidate in (f"{base}.py", f"{base}/__init__.py"):
                    if (root / candidate).is_file() and candidate not in files:
                        pending.append(candidate)
    return dict(sorted(files.items()))


def producer_spec_v2(*, source_root: str | Path, backbone: dict, environment: dict,
                     data: dict, policy: dict, sites: list[dict], views: list[dict],
                     data_role: str, disjointness: dict) -> dict:
    """Create a clean producer identity with no objective/optimizer/channel fields.

    ``views`` freezes complete per-microbatch metadata plus every input tensor
    digest. ``data`` binds memberships, fingerprints and task-native references.
    Site records come from the shared encoder-site resolver.
    """
    spec = {"schema": SEQUENCE_SCHEMA, "clean_bypass": True,
            "source_files": capture_source_inventory(source_root),
            "backbone": backbone, "environment": environment, "data": data,
            "policy": policy, "sites": sites, "views": views,
            "data_role": data_role, "disjointness": disjointness,
            "capabilities": ["local_reconstruction", "objective_validation", "geometry"],
            "tensor_schema": "complete-padded-native-prefix-v2"}
    _validate_producer_v2(spec)
    return copy.deepcopy(spec)


def sequence_view_v2(batch: dict) -> dict:
    """Freeze the exact padded source/target/native-prefix view before capture."""
    tensors = {key: {"sha256": tensor_digest(batch[key]),
                     "shape": list(batch[key].shape), "dtype": str(batch[key].dtype)}
               for key in _SEQUENCE_TENSORS if key != "activation"}
    if "pixel_values" in batch:
        tensors["pixel_values"] = {"sha256": tensor_digest(batch["pixel_values"]),
                                  "shape": list(batch["pixel_values"].shape),
                                  "dtype": str(batch["pixel_values"].dtype)}
    return {**{key: copy.deepcopy(batch[key]) for key in _SEQUENCE_METADATA}, "tensors": tensors}


def _validate_producer_v2(spec: dict) -> None:
    if spec.get("schema") != SEQUENCE_SCHEMA or spec.get("clean_bypass") is not True:
        raise ValueError("unsupported clean sequence producer schema")
    if spec.get("data_role") not in _DATA_ROLES or not spec.get("disjointness"):
        raise ValueError("producer requires data role and disjointness evidence")
    for field in ("backbone", "environment", "data", "policy", "sites", "views", "source_files"):
        if not spec.get(field):
            raise ValueError(f"producer missing {field}")
    for field in ("model_revision", "tokenizer_revision", "weights_sha256", "config_sha256"):
        if not spec["backbone"].get(field):
            raise ValueError(f"producer backbone missing {field}")
    for field in ("libraries", "backend", "activation_dtype"):
        if not spec["environment"].get(field):
            raise ValueError(f"producer environment missing {field}")
    if spec["data"].get("task") not in {"hellaswag", "coco"} or not spec["data"].get("fingerprints"):
        raise ValueError("producer requires native task and dataset fingerprints")
    if spec["data"]["task"] == "coco" and not spec["backbone"].get("processor_revision"):
        raise ValueError("COCO producer requires processor identity")
    sites = [site["site_id"] for site in spec["sites"]]
    views = [view["batch_view_id"] for view in spec["views"]]
    if len(sites) != len(set(sites)) or len(views) != len(set(views)):
        raise ValueError("duplicate producer sites or batch views")
    for name, record in spec["source_files"].items():
        content = bytes.fromhex(record["content_hex"])
        if len(content) != record["bytes"] or hashlib.sha256(content).hexdigest() != record["sha256"]:
            raise ValueError(f"producer source archive corrupted: {name}")
    if "uv.lock" not in spec["source_files"] or "jscc/activation_replay.py" not in spec["source_files"]:
        raise ValueError("producer source closure missing capture or lockfile")
    # Source-family exclusions are explicit; optimization demos may use training
    # families but cannot use selection/development/confirmation families.
    forbidden = set(spec["disjointness"].get("excluded_source_family_ids", []))
    allowed = set(spec["data"].get("source_family_ids", []))
    if not allowed:
        raise ValueError("producer requires source-family membership")
    for view in spec["views"]:
        families = set(view["source_family_ids"])
        if not families.issubset(allowed) or families & forbidden:
            raise ValueError("producer source-family exclusion or membership violation")
        demo_families = {family for row in view["source_views"] for family in row["demo_source_family_ids"]}
        if demo_families & forbidden:
            raise ValueError("producer demonstration source-family exclusion violation")
        if spec["data"]["task"] == "coco" and any(
                not row.get("image_sha256") or not row.get("processor_sha256") for row in view["source_views"]):
            raise ValueError("COCO views require verified image and processor references")


def _validate_sequence_v2(batch: dict, spec: dict, view: dict, site: dict) -> None:
    mask, activation = batch["attention_mask"], batch["activation"]
    if mask.ndim != 2 or activation.ndim != 3 or activation.shape[:2] != mask.shape:
        raise ValueError("sequence activation/mask shape mismatch")
    if not bool(((mask == 0) | (mask == 1)).all()) or not bool(mask.any(dim=1).all()):
        raise ValueError("invalid binary encoder mask")
    if not bool(torch.isfinite(activation).all()):
        raise ValueError("nonfinite sequence activation")
    if str(activation.dtype) != spec["environment"]["activation_dtype"]:
        raise ValueError("activation precision differs from producer")
    if activation.shape[-1] != site["hidden_dim"]:
        raise ValueError("activation dimension differs from resolved site")
    nrows = mask.shape[0]
    if batch["input_ids"].shape != mask.shape or batch["row_ids"].shape != (nrows,):
        raise ValueError("sequence source shape mismatch")
    labels, prefix, valid = (batch[key] for key in ("labels", "decoder_input_ids", "decoder_attention_mask"))
    if labels.ndim != 2 or labels.shape[0] != nrows or prefix.shape != labels.shape or valid.shape != labels.shape:
        raise ValueError("native target/prefix shape mismatch")
    if not bool(((valid == 0) | (valid == 1)).all()):
        raise ValueError("invalid decoder prefix mask")
    for key in _SEQUENCE_METADATA[1:]:
        if len(batch[key]) != nrows:
            raise ValueError(f"sequence metadata row count mismatch: {key}")
    if any(len(roles) != mask.shape[1] for roles in batch["position_roles"]):
        raise ValueError("position roles must retain padded source shape")
    if sequence_view_v2(batch) != view:
        raise ValueError("sequence view/input/native-target identity mismatch")


def capture_sequences(directory: str | Path, producer_spec: dict, ordered_microbatches,
                      model, *, max_cache_bytes: int | None = None,
                      deadline: float | None = None) -> dict:
    """Write an immutable complete-sequence bank; failures retain incomplete evidence."""
    spec = copy.deepcopy(producer_spec)
    _validate_producer_v2(spec)
    directory = Path(directory)
    if max_cache_bytes is not None and max_cache_bytes < 1:
        raise ValueError("max_cache_bytes must be positive")
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": SEQUENCE_SCHEMA, "producer": spec,
                "producer_sha256": canonical_digest(spec), "status": "incomplete", "shards": []}
    shards = manifest["shards"]
    iterator = iter(ordered_microbatches)
    used = 0
    try:
        for index, view in enumerate(spec["views"]):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("sequence capture deadline reached")
            batch = next(iterator)
            if sequence_view_v2(batch) != view:
                raise ValueError("ordered batch view differs from producer")
            if spec["policy"].get("native_prefix") == "reference":
                native = model.base.prepare_decoder_input_ids_from_labels(labels=batch["labels"])
                if (not torch.equal(native.cpu(), batch["decoder_input_ids"].cpu())
                        or not torch.equal(batch["labels"].ne(-100), batch["decoder_attention_mask"].bool())):
                    raise ValueError("reference-prefix records differ from native decoder preparation")
            values = capture_clean_sites(model, batch, spec["sites"], spec["policy"]["site_authorization"])
            for site_index, site in enumerate(spec["sites"]):
                saved = {key: value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value)
                         for key, value in batch.items()}
                saved["activation"] = values[site["site_id"]].detach().cpu().clone()
                saved["site_id"] = site["site_id"]
                _validate_sequence_v2(saved, spec, view, site)
                path = directory / f"sequence_{index:08d}_{site_index:03d}.pt"
                temporary = path.with_suffix(".partial")
                torch.save(saved, temporary)
                size = temporary.stat().st_size
                if max_cache_bytes is not None and used + size > max_cache_bytes:
                    temporary.unlink()
                    raise RuntimeError("sequence capture byte cap exceeded")
                temporary.rename(path)
                used += size
                shards.append({"file": path.name, "sha256": file_digest(path), "bytes": size,
                               "site_id": site["site_id"], "batch_view_id": view["batch_view_id"],
                               "activation_sha256": tensor_digest(saved["activation"]),
                               "activation_shape": list(saved["activation"].shape),
                               "activation_dtype": str(saved["activation"].dtype),
                               "valid_vectors": int(saved["attention_mask"].sum()),
                               "allocated_vectors": saved["attention_mask"].numel()})
        if next(iterator, None) is not None:
            raise ValueError("extra microbatches outside producer layout")
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("sequence capture deadline reached before publication")
        manifest.update(status="complete", cache_bytes=used)
        manifest["manifest_sha256"] = canonical_digest(manifest)
        temporary_manifest = directory / "manifest.partial"
        temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary_manifest.rename(directory / "manifest.json")
        return manifest
    except Exception as exc:
        manifest.update(status="incomplete", reason=str(exc), cache_bytes=used)
        (directory / "incomplete.json").write_text(json.dumps(manifest, indent=2) + "\n")
        raise


class SequenceReplay:
    """Verified, complete records only; functional cached suffix is unsupported."""

    def __init__(self, directory: str | Path, requirement: dict):
        self.directory = Path(directory)
        manifest = json.loads((self.directory / "manifest.json").read_text())
        expected_digest = manifest.pop("manifest_sha256", None)
        if canonical_digest(manifest) != expected_digest or expected_digest != requirement.get("manifest_sha256"):
            raise ValueError("sequence manifest identity mismatch")
        if manifest.get("status") != "complete" or manifest.get("schema") != SEQUENCE_SCHEMA:
            raise ValueError("incomplete or unsupported sequence bank")
        spec = manifest["producer"]
        _validate_producer_v2(spec)
        if canonical_digest(spec) != manifest["producer_sha256"] or manifest["producer_sha256"] != requirement.get("producer_sha256"):
            raise ValueError("sequence producer identity mismatch")
        if requirement.get("capability") not in {"local_reconstruction", "objective_validation", "geometry"}:
            raise ValueError("functional cached suffix replay is disabled")
        receipt = requirement.get("parity_receipt", {})
        if (receipt.get("status") != "passed"
                or receipt.get("producer_sha256") != manifest["producer_sha256"]
                or receipt.get("capability") != requirement["capability"]
                or not requirement.get("learner_source_sha256")
                or receipt.get("learner_source_sha256") != requirement["learner_source_sha256"]
                or not receipt.get("evidence_sha256")):
            raise ValueError("consumer requires a compatible passed parity receipt")
        if requirement.get("data_role") != spec["data_role"]:
            raise ValueError("consumer cannot relabel producer data role")
        if requirement["capability"] == "local_reconstruction" and spec["data_role"] != "optimization":
            raise ValueError("local optimization requires optimization data")
        if requirement["capability"] == "objective_validation" and spec["data_role"] != "objective_validation":
            raise ValueError("objective validation requires objective_validation data")
        if requirement.get("sites") != [site["site_id"] for site in spec["sites"]]:
            raise ValueError("consumer site requirement mismatch")
        if requirement.get("views") != [view["batch_view_id"] for view in spec["views"]]:
            raise ValueError("consumer view/layout requirement mismatch")
        if requirement.get("activation_dtype") != spec["environment"]["activation_dtype"]:
            raise ValueError("consumer precision requirement mismatch")
        self._manifest = manifest
        self._shards = {}
        expected = [(view["batch_view_id"], site["site_id"]) for view in spec["views"] for site in spec["sites"]]
        actual = [(shard["batch_view_id"], shard["site_id"]) for shard in manifest["shards"]]
        if actual != expected or manifest["cache_bytes"] != sum(shard["bytes"] for shard in manifest["shards"]):
            raise ValueError("incomplete sequence bank or changed layout")
        for shard in manifest["shards"]:
            name = shard["file"]
            if Path(name).name != name or (self.directory / name).is_symlink():
                raise ValueError("invalid sequence shard path")
            self._shards[(shard["batch_view_id"], shard["site_id"])] = shard
            self.read(shard["batch_view_id"], shard["site_id"])

    @property
    def manifest(self) -> dict:
        return copy.deepcopy(self._manifest)

    def read(self, batch_view_id: str, site_id: str) -> dict:
        shard = self._shards[(batch_view_id, site_id)]
        path = self.directory / shard["file"]
        if path.stat().st_size != shard["bytes"] or file_digest(path) != shard["sha256"]:
            raise ValueError("sequence shard byte/checksum mismatch")
        batch = torch.load(path, map_location="cpu", weights_only=True)
        spec = self._manifest["producer"]
        view = next(view for view in spec["views"] if view["batch_view_id"] == batch_view_id)
        site = next(site for site in spec["sites"] if site["site_id"] == site_id)
        _validate_sequence_v2(batch, spec, view, site)
        if (batch["site_id"] != site_id or tensor_digest(batch["activation"]) != shard["activation_sha256"]
                or list(batch["activation"].shape) != shard["activation_shape"]
                or str(batch["activation"].dtype) != shard["activation_dtype"]
                or int(batch["attention_mask"].sum()) != shard["valid_vectors"]
                or batch["attention_mask"].numel() != shard["allocated_vectors"]):
            raise ValueError("sequence shard activation/provenance mismatch")
        return batch


def open_replay(directory: str | Path, requirement: dict) -> SequenceReplay:
    return SequenceReplay(directory, requirement)


@torch.no_grad()
def capture_clean_sites(model, batch: dict, sites: list[dict], authorization: dict) -> dict:
    """Clean encoder capture through the same resolver and authorization seam."""
    from dataclasses import asdict
    from .models.split_model import resolve_encoder_site

    parameter = next(model.base.parameters())
    kwargs = {key: batch[key].to(parameter.device) for key in ("input_ids", "attention_mask")}
    if "pixel_values" in batch:
        pixels = batch["pixel_values"]
        if pixels.ndim == 5:
            pixels = pixels.flatten(0, 1)
        kwargs["pixel_values"] = pixels.to(device=parameter.device, dtype=parameter.dtype)
    values = {}
    for record in sites:
        resolved = resolve_encoder_site(model.base, record["split"], record["model_revision"])
        if record != {**asdict(resolved), "site_id": resolved.site_id, "split": resolved.split}:
            raise ValueError("capture site metadata differs from resolved backbone")
        observed = []
        def observe(_module, _args, output):
            value = output[0] if isinstance(output, tuple) else output
            observed.append(value.detach().clone())
        stack = stack_module(model.base, "enc")
        module = stack.norm if resolved.index is None else stack.layers[resolved.index]
        with model.at_site(resolved, purpose="capture", authorization=authorization):
            handle = module.register_forward_hook(observe, prepend=True)
            try:
                with model.transmission(bypass=True, encoder_mask=kwargs["attention_mask"]):
                    model.base.get_encoder()(**kwargs, return_dict=True)
            finally:
                handle.remove()
        if len(observed) != 1:
            raise RuntimeError("capture site was not visited exactly once")
        values[resolved.site_id] = observed[0]
    return values
