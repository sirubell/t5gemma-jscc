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
