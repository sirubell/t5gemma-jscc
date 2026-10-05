"""Trust-boundary and durable state helpers for ordinary exact continuation."""
from contextlib import contextmanager
import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import cast

import numpy as np
import torch

from .evaluation_policy import file_digest
from .experiment_records import write_json_atomic
from .experiment_state import parameter_map
from .run_records import runtime_identity
from .presentation import tensor_digest, training_source_digest
from .runtime import source_state
from .models.channel import AWGNChannel, IdentityChannel

SCHEMA = "ordinary-continuation-v1"


def capture_rng():
    numpy = cast(tuple, np.random.get_state())
    return {"python": random.getstate(), "numpy": [numpy[0], numpy[1].tolist(), *numpy[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state):
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.array(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("incompatible CUDA RNG device topology")
        torch.cuda.set_rng_state_all(state["cuda"])


def communication_identity(model):
    def value_identity(value):
        if torch.is_tensor(value):
            return {"shape": list(value.shape), "dtype": str(value.dtype),
                    "sha256": tensor_digest(value.reshape(-1))}
        # Extra state is part of nn.Module's supported checkpoint contract.
        try:
            return {"extra_state": json.dumps(value, sort_keys=True, allow_nan=False)}
        except (TypeError, ValueError) as error:
            raise ValueError("exact continuation requires JSON-compatible custom channel extra state") from error
    return {name: {key: value_identity(value) for key, value in component.state_dict().items()}
            for name, component in (("codec", model.codec), ("memory_codec", model.memory_codec),
                                    ("channel", model.channel)) if component is not None}


def compatibility(config, model, processor, data, optimizer, determinism):
    dataset = data.train.dataset
    raw = getattr(dataset, "rows", dataset)
    channel_file = inspect.getsourcefile(type(model.channel))
    source = source_state()
    if not isinstance(source["revision"], str) or len(source["revision"]) != 40 or source["dirty"] is None:
        raise ValueError("exact continuation requires a known full source revision")
    return {"source": source, "training_source_sha256": training_source_digest(),
            "runtime": runtime_identity(), "config": config, "data_ids": data.ids,
            "data": {"rows": len(dataset), "fingerprint": getattr(raw, "_fingerprint", None),
                     "dataset_class": f"{type(dataset).__module__}.{type(dataset).__qualname__}",
                     "prompt_evidence": getattr(data, "prompt_evidence", None)},
            "initializer": communication_identity(model),
            "model": {"name": getattr(model.base.config, "_name_or_path", None),
                      "revision": getattr(model.base.config, "_commit_hash", None),
                      "processor": f"{type(processor).__module__}.{type(processor).__qualname__}",
                      "tokenizer": getattr(getattr(processor, "tokenizer", processor), "name_or_path", None),
                      "attention_backend": getattr(getattr(model.base.config, "decoder", model.base.config),
                                                   "_attn_implementation", None)},
            "channel": {"class": f"{type(model.channel).__module__}.{type(model.channel).__qualname__}",
                        "source_sha256": file_digest(Path(channel_file)) if channel_file else None},
            "parameters": parameter_map(model, optimizer), "determinism": determinism,
            "cuda_devices": torch.cuda.device_count() if config["model"]["device"] == "cuda" else 0,
            "numerics": {"threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                         "matmul_precision": torch.get_float32_matmul_precision(),
                         "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
                         "cudnn_deterministic": torch.backends.cudnn.deterministic,
                         "cudnn_benchmark": torch.backends.cudnn.benchmark,
                         "cuda_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
                         if config["model"]["device"] == "cuda" else []}}


def validate_compatibility(saved, current):
    for key in current:
        if saved.get(key) != current[key]:
            raise ValueError(f"incompatible ordinary continuation {key}")


def load_checkpoint(path, expected=None):
    """Digest and deserialize through one descriptor; only published saves qualify."""
    path = Path(path).resolve()
    try:
        receipt = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError("missing valid durable checkpoint receipt/cursor; legacy continuation unsupported") from error
    if (receipt.get("schema") != "ordinary-checkpoint-v1" or receipt.get("role") != "ordinary-post-update"
            or type(receipt.get("step")) is not int or type(receipt.get("size_bytes")) is not int
            or receipt["size_bytes"] <= 0 or not isinstance(receipt.get("sha256"), str)):
        raise ValueError("invalid ordinary durable checkpoint receipt/role")
    if expected is not None and any(receipt[key] != expected[key] for key in ("sha256", "size_bytes", "step")):
        raise ValueError("ordinary continuation checkpoint identity changed")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != receipt["sha256"] or os.fstat(stream.fileno()).st_size != receipt["size_bytes"]:
            raise ValueError("ordinary continuation checkpoint bytes changed")
        stream.seek(0)
        state = torch.load(stream, map_location="cpu", weights_only=True)
    continuation = state.get("continuation")
    if (not isinstance(continuation, dict) or continuation.get("schema") != SCHEMA
            or not continuation.get("cursor") or not continuation.get("rng")
            or state.get("step") != receipt["step"] or type(state.get("step")) is not int
            or state["step"] < 1 or continuation.get("completed_updates") != state["step"]):
        raise ValueError("missing/unsupported ordinary continuation cursor or training state")
    for key in ("codec", "channel", "memory_codec", "optimizer", "scheduler", "scaler",
                "config", "data_ids", "best", "bad_evaluations"):
        if key not in state:
            raise ValueError(f"incomplete ordinary checkpoint: {key}")
    required = {"compatibility", "cursor", "counters", "rng", "completed_updates", "best_step", "best_checkpoint"}
    if not required <= continuation.keys() or not isinstance(continuation["compatibility"], dict):
        raise ValueError("incomplete ordinary continuation runtime state")
    best, bad, best_step = state["best"], state["bad_evaluations"], continuation["best_step"]
    if (type(bad) is not int or bad < 0 or type(best) not in (int, float)
            or (best_step is None and (best != float("inf") or bad != 0))
            or (best_step is not None and (type(best_step) is not int
                 or not 1 <= best_step <= state["step"] or not math.isfinite(best)))):
        raise ValueError("invalid ordinary checkpoint selection state")
    patience = state["config"]["training"]["patience"]
    if patience and bad >= patience:
        raise ValueError("ordinary checkpoint already reached terminal early-stop patience")
    counter_names = {"presentations", "source_tokens_valid", "source_tokens_allocated",
                     "target_tokens_valid", "target_tokens_allocated", "hidden_latent_valid",
                     "hidden_latent_allocated", "memory_latent_valid", "memory_latent_allocated"}
    if (not isinstance(continuation["counters"], dict)
            or set(continuation["counters"]) != counter_names
            or any(type(value) is not int or value < 0 for value in continuation["counters"].values())
            or not {"python", "numpy", "torch", "cuda"} <= continuation["rng"].keys()):
        raise ValueError("invalid ordinary continuation counters/RNG")
    return state, {"path": str(path), **receipt}


def atomic_torch_save(payload, path):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def publish_checkpoint(path, step):
    receipt = {"schema": "ordinary-checkpoint-v1", "sha256": file_digest(path), "size_bytes": path.stat().st_size, "step": step,
               "role": "ordinary-post-update"}
    write_json_atomic(path.with_suffix(path.suffix + ".json"), receipt)
    return {"path": str(path.resolve()), **receipt}


def link_checkpoint(reference, destination):
    """Retain earlier bytes; aliases refer to the durable immutable save."""
    destination = Path(destination)
    temporary = destination.with_name(f".{destination.name}.link")
    try:
        os.link(reference["path"], temporary)
        os.replace(temporary, destination)
        write_json_atomic(destination.with_suffix(destination.suffix + ".json"),
                          {key: value for key, value in reference.items() if key != "path"})
    finally:
        temporary.unlink(missing_ok=True)


def channel_support(model):
    channel = model.channel
    if type(channel) in (AWGNChannel, IdentityChannel):
        return None
    if type(channel).__module__ == "examples.wireless_channel":
        return None
    if getattr(channel, "supports_exact_continuation", False) is True:
        return None
    return "custom channel must declare supports_exact_continuation=True and capture all runtime state"


@contextmanager
def isolated_observation(model):
    """Observers cannot advance a custom channel's durable simulation state."""
    saved = copy.deepcopy({"codec": model.codec.state_dict(), "channel": model.channel.state_dict(),
                           "memory_codec": model.memory_codec.state_dict() if model.memory_codec is not None else None})
    rng = capture_rng()
    was_training = model.training
    try:
        yield
    finally:
        model.load_communication_state(saved)
        restore_rng(rng)
        model.train(was_training)
