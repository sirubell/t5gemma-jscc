"""Immutable, hash-bound full-state carry for the versioned baseline protocol."""

from contextlib import contextmanager
from dataclasses import dataclass
import copy
import hashlib
import io
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

IDENTITIES = (
    "source_identity",
    "config_identity",
    "parent_identity",
    "initialization_identity",
    "stream_identity",
    "protocol_identity",
)
REQUIRED = (*IDENTITIES, "completed_updates", "phase", "lineage")


@dataclass(frozen=True)
class CheckpointRef:
    path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class ValidatedState:
    payload: dict
    reference: CheckpointRef


def capture_rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA RNG state cannot be restored on this runtime")
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def isolated_rng(seed=None):
    saved = capture_rng()
    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.manual_seed(seed)
        yield
    finally:
        restore_rng(saved)


def parameter_map(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    groups = []
    for group in optimizer.param_groups:
        mapped = []
        for p in group["params"]:
            if id(p) not in names:
                raise ValueError("optimizer parameter outside communication model")
            mapped.append((names[id(p)], tuple(p.shape), str(p.dtype)))
        groups.append(mapped)
    return groups


def _validate(payload):
    required = {
        "schema",
        "kind",
        "metadata",
        "parameter_map",
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "rng",
        "stream",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("incomplete or unknown state payload")
    if (
        payload["schema"] != "experiment-state-v2"
        or payload["kind"] not in {"single_site_v2", "shared_encoder_v2"}
    ):
        raise ValueError("unsupported state schema/kind")
    metadata = payload["metadata"]
    if not isinstance(metadata, dict) or any(k not in metadata for k in REQUIRED):
        raise ValueError("missing state identity or lineage")
    for key in IDENTITIES:
        if key != "parent_identity" and (
            not isinstance(metadata[key], str) or not metadata[key]
        ):
            raise ValueError(f"missing {key}")
    count = metadata["completed_updates"]
    if payload["kind"] == "shared_encoder_v2" or "sharing" in metadata:
        from .sharing_state import validate_sharing_state
        validate_sharing_state(payload)
    else:
        if type(count) is not int or not 0 <= count <= 400:
            raise ValueError("invalid completed update count")
        if metadata["phase"] not in {
            "both",
            "reconstruction",
            "staged",
            "reconstruction-prefix",
        }:
            raise ValueError("unknown phase")
        if not isinstance(metadata["lineage"], list):
            raise ValueError("phase lineage must be explicit")
        stream = payload["stream"]
        if (
            not isinstance(stream, dict)
            or stream.get("completed_updates") != count
            or stream.get("offset") != count * 64
        ):
            raise ValueError("inconsistent stream/update counters")
    if not isinstance(metadata["lineage"], list):
        raise ValueError("phase lineage must be explicit")
    if payload["scheduler"].get("last_epoch") != count:
        raise ValueError("scheduler/global update mismatch")
    if set(payload["rng"]) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("incomplete RNG state")
    if count > 0 and not payload["optimizer"].get("state"):
        raise ValueError("missing optimizer moments")
    if payload["scaler"] is not None and not isinstance(payload["scaler"], dict):
        raise ValueError("invalid scaler state")
    scaler_state = payload["scaler"]
    if scaler_state and set(scaler_state) != {
        "scale",
        "growth_factor",
        "backoff_factor",
        "growth_interval",
        "_growth_tracker",
    }:
        raise ValueError("incomplete scaler state")
    if (
        not isinstance(payload["rng"]["torch"], torch.Tensor)
        or payload["rng"]["torch"].dtype != torch.uint8
    ):
        raise ValueError("invalid torch RNG state")
    model_state = payload["model"]
    if not isinstance(model_state, dict) or not model_state:
        raise ValueError("missing communication weights")
    if any(
        not isinstance(v, torch.Tensor) or not torch.isfinite(v).all()
        for v in model_state.values()
    ):
        raise ValueError("nonfinite or invalid communication weights")
    groups = payload["optimizer"].get("param_groups", [])
    mapping = payload["parameter_map"]
    if len(groups) != len(mapping) or not groups:
        raise ValueError("incomplete optimizer parameter map")
    seen_names = set()
    seen_ids = set()
    for group, entries in zip(groups, mapping):
        if len(group["params"]) != len(entries):
            raise ValueError("incomplete optimizer parameter map")
        for identifier, (name, shape, dtype) in zip(group["params"], entries):
            if name in seen_names or identifier in seen_ids or name not in model_state:
                raise ValueError("duplicate or missing optimizer parameter")
            seen_names.add(name)
            seen_ids.add(identifier)
            if (
                tuple(model_state[name].shape) != shape
                or str(model_state[name].dtype) != dtype
            ):
                raise ValueError("state parameter-map shape/dtype mismatch")
            if count:
                moments = payload["optimizer"]["state"].get(identifier, {})
                if not {"step", "exp_avg", "exp_avg_sq"} <= moments.keys():
                    raise ValueError("missing AdamW moments")
                if moments["step"].item() != count:
                    raise ValueError("optimizer/global update mismatch")
                for key in ("exp_avg", "exp_avg_sq"):
                    value = moments[key]
                    if tuple(value.shape) != shape or not torch.isfinite(value).all():
                        raise ValueError("invalid optimizer moments")
    if metadata["parent_identity"] is not None:
        if (
            not isinstance(metadata["parent_identity"], str)
            or not metadata["parent_identity"]
        ):
            raise ValueError("invalid parent identity")
        if (
            not metadata["lineage"]
            or metadata["lineage"][-1].get("parent_sha256")
            != metadata["parent_identity"]
        ):
            raise ValueError("parent identity missing from lineage")


def save_state(path, *, model, optimizer, scheduler, scaler, metadata, stream_state):
    """Durably publish once. A failed save never replaces any existing parent."""
    payload = dict(
        schema="experiment-state-v2",
        kind=("shared_encoder_v2" if metadata.get("sharing", {}).get("learner") == "shared"
              else "single_site_v2"),
        metadata=copy.deepcopy(metadata),
        parameter_map=parameter_map(model, optimizer),
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),
        scaler=scaler.state_dict() if scaler is not None else None,
        rng=capture_rng(),
        stream=copy.deepcopy(stream_state),
    )
    _validate(payload)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    data = buffer.getvalue()
    path = Path(path).absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".incomplete-", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # Atomic no-clobber publication.
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return CheckpointRef(str(path), hashlib.sha256(data).hexdigest(), len(data))


def open_state(reference: CheckpointRef, *, expected):
    """Only trusted local torch artifacts; hashes are integrity, not trust proofs."""
    if any(k not in expected for k in IDENTITIES):
        raise ValueError("all compatibility identities must be supplied")
    data = Path(reference.path).read_bytes()
    if (
        len(data) != reference.size
        or hashlib.sha256(data).hexdigest() != reference.sha256
    ):
        raise ValueError("checkpoint bytes incomplete or corrupt")
    try:
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
        _validate(payload)
    except Exception as error:
        raise ValueError("invalid checkpoint state") from error
    if any(payload["metadata"].get(k) != v for k, v in expected.items()):
        raise ValueError("state source/config/parent/lineage compatibility mismatch")
    return ValidatedState(payload, reference)


def restore_state(validated: ValidatedState, *, model, optimizer, scheduler, scaler):
    # Reopen immutable bytes so a mutable in-memory payload cannot bypass integrity.
    payload = open_state(
        validated.reference, expected=validated.payload["metadata"]
    ).payload
    _validate(payload)
    if parameter_map(model, optimizer) != payload["parameter_map"]:
        raise ValueError("optimizer parameter-map mismatch")
    current = model.state_dict()
    if current.keys() != payload["model"].keys() or any(
        current[k].shape != v.shape or current[k].dtype != v.dtype
        for k, v in payload["model"].items()
    ):
        raise ValueError("communication architecture mismatch")
    if (scaler is None) != (payload["scaler"] is None):
        raise ValueError("scaler presence mismatch")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(copy.deepcopy(payload["optimizer"]))
    scheduler.load_state_dict(copy.deepcopy(payload["scheduler"]))
    if scaler is not None:
        scaler.load_state_dict(copy.deepcopy(payload["scaler"]))
    restore_rng(payload["rng"])
    return copy.deepcopy(payload["stream"])


def branch_metadata(validated: ValidatedState, *, phase):
    """Describe one exact 200/400 prefix branch; caller must restore full state."""
    payload = validated.payload
    _validate(payload)
    metadata = copy.deepcopy(payload["metadata"])
    if (
        metadata["completed_updates"] != 200
        or metadata["phase"] != "reconstruction-prefix"
    ):
        raise ValueError(
            "carry branch requires the exact reconstruction prefix at 200/400"
        )
    if phase not in {"reconstruction", "staged"}:
        raise ValueError("unplanned phase transition")
    metadata["lineage"].append(
        {
            "parent_sha256": validated.reference.sha256,
            "completed_updates": 200,
            "from_phase": metadata["phase"],
            "to_phase": phase,
        }
    )
    metadata["phase"] = phase
    metadata["parent_identity"] = validated.reference.sha256
    return metadata
