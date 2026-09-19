"""Small run folders, random seeds, and device helpers."""
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import json
import os
import random
import subprocess
import uuid

import numpy as np
import torch


def configure_training_determinism(settings):
    """Select an explicit repeatable training policy before CUDA initialization."""
    enabled = bool(settings.get("deterministic_algorithms", False))
    if enabled:
        workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if workspace not in {":4096:8", ":16:8"}:
            raise ValueError("Deterministic training requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8")
    torch.use_deterministic_algorithms(enabled)
    return {"deterministic_algorithms": enabled,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@contextmanager
def isolated_rng(seed):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng():
        seed_everything(seed)
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def new_run(parent, name):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(parent) / f"{stamp}-{name}-{uuid.uuid4().hex[:6]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def source_state():
    """Return lightweight source provenance for a run manifest.

    Runs may be created from an exported plan or a source archive, so Git
    metadata is intentionally best-effort. A missing repository is recorded as
    unknown instead of being inferred from a directory name.
    """
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip())
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def append_metrics(path, values):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(values, allow_nan=False) + "\n")


def model_inputs(batch, model):
    parameter = next(model.base.parameters())
    device = parameter.device
    labels = batch["labels"].to(device)
    prepare_decoder = getattr(model.base, "prepare_decoder_input_ids_from_labels", None)
    if callable(prepare_decoder):
        # Native preparation owns model-specific BOS/shift/padding semantics.
        # T5Gemma 2 stores BOS on config.decoder, not decoder_start_token_id.
        decoder = prepare_decoder(labels=labels)
    else:
        # Compatibility for lightweight/custom backbones without the HF API.
        pad_id = model.base.config.pad_token_id
        start_id = getattr(model.base.config, "decoder_start_token_id", None)
        if start_id is None:
            start_id = pad_id
        decoder = torch.full_like(labels, pad_id)
        decoder[:, 0] = start_id
        decoder[:, 1:] = labels[:, :-1].masked_fill(labels[:, :-1] == -100, pad_id)
    kwargs = {"input_ids": batch["input_ids"].to(device),
              "attention_mask": batch["attention_mask"].to(device),
              "decoder_input_ids": decoder, "use_cache": False}
    if "pixel_values" in batch:
        pixels = batch["pixel_values"]
        if pixels.ndim == 5:
            pixels = pixels.flatten(0, 1)
        kwargs["pixel_values"] = pixels.to(device=device, dtype=parameter.dtype)
    return kwargs, labels


def autocast_for(model):
    parameter = next(model.base.parameters())
    return torch.autocast(device_type=parameter.device.type, dtype=parameter.dtype,
                          enabled=parameter.dtype in (torch.bfloat16, torch.float16))


def prepare_trainable_parameters(model):
    """Keep communication parameters in FP32 beside a BF16 frozen backbone.

    ``SplitModel.build_model`` applies the requested execution dtype to the
    complete wrapper.  Training has a more specific precision contract: the
    frozen T5Gemma backbone can stay in BF16, while the trainable codec(s) and
    any trainable channel parameters use FP32.  Casting parameters by name
    would be brittle for custom channels, so cast every trainable module
    reachable through the communication components only.
    """

    for component_name in ("codec", "memory_codec", "channel"):
        component = getattr(model, component_name, None)
        if component is not None:
            component.float()
    return model


def promote_optimizer_state(optimizer):
    """Upcast floating AdamW state after loading an older checkpoint."""

    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value) and value.is_floating_point():
                state[key] = value.float()
    return optimizer


def _dtype_counts(values):
    counts = {}
    for value in values:
        key = str(value.dtype).replace("torch.", "")
        counts[key] = counts.get(key, 0) + 1
    return counts


def precision_telemetry(model, optimizer=None):
    """Return JSON-safe parameter/optimizer dtype evidence for a run."""

    parameters = list(model.named_parameters())
    trainable = [(name, value) for name, value in parameters if value.requires_grad]
    frozen = [(name, value) for name, value in parameters if not value.requires_grad]
    telemetry = {
        "parameters": _dtype_counts(value for _, value in parameters),
        "trainable_parameters": _dtype_counts(value for _, value in trainable),
        "frozen_parameters": _dtype_counts(value for _, value in frozen),
        "trainable_parameter_names": [name for name, _ in trainable],
    }
    if optimizer is not None:
        optimizer_values = [value for state in optimizer.state.values()
                            for value in state.values()
                            if torch.is_tensor(value) and value.is_floating_point()]
        telemetry["optimizer_state"] = _dtype_counts(optimizer_values)
    else:
        telemetry["optimizer_state"] = {}
    return telemetry


def parameter_update_l2(parameters, before, *, on_device=False):
    """Return the FP32 L2 magnitude of an optimizer update.

    The historical path copied every parameter delta to CPU.  The optional
    on-device reduction preserves the scalar value while moving only one
    final result across the device boundary, which is useful for low-sync
    logging benchmarks.
    """

    first = next(iter(parameters), None)
    if first is None:
        return 0.0
    device = first.device if on_device else torch.device("cpu")
    squared = torch.zeros((), dtype=torch.float64, device=device)
    for parameter, previous in zip(parameters, before):
        current = parameter.detach().float()
        prior = previous.float()
        if not on_device:
            current = current.cpu()
            prior = prior.cpu()
        squared += (current - prior).square().sum().double()
    return float(torch.sqrt(squared).item())


def sample_snr_db(model, low, high, batch_size):
    """Draw one SNR per sample, shaped for ``[batch, positions, channels]``."""

    parameter = next(model.base.parameters())
    return torch.empty((batch_size, 1, 1), device=parameter.device).uniform_(low, high)
