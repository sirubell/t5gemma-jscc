"""Small run folders, random seeds, and device helpers."""
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import json
import random
import uuid

import numpy as np
import torch


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


def append_metrics(path, values):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(values, allow_nan=False) + "\n")


def model_inputs(batch, model):
    parameter = next(model.base.parameters())
    device = parameter.device
    labels = batch["labels"].to(device)
    pad_id = model.base.config.pad_token_id
    # Teacher forcing falls back to pad when the backbone omits decoder_start_token_id.
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
