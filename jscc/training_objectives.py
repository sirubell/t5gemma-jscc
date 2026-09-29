"""Shared effective-batch objective accounting for functional and local training.

Keep numerical reductions and legacy/explicit loss weighting in one place.
Model forwards, optimizer steps and checkpoint lifecycles belong to the learners.
"""
from typing import NotRequired, TypedDict, cast

import torch

from .losses import aggregate_stream_numerators


class BatchValues(TypedDict):
    loss: torch.Tensor
    kl: torch.Tensor
    nmse: torch.Tensor
    kl_numerator: NotRequired[torch.Tensor]
    kl_denominator: NotRequired[torch.Tensor]
    hidden_numerator: NotRequired[torch.Tensor]
    hidden_denominator: NotRequired[torch.Tensor]
    memory_numerator: NotRequired[torch.Tensor | None]
    memory_denominator: NotRequired[torch.Tensor | None]

def objective_components(kl, hidden, memory, settings, *, stream_count=None):
    """Explicit weights override legacy kl/mse; no second stream averaging."""
    count = stream_count if stream_count is not None else (2 if memory is not None else 1)
    explicit = settings.get("loss_weights")
    weights = explicit or {"kl": settings["kl_weight"],
                           "hidden": settings["mse_weight"] / count,
                           "memory": settings["mse_weight"] / count}
    memory_value = memory if memory is not None else torch.zeros_like(hidden)
    terms = {"weighted_kl": weights["kl"] * kl,
             "weighted_hidden": weights["hidden"] * hidden,
             "weighted_memory": weights["memory"] * memory_value}
    return {**terms, "hidden": hidden, "memory": memory_value,
            "weight_kl": kl.new_tensor(weights["kl"]),
            "weight_hidden": kl.new_tensor(weights["hidden"]),
            "weight_memory": kl.new_tensor(weights["memory"]),
            "loss": terms["weighted_kl"] + terms["weighted_hidden"] + terms["weighted_memory"]}

def _valid_sample_count(mask: torch.Tensor) -> torch.Tensor:
    """Count samples with at least one valid position without model work."""

    if mask.ndim == 1:
        mask = mask[:, None]
    return mask.to(dtype=torch.bool).reshape(mask.shape[0], -1).any(dim=1).sum()

def effective_batch_denominators(model, batches: list[dict], device) -> dict[str, torch.Tensor | int]:
    """Return the fixed denominators required by streamed backward.

    Counts depend only on labels and validity masks, so they can be collected
    before the first forward without retaining a computation graph.  The
    resulting objective is the same global effective-batch objective used by
    ``aggregate_batch_losses``.
    """

    kl = torch.zeros((), device=device, dtype=torch.float32)
    hidden = torch.zeros((), device=device, dtype=torch.float32)
    memory = torch.zeros((), device=device, dtype=torch.float32)
    has_memory = getattr(model, "memory_codec", None) is not None
    stack = model.split.get("stack") if hasattr(model, "split") else "enc"
    for batch in batches:
        labels = batch["labels"].to(device=device)
        encoder_mask = batch["attention_mask"].to(device=device, dtype=torch.bool)
        kl = kl + (labels != -100).sum().to(dtype=torch.float32)
        hidden_mask = labels != -100 if stack == "dec" else encoder_mask
        hidden = hidden + _valid_sample_count(hidden_mask).to(dtype=torch.float32)
        if has_memory:
            memory = memory + _valid_sample_count(encoder_mask).to(dtype=torch.float32)
    return {"kl": kl, "hidden": hidden, "memory": memory,
            "stream_count": 2 if has_memory else 1}

def scaled_batch_loss(values: BatchValues, training, denominators) -> torch.Tensor:
    """Scale one microbatch so its backward contributes to the global loss."""

    kl_numerator = _required_stat(values, "kl_numerator")
    hidden_numerator = _required_stat(values, "hidden_numerator")
    kl_denominator = cast(torch.Tensor, denominators["kl"]).clamp_min(1).to(
        dtype=kl_numerator.dtype)
    kl = kl_numerator / kl_denominator
    streams = [
        hidden_numerator /
        cast(torch.Tensor, denominators["hidden"]).clamp_min(1).to(
            dtype=hidden_numerator.dtype)
    ]
    memory_numerator = values.get("memory_numerator")
    if memory_numerator is not None:
        memory_denominator = cast(torch.Tensor, denominators["memory"]).clamp_min(1).to(
            dtype=memory_numerator.dtype)
        streams.append(memory_numerator / memory_denominator)
    return objective_components(kl, streams[0], streams[1] if len(streams) > 1 else None,
                                training, stream_count=denominators["stream_count"])["loss"]

def detached_batch_values(values: BatchValues) -> BatchValues:
    """Drop computation graphs before retaining stats for a later log row."""

    return cast(BatchValues, {
        key: value.detach() if torch.is_tensor(value) else value
        for key, value in values.items()
    })

def _required_stat(values: BatchValues, name: str) -> torch.Tensor:
    value = values.get(name)
    if value is None:
        raise RuntimeError(f"missing required batch statistic: {name}")
    return value

def aggregate_batch_losses(values: list[BatchValues], training) -> dict[str, torch.Tensor]:
    """Form the objective over all microbatches in one optimizer update."""

    if not values:
        raise ValueError("at least one microbatch is required")
    kl_numerator = _required_stat(values[0], "kl_numerator")
    kl_numerator = sum((_required_stat(item, "kl_numerator") for item in values[1:]), kl_numerator)
    kl_denominator = _required_stat(values[0], "kl_denominator")
    kl_denominator = sum((_required_stat(item, "kl_denominator") for item in values[1:]), kl_denominator)
    kl = kl_numerator / kl_denominator.clamp_min(1).to(kl_numerator.dtype)
    hidden = [(_required_stat(item, "hidden_numerator"),
               _required_stat(item, "hidden_denominator")) for item in values]
    streams = [(sum((item[0] for item in hidden[1:]), hidden[0][0]),
               sum((item[1] for item in hidden[1:]), hidden[0][1]))]
    memory: list[tuple[torch.Tensor, torch.Tensor]] = []
    for item in values:
        memory_numerator = item.get("memory_numerator")
        memory_denominator = item.get("memory_denominator")
        if memory_numerator is not None:
            if memory_denominator is None:
                raise RuntimeError("memory denominator is missing for a memory reconstruction")
            memory.append((memory_numerator, memory_denominator))
    if memory:
        streams.append((sum((item[0] for item in memory[1:]), memory[0][0]),
                        sum((item[1] for item in memory[1:]), memory[0][1])))
    nmse = aggregate_stream_numerators(streams)
    components = objective_components(kl, streams[0][0] / streams[0][1].clamp_min(1),
        streams[1][0] / streams[1][1].clamp_min(1) if memory else None, training)
    result = {**components,
              "kl": kl, "nmse": nmse, "kl_numerator": kl_numerator,
              "kl_denominator": kl_denominator,
              "hidden_numerator": streams[0][0], "hidden_denominator": streams[0][1]}
    if memory:
        result.update(memory_numerator=streams[1][0], memory_denominator=streams[1][1])
    return result
