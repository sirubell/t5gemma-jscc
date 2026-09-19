"""Losses used by the frozen-backbone JSCC distillation loop.

The public helpers return a mean by default for compatibility with the old
training code.  The ``*_stats`` helpers expose numerators and denominators so
the caller can form one objective over an effective batch (including gradient
accumulation) instead of averaging already-normalised microbatch losses.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _valid_token_mask(labels: torch.Tensor) -> torch.Tensor:
    """Return the labels mask used by both the KL loss and its accounting."""

    return labels != -100


def distillation_loss_stats(
    student: torch.Tensor,
    teacher: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    *,
    valid_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(sum of token KLs, number of valid target tokens)``.

    KL is evaluated in float32 even when the backbone and the forward pass use
    BF16.  Returning the unnormalised numerator is intentional: an effective
    batch may contain different amounts of padding in its microbatches.
    """

    valid = _valid_token_mask(labels)
    if valid_only:
        # Selecting complete vocabulary rows after masking preserves the
        # exact KL objective while avoiding softmax/log-softmax work for
        # padded target positions.  Keep a differentiable zero for the
        # degenerate all-padding fixture.
        if not bool(valid.any()):
            return student.float().sum() * 0.0, valid.sum()
        student_float = student[valid].float() / temperature
        teacher_float = teacher[valid].float() / temperature
    else:
        student_float = student.float() / temperature
        teacher_float = teacher.float() / temperature
    token_kl = F.kl_div(
        F.log_softmax(student_float, dim=-1),
        F.softmax(teacher_float, dim=-1),
        reduction="none",
    ).sum(dim=-1) * temperature**2
    numerator = token_kl.sum() if valid_only else token_kl.masked_select(valid).sum()
    denominator = valid.sum()
    return numerator, denominator


def distillation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    *,
    reduction: str = "mean",
    valid_only: bool = False,
) -> torch.Tensor:
    """Compute KL(teacher || student) over non-padding target tokens."""

    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be mean or sum")
    numerator, denominator = distillation_loss_stats(
        student, teacher, labels, temperature, valid_only=valid_only
    )
    if reduction == "sum":
        return numerator
    return numerator / denominator.clamp_min(1).to(numerator.dtype)


def _expanded_mask(mask: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
    """Broadcast a ``[batch, positions]`` validity mask over feature axes."""

    if reference.ndim < 2:
        raise ValueError("reconstruction tensors must have batch and feature dimensions")
    if mask is None:
        return torch.ones(reference.shape, dtype=torch.bool, device=reference.device)
    mask = mask.to(device=reference.device, dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask[:, None]
    while mask.ndim < reference.ndim:
        mask = mask.unsqueeze(-1)
    try:
        return torch.broadcast_to(mask, reference.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"valid mask shape {tuple(mask.shape)} cannot broadcast to "
            f"reconstruction shape {tuple(reference.shape)}"
        ) from exc


def reconstruction_loss_stats(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a per-sample nMSE numerator and valid-sample count."""

    if reconstructed.shape != original.shape:
        raise ValueError(
            f"reconstruction shape {tuple(reconstructed.shape)} does not match "
            f"original shape {tuple(original.shape)}"
        )
    valid = _expanded_mask(mask, original)
    valid_float = valid.to(dtype=torch.float32)
    reduce_axes = tuple(range(1, original.ndim))
    counts = valid_float.sum(dim=reduce_axes)
    sample_valid = counts > 0
    error = (reconstructed.float() - original.float()).square() * valid_float
    power = original.float().square() * valid_float
    sample_error = error.sum(dim=reduce_axes) / counts.clamp_min(1.0)
    sample_power = power.sum(dim=reduce_axes) / counts.clamp_min(1.0)
    ratios = sample_error / (sample_power + 1e-8)
    numerator = ratios.masked_select(sample_valid).sum()
    denominator = sample_valid.sum()
    return numerator, denominator


def reconstruction_loss(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute normalized MSE.

    The no-mask call keeps the historical global nMSE contract for direct
    callers.  Corrected training passes a stream mask and uses the new
    per-sample statistic instead.
    """

    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be mean or sum")
    if mask is None:
        original_float = original.float()
        value = F.mse_loss(reconstructed.float(), original_float) / (
            original_float.square().mean() + 1e-8
        )
        if reduction == "sum":
            return value * reconstructed.numel()
        return value
    numerator, denominator = reconstruction_loss_stats(reconstructed, original, mask)
    if reduction == "sum":
        return numerator
    return numerator / denominator.clamp_min(1).to(numerator.dtype)


def aggregate_stream_numerators(
    streams: list[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Average stream nMSE means, keeping each stream equally weighted."""

    if not streams:
        raise ValueError("at least one reconstruction stream is required")
    means = [numerator / count.clamp_min(1).to(numerator.dtype)
             for numerator, count in streams]
    return torch.stack(means).mean()
