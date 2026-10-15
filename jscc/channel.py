"""Channels and transmit-power normalization.

Channel contract (any ``nn.Module``; see docs/channel-integration.md):

    forward(z, snr_db) -> same shape as z
        z:      real tensor [batch, tokens, b]
        snr_db: None (no noise), a float, or a per-sample tensor [batch, 1, 1]

Optional class attributes:
    normalizes_power = True   the caller skips power normalization
    snr_mode = "per_batch"    the caller passes one float per call
After ``forward`` a channel may set ``self.aux_loss`` (scalar tensor); the
trainer adds ``channel.aux_weight * aux_loss`` to the objective.
Trainable parameters are optimized and checkpointed with the codec.
"""
import importlib

import torch
from torch import nn


def token_mask(mask: torch.Tensor | None, z: torch.Tensor) -> torch.Tensor | None:
    """Reduce a 2D 0/1 mask or a 4D additive attention mask to [batch, tokens] bool."""
    if mask is None:
        return None
    mask = mask.to(z.device)
    if mask.ndim == 2:
        return mask != 0
    if mask.ndim == 4:
        visible = torch.isfinite(mask) & (mask > -1e4) if mask.is_floating_point() else mask != 0
        return visible.any(dim=1).any(dim=-2)
    raise ValueError(f"unsupported mask shape {tuple(mask.shape)}")


def normalize_power(z: torch.Tensor, mask: torch.Tensor | None = None, *, token_wise: bool = False):
    """Unit average power per sample (over valid tokens), or per token if ``token_wise``.

    BF16/FP16 inputs are promoted to FP32.
    """
    if z.dtype in (torch.bfloat16, torch.float16):
        z = z.float()
    if token_wise:
        return z / torch.sqrt(z.pow(2).mean(dim=-1, keepdim=True) + 1e-8)
    if mask is None:
        power = z.pow(2).mean(dim=tuple(range(1, z.ndim)), keepdim=True)
    else:
        m = mask.to(z.dtype).unsqueeze(-1)
        count = mask.sum(dim=1, dtype=torch.int64).reshape(-1, 1, 1) * z.shape[-1]
        power = (z.pow(2) * m).sum(dim=(1, 2), keepdim=True) / count.clamp_min(1)
        z = torch.where(count > 0, z, torch.zeros_like(z))
    return z / torch.sqrt(power + 1e-8)


class AWGNChannel(nn.Module):
    """Real AWGN with noise std 10^(-SNR/20), assuming unit signal power."""

    def forward(self, z, snr_db):
        if snr_db is None:
            return z
        return z + torch.randn_like(z) * (10.0 ** (-torch.as_tensor(snr_db, dtype=z.dtype, device=z.device) / 20.0))


class IdentityChannel(nn.Module):
    def forward(self, z, snr_db):
        return z


def build_channel(config: dict) -> nn.Module:
    name = config["type"]
    if name == "awgn":
        return AWGNChannel()
    if name == "identity":
        return IdentityChannel()
    module, _, cls = name.partition(":")
    if not cls:
        raise ValueError("custom channel.type must be 'package.module:ClassName'")
    return getattr(importlib.import_module(module), cls)(**config.get("kwargs", {}))
