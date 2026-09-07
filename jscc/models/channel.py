"""Channel input/output: real tensor [batch, tokens, bottleneck], same shape.

The caller normalizes transmit power per sample. snr_db=None means no noise.
Custom Python channels implement the same forward(z, snr_db) method.
"""
import importlib

import torch
from torch import nn


def normalize_power(z):
    # Normalize over all token/channel dimensions within each sample.
    power = z.pow(2).mean(dim=tuple(range(1, z.ndim)), keepdim=True)
    return z / torch.sqrt(power + 1e-8)


class AWGNChannel(nn.Module):
    def forward(self, z, snr_db):
        if snr_db is None:
            return z
        return z + torch.randn_like(z) * (10.0 ** (-snr_db / 20.0))


class IdentityChannel(nn.Module):
    def forward(self, z, snr_db):
        return z


def build_channel(config):
    name = config["type"]
    if name == "awgn":
        cls = AWGNChannel
    elif name == "identity":
        cls = IdentityChannel
    else:
        module, cls_name = name.split(":")
        cls = getattr(importlib.import_module(module), cls_name)
    return cls(**config.get("kwargs", {}))
