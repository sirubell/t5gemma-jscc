"""Check a channel without downloading a model or dataset.

uv run --locked python -m examples.check_channel --config configs/evaluation/wireless.yaml
"""
import argparse
from pathlib import Path

import torch
import yaml

from jscc.models.channel import build_channel, normalize_power


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    channel = build_channel(config["channel"])
    z = normalize_power(torch.randn(2, 7, 512)).detach().requires_grad_(True)
    received = channel(z, 0.0)
    assert received.shape == z.shape, "channel must restore [batch, tokens, bottleneck]"
    assert received.device == z.device and received.dtype == z.dtype
    assert torch.isfinite(received).all(), "channel output contains NaN/Inf"
    print("Tensor shape, dtype, device and finite values: PASS")
    if received.requires_grad:
        (received * torch.randn_like(received)).sum().backward()
    if z.grad is not None and torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0:
        print("Gradient reaches transmitted tensor: PASS")
    else:
        print("No usable gradient reached the input; use this channel for evaluation until resolved.")


if __name__ == "__main__":
    main()
