"""Check a channel class without loading a model or dataset.

    python -m examples.check_channel examples.qam_channel:LearnedQAMChannel
"""
import importlib
import sys

import torch

from jscc.channel import normalize_power


def main(name):
    module, _, cls = name.partition(":")
    channel = getattr(importlib.import_module(module), cls)()
    per_batch = getattr(channel, "snr_mode", "per_sample") == "per_batch"
    z = normalize_power(torch.randn(2, 7, 64)).requires_grad_(True)
    for snr in (None, 0.0 if per_batch else torch.zeros(2, 1, 1)):
        out = channel(z, snr)
        assert out.shape == z.shape and out.dtype == z.dtype, "output must keep shape and dtype"
        assert torch.isfinite(out).all(), "output contains NaN/Inf"
    out.sum().backward()
    ok = z.grad is not None and z.grad.abs().sum() > 0
    print("shape/dtype/finite: PASS")
    print("gradient reaches z:", "PASS" if ok else "NO (evaluation-only channel)")
    print("trainable parameters:", sum(p.numel() for p in channel.parameters() if p.requires_grad))
    print("own power normalization:", getattr(channel, "normalizes_power", False), "| snr_mode:",
          getattr(channel, "snr_mode", "per_sample"), "| aux_loss:", getattr(channel, "aux_loss", None) is not None)


if __name__ == "__main__":
    main(sys.argv[1])
