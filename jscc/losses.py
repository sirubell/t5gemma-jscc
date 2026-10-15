"""Distillation (KL) and reconstruction (nMSE) statistics.

Each helper returns ``(numerator, denominator)`` so an update made of several
microbatches can be normalized over the whole effective batch.
- KL: sum over valid target tokens of KL(teacher || student), FP32; denominator
  is the number of valid target tokens.
- nMSE: per sample, masked MSE divided by masked mean power of the original;
  numerator sums those ratios, denominator counts samples with a valid position.
"""
import torch
import torch.nn.functional as F


def kl_stats(student, teacher, labels, temperature=1.0):
    valid = labels != -100
    if not bool(valid.any()):
        return student.float().sum() * 0.0, valid.sum()
    parts = []
    for s, t in zip(student[valid].split(32), teacher[valid].split(32)):
        token_kl = F.kl_div(F.log_softmax(s.float() / temperature, dim=-1),
                            F.softmax(t.float() / temperature, dim=-1),
                            reduction="none").sum(dim=-1) * temperature ** 2
        parts.append(token_kl.sum())
    return torch.stack(parts).sum(), valid.sum()


def nmse_stats(reconstructed, original, mask):
    """``mask``: [batch, tokens] bool."""
    m = mask.to(device=original.device, dtype=torch.float32).unsqueeze(-1).expand_as(original)
    counts = m.sum(dim=(1, 2))
    error = ((reconstructed.float() - original.float()).square() * m).sum(dim=(1, 2)) / counts.clamp_min(1)
    power = (original.float().square() * m).sum(dim=(1, 2)) / counts.clamp_min(1)
    ratios = error / (power + 1e-8)
    has = counts > 0
    return ratios.masked_select(has).sum(), has.sum()
