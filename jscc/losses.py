"""Distillation loss: KL(teacher || student) + normalized MSE."""
import torch.nn.functional as F


def distillation_loss(student, teacher, labels, temperature=1.0):
    valid = labels != -100
    # Use fp32 for log probabilities even when the model runs in bfloat16.
    s = student[valid].float() / temperature
    t = teacher[valid].float() / temperature
    return F.kl_div(F.log_softmax(s, dim=-1), F.softmax(t, dim=-1),
                    reduction="batchmean") * temperature ** 2


def reconstruction_loss(reconstructed, original):
    original = original.float()
    return F.mse_loss(reconstructed.float(), original) / (original.square().mean() + 1e-8)
