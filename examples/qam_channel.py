"""Example physical-layer channel: trainable mapping to 16-QAM, complex AWGN, trainable demapping.

It shows the parts of the channel contract a physical-layer simulator usually needs:
- trainable parameters (optimized and saved with the codec),
- its own power handling (``normalizes_power = True``: the constellation has unit power),
- one SNR per call (``snr_mode = "per_batch"``),
- an auxiliary loss (``self.aux_loss``), weighted by ``training.channel_aux_weight``.

Each pair of latent coordinates becomes one complex symbol, so b coordinates use b/2 symbols.
Symbol choice uses straight-through Gumbel-softmax, as in a learned-modulation design.
"""
import torch
import torch.nn.functional as F
from torch import nn


def qam16():
    levels = torch.tensor([-3.0, -1.0, 1.0, 3.0])
    i, q = torch.meshgrid(levels, levels, indexing="xy")
    points = torch.stack([i.reshape(-1), q.reshape(-1)], dim=1)
    return points / points.pow(2).sum(1).mean().sqrt()          # unit average symbol energy


class LearnedQAMChannel(nn.Module):
    normalizes_power = True
    snr_mode = "per_batch"

    def __init__(self, tau: float = 1.0, hidden: int = 32):
        super().__init__()
        self.tau = tau
        self.mapper = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, 16))
        self.demapper = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, 2))
        self.register_buffer("points", qam16())
        self.aux_loss = None

    def forward(self, z, snr_db):
        if z.shape[-1] % 2:
            raise ValueError("bottleneck width must be even (two coordinates per complex symbol)")
        pairs = z.reshape(*z.shape[:-1], -1, 2).float()           # [batch, tokens, b/2, (I, Q)]
        logits = self.mapper(pairs)
        onehot = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)
        points: torch.Tensor = self.points  # type: ignore[assignment]
        symbols = onehot @ points                             # [.., b/2, 2], unit power
        # Encourage use of the whole constellation (entropy of the average symbol distribution).
        usage = logits.softmax(-1).reshape(-1, 16).mean(0)
        self.aux_loss = (usage * usage.clamp_min(1e-9).log()).sum()
        if snr_db is not None:
            sigma = 10.0 ** (-float(snr_db) / 20.0) / 2 ** 0.5    # per real dimension
            symbols = symbols + sigma * torch.randn_like(symbols)
        return self.demapper(symbols).reshape(z.shape).to(z.dtype)
