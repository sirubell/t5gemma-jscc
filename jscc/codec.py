"""Codec designs: transmitter encoder D -> b and receiver decoder b -> D.

Architectures (``codec.architecture``):
- ``linear``: one Linear each side (the October "direct affine" design).
- ``mlp``: ``depth`` hidden layers of width ``hidden_dim`` with an activation.
- ``residual``: Linear to ``hidden_dim``, ``n_res_blocks`` pre-norm residual
  blocks, Linear out (the July-September design).

Optional ``norm`` (none, layernorm) is applied at the D boundary on both sides.
Optional ``film`` conditions the receiver's first layer output on SNR.
Module layout keeps ``encoder``/``decoder`` as ``nn.Sequential`` so saved
state dicts of the earlier linear and residual designs load unchanged.
"""

import torch
from torch import nn

ACTIVATIONS = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}


class ResidualBlock(nn.Module):
    def __init__(self, width, activation, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.act = activation()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(width, width)
        nn.init.zeros_(self.fc2.bias)
        with torch.no_grad():
            self.fc2.weight.mul_(0.1)

    def forward(self, x):
        return x + self.fc2(self.drop(self.act(self.fc1(self.norm(x)))))


def _stack(dims, activation, dropout):
    layers = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2:
            layers.append(activation())
            if dropout:
                layers.append(nn.Dropout(dropout))
    return layers


class Codec(nn.Module):
    def __init__(self, input_dim: int, config: dict):
        super().__init__()
        self.config = dict(config)
        arch = config["architecture"]
        b = config["bottleneck_dim"]
        activation = ACTIVATIONS[config.get("activation", "gelu")]
        dropout = config.get("dropout", 0.0)
        if arch == "linear":
            self.encoder = nn.Sequential(nn.Linear(input_dim, b))
            self.decoder = nn.Sequential(nn.Linear(b, input_dim))
        elif arch == "mlp":
            h, depth = config["hidden_dim"], config["depth"]
            if depth < 1:
                raise ValueError("mlp depth must be >= 1; use architecture: linear for none")
            self.encoder = nn.Sequential(*_stack([input_dim] + [h] * depth + [b], activation, dropout))
            self.decoder = nn.Sequential(*_stack([b] + [h] * depth + [input_dim], activation, dropout))
        elif arch == "residual":
            h, n = config["hidden_dim"], config["n_res_blocks"]
            self.encoder = nn.Sequential(nn.Linear(input_dim, h),
                                         *[ResidualBlock(h, activation, dropout) for _ in range(n)],
                                         nn.Linear(h, b))
            self.decoder = nn.Sequential(nn.Linear(b, h),
                                         *[ResidualBlock(h, activation, dropout) for _ in range(n)],
                                         nn.Linear(h, input_dim))
        else:
            raise ValueError(f"unknown codec.architecture: {arch}")
        norm = config.get("norm", "none")
        if norm not in ("none", "layernorm"):
            raise ValueError("codec.norm must be none or layernorm")
        self.input_norm = nn.LayerNorm(input_dim) if norm == "layernorm" else nn.Identity()
        self.output_norm = nn.LayerNorm(input_dim) if norm == "layernorm" else nn.Identity()
        self.film = None
        if config.get("film", False):
            first: nn.Linear = self.decoder[0]  # type: ignore[assignment]
            width = first.out_features
            hidden = config.get("film_hidden", 64)
            out = nn.Linear(hidden, 2 * width)
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)
            self.film = nn.Sequential(nn.Linear(1, hidden), activation(), out)

    def encode(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.input_norm(hidden))

    def decode(self, received: torch.Tensor, snr_db=None) -> torch.Tensor:
        """``snr_db``: None, a float, or a per-sample tensor broadcastable to [batch, 1, 1]."""
        x = self.decoder[0](received)
        if self.film is not None:
            snr = self.config.get("film_clean_snr", 18.0) if snr_db is None else snr_db
            snr = torch.as_tensor(snr, dtype=x.dtype, device=x.device).reshape(-1, 1, 1) / 20.0
            gamma, beta = self.film(snr).chunk(2, dim=-1)
            x = x * (1.0 + gamma) + beta
        for layer in list(self.decoder.children())[1:]:
            x = layer(x)
        return self.output_norm(x)
