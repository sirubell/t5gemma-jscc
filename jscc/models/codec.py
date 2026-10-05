"""Residual/factorized and direct affine codecs with explicit YAML designs."""

import copy
from contextlib import contextmanager
from unittest.mock import patch
from jscc.presentation import derived_seed
import torch
from torch import nn

from jscc.config import validate_codec_architecture


class BackboneRMSNorm(nn.Module):
    """Exact installed T5Gemma2RMSNorm semantics, including offset scale.

    Verified against Transformers 5.10.2 on the recorded 5090B runtime.
    RMS reduction and scale multiplication precede the input-dtype cast.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


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


class Codec(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        validate_codec_architecture(config)
        self.config = config
        self.film: nn.Sequential | None = None
        architecture = config.get("architecture", "residual_mlp")
        if architecture in ("direct_affine", "direct_outer_ln", "direct_outer_rms", "two_linear_gelu"):
            # Retain the public Sequential interface used by diagnostics. Each
            # boundary norm is at D, never at H or the transmitted bottleneck.
            norm = config["layernorm"]
            def boundary_norm():
                if norm == "both":
                    return nn.LayerNorm(input_dim)
                if norm == "rms_both":
                    return BackboneRMSNorm(input_dim, eps=config["rms_norm_eps"])
                return nn.Identity()
            self.input_norm = boundary_norm()
            self.output_norm = boundary_norm()
            bottleneck = config["bottleneck_dim"]
            if architecture == "two_linear_gelu":
                width = config["hidden_dim"]
                self.encoder = nn.Sequential(nn.Linear(input_dim, width), nn.GELU(),
                                             nn.Linear(width, bottleneck))
                self.decoder = nn.Sequential(nn.Linear(bottleneck, width), nn.GELU(),
                                             nn.Linear(width, input_dim))
            else:
                self.encoder = nn.Sequential(nn.Linear(input_dim, bottleneck))
                self.decoder = nn.Sequential(nn.Linear(bottleneck, input_dim))
            return
        width, bottleneck = config["hidden_dim"], config["bottleneck_dim"]
        activation = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[config["activation"]]
        norm = config["layernorm"]
        if norm not in ("none", "pre", "post", "both"):
            raise ValueError("codec.layernorm must be none, pre, post, or both")
        self.input_norm = nn.LayerNorm(input_dim) if norm in ("pre", "both") else nn.Identity()
        self.output_norm = nn.LayerNorm(input_dim) if norm in ("post", "both") else nn.Identity()

        def trunk():
            return [ResidualBlock(width, activation, config.get("dropout", 0.0))
                    for _ in range(config["n_res_blocks"])]

        self.encoder = nn.Sequential(nn.Linear(input_dim, width), *trunk(), nn.Linear(width, bottleneck))
        self.decoder = nn.Sequential(nn.Linear(bottleneck, width), *trunk(), nn.Linear(width, input_dim))
        if config["snr_film"]:
            film_input = nn.Linear(1, config["film_hidden"])
            film_output = nn.Linear(config["film_hidden"], 2 * width)
            self.film = nn.Sequential(film_input, activation(), film_output)
            nn.init.zeros_(film_output.weight)
            nn.init.zeros_(film_output.bias)

    def encode(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.input_norm(hidden))

    def decode(self, received: torch.Tensor, snr_db: float | None) -> torch.Tensor:
        hidden = self.decoder[0](received)
        if self.film is not None and snr_db is not None:
            snr = hidden.new_full((1, 1), snr_db / 20.0)
            conditioning: torch.Tensor = self.film(snr)
            gamma, beta = conditioning.chunk(2, dim=-1)
            hidden = hidden * (1.0 + gamma) + beta
        for layer in list(self.decoder.children())[1:]:
            hidden = layer(hidden)
        hidden = self.output_norm(hidden)
        return hidden


FAMILIES = ("affine_core", "shallow_nonlinear", "current_residual")


EXPECTED_PARAMETERS = dict(zip(FAMILIES, (1181312, 3837824, 14473088)))


class SimpleCodec(Codec):
    def __init__(self, input_dim, config, family):
        nn.Module.__init__(self)
        self.config = copy.deepcopy(config)
        self.config["n_res_blocks"] = 0
        width, bottleneck = config["hidden_dim"], config["bottleneck_dim"]
        self.input_norm, self.output_norm = nn.Identity(), nn.Identity()
        self.film = None
        if family == "affine_core":
            self.encoder = nn.Sequential(nn.Linear(input_dim, bottleneck))
            self.decoder = nn.Sequential(nn.Linear(bottleneck, input_dim))
        else:
            self.encoder = nn.Sequential(nn.Linear(input_dim, width), nn.GELU(), nn.Linear(width, bottleneck))
            self.decoder = nn.Sequential(nn.Linear(bottleneck, width), nn.GELU(), nn.Linear(width, input_dim))

    def encode(self, hidden):
        return self.encoder(hidden)

    def decode(self, received, snr_db):
        return self.decoder(received)


def projection_roles(codec, family) -> dict[str, nn.Linear]:
    roles = {"encoder_to_latent": codec.encoder[-1], "decoder_from_latent": codec.decoder[0]}
    if family != "affine_core":
        roles.update(encoder_from_hidden=codec.encoder[0], decoder_to_hidden=codec.decoder[-1])
    if not all(isinstance(module, nn.Linear) for module in roles.values()):
        raise TypeError("projection roles must identify Linear modules")
    return {role: module for role, module in roles.items() if isinstance(module, nn.Linear)}


def codec_factory(input_dim, config, family, seed=0):
    if family not in FAMILIES:
        raise ValueError(f"unknown experimental codec family: {family}")
    if config["layernorm"] != "none" or config["snr_film"] or config.get("dropout", 0) != 0:
        raise ValueError("E5 requires outer none, FiLM off, dropout zero")
    if config["activation"] != "gelu" or config["n_res_blocks"] != 2:
        raise ValueError("E5 reference requires GELU and two residual blocks")
    # Construction and role initialization never consume the caller RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator().manual_seed(derived_seed(seed, "e5:private:" + family)).get_state())
        codec = Codec(input_dim, config) if family == "current_residual" else SimpleCodec(input_dim, config, family)
        for role, module in projection_roles(codec, family).items():
            torch.set_rng_state(torch.Generator().manual_seed(
                derived_seed(seed, f"e5:projection:{role}:{tuple(module.weight.shape)}")).get_state())
            module.reset_parameters()
    return codec


@contextmanager
def codec_family_context(family, seed=0):
    """Only wrap model construction; restore the production factory on exit."""
    from jscc.models import split_model
    with patch.object(split_model, "Codec", lambda dim, cfg: codec_factory(dim, cfg, family, seed)):
        yield
