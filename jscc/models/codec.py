"""One residual MLP codec for COCO and HellaSwag; all choices live in YAML."""
import torch
from torch import nn


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
        self.config = config
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
        self.film: nn.Sequential | None = None
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
