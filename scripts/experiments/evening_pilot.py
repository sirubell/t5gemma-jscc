"""Retained E5 codec-family initialization, configuration and selection helpers.

The historical executor is retired; imports preserve scientific interpretation.
"""

if __name__ == "__main__":
    raise SystemExit(
        "evening_pilot executor is retired; retained helpers are available by import."
    )
import copy
import random
from contextlib import contextmanager
from unittest.mock import patch

import torch
from torch import nn

from jscc.config import resolve_codec_configs
from jscc.models import split_model
from jscc.models.codec import Codec
from jscc.presentation import derived_seed

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
    with patch.object(split_model, "Codec", lambda dim, cfg: codec_factory(dim, cfg, family, seed)):
        yield


def experimental_build_model(config):
    family = config["experiment"]["codec_family"]
    if config["split"] != {"stack": "enc", "where": "after_layer", "index": 9}:
        raise ValueError("E5 factory only supports enc_l9")
    with codec_family_context(family, config["seed"]):
        return split_model.build_model(config)


def pilot_config(base, output, family):
    if family not in FAMILIES:
        raise ValueError("unknown family")
    cfg = copy.deepcopy(base)
    if cfg["split"] != {"stack": "enc", "where": "after_layer", "index": 9}:
        raise ValueError("E5 requires enc_l9")
    hidden, _ = resolve_codec_configs(cfg["codec"])
    if (hidden["hidden_dim"], hidden["bottleneck_dim"], hidden["layernorm"], hidden["snr_film"]) != (1152, 512, "none", False):
        raise ValueError("requires H1152/B512 outer none, FiLM off reference config")
    if cfg["protocol"] != "corrected-baseline-v2-native-decoder-inputs" or cfg["task"] != "hellaswag":
        raise ValueError("requires HellaSwag native-v2")
    policy = cfg["data"].get("prompt_policy", {})
    if policy.get("mode") != "five_shot" or policy.get("source_max_length") != 2048 or cfg["data"]["max_length"] != 512:
        raise ValueError("requires unchanged five-shot source2048/target512 policy")
    if cfg["seed"] != 0 or cfg["model"]["dtype"] != "bfloat16":
        raise ValueError("requires seed0/BF16 reference")
    if cfg["channel"]["type"] != "awgn" or not cfg["channel"]["train_noise"] or cfg["channel"]["train_snr_range"] != [-6, 18]:
        raise ValueError("requires production Uniform[-6,18] AWGN")
    if (cfg["training"]["kl_weight"], cfg["training"]["mse_weight"]) != (1., .1):
        raise ValueError("requires original KL + 0.1 nMSE objective")
    cfg["run"].update(name=f"evening-e5-{family}", output_dir=str(output / "runs"), wandb_project=None)
    cfg["experiment"] = {"name": "EVENING_SMALL_SUITE_V1", "codec_family": family,
                         "policy": "fresh-500-horizon-family-screen", "long_execution_authorized": False}
    cfg["training"].update(max_steps=500, schedule_steps=500, batch_size=64, gradient_accumulation=1,
        warmup_ratio=.05, lr=2e-4, log_every=50, eval_every=250, save_steps=[250, 500],
        validation_batches=0, validation_snrs=["no_noise"], patience=None, max_minutes=None,
        presentation_stream={"policy": "epoch-permutations-v1", "seed": 0, "total_presentations": 32000},
        paired_randomness={"policy": "step-microbatch-stream-v1", "seed": 0, "audit_steps": list(range(1, 501))})
    cfg["training"].pop("feature_summary", None)
    return cfg


def selection_indices(ids):
    if len(ids) < 128 or len(set(ids)) != len(ids):
        raise ValueError("selection requires at least128 unique original holdout IDs")
    positions = list(range(len(ids)))
    random.Random(20260921).shuffle(positions)
    return positions[:128]
