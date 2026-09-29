"""Opt-in E5 codec-family pilot; never submits jobs or resumes checkpoints.

The coordinator must reserve budget for all three families and six evaluation
panels before invoking any run. Production model/data/loss defaults are untouched.
"""
import argparse
import copy
import gzip
import hashlib
import json
import random
import time
from contextlib import contextmanager, ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from jscc import training
from jscc.config import load_config, resolve_codec_configs
from jscc.models import split_model
from jscc.models.codec import Codec
from jscc.presentation import derived_seed, tensor_digest

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


def run_pilot(config, output, seconds=2600):
    """Run one reserved family; all scoring stays in the separate evaluator."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = pilot_config(config, output, config["experiment"]["codec_family"])
    family = config["experiment"]["codec_family"]
    report = {"experiment": "E5", "family": family, "status": "RUNNING", "training_presentations": 0,
              "selection_presentations": 0, "optimizer_calls": 0, "completed_updates": 0,
              "main_codec_calls": 0, "channel_calls": 0, "source": training.source_state(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "timing_boundary": "host wall time includes setup, training, selection and checkpoint I/O"}
    started = time.monotonic()
    original_load, original_loss = training.load_data, training.batch_losses
    original_step, original_build = torch.optim.AdamW.step, experimental_build_model
    handles = []

    def flush():
        report["wall_seconds"] = time.monotonic() - started
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def deadline():
        if time.monotonic() - started >= seconds:
            raise TimeoutError("E5 reserved wall-time exhausted; no retry")

    def build(cfg):
        processor, model = original_build(cfg)
        params = sum(p.numel() for p in model.codec.parameters())
        if params != EXPECTED_PARAMETERS[family] or model.memory_codec is not None:
            raise ValueError("unexpected E5 parameter count or memory stream")
        report["parameters"] = params
        report["initial_state"] = {k: tensor_digest(v) for k, v in model.codec.state_dict().items()}
        report["projection_initialization"] = {role: {k: tensor_digest(v) for k, v in mod.state_dict().items()}
                                                for role, mod in projection_roles(model.codec, family).items()}
        def count_channel(*args):
            report["channel_calls"] += 1
        handles.append(model.channel.register_forward_pre_hook(count_channel))
        # encode is called directly, so a module forward hook would never count it.
        encode = model.codec.encode
        def counted_encode(*args, **kwargs):
            report["main_codec_calls"] += 1
            return encode(*args, **kwargs)
        model.codec.encode = counted_encode
        return processor, model

    def load(*args, **kwargs):
        data = original_load(*args, **kwargs)
        positions = selection_indices(data.ids["validation_rows"])
        report["selection_ids"] = [data.ids["validation_rows"][i] for i in positions]
        loader = data.validation
        data.validation = DataLoader(Subset(loader.dataset, positions), batch_size=64, shuffle=False,
                                      collate_fn=loader.collate_fn, num_workers=0)
        # Retain full original reserved holdout in data.ids to exclude every reserved
        # row from optimization/demonstration pools; subset is recorded separately.
        return data

    def losses(model, batch, settings, snr, **kwargs):
        deadline()
        is_train = "row_ids" in batch
        key = "training_presentations" if is_train else "selection_presentations"
        count = batch["labels"].shape[0]
        cap = 32000 if is_train else 256
        if report[key] + count > cap:
            raise RuntimeError("E5 presentation cap exceeded before dispatch")
        report[key] += count
        native, _ = training.model_inputs(batch, model)
        record = {"phase": "train" if is_train else "selection", "update": report["completed_updates"],
                  "batch": {k: v.cpu().tolist() for k, v in batch.items() if torch.is_tensor(v)},
                  "native_decoder_input_ids": native["decoder_input_ids"].cpu().tolist(),
                  "native_decoder_sha256": tensor_digest(native["decoder_input_ids"]),
                  "snr_sha256": tensor_digest(snr) if torch.is_tensor(snr) else None}
        with gzip.open(output / "presented_native_inputs.jsonl.gz", "at") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        flush()
        values = original_loss(model, batch, settings, snr, **kwargs)
        if not is_train:
            evidence = {"update": report["completed_updates"], "count": count,
                        "components": {k: float(v.detach()) if torch.is_tensor(v) else v
                                       for k, v in values.items()},
                        "hidden_allocated": model.channel_uses["hidden"],
                        "hidden_valid": model.channel_uses_valid["hidden"]}
            with (output / "selection_components.jsonl").open("a") as stream:
                stream.write(json.dumps(evidence, allow_nan=False) + "\n")
        return values

    def step(optimizer, *args, **kwargs):
        deadline()
        if report["optimizer_calls"] >= 500 or (report["optimizer_calls"] == 0 and optimizer.state):
            raise RuntimeError("E5 update cap or fresh optimizer contract violated")
        report["optimizer_calls"] += 1
        result = original_step(optimizer, *args, **kwargs)
        report["completed_updates"] += 1
        return result

    try:
        with ExitStack() as stack:
            for target, name, replacement in ((training, "build_model", build), (training, "load_data", load),
                    (training, "batch_losses", losses), (torch.optim.AdamW, "step", step)):
                stack.enter_context(patch.object(target, name, replacement))
            report["run"] = str(training.train(config))
        if (report["completed_updates"], report["training_presentations"], report["selection_presentations"]) != (500, 32000, 256):
            raise RuntimeError("pilot stopped short of its declared budget")
        report["status"] = "MEASURED"
    except BaseException as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for handle in handles:
            handle.remove()
        flush()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--seconds", type=int, default=2600)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 2600:
        parser.error("seconds must be within1..2600; allocation cap is45 minutes")
    config = load_config(args.config)
    config["experiment"] = {"codec_family": args.family}
    run_pilot(config, args.output, args.seconds)


if __name__ == "__main__":
    main()
