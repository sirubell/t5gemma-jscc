"""Bounded full-weight HellaSwag encoder-hook versus local-replay probe.

The probe performs no optimizer update, cache write, or task evaluation. A
separate scheduler limit is required because Python cannot preempt a GPU call.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jscc.activation_replay import canonical_digest, capture_encoder_microbatch
from jscc.config import load_config
from jscc.local_reconstruction import local_batch_values, paired_snr
from jscc.losses import reconstruction_loss_stats
from jscc.models.channel import AWGNChannel
from jscc.models.split_model import stack_module
from jscc.presentation import training_source_digest
from jscc.runtime import (autocast_for, configure_training_determinism,
                          prepare_trainable_parameters, seed_everything, source_state)
from scripts.hellaswag_two_stage import check_recipe


# Declared before any target data or result is observed. Activation capture uses
# the identical frozen encoder, so it must be bitwise equal; nMSE and gradient
# reductions may differ slightly at the BF16/FP32 boundary.
TOLERANCES = {"activation_atol": 0.0, "activation_rtol": 0.0,
              "nmse_atol": 1e-5, "nmse_rtol": 1e-4,
              "gradient_atol": 1e-5, "gradient_rtol": 1e-3}
BASELINE_PATH = Path(__file__).resolve().parents[1] / "configs/tasks/hellaswag.yaml"
SCIENTIFIC_FIELDS = ("task", "protocol", "seed", "model", "split", "codec",
                     "channel", "data", "evaluation")
ALLOWED_TRAINING_CHANGES = {"max_steps", "schedule_steps", "eval_every", "min_steps",
                            "patience", "save_steps", "selection_steps", "feature_summary",
                            "log_every", "validation_batches"}


class _EncoderBoundaryReached(Exception):
    pass


def check_preflight_recipe(config: dict) -> dict:
    budget = check_recipe(config)
    baseline = load_config(BASELINE_PATH)
    for field in SCIENTIFIC_FIELDS:
        current = copy.deepcopy(config[field])
        expected = copy.deepcopy(baseline[field])
        if field == "codec":
            current.pop("input_dim", None)
            expected.pop("input_dim", None)
        if current != expected:
            raise ValueError(f"preflight {field} differs from current HellaSwag baseline")
    current_training = copy.deepcopy(config["training"])
    baseline_training = copy.deepcopy(baseline["training"])
    for settings in (current_training, baseline_training):
        for key in ALLOWED_TRAINING_CHANGES:
            settings.pop(key, None)
        settings["presentation_stream"].pop("total_presentations", None)
        settings["presentation_stream"].pop("start_presentation", None)
        settings["paired_randomness"].pop("audit_steps", None)
    if current_training != baseline_training:
        raise ValueError("preflight training semantics differ from current HellaSwag baseline")
    if budget["batch_size"] != 64 or budget["accumulation"] != 1:
        raise ValueError("full-weight preflight requires the unchanged 64x1 batch policy")
    if budget["presentation_start"] != 0:
        raise ValueError("preflight requires the first fixed-stream batch")
    if config["model"]["device"] != "cuda" or config["model"]["dtype"] != "bfloat16":
        raise ValueError("full-weight preflight requires CUDA BF16")
    if not config["training"].get("deterministic_algorithms"):
        raise ValueError("preflight requires deterministic training policy")
    if not config["training"].get("streamed_backward"):
        raise ValueError("preflight requires the production streamed-backward policy")
    scientific_identity = {field: copy.deepcopy(baseline[field]) for field in SCIENTIFIC_FIELDS}
    scientific_identity["codec"].pop("input_dim", None)
    return {**budget,
            "baseline_path": str(BASELINE_PATH),
            "baseline_resolved_sha256": canonical_digest(baseline),
            "scientific_identity_sha256": canonical_digest(scientific_identity)}


def _deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("preflight wall deadline reached")


def _sync(model) -> None:
    if next(model.base.parameters()).device.type == "cuda":
        torch.cuda.synchronize()


def _timed(model, action):
    _sync(model)
    started = time.monotonic()
    result = action()
    _sync(model)
    return result, time.monotonic() - started


def _gradient_stats(reference: torch.Tensor, observed: torch.Tensor) -> dict:
    reference = reference.detach().to(device="cpu", dtype=torch.float32)
    observed = observed.detach().to(device="cpu", dtype=torch.float32)
    difference = observed - reference
    return {"reference_l2": float(torch.linalg.vector_norm(reference)),
            "observed_l2": float(torch.linalg.vector_norm(observed)),
            "difference_l2": float(torch.linalg.vector_norm(difference)),
            "max_abs_error": float(difference.abs().max()),
            "allclose": bool(torch.allclose(reference, observed,
                atol=TOLERANCES["gradient_atol"], rtol=TOLERANCES["gradient_rtol"]))}


def _online_values(model, batch: dict, snr, replay_seed: int):
    """Use the production encoder hook and stop immediately after its codec."""
    layer = stack_module(model.base, "enc").layers[9]
    reached = []
    masks = []

    def stop(_module, _args, _output):
        if model.activation is None or model.reconstruction is None:
            raise RuntimeError("production codec hook did not record its boundary")
        mask = model._active_encoder_valid_mask
        if mask is None:
            raise RuntimeError("production codec hook has no encoder mask")
        masks.append(mask.detach().cpu().clone())
        reached.append(True)
        raise _EncoderBoundaryReached

    handle = layer.register_forward_hook(stop)
    device = next(model.base.parameters()).device
    model.activation = model.reconstruction = None
    try:
        context = (model.channel.replay(replay_seed, "train:1:0", capture=snr is not None)
                   if isinstance(model.channel, AWGNChannel) else nullcontext())
        with torch.enable_grad(), context, autocast_for(model), model.transmission(
                snr, encoder_mask=batch["attention_mask"]), model.attention_context():
            try:
                model.base.get_encoder()(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    return_dict=True,
                )
            except _EncoderBoundaryReached:
                pass
    finally:
        handle.remove()
    if len(reached) != 1:
        raise RuntimeError("production codec boundary was not reached exactly once")
    activation, reconstructed = model.activation, model.reconstruction
    assert activation is not None and reconstructed is not None
    numerator, denominator = reconstruction_loss_stats(
        reconstructed, activation, batch["attention_mask"].to(device))
    draws = [dict(item) for item in model.channel.draw_summaries]
    return activation, numerator / denominator.clamp_min(1), denominator, masks[0], draws


def compare_one_batch(model, batch: dict, config: dict,
                      *, deadline: float | None = None, on_condition=None) -> dict:
    """Compare one exact production microbatch without stepping parameters."""
    _deadline(deadline)
    expected = config["training"]["batch_size"]
    if batch["input_ids"].shape[0] != expected or batch["row_ids"].numel() != expected:
        raise ValueError("preflight requires one complete declared microbatch")
    if batch["attention_mask"].shape != batch["input_ids"].shape:
        raise ValueError("source mask and token shapes differ")
    if not bool(torch.all((batch["attention_mask"] == 0) | (batch["attention_mask"] == 1))):
        raise ValueError("source mask is not binary")
    model.train()
    model.base.eval()
    parameters = [(name, parameter) for name, parameter in model.codec.named_parameters()
                  if parameter.requires_grad]
    if not parameters:
        raise ValueError("codec has no trainable parameters")
    if any(parameter.grad is not None for parameter in model.base.parameters()):
        raise RuntimeError("frozen backbone has an existing gradient")
    if next(model.base.parameters()).device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    captured, capture_seconds = _timed(model, lambda: capture_encoder_microbatch(model, batch))
    if not bool(torch.isfinite(captured).all()):
        raise FloatingPointError("nonfinite captured activation")
    shard = {**batch, "activation": captured.detach().cpu().clone()}
    result = {"batch_size": expected,
              "row_ids_sha256": canonical_digest(batch["row_ids"].tolist()),
              "source_shape": list(batch["input_ids"].shape),
              "activation_shape": list(captured.shape),
              "activation_dtype": str(captured.dtype),
              "activation_bytes": captured.numel() * captured.element_size(),
              "valid_vectors": int(batch["attention_mask"].sum()),
              "capture_seconds": capture_seconds,
              "conditions": {}}
    for label in ("no_noise", "fixed_awgn"):
        _deadline(deadline)
        snr = None if label == "no_noise" else paired_snr(model, config, 1, 0, expected)
        replay_seed = config["training"]["paired_randomness"]["seed"]
        (online, online_loss, denominator, online_mask, online_draws), online_seconds = _timed(
            model, lambda: _online_values(model, batch, snr, replay_seed))
        if not bool(torch.isfinite(online).all()) or not bool(torch.isfinite(online_loss)):
            raise FloatingPointError(f"nonfinite online tensor or nMSE: {label}")
        online_grads, online_grad_seconds = _timed(model, lambda: torch.autograd.grad(
            online_loss, tuple(parameter for _, parameter in parameters), allow_unused=True))
        online_grads = tuple(grad.detach().cpu() if grad is not None else None for grad in online_grads)
        activation_error = (online.detach().float() - captured.float()).abs()
        activation_equal = bool(torch.equal(online.detach(), captured))
        mask_equal = bool(torch.equal(online_mask, shard["attention_mask"]))
        model.activation = model.reconstruction = None
        _deadline(deadline)
        context = (model.channel.replay(replay_seed, "train:1:0", capture=snr is not None)
                   if isinstance(model.channel, AWGNChannel) else nullcontext())
        def replay():
            with torch.enable_grad(), context:
                return local_batch_values(model, shard, snr, config["training"])
        local, replay_seconds = _timed(model, replay)
        replay_draws = [dict(item) for item in model.channel.draw_summaries]
        if not bool(torch.isfinite(local["nmse"])):
            raise FloatingPointError(f"nonfinite replay nMSE: {label}")
        replay_grads, replay_grad_seconds = _timed(model, lambda: torch.autograd.grad(
            local["nmse"], tuple(parameter for _, parameter in parameters), allow_unused=True))
        gradient_rows = {}
        gradient_ok = False
        for (name, _), reference, observed in zip(parameters, online_grads, replay_grads):
            if reference is None or observed is None:
                raise RuntimeError(f"missing codec gradient: {name}")
            if not bool(torch.isfinite(reference).all()) or not bool(torch.isfinite(observed).all()):
                raise FloatingPointError(f"nonfinite codec gradient: {name}")
            stats = _gradient_stats(reference, observed)
            gradient_rows[name] = stats
            gradient_ok = gradient_ok or stats["reference_l2"] > 0
        if not gradient_ok:
            raise RuntimeError("all codec gradients are zero")
        nmse_reference = float(online_loss.detach())
        nmse_observed = float(local["nmse"].detach())
        noise_draws_equal = online_draws == replay_draws and (
            len(online_draws) == 1 if snr is not None else len(online_draws) == 0)
        condition = {"activation_equal": activation_equal, "mask_equal": mask_equal,
                     "noise_draws_equal": noise_draws_equal,
                     "noise_draws": online_draws,
                     "activation_max_abs_error": float(activation_error.max()),
                     "nmse_reference": nmse_reference, "nmse_replay": nmse_observed,
                     "nmse_abs_error": abs(nmse_observed - nmse_reference),
                     "valid_samples_online": int(denominator),
                     "valid_samples_replay": int(local.get("hidden_denominator", 0)),
                     "gradients": gradient_rows,
                     "online_seconds": online_seconds + online_grad_seconds,
                     "replay_seconds": replay_seconds + replay_grad_seconds}
        condition["passed"] = (activation_equal and mask_equal and noise_draws_equal and
            condition["valid_samples_online"] == condition["valid_samples_replay"] == expected and
            math.isclose(nmse_reference, nmse_observed,
                         abs_tol=TOLERANCES["nmse_atol"], rel_tol=TOLERANCES["nmse_rtol"]) and
            all(row["allclose"] for row in gradient_rows.values()))
        result["conditions"][label] = condition
        if on_condition is not None:
            on_condition(label, condition)
        if not condition["passed"]:
            raise AssertionError(f"online/replay parity failed: {label}")
    if any(parameter.grad is not None for parameter in model.base.parameters()):
        raise RuntimeError("frozen backbone acquired a gradient")
    if next(model.base.parameters()).device.type == "cuda":
        result["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_memory_reserved_bytes"] = torch.cuda.max_memory_reserved()
    return result


def _write_report(path: Path, report: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, help="fresh JSON receipt path")
    parser.add_argument("--max-seconds", required=True, type=float)
    parser.add_argument("--check-only", action="store_true", help="validate recipe without loading model/data/GPU")
    args = parser.parse_args()
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("--max-seconds must be finite and positive")
    output = Path(args.output).resolve()
    if output.exists() or output.with_name(output.name + ".partial").exists():
        parser.error("--output must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + args.max_seconds
    report = {"status": "started", "config": str(Path(args.config).resolve()),
              "source": source_state(), "tolerances": TOLERANCES,
              "max_seconds": args.max_seconds, "optimizer_updates": 0,
              "torch_version": str(torch.__version__),
              "training_source_sha256": training_source_digest(),
              "preflight_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    _write_report(output, report)
    try:
        config = load_config(args.config)
        report["budget"] = check_preflight_recipe(config)
        report["config_sha256"] = canonical_digest(config)
        _deadline(deadline)
        _write_report(output, report)
        if args.check_only:
            report["status"] = "checked"
        else:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            configure_training_determinism(config["training"])
            seed_everything(config["seed"])
            from jscc.models.split_model import build_model
            from jscc.data import load_data
            tokenizer, model = build_model(config)
            prepare_trainable_parameters(model)
            backbone_width = int(model.base.config.decoder.hidden_size)
            codec_input_width = int(config["codec"]["input_dim"])
            if backbone_width != 1152 or codec_input_width != 1152:
                raise ValueError("loaded backbone/codec input width differs from pinned 1152")
            report["loaded_backbone_width"] = backbone_width
            report["loaded_codec_input_width"] = codec_input_width
            _deadline(deadline)
            data = load_data(config, tokenizer)
            _deadline(deadline)
            batch = next(iter(data.train))
            sampler = getattr(data.train, "sampler", None)
            if sampler is None or not torch.equal(batch["row_ids"], sampler.actual_ids[:64]):
                raise ValueError("first batch differs from declared presentation stream")
            report["data_ids_sha256"] = canonical_digest(data.ids)
            report["setup_seconds"] = time.monotonic() - started
            _deadline(deadline)
            _write_report(output, report)
            def save_condition(label, condition):
                report.setdefault("completed_conditions", {})[label] = condition
                _write_report(output, report)
            report["probe"] = compare_one_batch(model, batch, config,
                deadline=deadline, on_condition=save_condition)
            _deadline(deadline)
            report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        if not args.check_only and torch.cuda.is_initialized():
            report["cuda_peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["cuda_peak_memory_reserved_bytes"] = torch.cuda.max_memory_reserved()
        _write_report(output, report)


if __name__ == "__main__":
    main()
