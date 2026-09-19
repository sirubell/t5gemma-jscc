"""Shared PyTorch loop: frozen teacher, JSCC student, AdamW, validation, checkpoints.

A step always means one optimizer update. Gradient accumulation is internal.
"""
import copy
import importlib
import json
import math
import random
import time
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast, overload

import torch
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm

from .config import save_config
from .data import load_data
from .losses import (
    aggregate_stream_numerators,
    distillation_loss_stats,
    reconstruction_loss_stats,
)
from .models.split_model import build_model
from .runtime import (
    append_metrics,
    autocast_for,
    isolated_rng,
    model_inputs,
    new_run,
    parameter_update_l2,
    precision_telemetry,
    prepare_trainable_parameters,
    promote_optimizer_state,
    sample_snr_db,
    seed_everything,
    source_state,
)



class BatchValues(TypedDict):
    loss: torch.Tensor
    kl: torch.Tensor
    nmse: torch.Tensor
    kl_numerator: NotRequired[torch.Tensor]
    kl_denominator: NotRequired[torch.Tensor]
    hidden_numerator: NotRequired[torch.Tensor]
    hidden_denominator: NotRequired[torch.Tensor]
    memory_numerator: NotRequired[torch.Tensor | None]
    memory_denominator: NotRequired[torch.Tensor | None]


def _stream_mask(model, batch, labels, stream, reference):
    """Resolve a communication stream mask with a backwards-compatible fallback.

    The corrected SplitModel may expose a stream-specific mask.  Older model
    objects do not, so derive the same masks from the task batch: encoder
    streams use ``attention_mask`` and decoder hidden states use valid labels.
    """

    names = (
        f"{stream}_reconstruction_mask",
        f"{stream}_mask",
        f"{stream}_valid_mask",
    )
    value = next((getattr(model, name, None) for name in names
                  if getattr(model, name, None) is not None), None)
    if value is None:
        stack = model.split.get("stack") if hasattr(model, "split") else "enc"
        value = labels != -100 if stream == "hidden" and stack == "dec" else batch["attention_mask"]
    value = value.to(device=reference.device, dtype=torch.bool)
    # A mask recorded by a hook can include feature axes already.  The loss
    # helper handles broadcasting and reports a useful shape error otherwise.
    return value


def _stream_stats(model, batch, labels, stream, reconstructed, original):
    if reconstructed is None or original is None:
        return None
    mask = _stream_mask(model, batch, labels, stream, original)
    return reconstruction_loss_stats(reconstructed, original, mask)


@overload
def batch_losses(model, batch, training, snr_db, *, return_stats: Literal[False] = False,
                 valid_only_kl: bool = False) -> dict[str, torch.Tensor]: ...


@overload
def batch_losses(model, batch, training, snr_db, *, return_stats: Literal[True],
                 valid_only_kl: bool = False) -> BatchValues: ...


def batch_losses(model, batch, training, snr_db, *, return_stats=False,
                 valid_only_kl=False) -> dict[str, torch.Tensor] | BatchValues:
    kwargs, labels = model_inputs(batch, model)
    encoder_mask = batch.get("attention_mask")
    decoder_mask = labels != -100
    with autocast_for(model):
        with torch.no_grad(), model.transmission(
            bypass=True, encoder_mask=encoder_mask, decoder_mask=decoder_mask
        ):
            teacher = model(**kwargs).logits
        with model.transmission(
            snr_db, encoder_mask=encoder_mask, decoder_mask=decoder_mask
        ):
            student = model(**kwargs).logits
        kl_numerator, kl_denominator = distillation_loss_stats(
            student, teacher, labels, training["temperature"],
            valid_only=valid_only_kl)
        hidden_stats = _stream_stats(model, batch, labels, "hidden",
                                     model.reconstruction, model.activation)
        if hidden_stats is None:
            raise RuntimeError("student transmission did not produce a hidden reconstruction")
        memory_stats = _stream_stats(model, batch, labels, "memory",
                                     model.memory_reconstruction, model.memory_activation)
        streams = [hidden_stats]
        if memory_stats is not None:
            streams.append(memory_stats)
        kl = kl_numerator / kl_denominator.clamp_min(1).to(kl_numerator.dtype)
        nmse = aggregate_stream_numerators(streams)
        loss = training["kl_weight"] * kl + training["mse_weight"] * nmse
    values = {"loss": loss, "kl": kl, "nmse": nmse}
    if return_stats:
        values.update({
            "kl_numerator": kl_numerator,
            "kl_denominator": kl_denominator,
            "hidden_numerator": hidden_stats[0],
            "hidden_denominator": hidden_stats[1],
            "memory_numerator": memory_stats[0] if memory_stats is not None else None,
            "memory_denominator": memory_stats[1] if memory_stats is not None else None,
        })
    return cast(BatchValues, values)


def _valid_sample_count(mask: torch.Tensor) -> torch.Tensor:
    """Count samples with at least one valid position without model work."""

    if mask.ndim == 1:
        mask = mask[:, None]
    return mask.to(dtype=torch.bool).reshape(mask.shape[0], -1).any(dim=1).sum()


def effective_batch_denominators(model, batches: list[dict], device) -> dict[str, torch.Tensor | int]:
    """Return the fixed denominators required by streamed backward.

    Counts depend only on labels and validity masks, so they can be collected
    before the first forward without retaining a computation graph.  The
    resulting objective is the same global effective-batch objective used by
    ``aggregate_batch_losses``.
    """

    kl = torch.zeros((), device=device, dtype=torch.float32)
    hidden = torch.zeros((), device=device, dtype=torch.float32)
    memory = torch.zeros((), device=device, dtype=torch.float32)
    has_memory = getattr(model, "memory_codec", None) is not None
    stack = model.split.get("stack") if hasattr(model, "split") else "enc"
    for batch in batches:
        labels = batch["labels"].to(device=device)
        encoder_mask = batch["attention_mask"].to(device=device, dtype=torch.bool)
        kl = kl + (labels != -100).sum().to(dtype=torch.float32)
        hidden_mask = labels != -100 if stack == "dec" else encoder_mask
        hidden = hidden + _valid_sample_count(hidden_mask).to(dtype=torch.float32)
        if has_memory:
            memory = memory + _valid_sample_count(encoder_mask).to(dtype=torch.float32)
    return {"kl": kl, "hidden": hidden, "memory": memory,
            "stream_count": 2 if has_memory else 1}


def scaled_batch_loss(values: BatchValues, training, denominators) -> torch.Tensor:
    """Scale one microbatch so its backward contributes to the global loss."""

    kl_numerator = _required_stat(values, "kl_numerator")
    hidden_numerator = _required_stat(values, "hidden_numerator")
    kl_denominator = cast(torch.Tensor, denominators["kl"]).clamp_min(1).to(
        dtype=kl_numerator.dtype)
    kl = kl_numerator / kl_denominator
    streams = [
        hidden_numerator /
        cast(torch.Tensor, denominators["hidden"]).clamp_min(1).to(
            dtype=hidden_numerator.dtype)
    ]
    memory_numerator = values.get("memory_numerator")
    if memory_numerator is not None:
        memory_denominator = cast(torch.Tensor, denominators["memory"]).clamp_min(1).to(
            dtype=memory_numerator.dtype)
        streams.append(memory_numerator / memory_denominator)
    nmse = torch.stack(streams).mean()
    return training["kl_weight"] * kl + training["mse_weight"] * nmse


def detached_batch_values(values: BatchValues) -> BatchValues:
    """Drop computation graphs before retaining stats for a later log row."""

    return cast(BatchValues, {
        key: value.detach() if torch.is_tensor(value) else value
        for key, value in values.items()
    })


def _required_stat(values: BatchValues, name: str) -> torch.Tensor:
    value = values.get(name)
    if value is None:
        raise RuntimeError(f"missing required batch statistic: {name}")
    return value


def aggregate_batch_losses(values: list[BatchValues], training) -> dict[str, torch.Tensor]:
    """Form the objective over all microbatches in one optimizer update."""

    if not values:
        raise ValueError("at least one microbatch is required")
    kl_numerator = _required_stat(values[0], "kl_numerator")
    kl_numerator = sum((_required_stat(item, "kl_numerator") for item in values[1:]), kl_numerator)
    kl_denominator = _required_stat(values[0], "kl_denominator")
    kl_denominator = sum((_required_stat(item, "kl_denominator") for item in values[1:]), kl_denominator)
    kl = kl_numerator / kl_denominator.clamp_min(1).to(kl_numerator.dtype)
    hidden = [(_required_stat(item, "hidden_numerator"),
               _required_stat(item, "hidden_denominator")) for item in values]
    streams = [(sum((item[0] for item in hidden[1:]), hidden[0][0]),
               sum((item[1] for item in hidden[1:]), hidden[0][1]))]
    memory: list[tuple[torch.Tensor, torch.Tensor]] = []
    for item in values:
        memory_numerator = item.get("memory_numerator")
        memory_denominator = item.get("memory_denominator")
        if memory_numerator is not None:
            if memory_denominator is None:
                raise RuntimeError("memory denominator is missing for a memory reconstruction")
            memory.append((memory_numerator, memory_denominator))
    if memory:
        streams.append((sum((item[0] for item in memory[1:]), memory[0][0]),
                        sum((item[1] for item in memory[1:]), memory[0][1])))
    nmse = aggregate_stream_numerators(streams)
    return {"loss": training["kl_weight"] * kl + training["mse_weight"] * nmse,
            "kl": kl, "nmse": nmse,
            "kl_numerator": kl_numerator, "kl_denominator": kl_denominator}


@torch.no_grad()
def validate(model, loader, config):
    settings = config["training"]
    was_training = model.training
    model.eval()
    records = []
    with isolated_rng(config["seed"]):
        for condition in settings["validation_snrs"]:
            snr = None if condition == "no_noise" else float(condition)
            details = []
            for batch_index, batch in enumerate(loader):
                if settings["validation_batches"] and batch_index >= settings["validation_batches"]:
                    break
                details.append(batch_losses(model, batch, settings, snr, return_stats=True))
            if not details:
                raise ValueError("validation dataset is empty")
            aggregated = aggregate_batch_losses(details, settings)
            records.append({key: aggregated[key].item() for key in ("loss", "kl", "nmse")})
    model.train(was_training)
    return {key: sum(record[key] for record in records) / len(records) for key in records[0]}


def make_scheduler(optimizer, settings):
    steps = settings.get("schedule_steps", settings["max_steps"])
    warmup = int(steps * settings["warmup_ratio"])
    minimum = settings["min_lr_ratio"]
    def factor(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, steps - warmup))
        return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def save_checkpoint(path, model, optimizer, scheduler, scaler, config, ids, step, best, bad):
    torch.save({"codec": model.codec.state_dict(), "channel": model.channel.state_dict(),
                "memory_codec": model.memory_codec.state_dict() if model.memory_codec is not None else None,
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "config": config, "data_ids": ids,
                "step": step, "best": best, "bad_evaluations": bad,
                "python_rng": random.getstate(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, path)


def train(config, resume: str | Path | None = None):
    resume_path = Path(resume) if resume is not None else None
    state = torch.load(resume_path, map_location="cpu", weights_only=True) if resume_path else None
    if state:
        # Resume the saved recipe. The supplied YAML chooses output location/device only.
        destination, device = config["run"], config["model"]["device"]
        config = copy.deepcopy(state["config"])
        config["run"], config["model"]["device"] = destination, device
        print(f"Resuming saved configuration at optimizer step {state['step']}")
    settings = config["training"]
    if state and state["step"] >= settings["max_steps"]:
        raise ValueError("This checkpoint has already reached max_steps; use a new config for a new experiment")
    seed_everything(config["seed"])
    processor, model = build_model(config)
    prepare_trainable_parameters(model)
    data = load_data(config, processor, state["data_ids"] if state else None)
    run = new_run(config["run"]["output_dir"], config["run"]["name"])
    save_config(config, run / "config.yaml")
    (run / "data_ids.json").write_text(json.dumps(data.ids, indent=2) + "\n")
    print(f"Run: {run}", flush=True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=settings["lr"], weight_decay=settings["weight_decay"])
    scheduler = make_scheduler(optimizer, settings)
    scaler = GradScaler("cuda", enabled=config["model"]["dtype"] == "float16"
                                 and config["model"]["device"] == "cuda")
    start, best, bad = 0, float("inf"), 0
    if state and resume_path is not None:
        model.load_communication_state(state)
        optimizer.load_state_dict(state["optimizer"])
        promote_optimizer_state(optimizer)
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start, best, bad = state["step"], state["best"], state["bad_evaluations"]
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        # Carry the earlier best into the resumed run when it precedes this checkpoint.
        best_path = resume_path.with_name("best.pt")
        earlier_best = torch.load(best_path, map_location="cpu", weights_only=True) if best_path.exists() else None
        if earlier_best and earlier_best["step"] <= start:
            torch.save(earlier_best, run / "best.pt")
        else:
            best, bad = float("inf"), 0
    run_metadata = {
        "resume_from": str(Path(resume).resolve()) if resume else None,
        "torch_version": str(torch.__version__),
        "source": source_state(),
        "training_budget": {
            "max_steps": settings["max_steps"],
            "schedule_steps": settings.get("schedule_steps", settings["max_steps"]),
        },
        "precision": precision_telemetry(model, optimizer),
        "objective": {
            "kl": "global valid-label token mean over the effective batch",
            "nmse": "equal mean of per-sample masked stream nMSE means",
            "snr": "one sampled SNR per training sample",
        },
        "execution_options": {
            "streamed_backward": bool(settings.get("streamed_backward", False)),
            "valid_only_kl": bool(settings.get("valid_only_kl", False)),
            "low_sync_logging": bool(settings.get("low_sync_logging", False)),
        },
    }
    (run / "run.json").write_text(json.dumps(run_metadata, indent=2) + "\n")
    tracker = None
    if config["run"].get("wandb_project"):
        wandb = importlib.import_module("wandb")  # Optional extra, loaded only when requested.
        tracker = wandb.init(project=config["run"]["wandb_project"], name=run.name, config=config)
    iterator = iter(data.train)
    model.train()
    progress = tqdm(range(start + 1, settings["max_steps"] + 1), desc="Optimizer steps")
    started = time.monotonic()
    last_log_time, last_log_step = started, start
    stop_reason = "max_steps"
    step = start
    def save(name, step):
        save_checkpoint(run / name, model, optimizer, scheduler, scaler, config, data.ids,
                        step, best, bad)
    for step in progress:
        step_started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        values = []
        sampled_snrs = []
        capture_update = step % settings["log_every"] == 0 or step == 1
        before_parameters = [parameter.detach().clone() for parameter in parameters] if capture_update else None
        def take_batch():
            nonlocal iterator
            try:
                return next(iterator)
            except StopIteration:
                iterator = iter(data.train)
                return next(iterator)

        def snr_for_batch(batch):
            snr = None
            if config["channel"]["train_noise"]:
                low, high = config["channel"]["train_snr_range"]
                snr = sample_snr_db(model, low, high, batch["labels"].shape[0])
                sampled_snrs.append(snr.detach())
                # The legacy FiLM implementation accepts only one scalar SNR.
                # Corrected baseline recipes disable FiLM and use the per-sample
                # tensor; retain scalar compatibility for older experiments.
                if any(getattr(component, "film", None) is not None
                       for component in (getattr(model, "codec", None),
                                         getattr(model, "memory_codec", None))
                       if component is not None):
                    snr = float(snr.mean().item())
            return snr

        if settings.get("streamed_backward", False):
            # Materialise only the CPU batches, not their computation graphs;
            # global counts are known from labels/masks before any forward.
            batches = [take_batch() for _ in range(settings["gradient_accumulation"])]
            device = next(model.base.parameters()).device
            denominators = effective_batch_denominators(model, batches, device)
            for batch in batches:
                batch_values = batch_losses(
                    model, batch, settings, snr_for_batch(batch), return_stats=True,
                    valid_only_kl=settings.get("valid_only_kl", False),
                )
                scaler.scale(scaled_batch_loss(batch_values, settings, denominators)).backward()
                values.append(detached_batch_values(batch_values))
            totals = aggregate_batch_losses(values, settings)
        else:
            for _ in range(settings["gradient_accumulation"]):
                batch = take_batch()
                values.append(batch_losses(
                    model, batch, settings, snr_for_batch(batch), return_stats=True,
                    valid_only_kl=settings.get("valid_only_kl", False),
                ))
            totals = aggregate_batch_losses(values, settings)
            scaler.scale(totals["loss"]).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, settings["grad_clip"])
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            scheduler.step()
        update_l2 = parameter_update_l2(
            parameters, before_parameters,
            on_device=settings.get("low_sync_logging", False),
        ) if before_parameters is not None else None
        if not settings.get("low_sync_logging", False) or capture_update:
            progress.set_postfix(loss=f"{totals['loss'].detach().item():.4f}")
        if step % settings["log_every"] == 0 or step == 1:
            logged_at = time.monotonic()
            # Loop wall time includes validation/checkpoint/logging overhead since
            # the previous training row, but excludes model/data setup before it.
            interval_seconds = logged_at - last_log_time
            row = {"step": step, "phase": "train", "lr": optimizer.param_groups[0]["lr"],
                   "seconds_per_update": logged_at - step_started,
                   "elapsed_seconds": logged_at - started,
                   "interval_wall_seconds": interval_seconds,
                   "interval_updates": step - last_log_step,
                   "interval_wall_updates_per_second": (step - last_log_step) / interval_seconds,
                   "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30
                   if config["model"]["device"] == "cuda" else None,
                   "peak_memory_reserved_gib": torch.cuda.max_memory_reserved() / 2**30
                   if config["model"]["device"] == "cuda" else None,
                   "loss": totals["loss"].item(), "kl": totals["kl"].item(),
                   "nmse": totals["nmse"].item()}
            row["gradient_norm"] = float(gradient_norm)
            row["parameter_update_l2"] = update_l2
            if sampled_snrs:
                sampled = torch.cat(sampled_snrs)
                row.update({"sampled_snr_mean": sampled.mean().item(),
                            "sampled_snr_min": sampled.min().item(),
                            "sampled_snr_max": sampled.max().item()})
            append_metrics(run / "metrics.jsonl", row)
            if tracker:
                tracker.log({f"train/{key}": value for key, value in row.items()
                             if key not in {"step", "phase"} and value is not None}, step=step)
            last_log_time, last_log_step = logged_at, step
        time_limit = settings.get("max_minutes")
        timed_out = time_limit is not None and time.monotonic() - started >= time_limit * 60
        if step % settings["eval_every"] == 0 or step == settings["max_steps"] or timed_out:
            validation_started = time.monotonic()
            metrics = validate(model, data.validation, config)
            validation_finished = time.monotonic()
            validation_row = {"elapsed_seconds": validation_finished - started,
                              "validation_seconds": validation_finished - validation_started, **metrics}
            append_metrics(run / "metrics.jsonl", {"step": step, "phase": "validation", **validation_row})
            score = metrics[settings["monitor"]]
            improved = score < best * (1.0 - settings["min_delta"])
            if improved:
                best, bad = score, 0
            elif step >= settings["min_steps"]:
                bad += 1
            save("last.pt", step)
            if improved:
                save("best.pt", step)
            print(f"step={step} validation={metrics} best={best:.6f}", flush=True)
            if tracker:
                tracker.log({f"validation/{key}": value for key, value in validation_row.items()}, step=step)
            if settings["patience"] and bad >= settings["patience"]:
                if step in settings["save_steps"]:
                    save(f"step_{step:06d}.pt", step)
                print("Early stopping", flush=True)
                stop_reason = "early_stopping"
                break
        if step in settings["save_steps"]:
            save(f"step_{step:06d}.pt", step)
        if timed_out:
            stop_reason = "time_budget"
            print("Training time budget reached; checkpoint saved for evaluation", flush=True)
            break
    run_metadata["precision"] = precision_telemetry(model, optimizer)
    (run / "run.json").write_text(json.dumps(run_metadata, indent=2) + "\n")
    (run / "completion.json").write_text(json.dumps({"step": step, "reason": stop_reason,
        "elapsed_seconds": time.monotonic() - started}, indent=2) + "\n")
    if tracker:
        tracker.finish()
    return run
