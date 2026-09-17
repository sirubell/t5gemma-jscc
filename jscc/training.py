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

import torch
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm

from .config import save_config
from .data import load_data
from .losses import distillation_loss, reconstruction_loss
from .models.split_model import build_model
from .runtime import append_metrics, autocast_for, isolated_rng, model_inputs, new_run, seed_everything


def batch_losses(model, batch, training, snr_db):
    kwargs, labels = model_inputs(batch, model)
    with autocast_for(model):
        with torch.no_grad(), model.transmission(bypass=True):
            teacher = model(**kwargs).logits
        with model.transmission(snr_db):
            student = model(**kwargs).logits
        kl = distillation_loss(student, teacher, labels, training["temperature"])
        nmse = reconstruction_loss(model.reconstruction, model.activation)
        if model.memory_reconstruction is not None:
            nmse = (nmse + reconstruction_loss(model.memory_reconstruction, model.memory_activation)) / 2
        loss = training["kl_weight"] * kl + training["mse_weight"] * nmse
    return {"loss": loss, "kl": kl, "nmse": nmse}


@torch.no_grad()
def validate(model, loader, config):
    settings = config["training"]
    was_training = model.training
    model.eval()
    records = []
    with isolated_rng(config["seed"]):
        for condition in settings["validation_snrs"]:
            snr = None if condition == "no_noise" else float(condition)
            totals = {"loss": 0.0, "kl": 0.0, "nmse": 0.0}
            count = 0
            for batch_index, batch in enumerate(loader):
                if settings["validation_batches"] and batch_index >= settings["validation_batches"]:
                    break
                values = batch_losses(model, batch, settings, snr)
                for key, value in values.items():
                    totals[key] += value.item()
                count += 1
            if count == 0:
                raise ValueError("validation dataset is empty")
            records.append({key: value / count for key, value in totals.items()})
    model.train(was_training)
    return {key: sum(record[key] for record in records) / len(records) for key in records[0]}


def make_scheduler(optimizer, settings):
    steps = settings["max_steps"]
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
    data = load_data(config, processor, state["data_ids"] if state else None)
    run = new_run(config["run"]["output_dir"], config["run"]["name"])
    save_config(config, run / "config.yaml")
    (run / "data_ids.json").write_text(json.dumps(data.ids, indent=2) + "\n")
    (run / "run.json").write_text(json.dumps({"resume_from": str(Path(resume).resolve()) if resume else None,
                                             "torch_version": str(torch.__version__)}, indent=2) + "\n")
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
    tracker = None
    if config["run"].get("wandb_project"):
        wandb = importlib.import_module("wandb")  # Optional extra, loaded only when requested.
        tracker = wandb.init(project=config["run"]["wandb_project"], name=run.name, config=config)
    iterator = iter(data.train)
    model.train()
    progress = tqdm(range(start + 1, settings["max_steps"] + 1), desc="Optimizer steps")
    started = time.monotonic()
    stop_reason = "max_steps"
    step = start
    def save(name, step):
        save_checkpoint(run / name, model, optimizer, scheduler, scaler, config, data.ids,
                        step, best, bad)
    for step in progress:
        step_started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        totals = {"loss": 0.0, "kl": 0.0, "nmse": 0.0}
        for _ in range(settings["gradient_accumulation"]):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(data.train)
                batch = next(iterator)
            snr = None
            if config["channel"]["train_noise"]:
                low, high = config["channel"]["train_snr_range"]
                snr = float(torch.empty(()).uniform_(low, high))
            losses = batch_losses(model, batch, settings, snr)
            scaler.scale(losses["loss"] / settings["gradient_accumulation"]).backward()
            for key, value in losses.items():
                totals[key] += value.item() / settings["gradient_accumulation"]
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, settings["grad_clip"])
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            scheduler.step()
        progress.set_postfix(loss=f"{totals['loss']:.4f}")
        if step % settings["log_every"] == 0 or step == 1:
            row = {"step": step, "phase": "train", "lr": optimizer.param_groups[0]["lr"],
                   "seconds_per_update": time.monotonic() - step_started,
                   "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30
                   if config["model"]["device"] == "cuda" else None, **totals}
            append_metrics(run / "metrics.jsonl", row)
            if tracker:
                tracker.log({f"train/{key}": value for key, value in totals.items()}, step=step)
        time_limit = settings.get("max_minutes")
        timed_out = time_limit is not None and time.monotonic() - started >= time_limit * 60
        if step % settings["eval_every"] == 0 or step == settings["max_steps"] or timed_out:
            metrics = validate(model, data.validation, config)
            append_metrics(run / "metrics.jsonl", {"step": step, "phase": "validation", **metrics})
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
                tracker.log({f"validation/{key}": value for key, value in metrics.items()}, step=step)
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
    (run / "completion.json").write_text(json.dumps({"step": step, "reason": stop_reason,
        "elapsed_seconds": time.monotonic() - started}, indent=2) + "\n")
    if tracker:
        tracker.finish()
    return run
