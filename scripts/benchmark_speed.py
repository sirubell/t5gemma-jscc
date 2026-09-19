#!/usr/bin/env python3
"""Measure corrected HellaSwag training throughput on a fixed checkpoint.

This is a short benchmark, not a quality-training entry point.  Every repeat
loads the same communication checkpoint and optimizer state, replays the same
data recipe, and records the effective-batch objective, payload accounting and
GPU telemetry.  The ``reference`` and ``streamed`` variants exercise the
production training helpers rather than a toy model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import threading
import time
from typing import Any

import torch

from jscc.config import load_config
from jscc.data import load_data
from jscc.runtime import prepare_trainable_parameters, promote_optimizer_state, seed_everything
from jscc.models.split_model import build_model
from jscc import training


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GpuSampler:
    def __init__(self):
        self.samples: list[dict[str, float]] = []
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.binary = shutil.which("nvidia-smi")

    def _read(self):
        if self.binary is None:
            return
        query = ["utilization.gpu", "power.draw", "temperature.gpu", "clocks.sm",
                 "memory.free"]
        try:
            result = subprocess.run(
                [self.binary, f"--query-gpu={','.join(query)}", "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True, timeout=2,
            )
            fields = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
            if len(fields) != len(query):
                return
            self.samples.append({key: float(value) for key, value in zip(query, fields)})
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return

    def _run(self):
        while not self.stop.is_set():
            self._read()
            self.stop.wait(0.5)

    def __enter__(self):
        if self.binary is not None:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=3)
        self._read()

    def summary(self) -> dict[str, float | None]:
        def mean(key):
            values = [item[key] for item in self.samples if key in item]
            return sum(values) / len(values) if values else None

        def maximum(key):
            values = [item[key] for item in self.samples if key in item]
            return max(values) if values else None

        return {
            "gpu_utilization_mean": mean("utilization.gpu"),
            "gpu_power_mean_w": mean("power.draw"),
            "temperature_max_c": maximum("temperature.gpu"),
            "clocks_sm_mean_mhz": mean("clocks.sm"),
            "minimum_free_gib": (min(item["memory.free"] for item in self.samples) / 1024
                                  if self.samples else None),
        }


def _next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _batch_counts(model, batches: list[dict[str, torch.Tensor]]) -> dict[str, int]:
    stack = model.split.get("stack", "enc")
    bottleneck = int(model.codec.config["bottleneck_dim"])
    allocated_hidden = valid_hidden = allocated_memory = valid_memory = 0
    source_tokens = target_tokens = 0
    for batch in batches:
        source = batch["attention_mask"]
        target = batch["labels"] != -100
        source_tokens += int(source.sum().item())
        target_tokens += int(target.sum().item())
        if stack == "dec":
            allocated_hidden += int(target.numel()) * bottleneck
            valid_hidden += int(target.sum().item()) * bottleneck
            allocated_memory += int(source.numel()) * bottleneck
            valid_memory += int(source.sum().item()) * bottleneck
        else:
            allocated_hidden += int(source.numel()) * bottleneck
            valid_hidden += int(source.sum().item()) * bottleneck
    return {
        "valid_source_tokens": source_tokens,
        "valid_target_tokens": target_tokens,
        "allocated_payload": allocated_hidden + allocated_memory,
        "valid_payload": valid_hidden + valid_memory,
        "allocated_hidden_payload": allocated_hidden,
        "valid_hidden_payload": valid_hidden,
        "allocated_memory_payload": allocated_memory,
        "valid_memory_payload": valid_memory,
    }


def _load_repeat(config_path: Path, checkpoint_path: Path, batch_size: int,
                 accumulation: int, seed: int):
    config = load_config(config_path)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = state["config"]
    config["training"] = dict(config["training"])
    config["training"].update({"batch_size": batch_size,
                                "gradient_accumulation": accumulation})
    config["model"] = dict(config["model"])
    config["model"].update({"device": "cuda", "dtype": "bfloat16"})
    seed_everything(seed)
    processor, model = build_model(config)
    prepare_trainable_parameters(model)
    model.load_communication_state(state)
    data = load_data(config, processor, state["data_ids"])
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["training"]["lr"],
                                  weight_decay=config["training"]["weight_decay"])
    optimizer.load_state_dict(state["optimizer"])
    promote_optimizer_state(optimizer)
    return config, model, data, optimizer, parameters


def _one_update(model, loader, iterator, optimizer, parameters, settings, variant,
                snr, valid_only_kl, logging_mode="none", capture_log=False):
    optimizer.zero_grad(set_to_none=True)
    data_wait = 0.0
    if variant == "streamed":
        batches = []
        for _ in range(settings["gradient_accumulation"]):
            fetch_started = time.perf_counter()
            batch, iterator = _next_batch(iterator, loader)
            data_wait += time.perf_counter() - fetch_started
            batches.append(batch)
        device = next(model.base.parameters()).device
        denominators = training.effective_batch_denominators(model, batches, device)
        values = []
        before = [parameter.detach().clone() for parameter in parameters] if capture_log else None
        for batch in batches:
            value = training.batch_losses(model, batch, settings, snr,
                                          return_stats=True, valid_only_kl=valid_only_kl)
            training.scaled_batch_loss(value, settings, denominators).backward()
            values.append(training.detached_batch_values(value))
        totals = training.aggregate_batch_losses(values, settings)
    else:
        values = []
        batches = []
        before = [parameter.detach().clone() for parameter in parameters] if capture_log else None
        for _ in range(settings["gradient_accumulation"]):
            fetch_started = time.perf_counter()
            batch, iterator = _next_batch(iterator, loader)
            data_wait += time.perf_counter() - fetch_started
            batches.append(batch)
            values.append(training.batch_losses(model, batch, settings, snr,
                                                return_stats=True, valid_only_kl=valid_only_kl))
        totals = training.aggregate_batch_losses(values, settings)
        totals["loss"].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, settings["grad_clip"])
    optimizer.step()
    if before is not None:
        update_l2 = training.parameter_update_l2(
            parameters, before, on_device=logging_mode == "low"
        )
    else:
        update_l2 = None
    return iterator, batches, totals, float(grad_norm), update_l2, data_wait


def _profile_update(model, loader, iterator, optimizer, parameters, settings, variant,
                    output: Path, valid_only_kl: bool):
    if not torch.cuda.is_available():
        return None
    trace_path = output / f"{variant}.trace.json"
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True, profile_memory=True, with_stack=False,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    ) as profiler:
        for _ in range(5):
            iterator, _, _, _, _, _ = _one_update(
                model, loader, iterator, optimizer, parameters, settings,
                variant, None, valid_only_kl,
            )
            profiler.step()
    profiler.export_chrome_trace(str(trace_path))
    table = profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=50)
    (output / f"{variant}.profile.txt").write_text(table)
    return trace_path


def run_repeat(config_path: Path, checkpoint_path: Path, output: Path, variant: str,
               batch_size: int, accumulation: int, warmup: int, measured: int,
               repeat: int, snr: float | None, valid_only_kl: bool,
               logging_mode: str, profile: bool) -> dict[str, Any]:
    config, model, data, optimizer, parameters = _load_repeat(
        config_path, checkpoint_path, batch_size, accumulation, 9000 + repeat)
    settings = config["training"]
    settings = dict(settings)
    settings.update({"batch_size": batch_size, "gradient_accumulation": accumulation,
                     "grad_clip": settings.get("grad_clip", 1.0),
                     "kl_weight": settings.get("kl_weight", 1.0),
                     "mse_weight": settings.get("mse_weight", 0.1)})
    model.train()
    torch.manual_seed(12345)
    iterator = iter(data.train)
    # Use no-noise for the throughput matrix so every variant follows exactly
    # the same deterministic channel path.  Fixed-noise equivalence is tested
    # separately in the CPU/CUDA protocol checks.
    for _ in range(warmup):
        iterator, _, _, _, _, _ = _one_update(
            model, data.train, iterator, optimizer, parameters, settings,
            variant, snr, valid_only_kl,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    sampler = GpuSampler()
    update_seconds = []
    examples = source_tokens = target_tokens = allocated = valid = 0
    data_wait = 0.0
    finite = True
    gradients_finite = True
    update_l2_values = []
    profile_path = None
    with sampler:
        for step in range(measured):
            compute_started = time.perf_counter()
            iterator, batches, totals, grad_norm, update_l2, waited = _one_update(
                model, data.train, iterator, optimizer, parameters, settings,
                variant, snr, valid_only_kl, logging_mode,
                capture_log=logging_mode != "none" and (step + 1) % 50 == 0,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            finished = time.perf_counter()
            # ``_one_update`` measures loader wait before model compute.
            update_seconds.append(finished - compute_started)
            data_wait += waited
            batch_counts = _batch_counts(model, batches)
            examples += batch_size * accumulation
            source_tokens += batch_counts["valid_source_tokens"]
            target_tokens += batch_counts["valid_target_tokens"]
            allocated += batch_counts["allocated_payload"]
            valid += batch_counts["valid_payload"]
            finite = finite and bool(torch.isfinite(totals["loss"]).item())
            gradients_finite = gradients_finite and bool(torch.isfinite(torch.tensor(grad_norm)).item())
            if update_l2 is not None:
                update_l2_values.append(update_l2)
            if profile and step == 0:
                profile_path = _profile_update(
                    model, data.train, iterator, optimizer, parameters, settings,
                    variant, output, valid_only_kl,
                )
    wall = sum(update_seconds)
    memory_allocated = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    memory_reserved = torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else None
    telemetry = sampler.summary()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    result = {
        "variant_id": variant,
        "status": "completed",
        "repeat_id": repeat,
        "batch_size": batch_size,
        "accumulation_steps": accumulation,
        "effective_batch_size": batch_size * accumulation,
        "warmup_updates": warmup,
        "measured_updates": measured,
        "measured_examples": examples,
        "valid_source_tokens": source_tokens,
        "valid_target_tokens": target_tokens,
        "allocated_payload": allocated,
        "valid_payload": valid,
        "wall_seconds": wall,
        "updates_per_second": measured / wall if wall else None,
        "examples_per_second": examples / wall if wall else None,
        "source_tokens_per_second": source_tokens / wall if wall else None,
        "target_tokens_per_second": target_tokens / wall if wall else None,
        "data_wait_seconds": data_wait,
        "peak_allocated_gib": memory_allocated,
        "peak_reserved_gib": memory_reserved,
        "minimum_free_gib": telemetry["minimum_free_gib"],
        "includes_dataloader": True,
        "includes_logging": logging_mode != "none",
        "includes_validation": False,
        "includes_checkpoint_io": False,
        "profiler_enabled": profile,
        **telemetry,
        "cpu_rss_max_mib": rss,
        "loss_finite": finite,
        "gradients_finite": gradients_finite,
        "oom": False,
        "exit_code": 0,
        "reason": None,
        "update_seconds": update_seconds,
        "update_l2_mean": (sum(update_l2_values) / len(update_l2_values)
                            if update_l2_values else None),
        "logging_mode": logging_mode,
        "profile_path": str(profile_path.relative_to(output)) if profile_path else None,
    }
    del model, data, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variant", choices=("reference", "streamed"), required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--measured", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--snr", type=float, default=None)
    parser.add_argument("--valid-only-kl", action="store_true")
    parser.add_argument("--logging-mode", choices=("none", "reference", "low"), default="none")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.batch_size * args.accumulation != 32:
        raise SystemExit("benchmark effective batch must remain 32")
    args.output.mkdir(parents=True, exist_ok=True)
    result = run_repeat(
        args.config.resolve(), args.checkpoint.resolve(), args.output.resolve(),
        args.variant, args.batch_size, args.accumulation, args.warmup,
        args.measured, args.repeat, args.snr, args.valid_only_kl,
        args.logging_mode, args.profile,
    )
    result.update({
        "source_hash": os.environ.get("T5GEMMA_SOURCE_HASH"),
        "config_hash": sha256_file(args.config.resolve()),
        "checkpoint_hash": sha256_file(args.checkpoint.resolve()),
        "noise_policy_id": "no_noise" if args.snr is None else f"fixed_{args.snr:g}db",
        "sample_order_hash": hashlib.sha256(
            json.dumps({"seed": 12345, "repeat": args.repeat}, sort_keys=True).encode()
        ).hexdigest(),
        "config_path": str(args.config.resolve()),
        "checkpoint_path": str(args.checkpoint.resolve()),
    })
    path = args.output / "benchmark-repeats.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps(result, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
