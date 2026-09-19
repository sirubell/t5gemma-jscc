"""Bounded production-entrypoint smoke/timing with measured branch counters."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time

import torch

from jscc import training
from jscc.config import load_config, save_config
from scripts.benchmark_speed import GpuSampler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--variant", choices=["reference", "candidate"], required=True)
    p.add_argument("--steps", type=int, default=70)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--profile", action="store_true")
    args = p.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    config["run"].update(
        name="closeout-" + args.variant, output_dir=str(args.output / "runs")
    )
    config["training"].update(
        max_steps=args.steps,
        eval_every=args.steps,
        save_steps=[],
        validation_batches=2,
        streamed_backward=args.variant == "candidate",
        valid_only_kl=args.variant == "candidate",
        low_sync_logging=False,
    )
    # Original train/selection split retained. Short validation scope is disclosed.
    config["data"]["num_workers"] = 0
    if args.steps == 2:
        config["training"]["log_every"] = 1
    path = args.output / "config.yaml"
    save_config(config, path)
    counters = {
        "batch_losses": 0,
        "valid_only_kl": 0,
        "scaled_batch_loss": 0,
        "optimizer_steps": 0,
        "validation_calls": 0,
        "save_calls": 0,
    }
    profiler = None
    if args.profile:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        )

        def tagged(original, name):
            def call(*a, **kw):
                with torch.profiler.record_function(name):
                    return original(*a, **kw)

            return call

        for name in [
            "model_inputs",
            "distillation_loss_stats",
            "reconstruction_loss_stats",
            "scaled_batch_loss",
            "aggregate_batch_losses",
        ]:
            setattr(training, name, tagged(getattr(training, name), "phase/" + name))
        torch.autograd.backward = tagged(torch.autograd.backward, "phase/backward")
        torch.nn.utils.clip_grad_norm_ = tagged(
            torch.nn.utils.clip_grad_norm_, "phase/clip"
        )
        build = training.build_model

        def instrumented_build(config):
            processor, model = build(config)
            forward = model.base.forward

            def call(*a, **kw):
                with torch.profiler.record_function(
                    "phase/teacher" if model.bypass else "phase/student"
                ):
                    return forward(*a, **kw)

            model.base.forward = call
            model._transmit = tagged(model._transmit, "phase/codec_channel")
            return processor, model

        training.build_model = instrumented_build
    measured = {}
    sample_order = []
    snr_draws = []
    original_snr = training.sample_snr_db
    def record_snr(*a, **kw):
        value = original_snr(*a, **kw)
        snr_draws.append(value.detach().clone())
        return value
    training.sample_snr_db = record_snr
    original_load = training.load_data

    def load(*a, **kw):
        data = original_load(*a, **kw)
        sampler = data.train.batch_sampler

        class RecordedSampler:
            def __iter__(self):
                for indices in sampler:
                    sample_order.extend(data.ids["train_rows"][i] for i in indices)
                    yield indices

            def __len__(self):
                return len(sampler)

        object.__setattr__(data.train, "batch_sampler", RecordedSampler())
        return data

    training.load_data = load
    original_losses = training.batch_losses

    def losses(*a, **kw):
        counters["batch_losses"] += 1
        counters["valid_only_kl"] += int(kw.get("valid_only_kl", False))
        return original_losses(*a, **kw)

    training.batch_losses = losses
    original_scaled = training.scaled_batch_loss

    def scaled(*a, **kw):
        counters["scaled_batch_loss"] += 1
        return original_scaled(*a, **kw)

    training.scaled_batch_loss = scaled
    original_zero = torch.optim.AdamW.zero_grad

    def zero(self, *a, **kw):
        if counters["optimizer_steps"] == args.warmup:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            measured["started"] = time.perf_counter()
        return original_zero(self, *a, **kw)

    torch.optim.AdamW.zero_grad = zero
    original_step = torch.optim.AdamW.step

    def step(self, *a, **kw):
        result = original_step(self, *a, **kw)
        counters["optimizer_steps"] += 1
        if profiler is not None:
            profiler.step()
        return result

    torch.optim.AdamW.step = step
    original_validate = training.validate

    def validate(*a, **kw):
        torch.cuda.synchronize()
        t = time.perf_counter()
        if "started" in measured and "wall_seconds" not in measured:
            measured["wall_seconds"] = t - measured["started"]
            measured["training_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            measured["training_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        result = original_validate(*a, **kw)
        torch.cuda.synchronize()
        measured["validation_seconds"] = time.perf_counter() - t
        counters["validation_calls"] += 1
        return result

    training.validate = validate
    original_save = training.save_checkpoint

    def save(*a, **kw):
        t = time.perf_counter()
        result = original_save(*a, **kw)
        measured["save_seconds"] = (
            measured.get("save_seconds", 0) + time.perf_counter() - t
        )
        counters["save_calls"] += 1
        return result

    training.save_checkpoint = save
    sys.argv = ["train.py", "--config", str(path)]
    started = time.perf_counter()
    status = "FAIL"
    sampler = GpuSampler()
    try:
        if profiler is not None:
            profiler.start()
        with sampler:
            runpy.run_path("train.py", run_name="__main__")
        status = "PASS"
    finally:
        if profiler is not None:
            profiler.stop()
            trace = args.output / "profile.json"
            profiler.export_chrome_trace(str(trace))
            with gzip.open(str(trace) + ".gz", "wb") as f:
                f.write(trace.read_bytes())
            phases = [
                {
                    "name": event.key,
                    "count": event.count,
                    "cpu_inclusive_us": event.cpu_time_total,
                    "device_inclusive_us": event.device_time_total,
                    "cpu_self_us": event.self_cpu_time_total,
                    "device_self_us": event.self_device_time_total,
                }
                for event in profiler.key_averages()
                if event.key.startswith("phase/")
            ]
            (args.output / "profiler-phase-summary.json").write_text(
                json.dumps(phases, indent=2)
            )
        measured.pop("started", None)
        out = {
            "status": status,
            "variant": args.variant,
            "entrypoint": "train.py via runpy",
            "profile": args.profile,
            "counters": counters,
            "measured": measured,
            "total_entrypoint_seconds": time.perf_counter() - started,
            "warmup_updates": args.warmup,
            "measured_updates": args.steps - args.warmup,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "update_equivalents": counters["optimizer_steps"],
            "data_order_hash": hashlib.sha256(
                json.dumps(sample_order).encode()
            ).hexdigest(),
            "actual_train_row_ids": sample_order,
            "snr_vectors": [v.cpu().tolist() for v in snr_draws],
            "snr_hash": hashlib.sha256(json.dumps([v.cpu().tolist() for v in snr_draws]).encode()).hexdigest(),
            "noise_identity": {
                "seed": config["seed"],
                "policy": "fresh production per-sample AWGN; same-seed same-shape reference/candidate; no explicit noise tensor replay",
            },
            "includes": "data movement, teacher/student, AWGN, losses, backward, clip, optimizer, scheduler and normal logging",
            "excludes": "model/data loading, final selection validation/checkpoint saving from measured loop; separately recorded",
        }
        if status == "PASS":
            out["telemetry"] = sampler.samples
            out["telemetry_summary"] = sampler.summary()
        (args.output / "production_check.json").write_text(
            json.dumps(out, indent=2, allow_nan=False)
        )


if __name__ == "__main__":
    main()
