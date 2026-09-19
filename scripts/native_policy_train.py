"""One-route bounded native-v2 timing experiment; never formal training.

Run with python -m scripts.native_policy_train --config ROUTE.yaml --output NEW.
The output must be on the execution host. This module never submits remote jobs.
"""
import argparse
import copy
import gc
import hashlib
import json
import random
import time
from contextlib import ExitStack, nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from jscc import training
from jscc.config import load_config, save_config, resolve_codec_configs
from scripts.native_preflight_train import IndexedCollator, IndexedRows, checked_gradient_names


PRESENTATION_CAP = 32768
UPDATE_CAP = 512


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def matrix():
    return [(f"{repeat}-b{batch}", batch, 5120, repeat, False)
            for repeat, batches in (("A", (64, 128)), ("B", (128, 64))) for batch in batches] + [
        (f"stress-b{batch}", batch, batch * 2, "stress", True) for batch in (64, 128)] + [("profile-b64", 64, 128, "profile", True)]


class BudgetDeadline(TimeoutError):
    """Only the authorized benchmark deadline, not a model/data I/O timeout."""


class Budget:
    """Charge before dispatch, including unsuccessful attempted forwards/updates."""
    def __init__(self, seconds=3300):
        self.presentations = 0
        self.updates = 0
        self.deadline = time.monotonic() + seconds

    def charge(self, examples=0, updates=0):
        if time.monotonic() >= self.deadline:
            raise BudgetDeadline("route time limit reached; no retry")
        if self.presentations + examples > PRESENTATION_CAP or self.updates + updates > UPDATE_CAP:
            raise RuntimeError("route work cap exceeded before dispatch")
        self.presentations += examples
        self.updates += updates


def order_for(dataset, repeat, count):
    indices = list(range(len(dataset)))
    if repeat == "stress":
        indices.sort(key=lambda i: (len(dataset[i]["input_ids"]) + len(dataset[i]["label_ids"]), i), reverse=True)
    else:
        random.Random(0 if repeat == "A" else 1).shuffle(indices)
    if len(indices) < count:
        raise ValueError("insufficient distinct rows for the declared paired workload")
    return indices[:count]


def candidate_config(base, output, name, batch, count):
    config = copy.deepcopy(base)
    config["run"].update(name=name, output_dir=str(output / "runs"), wandb_project=None)
    config["training"].update(max_steps=count // batch, batch_size=batch, gradient_accumulation=1,
        schedule_steps=20000, warmup_ratio=0.05, lr=2e-4, eval_every=count // batch,
        save_steps=[count // batch], validation_batches=0, streamed_backward=True,
        valid_only_kl=True, low_sync_logging=False, deterministic_algorithms=True,
        patience=None, max_minutes=None)
    return config


def validate_policy(config):
    if config.get("protocol") != "corrected-baseline-v2-native-decoder-inputs":
        raise ValueError("requires native-v2 protocol")
    hidden, memory = resolve_codec_configs(config["codec"])
    for codec in (hidden, memory):
        if codec["snr_film"] or (codec["hidden_dim"], codec["bottleneck_dim"], codec["n_res_blocks"]) != (1152, 512, 2):
            raise ValueError("requires FiLM off and H1152/B512/two blocks on both streams")
    if hidden["layernorm"] != "none" or memory["layernorm"] != "both":
        raise ValueError("requires raw-layer main none / memory both normalization")
    if config["task"] != "hellaswag" or config["data"]["num_validation"] != 512:
        raise ValueError("requires the fixed HellaSwag 512-row selection")
    if config["data"]["max_length"] != 512:
        raise ValueError("long-input stress requires the unchanged max_length512")
    channel = config["channel"]
    if channel["type"] != "awgn" or not channel["train_noise"] or channel["train_snr_range"] != [-6, 18]:
        raise ValueError("requires production Uniform[-6,18] AWGN")
    if config["training"]["validation_snrs"] != ["no_noise"]:
        raise ValueError("exactly one no_noise selection pass is allowed")
    split = config["split"]
    if (split["stack"], split["index"]) not in (("enc", 9), ("dec", 8)) or split["where"] != "after_layer":
        raise ValueError("only encoder layer9 or decoder layer8 is authorized")
    if config["model"]["device"] != "cuda" or config["model"]["dtype"] != "bfloat16":
        raise ValueError("requires CUDA with BF16 frozen backbone")


def start_profile(report):
    """Trace tooling is optional; its setup failure is not a training failure."""
    try:
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True, profile_memory=True)
        profiler.start()
    except Exception as exc:
        report["profiler"] = {"status": "unavailable", "stage": "start", "error": f"{type(exc).__name__}: {exc}"}
        return None
    report["profiler"] = {"status": "running"}
    return profiler


def finish_profile(profiler, output, report):
    """Catch only tool stop/export errors, outside actual training operations."""
    if profiler is None:
        return
    try:
        profiler.stop()
        profiler.export_chrome_trace(str(output / "trace.json"))
    except Exception as exc:
        report["profiler"] = {"status": "partial_or_unavailable", "stage": "stop_or_export",
                              "error": f"{type(exc).__name__}: {exc}",
                              "partial_trace_exists": (output / "trace.json").exists()}
        return
    report["trace"] = str(output / "trace.json")
    report["profiler"] = {"status": "complete"}


def run_candidate(config, output, descriptor, budget, select, expected_initial=None):
    name, batch_size, count, repeat, stress = descriptor
    report = {"name": name, "status": "RUNNING", "selection": "pending" if select else "suppressed",
              "config_sha256": digest(config), "attempted_presentations": 0, "optimizer_calls": 0,
              "completed_updates": 0, "selection_presentations": 0, "phase_seconds": {},
              "evidence_write_seconds_nested": 0.,
              "noise_policy": "production SNR per sample; same SNR shared across streams, fresh channel draws per stream"}
    output.mkdir()
    save_config(config, output / "config.yaml")
    records = output / "batches.jsonl"
    started = time.perf_counter()
    phase = "setup"
    phase_started = started
    held = {}
    consumed = []
    measured_lengths = {"input": [], "target": []}
    profiling = repeat == "profile"
    original_load, original_prepare = training.load_data, training.prepare_trainable_parameters
    original_loss, original_validate, original_save = training.batch_losses, training.validate, training.save_checkpoint
    original_step, original_clip = torch.optim.AdamW.step, torch.nn.utils.clip_grad_norm_

    def flush():
        write_started = time.perf_counter()
        report["route_charged"] = {"presentations": budget.presentations, "updates": budget.updates}
        temp = output / "report.tmp"
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temp.replace(output / "report.json")
        report["evidence_write_seconds_nested"] += time.perf_counter() - write_started

    def append_record(record):
        write_started = time.perf_counter()
        with records.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        report["evidence_write_seconds_nested"] += time.perf_counter() - write_started

    def transition(next_phase):
        nonlocal phase, phase_started
        torch.cuda.synchronize()
        now = time.perf_counter()
        report["phase_seconds"][phase] = report["phase_seconds"].get(phase, 0) + now - phase_started
        phase, phase_started = next_phase, now

    def prepare(model):
        original_prepare(model)
        held["model"] = model
        state = hashlib.sha256()
        for key, parameter in model.named_parameters():
            if parameter.requires_grad:
                state.update(key.encode())
                state.update(parameter.detach().cpu().float().numpy().tobytes())
        held["initial"] = {key: parameter.detach().cpu().clone() for key, parameter in model.named_parameters()
                           if parameter.requires_grad}
        report["initial_parameters_sha256"] = state.hexdigest()
        if expected_initial is not None and state.hexdigest() != expected_initial:
            raise RuntimeError("fresh candidate initialization differs")
        if profiling:
            original_forward = model.forward
            def forward(*args, **kwargs):
                with torch.profiler.record_function("teacher" if model.bypass else "student"):
                    return original_forward(*args, **kwargs)
            model.forward = forward
        report["backend"] = str(getattr(model.base.config, "_attn_implementation", None))

    def load(*args, **kwargs):
        data = original_load(*args, **kwargs)
        order = order_for(data.train.dataset, repeat, count)
        report["planned_ids"] = [data.ids["train_rows"][i] for i in order]
        report["planned_ids_sha256"] = digest(report["planned_ids"])
        report["selection_ids_sha256"] = digest(data.ids["validation_rows"])
        data.train = DataLoader(IndexedRows(data.train.dataset, data.ids["train_rows"]),
            batch_size=batch_size, sampler=order, num_workers=config["data"]["num_workers"],
            collate_fn=IndexedCollator(data.train.collate_fn), pin_memory=True)
        flush()
        return data

    def losses(model, batch, settings, snr, **kwargs):
        ids = batch.get("preflight_row_ids")
        if ids is None:
            budget.charge()
            report["selection_presentations"] += len(batch["labels"])
            if report["selection_presentations"] > 512:
                raise RuntimeError("selection cap exceeded")
            return original_loss(model, batch, settings, snr, **kwargs)
        if phase == "setup":
            transition("stress" if stress else "warmup")
            torch.cuda.reset_peak_memory_stats()
            if profiling:
                held["profiler"] = start_profile(report)
        elif not stress and report["attempted_presentations"] == 1024:
            transition("measured")
        budget.charge(examples=len(ids))
        report["attempted_presentations"] += len(ids)
        consumed.extend(ids)
        record = {"row_ids": ids, "phase": phase, "input_lengths": batch["attention_mask"].sum(1).tolist(),
                  "target_lengths": (batch["labels"] != -100).sum(1).tolist(),
                  "input_shape": list(batch["input_ids"].shape), "target_shape": list(batch["labels"].shape),
                  "snr_db": snr.detach().cpu().tolist(), "status": "attempted"}
        if phase == "measured":
            measured_lengths["input"].extend(record["input_lengths"])
            measured_lengths["target"].extend(record["target_lengths"])
        record["stream_snr_db"] = {"hidden": record["snr_db"]}
        if model.memory_codec is not None:
            record["stream_snr_db"]["memory"] = record["snr_db"]
        record["padding_tokens"] = {
            "input": batch["input_ids"].numel() - sum(record["input_lengths"]),
            "target": batch["labels"].numel() - sum(record["target_lengths"])}
        # Persist attempted rows before a forward can fail or an allocation ends.
        append_record(record)
        flush()
        with torch.profiler.record_function("batch_loss") if profiling else nullcontext():
            result = original_loss(model, batch, settings, snr, **kwargs)
        record = {"batch": report["attempted_presentations"] // batch_size, "status": "forward_complete",
                  "payload_allocated": {k: int(v) for k, v in model.channel_uses_allocated.items()},
                  "payload_valid": {k: int(v) for k, v in model.valid_payload_counts.items()}}
        append_record(record)
        if not all(torch.isfinite(result[k]).all().item() for k in ("loss", "kl", "nmse")):
            raise FloatingPointError("nonfinite production loss")
        return result

    def clip(*args, **kwargs):
        report["gradient_names"] = checked_gradient_names(held["model"])
        kwargs["error_if_nonfinite"] = True
        return original_clip(*args, **kwargs)

    def step(optimizer, *args, **kwargs):
        if report["optimizer_calls"] == 0 and optimizer.state:
            raise RuntimeError("optimizer must start empty")
        budget.charge(updates=1)
        report["optimizer_calls"] += 1
        flush()
        result = original_step(optimizer, *args, **kwargs)
        report["completed_updates"] += 1
        report["precision"] = training.precision_telemetry(held["model"], optimizer)
        return result

    def validate(*args, **kwargs):
        finish_profile(held.pop("profiler", None), output, report)
        budget.charge()
        transition("selection" if select else "finalize")
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        if not select:
            # Production loop requires a metric dict to finish. These sentinels
            # are explicitly excluded from quality reports and no best is saved.
            return {"loss": 0., "kl": 0., "nmse": 0.}
        result = original_validate(*args, **kwargs)
        if report["selection_presentations"] != 512:
            raise RuntimeError("selection did not consume exactly512")
        if not all(torch.isfinite(torch.tensor(v)).item() for v in result.values()):
            raise FloatingPointError("nonfinite selection metric")
        report["selection"] = "complete"
        report["selection_metrics"] = result
        transition("finalize")
        return result

    def save(path, *args, **kwargs):
        if not Path(path).name.startswith("step_"):
            return None
        budget.charge()
        transition("checkpoint")
        result = original_save(path, *args, **kwargs)
        report["checkpoint"] = str(path)
        transition("finalize")
        return result

    original_backward = torch.autograd.backward
    def backward(*args, **kwargs):
        with torch.profiler.record_function("backward"):
            return original_backward(*args, **kwargs)

    original_kl = training.distillation_loss_stats
    def kl(*args, **kwargs):
        with torch.profiler.record_function("loss_kl"):
            return original_kl(*args, **kwargs)

    try:
        budget.charge()  # Reject exhausted time before model/data setup.
        with ExitStack() as stack:
            if profiling:
                stack.enter_context(patch.object(torch.autograd, "backward", backward))
                stack.enter_context(patch.object(training, "distillation_loss_stats", kl))
            for target, attr, replacement in ((training, "load_data", load),
                (training, "prepare_trainable_parameters", prepare), (training, "batch_losses", losses),
                (training, "validate", validate), (training, "save_checkpoint", save),
                (torch.optim.AdamW, "step", step), (torch.nn.utils, "clip_grad_norm_", clip)):
                stack.enter_context(patch.object(target, attr, replacement))
            report["run"] = str(training.train(config))
        if report["completed_updates"] != count // batch_size:
            raise RuntimeError("production loop stopped before declared work completed")
        report["update_l2"] = sum(
            (parameter.detach().cpu().float() - held["initial"][key].float()).square().sum().item()
            for key, parameter in held["model"].named_parameters() if parameter.requires_grad) ** 0.5
        if not 0 < report["update_l2"] < float("inf"):
            raise FloatingPointError("no finite nonzero parameter update")
        precision = report["precision"]
        for group, dtype in (("trainable_parameters", "float32"), ("frozen_parameters", "bfloat16"),
                             ("optimizer_state", "float32")):
            if set(precision[group]) != {dtype}:
                raise RuntimeError(f"unexpected {group} precision")
        report["status"] = "PASS"
    except BudgetDeadline as exc:
        report.update(status="TIME_LIMIT", error=f"{type(exc).__name__}: {exc}")
    except torch.OutOfMemoryError as exc:
        report.update(status="OOM", error=str(exc))
    except BaseException as exc:
        report.update(status="STOP_ROUTE", error=f"{type(exc).__name__}: {exc}")
    finally:
        finish_profile(held.pop("profiler", None), output, report)
        try:
            transition("done")
        except RuntimeError:
            report["timing_limitation"] = "CUDA synchronization failed after error"
        report["consumed_ids_sha256"] = digest(consumed)
        if report["status"] == "PASS" and not stress:
            seconds = report["phase_seconds"]["measured"]
            report["measured_examples_per_second"] = 4096 / seconds
            report["measured_valid_tokens_per_second"] = sum(sum(v) for v in measured_lengths.values()) / seconds
            report["measured_length_quantiles"] = {
                key: torch.tensor(values, dtype=torch.float32).quantile(torch.tensor([0., .5, .9, .99, 1.])).tolist()
                for key, values in measured_lengths.items()}

        report["total_seconds"] = time.perf_counter() - started
        flush()
        held.clear()
        gc.collect()
        torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=3300)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 3300:
        parser.error("route runtime must be in1..3300 seconds, within the60-minute allocation")
    base = load_config(args.config)
    validate_policy(base)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    budget = Budget(args.seconds)
    reports = []
    initial = None
    selected = False
    failed128 = False
    summary = {"label": "BOUNDED_BENCHMARK_NOT_QUALITY", "formal_execution_authorized": False,
               "source": training.source_state(), "torch": str(torch.__version__), "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(), "matrix": matrix(), "candidates": reports,
               "limitations": ["Natural longest-row stress only; no synthetic512 boundary fixture.",
                                "Profiler is a separate2-update run, excluded from steady throughput; child spans are nested.",
                                "Suppressed selection candidates have sentinel validation metrics in production metrics.jsonl; do not use those as quality results."],
               "timing": "Synchronized wall-exclusive phases; training includes instrumentation and production logging; setup includes first iterator fetch; no nested spans added."}
    for descriptor in matrix():
        name, batch, count, _, stress = descriptor
        if failed128 and batch == 128:
            reports.append({"name": name, "status": "SKIPPED_AFTER_OOM"})
            continue
        config = candidate_config(base, output, name, batch, count)
        report = run_candidate(config, output / name, descriptor, budget,
                               select=not selected and batch == 64 and not stress, expected_initial=initial)
        reports.append(report)
        initial = initial or report.get("initial_parameters_sha256")
        selected |= report["selection"] == "complete"
        summary["charged"] = {"presentations": budget.presentations, "updates": budget.updates}
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        if report["status"] == "TIME_LIMIT":
            first = reports[0]
            if first.get("status") == "PASS" and Path(first.get("checkpoint", "")).is_file():
                break  # Preserve the completed reference for the separately bounded evaluator.
            raise SystemExit("deadline before reference checkpoint; partial evidence retained")
        if report["status"] == "STOP_ROUTE" or (report["status"] == "OOM" and batch == 64):
            raise SystemExit("route stopped; partial evidence retained")
        failed128 |= report["status"] == "OOM"
    if not selected:
        raise SystemExit("no full selection completed")


if __name__ == "__main__":
    main()
