"""Bounded native-v2 production training instrumentation; never a quality study.

Run as a module. Each invocation is fresh; the caller owns cumulative GPU/example
budgets across invocations. Dataset membership is unchanged; only order differs.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import runpy
import sys
import time
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Dataset

from jscc import training
from jscc.config import load_config, save_config
from scripts.benchmark_speed import GpuSampler


class IndexedRows(Dataset):
    def __init__(self, dataset, row_ids):
        self.dataset, self.row_ids = dataset, row_ids

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index], self.row_ids[index]


class IndexedCollator:
    def __init__(self, collator):
        self.collator = collator

    def __call__(self, rows):
        batch = self.collator([row for row, _ in rows])
        batch["preflight_row_ids"] = [row_id for _, row_id in rows]
        return batch


def selected_order(dataset, seed):
    """Alternate 16 longest and 16 shortest tokenized examples, then shuffle."""
    lengths = [(len(row["input_ids"]), len(row["label_ids"])) for row in dataset]
    ranked = sorted(range(len(lengths)), key=lambda i: (sum(lengths[i]), lengths[i], i))
    if len(ranked) < 32:
        raise ValueError("preflight requires at least 32 training examples")
    head = [i for pair in zip(reversed(ranked[-16:]), ranked[:16]) for i in pair]
    chosen = set(head)
    rest = [i for i in range(len(lengths)) if i not in chosen]
    random.Random(seed).shuffle(rest)
    return head + rest, lengths


def equal_state(left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_state(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal_state(a, b) for a, b in zip(left, right))
    return left == right


def checked_gradient_names(model):
    names = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise RuntimeError(f"missing trainable gradient: {name}")
        if not torch.isfinite(parameter.grad).all().item():
            raise FloatingPointError(f"nonfinite trainable gradient: {name}")
        names.append(name)
    if not names:
        raise RuntimeError("no trainable gradients")
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--selection-batches", type=int, default=0)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    examples = args.microbatch * args.accumulation * args.steps
    if min(args.microbatch, args.accumulation, args.steps) < 1 or examples > 1600:
        parser.error("positive dimensions and at most 1600 backward examples per invocation required")
    if args.selection_batches < 0:
        parser.error("selection-batches must be nonnegative (0 means full selection)")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    config = load_config(args.config)
    if config["task"] != "hellaswag" or config["data"]["num_validation"] != 512:
        raise ValueError("requires HellaSwag with the fixed 512-row selection split")
    if config["channel"]["type"] != "awgn" or not config["channel"]["train_noise"]:
        raise ValueError("normal production AWGN must be enabled")
    if config["model"]["device"] != "cuda":
        raise ValueError("this bounded full-weight preflight requires CUDA")
    config["run"].update(name="native-v2-preflight", output_dir=str(args.output / "runs"), wandb_project=None)
    config["data"]["num_workers"] = 4
    config["training"].update(
        max_steps=args.steps, schedule_steps=20000, batch_size=args.microbatch,
        gradient_accumulation=args.accumulation, eval_every=args.steps,
        save_steps=[args.steps], validation_batches=args.selection_batches,
        streamed_backward=True, valid_only_kl=True, low_sync_logging=False,
        log_every=1, patience=None, max_minutes=None,
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config_path = args.output / "config.yaml"
    save_config(config, config_path)
    report = {
        "status": "FAIL", "label": "TRAINING_BENCHMARK_NOT_QUALITY" if args.benchmark else "NATIVE_V2_EXECUTION_PREFLIGHT_NOT_QUALITY",
        "entrypoint": "train.py via runpy", "consumed_batches": [], "selection_calls": 0,
        "counters": {"training_batch_losses": 0, "valid_only_kl": 0, "scaled_batch_loss": 0,
                     "backward_calls": 0, "backward_examples": 0, "optimizer_steps": 0},
        "order_rule": "first32 alternate longest16/shortest16 by (input+label length,input length,label length,index); rest Random(seed) shuffle; membership unchanged",
        "noise_policy": "production per-sample SNR and fresh AWGN; no cross-batch noise replay",
        "timing_scope": "instrumented production loop including metadata transfer and finite checks; validation and checkpoint measured separately",
    }
    import importlib.metadata
    root = Path(__file__).resolve().parents[1]
    report["runtime"] = {
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "transformers": importlib.metadata.version("transformers"),
        "lm_eval": importlib.metadata.version("lm_eval"),
        "gpu": torch.cuda.get_device_name(),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }
    report["source_files"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted((root / "jscc").rglob("*.py"))}
    report["source_files"]["scripts/native_preflight_train.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    counts = report["counters"]
    held = {}
    original_load, original_prepare = training.load_data, training.prepare_trainable_parameters
    original_inputs, original_losses = training.model_inputs, training.batch_losses
    original_scaled, original_validate = training.scaled_batch_loss, training.validate
    original_save = training.save_checkpoint
    original_backward, original_step = torch.autograd.backward, torch.optim.AdamW.step
    original_clip = torch.nn.utils.clip_grad_norm_

    def load(*a, **kw):
        data = original_load(*a, **kw)
        order, lengths = selected_order(data.train.dataset, config["seed"])
        report["first32"] = [{"row_id": data.ids["train_rows"][i], "input_length": lengths[i][0], "label_length": lengths[i][1]} for i in order[:32]]
        report["membership"] = {"train_count": len(data.ids["train_rows"]), "selection_count": len(data.ids["validation_rows"])}
        data.train = DataLoader(IndexedRows(data.train.dataset, data.ids["train_rows"]),
                                batch_size=args.microbatch, sampler=order, num_workers=4,
                                collate_fn=IndexedCollator(data.train.collate_fn), pin_memory=True)
        return data

    def prepare(model):
        original_prepare(model)
        held["model"] = model
        held["initial"] = {name: {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
                           for name in ("codec", "memory_codec") if (module := getattr(model, name)) is not None}

    def inputs(batch, model):
        result = original_inputs(batch, model)
        if "preflight_row_ids" in batch:
            kwargs, labels = result
            native = model.base.prepare_decoder_input_ids_from_labels(labels=labels)
            if not torch.equal(native, kwargs["decoder_input_ids"]):
                raise RuntimeError("native decoder preparation mismatch")
            held["batch_record"].update(decoder_input_ids=kwargs["decoder_input_ids"].cpu().tolist(),
                                         native_ids_equal=True)
        return result

    def losses(model, batch, settings, snr, **kw):
        if "preflight_row_ids" in batch:
            counts["training_batch_losses"] += 1
            counts["valid_only_kl"] += int(kw.get("valid_only_kl", False))
            held["current_examples"] = len(batch["preflight_row_ids"])
            record = {"row_ids": batch["preflight_row_ids"],
                      "input_lengths": batch["attention_mask"].sum(1).tolist(),
                      "label_lengths": (batch["labels"] != -100).sum(1).tolist(),
                      "input_shape": list(batch["input_ids"].shape),
                      "label_shape": list(batch["labels"].shape), "snr": snr.detach().cpu().tolist()}
            report["consumed_batches"].append(record)
            held["batch_record"] = record
            if "training_started" not in held:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                held["training_started"] = time.perf_counter()
        else:
            report["selection_examples"] = report.get("selection_examples", 0) + len(batch["labels"])
        result = original_losses(model, batch, settings, snr, **kw)
        if not all(torch.isfinite(result[k]).all().item() for k in ("loss", "kl", "nmse")):
            raise FloatingPointError("nonfinite loss; stop without retry")
        return result

    def scaled(*a, **kw):
        counts["scaled_batch_loss"] += 1
        value = original_scaled(*a, **kw)
        if not torch.isfinite(value).all().item():
            raise FloatingPointError("nonfinite scaled loss")
        return value

    def backward(*a, **kw):
        size = held["current_examples"]
        if counts["backward_examples"] + size > examples:
            raise RuntimeError("backward example budget exceeded")
        # Count attempted backwards conservatively, including an OOM attempt.
        counts["backward_calls"] += 1
        counts["backward_examples"] += size
        return original_backward(*a, **kw)

    def clip(*a, **kw):
        names = checked_gradient_names(held["model"])
        report["gradient_parameter_names"] = names
        report["gradient_checks"] = report.get("gradient_checks", 0) + 1
        kw["error_if_nonfinite"] = True
        return original_clip(*a, **kw)

    def step(optimizer, *a, **kw):
        if counts["optimizer_steps"] == 0 and optimizer.state:
            raise RuntimeError("optimizer was not freshly initialized")
        result = original_step(optimizer, *a, **kw)
        held["optimizer"] = optimizer
        counts["optimizer_steps"] += 1
        return result

    def validate(*a, **kw):
        torch.cuda.synchronize()
        started = time.perf_counter()
        report["training_seconds"] = started - held["training_started"]
        report["training_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        report["training_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        report["selection_calls"] += 1
        result = original_validate(*a, **kw)
        torch.cuda.synchronize()
        report["selection_seconds"] = time.perf_counter() - started
        return result

    def save(*a, **kw):
        started = time.perf_counter()
        result = original_save(*a, **kw)
        report["checkpoint_seconds"] = report.get("checkpoint_seconds", 0) + time.perf_counter() - started
        return result

    sampler = GpuSampler()
    started = time.perf_counter()
    try:
        from contextlib import ExitStack
        with ExitStack() as stack:
            for target, name, replacement in (
                (training, "load_data", load), (training, "prepare_trainable_parameters", prepare),
                (training, "model_inputs", inputs), (training, "batch_losses", losses),
                (training, "scaled_batch_loss", scaled), (training, "validate", validate),
                (training, "save_checkpoint", save),
                (torch.autograd, "backward", backward), (torch.optim.AdamW, "step", step),
                (torch.nn.utils, "clip_grad_norm_", clip),
            ):
                stack.enter_context(patch.object(target, name, replacement))
            stack.enter_context(patch.object(sys, "argv", ["train.py", "--config", str(config_path),
                                                           "--run-path-file", str(args.output / "run-path.txt")]))
            with sampler:
                runpy.run_path(str(Path(__file__).resolve().parents[1] / "train.py"), run_name="__main__")
        run = Path((args.output / "run-path.txt").read_text().strip())
        model, optimizer = held["model"], held["optimizer"]
        report["precision"] = training.precision_telemetry(model, optimizer)
        if report["precision"]["trainable_parameters"].keys() != {"float32"}:
            raise RuntimeError("trainable parameters are not FP32")
        if report["precision"]["frozen_parameters"].keys() != {"bfloat16"}:
            raise RuntimeError("frozen backbone is not BF16")
        if report["precision"]["optimizer_state"].keys() != {"float32"}:
            raise RuntimeError("optimizer moments are not FP32")
        expected_microbatches = args.steps * args.accumulation
        if any(counts[k] != expected_microbatches for k in (
                "training_batch_losses", "valid_only_kl", "scaled_batch_loss", "backward_calls")):
            raise RuntimeError("B2 branch counters do not match actual work")
        if not args.selection_batches and report.get("selection_examples") != 512:
            raise RuntimeError("full selection did not consume exactly 512 rows")
        report["update_l2"] = {}
        for name, before in held["initial"].items():
            after = getattr(model, name).state_dict()
            delta = sum((after[k].detach().cpu().float() - v.float()).square().sum().item() for k, v in before.items()) ** 0.5
            if not 0 < delta < float("inf"):
                raise RuntimeError(f"{name} has no finite nonzero update: {delta}")
            report["update_l2"][name] = delta
        checkpoint = run / f"step_{args.steps:06d}.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        # Explicitly alter codec parameters, then restore through production API.
        with torch.no_grad():
            for name in held["initial"]:
                for parameter in getattr(model, name).parameters():
                    parameter.zero_()
        model.load_communication_state(state)
        reloaded = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        reloaded.load_state_dict(state["optimizer"])
        training.promote_optimizer_state(reloaded)
        codec_equal = all(equal_state(getattr(model, name).state_dict(), state[name]) for name in held["initial"])
        optimizer_equal = equal_state(expected_optimizer, reloaded.state_dict())
        report["reload"] = {"checkpoint": str(checkpoint), "codec_equal": codec_equal,
                            "optimizer_equal": optimizer_equal, "additional_backward": 0}
        if not codec_equal or not optimizer_equal or counts["optimizer_steps"] != args.steps:
            raise RuntimeError("checkpoint reload or completed step count mismatch")
        report["status"] = "PASS"
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["total_seconds"] = time.perf_counter() - started
        report["telemetry"] = sampler.samples
        report["telemetry_summary"] = sampler.summary()
        ids = [i for batch in report["consumed_batches"] for i in batch["row_ids"]]
        report["consumed_order_sha256"] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        report["effective_batch32_equivalents"] = counts["backward_examples"] / 32
        (args.output / "preflight.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
