#!/usr/bin/env python3
"""Bounded full-weight B0/B2 CUDA checks using production objective helpers."""

from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import types
import torch
from datasets import load_dataset
from jscc import training
from jscc.data.hellaswag import Collator
from jscc.models.split_model import build_model, stack_module
from jscc.runtime import prepare_trainable_parameters, seed_everything

TOL = {
    "registered_before_candidate": True,
    "scalar_atol": 5e-4,
    "scalar_rtol": 5e-3,
    "near_zero_norm": 1e-8,
    "gradient_relative_l2": 1e-2,
    "cosine_min": 0.999,
    "exp_avg_relative_l2": 1e-2,
    "exp_avg_sq_relative_l2": 3e-2,
    "delta_relative_l2": 2e-2,
    "near_zero_absolute": 1e-7,
    "near_zero_delta_absolute": 1e-6,
    "toy_atol": 1e-6,
    "toy_rtol": 1e-5,
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def filehash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def prepared_decoder_metadata(model, batch):
    from jscc.runtime import model_inputs

    kwargs, _ = model_inputs(batch, model)
    ids = kwargs["decoder_input_ids"].detach().cpu().tolist()
    return {
        "prepared_decoder_input_ids": ids,
        "prepared_decoder_start_tokens": [row[0] for row in ids],
        "prepared_decoder_input_hash": digest(ids),
        "native_preparation_method": type(model.base).__name__
        + ".prepare_decoder_input_ids_from_labels",
        "scope_note": "Training context/targets remain distinct from five-shot harness prompts.",
    }


class ReplayChannel(torch.nn.Module):
    """CPU noise coordinates keyed by sample occurrence and explicit stream."""

    def __init__(self):
        super().__init__()
        self.ids, self.occurrence, self.stream, self.records = [], 0, "", []

    def forward(self, z, snr_db):
        if snr_db is None:
            return z
        noise = []
        for row_id in self.ids:
            key = [row_id, self.occurrence, self.stream]
            generator = torch.Generator().manual_seed(int(digest(key)[:15], 16))
            # Generate a fixed 512-token coordinate grid: padding/partition never
            # changes values at an existing sample/token/feature coordinate.
            values = torch.randn((512, z.shape[-1]), generator=generator)[: z.shape[1]]
            noise.append(values)
        noise = torch.stack(noise).to(device=z.device, dtype=z.dtype)
        self.records.append(
            {
                "ids": self.ids,
                "occurrence": self.occurrence,
                "stream": self.stream,
                "shape": list(z.shape),
                "noise_sha256": hashlib.sha256(
                    noise.float().cpu().numpy().tobytes()
                ).hexdigest(),
                "snr": snr_db.flatten().cpu().tolist(),
            }
        )
        return z + noise * (10.0 ** (-snr_db / 20.0))


def make_batches(config, tokenizer, data_ids):
    raw = load_dataset(
        config["data"]["name"], revision=config["data"]["revision"], split="train"
    )
    ids = data_ids["train_rows"][:1024]
    rows = []
    for row_id in ids:
        row = raw[row_id]
        inp = tokenizer(
            row["ctx"], truncation=True, max_length=config["data"]["max_length"]
        )
        lab = tokenizer(
            row["endings"][int(row["label"])],
            truncation=True,
            max_length=config["data"]["max_length"],
        )
        rows.append(
            {
                "id": row_id,
                "input_ids": inp["input_ids"],
                "attention_mask": inp["attention_mask"],
                "label_ids": lab["input_ids"],
            }
        )
    ordered = sorted(rows, key=lambda r: len(r["input_ids"]) + len(r["label_ids"]))
    high_padding = [r for pair in zip(ordered[:16], ordered[-16:]) for r in pair]
    groups = [
        ("general", rows[:32]),
        ("high_padding", high_padding),
        ("long", ordered[-32:]),
    ]
    collator = Collator(tokenizer.pad_token_id)
    return [
        (
            name,
            [
                (collator(group[i : i + 16]), [r["id"] for r in group[i : i + 16]])
                for i in (0, 16)
            ],
        )
        for name, group in groups
    ]


def metrics(a, b, kind):
    a, b = a.double().flatten(), b.double().flatten()
    norm, other = a.norm().item(), b.norm().item()
    diff = a - b
    near = norm <= TOL["near_zero_norm"]
    rel = None if near else diff.norm().item() / norm
    cos = None if near or other == 0 else torch.dot(a, b).item() / (norm * other)
    maxabs = diff.abs().max().item() if a.numel() else 0.0
    absolute = (
        TOL["near_zero_delta_absolute"]
        if kind == "delta"
        else TOL["near_zero_absolute"]
    )
    passed = (
        maxabs <= absolute
        if near
        else rel <= TOL[kind + "_relative_l2"]
        and (
            kind in ("exp_avg", "exp_avg_sq")
            or (cos is not None and cos >= TOL["cosine_min"])
        )
    )
    return {
        "reference_norm": norm,
        "candidate_norm": other,
        "difference_norm": diff.norm().item(),
        "dot_product": torch.dot(a, b).item(),
        "numel": a.numel(),
        "max_abs": maxabs,
        "relative_l2": rel,
        "cosine": cos,
        "near_zero": near,
        "pass": passed,
    }


def snapshot(params, optimizer, origin):
    return {
        name: {
            "gradient": (
                p.grad.detach().cpu().clone()
                if p.grad is not None
                else torch.zeros_like(p, device="cpu")
            ),
            "delta": p.detach().cpu() - origin[name],
            "exp_avg": optimizer.state[p]["exp_avg"].detach().cpu().clone(),
            "exp_avg_sq": optimizer.state[p]["exp_avg_sq"].detach().cpu().clone(),
            "step": float(optimizer.state[p]["step"]),
            "parameter_dtype": str(p.dtype),
            "moment_dtype": str(optimizer.state[p]["exp_avg"].dtype),
            "gradient_missing": p.grad is None,
        }
        for name, p in params.items()
    }


def run_branch(model, config, batches, initial, condition, branch, temp, ledger):
    seed_everything(190919)
    model.load_state_dict(initial, strict=False)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    origin = {n: p.detach().cpu().clone() for n, p in params.items()}
    settings = config["training"]
    optimizer = torch.optim.AdamW(
        params.values(), lr=settings["lr"], weight_decay=settings["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    channel = model.channel
    rows, paths = [], []
    for step, (batch_name, microbatches) in enumerate(batches):
        optimizer.zero_grad(set_to_none=True)
        all_batches = [b for b, _ in microbatches]
        den = training.effective_batch_denominators(model, all_batches, "cuda")
        values = []
        channel.records = []
        torch.cuda.reset_peak_memory_stats()
        start_examples = ledger["completed_backward_examples"]
        try:
            for batch, ids in microbatches:
                channel.ids, channel.occurrence = ids, step
                snr = (
                    None
                    if condition == "no_noise"
                    else torch.tensor(
                        [-6.0, 0.0, 6.0, 18.0] * 4, device="cuda"
                    ).reshape(16, 1, 1)
                )
                val = training.batch_losses(
                    model,
                    batch,
                    settings,
                    snr,
                    return_stats=True,
                    valid_only_kl=branch == "candidate",
                )
                if branch == "candidate":
                    training.scaled_batch_loss(val, settings, den).backward()
                    ledger["completed_backward_examples"] += len(ids)
                    val = training.detached_batch_values(val)
                values.append(val)
            totals = training.aggregate_batch_losses(values, settings)
            if branch != "candidate":
                totals["loss"].backward()
                ledger["completed_backward_examples"] += 32
            # Capture unclipped gradients; snapshots after step retain clipped grads.
            gradients = {
                n: p.grad.detach().cpu().clone()
                for n, p in params.items()
                if p.grad is not None
            }
            pre = float(
                torch.nn.utils.clip_grad_norm_(
                    list(params.values()), settings["grad_clip"]
                )
            )
            post = (
                sum(
                    float(p.grad.detach().float().square().sum())
                    for p in params.values()
                    if p.grad is not None
                )
                ** 0.5
            )
            optimizer.step()
            scheduler.step()
            ledger["optimizer_steps"] += 1
            state = snapshot(params, optimizer, origin)
            for n in gradients:
                state[n]["gradient"] = gradients[n]
            path = temp / f"{condition}-{branch}-{step}.pt"
            torch.save(state, path)
            paths.append(path)
            row = {
                "step": step + 1,
                "batch_kind": batch_name,
                "status": "PASS",
                "lr": optimizer.param_groups[0]["lr"],
                "grad_norm_before_clip": pre,
                "grad_norm_after_clip": post,
                "clip_coefficient": min(1.0, settings["grad_clip"] / (pre + 1e-6)),
                "optimizer_parameter_states": {
                    n: {k: v for k, v in state[n].items() if not torch.is_tensor(v)}
                    for n in state
                },
                "optimizer_step_calls": 1,
                "scheduler_step_calls": 1,
                "losses": {k: float(v.detach()) for k, v in totals.items()},
                "microbatch_statistics": [
                    {
                        k: float(v.detach()) if torch.is_tensor(v) else v
                        for k, v in val.items()
                    }
                    for val in values
                ],
                "sample_order_hash": digest([ids for _, ids in microbatches]),
                "batches": [
                    {
                        "ids": ids,
                        **prepared_decoder_metadata(model, batch),
                        "shape": {k: list(v.shape) for k, v in batch.items()},
                        "valid_source": int(batch["attention_mask"].sum()),
                        "valid_target": int((batch["labels"] != -100).sum()),
                        "input_hash": digest({k: v.tolist() for k, v in batch.items()}),
                    }
                    for batch, ids in microbatches
                ],
                "noise_records": channel.records,
                "noise_hash": digest(channel.records),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
            rows.append(row)
            del gradients, state, values, totals
        except torch.cuda.OutOfMemoryError as exc:
            rows.append(
                {
                    "step": step + 1,
                    "batch_kind": batch_name,
                    "status": "BLOCKED_OOM",
                    "reason": str(exc),
                    "completed_backward_examples_this_step": ledger[
                        "completed_backward_examples"
                    ]
                    - start_examples,
                }
            )
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            break
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return rows, paths


def compare(paths_a, paths_b, scope, output):
    summaries = []
    for step, (pa, pb) in enumerate(zip(paths_a, paths_b)):
        a, b = torch.load(pa, weights_only=True), torch.load(pb, weights_only=True)
        streams = {}
        for name in a:
            stream = "memory" if name.startswith("memory_codec.") else "main"
            streams.setdefault(stream, []).append(name)
        summary = {"step": step + 1, "streams": {}}
        with output.open("a") as f:
            for name in a:
                for kind in ("gradient", "delta", "exp_avg", "exp_avg_sq"):
                    stat = metrics(a[name][kind], b[name][kind], kind)
                    f.write(
                        json.dumps(
                            {
                                "scope": scope,
                                "step": step + 1,
                                "parameter": name,
                                "quantity": kind,
                                **stat,
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
            for stream, names in streams.items():
                summary["streams"][stream] = {
                    kind: metrics(
                        torch.cat([a[n][kind].flatten() for n in names]),
                        torch.cat([b[n][kind].flatten() for n in names]),
                        kind,
                    )
                    for kind in ("gradient", "delta", "exp_avg", "exp_avg_sq")
                }
        summaries.append(summary)
    return summaries


def kl_and_reload(model, config, batches, temp, ledger):
    """Independent small real-input task-gradient and serialized-state checks."""
    from jscc.runtime import model_inputs, autocast_for

    batch = {k: v[:2] for k, v in batches[0][1][0][0].items()}
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    model.zero_grad(set_to_none=True)
    settings = dict(config["training"], mse_weight=0.0)
    val = training.batch_losses(
        model, batch, settings, None, return_stats=True, valid_only_kl=True
    )
    val["loss"].backward()
    ledger["completed_backward_examples"] += 2
    gradients = {}
    for stream in ("main", "memory"):
        members = {
            n: p
            for n, p in params.items()
            if (n.startswith("memory_codec.")) == (stream == "memory")
        }
        if members:
            gradients[stream] = {
                "gradient_norm": sum(
                    float(p.grad.float().square().sum())
                    for p in members.values()
                    if p.grad is not None
                )
                ** 0.5,
                "missing_parameters": [n for n, p in members.items() if p.grad is None],
                "zero_parameters": [
                    n
                    for n, p in members.items()
                    if p.grad is not None and not bool(p.grad.count_nonzero())
                ],
            }
    optimizer = torch.optim.AdamW(
        params.values(), lr=settings["lr"], weight_decay=settings["weight_decay"]
    )
    optimizer.step()
    ledger["optimizer_steps"] += 1

    def output():
        kwargs, labels = model_inputs(batch, model)
        with (
            torch.no_grad(),
            autocast_for(model),
            model.transmission(
                None, encoder_mask=batch["attention_mask"], decoder_mask=labels != -100
            ),
        ):
            return model(**kwargs).logits.float().cpu()

    before = output()
    path = temp / "reload.pt"
    torch.save(
        {
            "parameters": {n: p.detach().cpu() for n, p in params.items()},
            "optimizer": optimizer.state_dict(),
        },
        path,
    )
    saved = torch.load(path, weights_only=True, map_location="cpu")
    with torch.no_grad():
        for p in params.values():
            p.add_(1.0)
    model.load_state_dict(saved["parameters"], strict=False)
    optimizer.load_state_dict(saved["optimizer"])
    after = output()
    # state reload is compared elementwise for each Adam tensor, not merely dtype.
    moment_equal = all(
        torch.equal(
            optimizer.state[p][key].cpu(), saved["optimizer"]["state"][index][key].cpu()
        )
        for index, p in enumerate(params.values())
        for key in ("exp_avg", "exp_avg_sq", "step")
    )
    return {
        "kl_only_loss": float(val["loss"].detach()),
        "streams": gradients,
        "reload_output_max_abs": float((before - after).abs().max()),
        "reload_output_equal": torch.equal(before, after),
        "optimizer_reload_equal": moment_equal,
        "input_hash": digest({k: v.tolist() for k, v in batch.items()}),
        "examples": 2,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-id", default="corrected-baseline-v1")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument(
        "--reuse-reference",
        type=Path,
        help="Reuse already-passed same-policy no_noise reference/replay; parent run directory",
    )
    parser.add_argument("--condition", choices=["no_noise", "fixed_awgn"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route", choices=["enc_l9", "dec_l8"], required=True)
    parser.add_argument(
        "--general-repeat",
        action="store_true",
        help="Bounded fallback after preserved original-shape OOM: repeat normal batch for 3 updates",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tolerance_path = args.output.parent / "configs/equivalence-tolerances.json"
    if tolerance_path.exists() and json.loads(tolerance_path.read_text()) != TOL:
        raise ValueError("Pre-registered tolerance differs; refusing overwrite")
    dump(tolerance_path, TOL)
    seed_everything(190919)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = copy.deepcopy(ckpt["config"])
    config["model"].update(device="cuda", dtype="bfloat16")
    config["split"] = {
        "stack": "enc" if args.route == "enc_l9" else "dec",
        "where": "after_layer",
        "index": 9 if args.route == "enc_l9" else 8,
    }
    config["codec"]["layernorm"] = "none"
    config["codec"]["memory"] = {"layernorm": "both"}
    tokenizer, model = build_model(config)
    prepare_trainable_parameters(model)
    # Fresh shared initialization per route; same exact weights for all branches.
    # This is implementation validation, not checkpoint quality/resume evidence.
    model.train()
    original_transmit = model._transmit
    replay = ReplayChannel()
    model.channel = replay

    def wrapped(self, hidden, codec, stream, *a, **kw):
        replay.stream = stream
        return original_transmit(hidden, codec, stream, *a, **kw)

    model._transmit = types.MethodType(wrapped, model)
    initial = {
        n: p.detach().cpu().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    batches = make_batches(config, tokenizer, ckpt["data_ids"])
    if args.general_repeat:
        batches = [("general_repeat", batches[0][1])] * 3
    del ckpt
    temp = args.output.parent / f"cuda-private-{args.route}"
    temp.mkdir(exist_ok=True)
    ledger = {"completed_backward_examples": 0, "optimizer_steps": 0}
    result = {
        "route": args.route,
        "protocol_id": args.protocol_id,
        "general_only_fallback": args.general_repeat,
        "layer_counts": {
            "encoder": len(stack_module(model.base, "enc").layers),
            "decoder": len(stack_module(model.base, "dec").layers),
        },
        "config": config,
        "config_hash": digest(config),
        "source": os.environ.get("T5GEMMA_SOURCE_HASH"),
        "executed_source_files": {
            str(p): filehash(p) for p in sorted(Path("jscc").rglob("*.py"))
        }
        | {"scripts/closeout_cuda.py": filehash(Path(__file__))},
        "checkpoint_sha256": filehash(args.checkpoint),
        "initialization": "fresh seed 190919; checkpoint supplies config/data IDs only; fresh AdamW; constant nonzero LR for controlled updates",
        "tolerances": TOL,
        "dtype": {
            "backbone": str(next(model.base.parameters()).dtype),
            "codec": str(next(model.codec.parameters()).dtype),
        },
        "backend": {
            "attention": str(getattr(model.base.config, "_attn_implementation", None)),
            "tf32": torch.backends.cuda.matmul.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "conditions": {},
        "budget": ledger,
        "production_functions": [
            "training.batch_losses",
            "training.effective_batch_denominators",
            "training.scaled_batch_loss",
            "training.aggregate_batch_losses",
        ],
    }
    for condition in [args.condition] if args.condition else ("no_noise", "fixed_awgn"):
        branches = {}
        paths = {}
        for branch in (
            ("reference", "reference_replay")
            if args.reference_only
            else ("reference", "reference_replay", "candidate")
        ):
            if (
                args.reuse_reference
                and condition == "no_noise"
                and branch != "candidate"
            ):
                previous = json.loads(
                    (
                        args.reuse_reference / "results" / f"cuda-{args.route}.json"
                    ).read_text()
                )
                if (
                    previous["config_hash"] != result["config_hash"]
                    or previous["backend"] != result["backend"]
                ):
                    raise ValueError(
                        "Reference reuse requires identical config/backend"
                    )
                branches[branch] = previous["conditions"][condition]["branches"][branch]
                paths[branch] = [
                    args.reuse_reference
                    / f"cuda-private-{args.route}"
                    / f"{condition}-{branch}-{i}.pt"
                    for i in range(3)
                ]
                result.setdefault("reused_reference", {})[branch] = {
                    "path": str(args.reuse_reference),
                    "evidence_sha256": filehash(
                        args.reuse_reference / "results" / f"cuda-{args.route}.json"
                    ),
                }
            else:
                branches[branch], paths[branch] = run_branch(
                    model, config, batches, initial, condition, branch, temp, ledger
                )
            dump(
                args.output / f"cuda-{args.route}.json",
                result | {"in_progress": condition, "partial": branches},
            )
        comparisons = {
            kind: compare(
                paths["reference"],
                paths[branch],
                f"{args.route}/{condition}/{kind}",
                args.output / f"per-parameter-{args.route}.jsonl",
            )
            for kind, branch in (
                [("self_replay", "reference_replay")]
                if args.reference_only
                else [("self_replay", "reference_replay"), ("candidate", "candidate")]
            )
        }
        scalar_checks = []
        for a, b in zip(branches["reference"], branches.get("candidate", [])):
            if a["status"] != "PASS" or b["status"] != "PASS":
                continue
            scalar_checks.append(
                {
                    k: {
                        "reference": v,
                        "candidate": b["losses"][k],
                        "absolute_difference": abs(v - b["losses"][k]),
                        "pass": abs(v - b["losses"][k])
                        <= TOL["scalar_atol"] + TOL["scalar_rtol"] * abs(v),
                    }
                    for k, v in a["losses"].items()
                }
            )
        result["conditions"][condition] = {
            "branches": branches,
            "comparisons": comparisons,
            "scalar_checks": scalar_checks,
        }
        dump(args.output / f"cuda-{args.route}.json", result)
    if not args.reference_only:
        result["kl_only_and_reload"] = kl_and_reload(
            model, config, batches, temp, ledger
        )
    result["budget"]["effective_batch32_update_equivalents"] = (
        ledger["completed_backward_examples"] / 32
    )
    result["limitations"] = [
        "KL-only and checkpoint reload use a two-example real-input subset; not a full training resume.",
        "Fresh Adam state, not exact training continuation.",
        "Constant nonzero LR intentionally isolates optimizer equivalence from scheduler warmup.",
    ]
    result["exit_code"] = 0
    dump(args.output / f"cuda-{args.route}.json", result)
    print(json.dumps({"route": args.route, "budget": ledger}))


if __name__ == "__main__":
    main()
