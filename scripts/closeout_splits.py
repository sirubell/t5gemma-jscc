#!/usr/bin/env python3
"""Small full-weight split wiring checks; never a quality-training launcher."""

from __future__ import annotations
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import traceback
import types
import torch
import yaml
from closeout_cuda import (
    digest,
    filehash,
    kl_and_reload,
    make_batches,
    prepared_decoder_metadata,
)
from jscc import training
from jscc.models.split_model import build_model, stack_module
from jscc.runtime import prepare_trainable_parameters, seed_everything


def run_split(config, data_ids, entry, output, ledger):
    seed_everything(190919)
    config = copy.deepcopy(config)
    override = entry["model_overrides"]
    config["split"] = override["split"]
    config["codec"]["layernorm"] = override["codec"]["layernorm"]
    config["codec"]["memory"] = {"layernorm": "both"}
    config["model"].update(device="cuda", dtype="bfloat16")
    tokenizer, model = build_model(config)
    prepare_trainable_parameters(model)
    model.train()
    batches = make_batches(config, tokenizer, data_ids)
    # Real high-padding group pairs a short and long training example.
    batch = {k: v[:2] for k, v in batches[1][1][0][0].items()}
    ids = batches[1][1][0][1][:2]
    batch_groups = [("mixed_length", [(batch, ids)])]
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    origin = {n: p.detach().cpu().clone() for n, p in params.items()}
    traces = []
    original_transmit = model._transmit
    original_roundtrip = model._roundtrip

    def transmit(self, hidden, codec, stream, *args, **kwargs):
        traces.append(
            {
                "event": "transmit",
                "stream": stream,
                "bypass": self.bypass,
                "shape": list(hidden.shape),
            }
        )
        return original_transmit(hidden, codec, stream, *args, **kwargs)

    def roundtrip(self, hidden):
        traces.append(
            {"event": "split_hook", "bypass": self.bypass, "shape": list(hidden.shape)}
        )
        return original_roundtrip(hidden)

    model._transmit = types.MethodType(transmit, model)
    model._roundtrip = types.MethodType(roundtrip, model)
    torch.cuda.reset_peak_memory_stats()
    private = output.parent / "split-private" / entry["name"]
    private.mkdir(parents=True, exist_ok=True)
    kl_reload = kl_and_reload(model, config, batch_groups, private, ledger)
    no_noise_delta = {
        stream: sum(
            float((p.detach().cpu() - origin[n]).square().sum())
            for n, p in params.items()
            if n.startswith("memory_codec.") == (stream == "memory")
        )
        ** 0.5
        for stream in kl_reload["streams"]
    }
    no_noise_traces = list(traces)
    no_noise_payload = {
        "allocated": dict(model.channel_uses),
        "valid": dict(model.channel_uses_valid),
    }
    traces.clear()
    origin = {n: p.detach().cpu().clone() for n, p in params.items()}
    optimizer = torch.optim.AdamW(
        params.values(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    optimizer.zero_grad(set_to_none=True)
    snr = torch.tensor([-6.0, 18.0], device="cuda").reshape(2, 1, 1)
    values = training.batch_losses(
        model, batch, config["training"], snr, return_stats=True, valid_only_kl=True
    )
    values["loss"].backward()
    ledger["completed_backward_examples"] += 2
    gradient_norm = float(
        torch.nn.utils.clip_grad_norm_(
            list(params.values()), config["training"]["grad_clip"]
        )
    )
    optimizer.step()
    ledger["optimizer_steps"] += 1
    delta = {
        stream: sum(
            float((p.detach().cpu() - origin[n]).square().sum())
            for n, p in params.items()
            if n.startswith("memory_codec.") == (stream == "memory")
        )
        ** 0.5
        for stream in kl_reload["streams"]
    }
    finite = bool(torch.isfinite(values["loss"])) and all(
        torch.isfinite(p).all().item() for p in params.values()
    )
    has_memory = model.memory_codec is not None
    passed = (
        finite
        and kl_reload["reload_output_equal"]
        and kl_reload["optimizer_reload_equal"]
        and all(v["gradient_norm"] > 0 for v in kl_reload["streams"].values())
        and all(v > 0 for v in delta.values())
        and all(v > 0 for v in no_noise_delta.values())
    )
    return {
        "split": entry["name"],
        "status": "SMOKE_ONLY" if passed else "FAIL",
        "scope": "full-weight batch2 mixed real training examples, one KL-only no_noise update and one full-objective AWGN update; not batch16x2 equivalence or task quality",
        "config": config,
        "config_hash": digest(config),
        "split_boundary": config["split"],
        "layer_counts": {
            s: len(stack_module(model.base, s).layers) for s in ("enc", "dec")
        },
        "memory_codec_present": has_memory,
        "sample_ids": ids,
        **prepared_decoder_metadata(model, batch),
        "input_hash": digest({k: v.tolist() for k, v in batch.items()}),
        "shape": {k: list(v.shape) for k, v in batch.items()},
        "valid_source_tokens": int(batch["attention_mask"].sum()),
        "valid_target_tokens": int((batch["labels"] != -100).sum()),
        "no_noise": {
            "objective": "KL-only for task-gradient wiring",
            "kl_and_reload": kl_reload,
            "update_delta_l2": no_noise_delta,
            "payload": no_noise_payload,
            "trace": no_noise_traces,
        },
        "fixed_awgn": {
            "snr_db": [-6.0, 18.0],
            "objective": "configured KL+nMSE",
            "loss": float(values["loss"].detach()),
            "kl": float(values["kl"].detach()),
            "nmse": float(values["nmse"].detach()),
            "gradient_norm_preclip": gradient_norm,
            "update_delta_l2": delta,
            "payload": {
                "allocated": dict(model.channel_uses),
                "valid": dict(model.channel_uses_valid),
            },
            "trace": traces,
        },
        "lr": config["training"]["lr"],
        "dtype": {
            "backbone": str(next(model.base.parameters()).dtype),
            "codec": str(next(model.codec.parameters()).dtype),
        },
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "cache_policy": "teacher-forced training model_inputs; generation/cache reuse not exercised",
        "exit_code": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-id", default="corrected-baseline-v1")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before process launch")
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config, data_ids = state["config"], state["data_ids"]
    del state
    entries = yaml.safe_load(Path("configs/studies/splits.yaml").read_text())[
        "experiments"
    ]
    ledger = {"completed_backward_examples": 0, "optimizer_steps": 0}
    source_files = {str(p): filehash(p) for p in sorted(Path("jscc").rglob("*.py"))}
    source_files.update(
        {
            str(p): filehash(p)
            for p in [
                Path("scripts/closeout_splits.py"),
                Path("scripts/closeout_cuda.py"),
            ]
        }
    )
    for entry in entries:
        if entry["name"] in {"enc_l9", "dec_l8"}:
            continue
        start = dict(ledger)
        try:
            row = run_split(config, data_ids, entry, args.output, ledger)
        except Exception as exc:
            row = {
                "split": entry["name"],
                "status": "FAIL",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "exit_code": 1,
            }
        row["budget"] = {key: ledger[key] - start[key] for key in ledger} | {
            "update_equivalents": (
                ledger["completed_backward_examples"]
                - start["completed_backward_examples"]
            )
            / 32
        }
        row["executed_source_files"] = source_files
        row["protocol_id"] = args.protocol_id
        with (args.output / "split-smoke.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "split": entry["name"],
                    "status": row["status"],
                    "budget": row["budget"],
                }
            ),
            flush=True,
        )
        gc.collect()
        torch.cuda.empty_cache()
    (args.output / "split-smoke-budget.json").write_text(
        json.dumps(
            ledger
            | {
                "update_equivalents": ledger["completed_backward_examples"] / 32,
                "task_panel_equivalents": 0,
                "expected_model_forward_calls_if_all_pass": 66,
                "forward_batch_size": 2,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
