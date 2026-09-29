"""Fresh exact64x1 acceptance using the normal training and evaluation loops."""

import argparse
import copy
import gc
import json
from pathlib import Path
import time
from typing import cast

import torch

from jscc.config import load_config, save_config
from jscc.models.channel import AWGNChannel
from jscc.data.hellaswag import load_data
from jscc import training
from jscc.runtime import (
    prepare_trainable_parameters,
    seed_everything,
    configure_training_determinism,
)
from scripts.five_shot_preflight import check_gradients
from scripts.prompt_alignment_benchmark import inventory
from scripts.frozen_prompt_gap import write, sha


def run(config_path, output, measured):
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(config_path)
    t = config["training"]
    if (
        t["batch_size"],
        t["gradient_accumulation"],
        t["max_steps"],
        t["schedule_steps"],
    ) != (64, 1, 10000, 10000):
        raise ValueError("Require approved64x1/10000recipe")
    phase = "stress_setup"
    steps = 2 + measured
    try:
        configure_training_determinism(t)
        seed_everything(config["seed"])
        tokenizer, model = training.build_model(config)
        prepare_trainable_parameters(model)
        model.train()
        data = load_data(config, tokenizer)
        stress_info = inventory(data, 64)
        stress_info["stress_scope"] = (
            "prospective640000presentation stream; natural attention proxy"
        )
        write(output / "stress-inventory.json", stress_info)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=t["lr"],
            weight_decay=t["weight_decay"],
        )
        batch = data.train.collate_fn(
            [data.train.dataset[i] for i in stress_info["stress_dataset_indices"]]
        )
        phase = "stress_update"
        optimizer.zero_grad(set_to_none=True)
        snr = torch.empty(
            (64, 1, 1), device=next(model.base.parameters()).device
        ).uniform_(-6.0, 18.0)
        denominators = training.effective_batch_denominators(model, [batch], snr.device)
        with cast(AWGNChannel, model.channel).replay(0, "exact64-stress", capture=True):
            stats = training.batch_losses(
                model, batch, t, snr, return_stats=True, valid_only_kl=True
            )
            stress_loss = training.scaled_batch_loss(stats, t, denominators)
            if not torch.isfinite(stress_loss):
                raise FloatingPointError("Nonfinite stress loss")
            stress_loss.backward()
        norm = float(
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
        )
        optimizer.step()
        loss = float(stress_loss.detach())
        write(
            output / "stress.json",
            {
                "source_tokens_valid": int(batch["attention_mask"].sum()),
                "source_tokens_allocated": batch["attention_mask"].numel(),
                "target_tokens_valid": int((batch["labels"] != -100).sum()),
                "target_tokens_allocated": batch["labels"].numel(),
                "snr_min": float(snr.min()),
                "snr_max": float(snr.max()),
                "snr_mean": float(snr.mean()),
                "hidden_memory_payload": dict(model.valid_payload_counts),
                "backbone_dtype": str(next(model.base.parameters()).dtype),
                "codec_dtypes": sorted(
                    {str(p.dtype) for p in model.parameters() if p.requires_grad}
                ),
                "presentations": 64,
                "updates": 1,
                "loss": loss,
                "gradient_norm": norm,
                "stream_gradients": check_gradients(model),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            },
        )
        del data, model, tokenizer, optimizer, batch, stats, stress_loss, snr
        gc.collect()
        torch.cuda.empty_cache()
        phase = "normal_training"
        short = copy.deepcopy(config)
        st = short["training"]
        short["run"] = {
            "name": config["run"]["name"] + "-exact64",
            "output_dir": str((output / "training").resolve()),
            "wandb_project": None,
        }
        st.update(
            max_steps=steps,
            eval_every=steps,
            save_steps=[steps],
            feature_summary=None,
            max_minutes=15,
        )
        st["presentation_stream"]["total_presentations"] = steps * 64
        st["paired_randomness"]["audit_steps"] = [1, 2, steps]
        save_config(short, output / "executed-config.yaml")
        original_build = training.build_model
        original_step = torch.optim.AdamW.step
        state = {}
        timing = {}
        norms = []
        counter = 0

        def build(c):
            tokenizer, model = original_build(c)
            state["model"] = model
            return tokenizer, model

        def step(self, closure=None):
            nonlocal counter
            counter += 1
            if counter in (1, 2, steps):
                record = {}
                for name, codec in [
                    ("hidden", state["model"].codec),
                    ("memory", state["model"].memory_codec),
                ]:
                    if codec is not None:
                        record[name] = (
                            torch.stack(
                                [
                                    p.grad.detach().float().square().sum()
                                    for p in codec.parameters()
                                    if p.grad is not None
                                ]
                            )
                            .sum()
                            .sqrt()
                        )
                norms.append((counter, record))
            result = original_step(self, closure)
            if counter == 2:
                torch.cuda.synchronize()
                timing["start"] = time.perf_counter()
            if measured and counter == steps:
                torch.cuda.synchronize()
                timing["end"] = time.perf_counter()
            return result

        training.build_model = build
        torch.optim.AdamW.step = step
        try:
            run_path = training.train(short)
        finally:
            training.build_model = original_build
            torch.optim.AdamW.step = original_step
        n = [
            {"step": i, "stream_gradients": {k: float(v) for k, v in d.items()}}
            for i, d in norms
        ]
        assert all(
            v > 0 and torch.isfinite(torch.tensor(v))
            for x in n
            for v in x["stream_gradients"].values()
        )
        completion = json.loads((run_path / "completion.json").read_text())
        assert (
            completion["status"] == "FULL_BUDGET_COMPLETED"
            and completion["step"] == steps
            and completion["presentations"] == steps * 64
        )
        write(
            output / "completion.json",
            {
                "status": "PASS_EXACT64_TRAIN",
                "config_sha256": sha(config_path),
                "run_path": str(run_path),
                "microbatch": 64,
                "accumulation": 1,
                "effective_batch": 64,
                "normal_updates": steps,
                "stress_updates": 1,
                "normal_presentations": steps * 64,
                "stress_presentations": 64,
                "warmup_updates": 2,
                "measured_updates": measured,
                "interval_seconds": timing.get("end", 0) - timing.get("start", 0)
                if measured
                else None,
                "seconds_per_update": (timing["end"] - timing["start"]) / measured
                if measured
                else None,
                "timing_scope": "after optimizer step2 through final optimizer step; synchronized at boundaries; includes normal loop data/loss/backward/optimizer/logging, excludes finalselection/checkpoint",
                "stream_gradients": n,
                "checkpoint": f"step_{steps:06d}.pt",
                "expected_step": steps,
                "quality_evidence": False,
            },
        )
    except BaseException as exc:
        write(
            output / "failure.json",
            {"phase": phase, "type": type(exc).__name__, "message": str(exc)},
        )
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--measured", type=int, choices=[0, 20], required=True)
    a = p.parse_args()
    run(a.config, a.output, a.measured)


if __name__ == "__main__":
    main()
