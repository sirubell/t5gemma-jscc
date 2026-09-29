"""Fresh bounded dec_l20 stream-weight screen, or discarded two-update preflight."""
import argparse
import copy
import gzip
import hashlib
import json
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Subset

from jscc import training
from jscc.config import load_config, resolve_codec_configs, validate_config
from jscc.presentation import tensor_digest
from scripts.experiments.evening_gradients import selection_ids, validate_fixture

ARMS = {"W005": .05, "W05": .5, "W5": 5.}


def screen_config(base, output, arm, preflight=False):
    cfg = copy.deepcopy(base)
    if cfg["split"] != {"stack": "dec", "where": "after_layer", "index": 20}:
        raise ValueError("screen requires dec_l20 after_layer")
    main, memory = resolve_codec_configs(cfg["codec"])
    if main["layernorm"] != "none" or memory["layernorm"] != "both":
        raise ValueError("screen requires main none / memory both")
    for codec in (main, memory):
        if (codec["hidden_dim"], codec["bottleneck_dim"], codec["n_res_blocks"], codec["activation"], codec["snr_film"]) != (1152, 512, 2, "gelu", False):
            raise ValueError("screen architecture differs from approved residual codec")
    policy = cfg["data"].get("prompt_policy", {})
    if cfg["task"] != "hellaswag" or cfg["protocol"] != "corrected-baseline-v2-native-decoder-inputs" or cfg["seed"] != 0:
        raise ValueError("requires seed0 native-v2 HellaSwag")
    if policy.get("mode") != "five_shot" or policy.get("source_max_length") != 2048 or cfg["data"]["max_length"] != 512:
        raise ValueError("requires existing five-shot source2048 target512")
    if cfg["model"]["dtype"] != "bfloat16" or cfg["channel"]["type"] != "awgn" or not cfg["channel"]["train_noise"] or cfg["channel"]["train_snr_range"] != [-6, 18]:
        raise ValueError("requires BF16 backbone and paired U[-6,18] AWGN")
    steps = 2 if preflight else 1000
    cfg["run"].update(name=f"main-weight-{arm}-{'preflight' if preflight else 'screen'}", output_dir=str(Path(output) / "runs"), wandb_project=None)
    cfg["experiment"] = {"name": "MAIN_WEIGHT_EARLY_SCREEN_V1", "arm": arm, "preflight_discarded": preflight}
    cfg["training"].update(max_steps=steps, schedule_steps=10000, batch_size=64, gradient_accumulation=1,
        warmup_ratio=.05, lr=2e-4, grad_clip=1., log_every=50, eval_every=500 if preflight else 250,
        save_steps=[2] if preflight else [250, 500, 1000], selection_steps=[] if preflight else [250, 500, 1000], validation_batches=0,
        validation_snrs=["no_noise"], patience=None, max_minutes=None,
        loss_weights={"kl": 1., "hidden": ARMS[arm], "memory": .05},
        streamed_backward=True, valid_only_kl=True,
        presentation_stream={"policy": "epoch-permutations-v1", "seed": 0, "total_presentations": steps * 64},
        paired_randomness={"policy": "step-microbatch-stream-v1", "seed": 0, "audit_steps": list(range(1, steps+1))})
    cfg["training"].pop("feature_summary", None)
    validate_config(cfg)
    return cfg


def frozen_selection_order(saved_ids, fixture):
    """Validate E3 membership, then retain its canonical second-shuffle order."""
    rows = fixture["rows"]
    canonical = validate_fixture(rows, saved_ids)
    ids = [row["row_id"] for row in canonical]
    if ids != [row["row_id"] for row in rows]:
        raise ValueError("frozen fixture does not use canonical E3 presentation order")
    return ids


def run_screen(base, output, arm, *, preflight=False, seconds=2600, deadline_epoch=None, selection_fixture=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cfg = screen_config(base, output, arm, preflight)
    cap = cfg["training"]["max_steps"]
    started = time.monotonic()
    report = {"arm": arm, "status": "RUNNING", "preflight_discarded": preflight,
              "optimizer_calls": 0, "training_presentations": 0, "selection_presentations": 0,
              "weight_policy": "explicit loss_weights overrides legacy kl_weight/mse_weight",
              "loss_weights": cfg["training"]["loss_weights"], "execution_flags": {k: cfg["training"][k] for k in ("streamed_backward", "valid_only_kl")},
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    original_build, original_load, original_loss = training.build_model, training.load_data, training.batch_losses
    original_step, original_scheduler, original_validate = torch.optim.AdamW.step, training.make_scheduler, training.validate
    active = {}

    def flush():
        report["wall_seconds"] = time.monotonic() - started
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def deadline():
        if time.monotonic() - started >= seconds or (deadline_epoch is not None and time.time() >= deadline_epoch):
            raise TimeoutError("bounded screen deadline reached; preserve partial checkpoint, do not retry")

    def build(config):
        processor, model = original_build(config)
        if model.memory_codec is None:
            raise ValueError("receiver-memory codec is mandatory")
        active["model"] = model
        report["initial_state"] = {name: {k: tensor_digest(v) for k, v in module.state_dict().items()}
                                   for name, module in (("hidden", model.codec), ("memory", model.memory_codec))}
        return processor, model

    def load(*args, **kwargs):
        data = original_load(*args, **kwargs)
        ids = selection_ids(data.ids)
        if selection_fixture is not None:
            fixture = json.loads(Path(selection_fixture).read_text())
            ids = frozen_selection_order(data.ids, fixture)
            report["selection_fixture_sha256"] = hashlib.sha256(Path(selection_fixture).read_bytes()).hexdigest()
        positions = [data.ids["validation_rows"].index(row) for row in ids]
        loader = data.validation
        data.validation = DataLoader(Subset(loader.dataset, positions), batch_size=64, shuffle=False,
                                      collate_fn=loader.collate_fn, num_workers=0)
        report["selection_ids"] = ids
        active["data"] = data
        return data

    def loss(model, batch, settings, snr, **kwargs):
        deadline()
        phase = "training_presentations" if "row_ids" in batch else "selection_presentations"
        count = batch["labels"].shape[0]
        if report[phase] + count > (cap*64 if phase.startswith("training") else 512):
            raise RuntimeError("screen presentation budget exhausted")
        report[phase] += count
        native, _ = training.model_inputs(batch, model)
        record = {"phase": phase, "step": report["optimizer_calls"],
                  "batch": {k: v.cpu().tolist() for k, v in batch.items() if torch.is_tensor(v)},
                  "native_decoder_input_ids": native["decoder_input_ids"].cpu().tolist(),
                  "native_decoder_sha256": tensor_digest(native["decoder_input_ids"]),
                  "snr": snr.detach().cpu().reshape(-1).tolist() if torch.is_tensor(snr) else snr,
                  "snr_sha256": tensor_digest(snr) if torch.is_tensor(snr) else None}
        with gzip.open(output / "presented_native_inputs.jsonl.gz", "at") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        return original_loss(model, batch, settings, snr, **kwargs)

    def validate(model, loader, config):
        if preflight:
            return {"loss": 0., "kl": 0., "nmse": 0.}  # no selection GPU work in discarded preflight
        result = original_validate(model, loader, config)
        with (output / "selection_components.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": report["optimizer_calls"], **result}, allow_nan=False) + "\n")
        return result

    def scheduler(optimizer, settings):
        result = original_scheduler(optimizer, settings)
        model, data = active["model"], active["data"]
        report["initial_state_fp32"] = {name: {k: tensor_digest(v) for k, v in module.state_dict().items()}
                                        for name, module in (("hidden", model.codec), ("memory", model.memory_codec))}
        torch.save({"codec": model.codec.state_dict(), "memory_codec": model.memory_codec.state_dict(),
                    "channel": model.channel.state_dict(), "config": cfg, "data_ids": data.ids, "step": 0}, output / "init.pt")
        if preflight:
            # Two discarded nonzero-LR updates; formal schedule remains unchanged.
            for group in optimizer.param_groups:
                group["lr"] = settings["lr"]
        else:
            validate(model, data.validation, cfg)
        return result

    def step(optimizer, *args, **kwargs):
        deadline()
        if report["optimizer_calls"] >= cap:
            raise RuntimeError("optimizer update cap exceeded")
        model = active["model"]
        if preflight:
            for group in optimizer.param_groups:
                group["lr"] = cfg["training"]["lr"]
        norms = {}
        for name, module in (("hidden", model.codec), ("memory", model.memory_codec)):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all().item() for g in grads):
                raise FloatingPointError(f"missing/nonfinite {name} gradients")
            norms[name] = float(torch.stack([g.float().square().sum() for g in grads]).sum().sqrt())
            if norms[name] == 0:
                raise FloatingPointError(f"zero {name} gradient")
        result = original_step(optimizer, *args, **kwargs)
        report["optimizer_calls"] += 1
        if preflight:
            states = [value for state in optimizer.state.values() for value in state.values() if torch.is_tensor(value)]
            if not states or not all(value.dtype == torch.float32 and torch.isfinite(value).all().item() for value in states):
                raise FloatingPointError("Adam states must be finite FP32")
            with (output / "preflight_updates.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": report["optimizer_calls"], "lr": optimizer.param_groups[0]["lr"], "postclip_gradient_norms": norms, "adam_states_fp32_finite": True}) + "\n")
        flush()
        return result

    try:
        with ExitStack() as stack:
            for target, name, replacement in ((training, "build_model", build), (training, "load_data", load),
                (training, "batch_losses", loss), (training, "validate", validate),
                (training, "make_scheduler", scheduler), (torch.optim.AdamW, "step", step)):
                stack.enter_context(patch.object(target, name, replacement))
            report["run"] = str(training.train(cfg))
        if report["optimizer_calls"] != cap or report["training_presentations"] != cap * 64 or report["selection_presentations"] != (0 if preflight else 512):
            raise RuntimeError("screen counts differ from fixed budget")
        report["status"] = "MEASURED"
    except BaseException as exc:
        report.update(status="FAILED_OR_PARTIAL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        flush()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--seconds", type=int, default=2600)
    parser.add_argument("--deadline-epoch", type=float)
    parser.add_argument("--selection-fixture", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 2700:
        parser.error("seconds must be in1..2700")
    run_screen(load_config(args.config), args.output, args.arm, preflight=args.preflight,
               seconds=args.seconds, deadline_epoch=args.deadline_epoch, selection_fixture=args.selection_fixture)


if __name__ == "__main__":
    main()
