"""Weight-only phase transfer and encoder-local reconstruction training."""

from __future__ import annotations

import copy
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import cast

import torch
from torch.amp.grad_scaler import GradScaler

from .activation_replay import canonical_digest, file_digest, load_manifest, load_shard
from .losses import reconstruction_loss_stats
from .models.channel import AWGNChannel
from .presentation import derived_seed, initialization_evidence, training_source_digest
from .runtime import (append_metrics, autocast_for, configure_training_determinism,
                      new_run, prepare_trainable_parameters, seed_everything)
from .training import make_scheduler, save_checkpoint
from .training_objectives import (BatchValues, aggregate_batch_losses, detached_batch_values,
                                  effective_batch_denominators, objective_components, scaled_batch_loss)


def _recipe(config: dict) -> dict:
    """Scientific B recipe, excluding only run destination and derived width."""
    codec = copy.deepcopy(config["codec"])
    codec.pop("input_dim", None)
    training = copy.deepcopy(config["training"])
    training.pop("phase_transfer", None)
    return {key: copy.deepcopy(config[key]) for key in
            ("task", "protocol", "seed", "model", "split", "channel", "data", "evaluation")} | {
                "codec": codec, "training": training}


def parent_recipe_sha256(config: dict) -> str:
    return canonical_digest(_recipe(config))


def validate_phase_transfer(checkpoint: str | Path, config: dict) -> dict:
    """Approve a declared local or functional terminal state; never import optimizer state."""
    path = Path(checkpoint).resolve()
    state = torch.load(path, map_location="cpu", weights_only=True)
    expected = config["training"].get("phase_transfer")
    parent_kind = expected.get("parent_kind", "local") if isinstance(expected, dict) else "local"
    if parent_kind == "functional":
        return _validate_functional_parent(path, state, config, expected)
    if parent_kind != "local":
        raise ValueError("phase transfer parent_kind must be local or functional")
    record = state.get("phase_record")
    if not isinstance(record, dict) or record.get("phase") != "B-local-reconstruction":
        raise ValueError("phase transfer requires a completed B local checkpoint")
    if record.get("terminal_step") != state.get("step"):
        raise ValueError("local checkpoint is not its declared terminal state")
    if record.get("source_sha256") != training_source_digest():
        raise ValueError("local checkpoint source differs from current source")
    expected = config["training"].get("phase_transfer")
    if not isinstance(expected, dict) or not isinstance(expected.get("parent_recipe_sha256"), str) or not isinstance(expected.get("parent_step"), int):
        raise ValueError("phase two requires a declared B parent recipe digest and terminal step")
    if expected["parent_recipe_sha256"] != parent_recipe_sha256(state["config"]) or expected["parent_step"] != state["step"]:
        raise ValueError("declared B parent recipe or terminal step differs from checkpoint")
    for key in ("task", "protocol", "seed", "model", "split", "channel", "data", "evaluation"):
        if state["config"][key] != config[key]:
            raise ValueError(f"phase-two {key} differs from declared B recipe")
    old_codec, new_codec = copy.deepcopy(state["config"]["codec"]), copy.deepcopy(config["codec"])
    old_codec.pop("input_dim", None)
    new_codec.pop("input_dim", None)
    if old_codec != new_codec:
        raise ValueError("phase-two recipe differs in model, task, data or communication design")
    if canonical_digest(state["data_ids"]) != record.get("data_ids_sha256"):
        raise ValueError("local checkpoint data IDs changed")
    if expected.get("cache_identity_sha256") is not None and expected["cache_identity_sha256"] != record.get("cache_identity_sha256"):
        raise ValueError("declared phase parent/cache does not match checkpoint")
    completion_path = path.with_name("completion.json")
    if not completion_path.is_file():
        raise ValueError("B parent has no completion receipt")
    completion = json.loads(completion_path.read_text())
    parent_sha256 = file_digest(path)
    if (completion.get("status") != "complete" or completion.get("checkpoint") != path.name or
            completion.get("checkpoint_sha256") != parent_sha256):
        raise ValueError("B parent is partial or completion receipt does not match checkpoint")
    if config["training"].get("presentation_stream") is None:
        raise ValueError("phase two requires an exact presentation stream")
    stream = config["training"]["presentation_stream"]
    if stream.get("start_presentation", 0) != record["presentations"]:
        raise ValueError("phase-two stream must start after B's exact presentations")
    if stream.get("seed") != state["config"]["training"]["presentation_stream"].get("seed"):
        raise ValueError("phase-two presentation permutation seed differs from B")
    metadata = {"policy": "communication-weights-only-fresh-optimizer-scheduler-scaler-v1",
                "parent_checkpoint": str(path), "parent_checkpoint_sha256": parent_sha256,
                "parent_recipe_sha256": expected["parent_recipe_sha256"],
                "parent_step": state["step"], "parent_presentations": record["presentations"],
                "cache_identity_sha256": record["cache_identity_sha256"],
                "parent_source_sha256": record["source_sha256"],
                "phase_two_presentations": config["training"]["presentation_stream"],
                "optimizer_state_loaded": False, "scheduler_state_loaded": False,
                "scaler_state_loaded": False}
    return {"state": state, "metadata": metadata}



def _validate_functional_parent(path: Path, state: dict, config: dict, expected: dict) -> dict:
    """Accept a real fresh functional terminal receipt, without relabeling it as B."""
    from .config import load_config

    if state.get("schema") == "experiment-state-v2":
        raise ValueError("versioned state requires carry-state lifecycle, not legacy transfer")
    parent = state["config"]
    settings = parent["training"]
    step = state.get("step")
    if (state.get("phase_record") is not None or state.get("phase_transfer") is not None
            or settings.get("phase_transfer")):
        raise ValueError("functional parent must be a fresh functional first stage")
    if (not isinstance(expected.get("parent_recipe_sha256"), str)
            or type(expected.get("parent_step")) is not int
            or expected["parent_recipe_sha256"] != parent_recipe_sha256(parent)
            or expected["parent_step"] != step or step != settings["max_steps"]):
        raise ValueError("declared functional parent recipe or terminal step differs from checkpoint")
    if expected.get("cache_identity_sha256") is not None:
        raise ValueError("functional parent cannot declare a local replay cache identity")
    for key in ("task", "protocol", "seed", "model", "split", "channel", "data", "evaluation"):
        if parent[key] != config[key]:
            raise ValueError(f"phase-two {key} differs from declared functional recipe")
    old_codec, new_codec = copy.deepcopy(parent["codec"]), copy.deepcopy(config["codec"])
    old_codec.pop("input_dim", None)
    new_codec.pop("input_dim", None)
    if old_codec != new_codec:
        raise ValueError("phase-two communication design differs from functional parent")
    stream, next_stream = settings.get("presentation_stream"), config["training"].get("presentation_stream")
    if not isinstance(stream, dict) or not isinstance(next_stream, dict):
        raise ValueError("functional transfer requires exact presentation streams")
    presentations = step * settings["batch_size"] * settings["gradient_accumulation"]
    if (stream.get("policy") != "epoch-permutations-v1" or next_stream.get("policy") != stream["policy"]
            or stream.get("start_presentation", 0) != 0
            or stream.get("total_presentations") != presentations
            or next_stream.get("start_presentation", 0) != presentations
            or next_stream.get("seed") != stream.get("seed")):
        raise ValueError("functional phase-two stream must follow the parent's exact presentations")
    required = ("completion.json", "run.json", "config.yaml", "data_ids.json")
    if any(not path.with_name(name).is_file() for name in required):
        raise ValueError("functional parent lacks completion, source, recipe or data receipt")
    completion = json.loads(path.with_name("completion.json").read_text())
    run = json.loads(path.with_name("run.json").read_text())
    digest = file_digest(path)
    if expected.get("parent_checkpoint_sha256") != digest:
        raise ValueError("declared functional parent checkpoint SHA-256 differs from weights")
    final = completion.get("final_checkpoint", {})
    if (completion.get("status") != "FULL_BUDGET_COMPLETED"
            or completion.get("reason") != "max_steps"
            or completion.get("source_verified") is not True
            or completion.get("presentation_budget_verified") is not True
            or completion.get("step") != step or completion.get("optimizer_updates") != step
            or completion.get("presentations") != presentations
            or final != {"file": path.name, "step": step, "sha256": digest}):
        raise ValueError("functional parent is partial or terminal completion receipt does not match")
    if run.get("training_source_sha256") != training_source_digest():
        raise ValueError("functional parent source differs from current source")
    if run.get("resume_from") is not None or run.get("phase_transfer") is not None:
        raise ValueError("functional parent must start fresh")
    saved_config = path.with_name("config.yaml")
    if (run.get("resolved_config_sha256") != file_digest(saved_config)
            or parent_recipe_sha256(load_config(saved_config)) != parent_recipe_sha256(parent)):
        raise ValueError("functional parent resolved recipe receipt differs from checkpoint")
    if json.loads(path.with_name("data_ids.json").read_text()) != state["data_ids"]:
        raise ValueError("functional parent data IDs differ from receipt")
    metadata = {"policy": "communication-weights-only-fresh-optimizer-scheduler-scaler-v1",
                "parent_kind": "functional", "parent_checkpoint": str(path),
                "parent_checkpoint_sha256": digest,
                "parent_recipe_sha256": expected["parent_recipe_sha256"],
                "parent_step": step, "parent_presentations": presentations,
                "parent_source_sha256": run["training_source_sha256"],
                "parent_data_ids_sha256": canonical_digest(state["data_ids"]),
                "parent_completion_sha256": file_digest(path.with_name("completion.json")),
                "parent_run_sha256": file_digest(path.with_name("run.json")),
                "phase_two_presentations": next_stream,
                "optimizer_state_loaded": False, "scheduler_state_loaded": False,
                "scaler_state_loaded": False}
    return {"state": state, "metadata": metadata}


def local_batch_values(model, shard: dict, snr_db, settings: dict) -> BatchValues:
    """Run the existing communication path only; no backbone forward or logits."""
    device = next(model.base.parameters()).device
    activation = shard["activation"].to(device)
    mask = shard["attention_mask"].to(device)
    with autocast_for(model), model.transmission(snr_db, encoder_mask=mask):
        model._roundtrip(activation)
        reconstructed = model.reconstruction
        numerator, denominator = reconstruction_loss_stats(reconstructed, activation, mask)
    zero = numerator.detach().new_zeros(())
    hidden = numerator / denominator.clamp_min(1).to(numerator.dtype)
    components = objective_components(zero, hidden, None, settings)
    return cast(BatchValues, {**components, "loss": components["loss"], "kl": zero, "nmse": hidden,
            "kl_numerator": zero, "kl_denominator": zero,
            "hidden_numerator": numerator, "hidden_denominator": denominator})


def paired_snr(model, config: dict, step: int, microbatch: int, batch_size: int):
    channel = config["channel"]
    if not channel["train_noise"]:
        return None
    paired = config["training"].get("paired_randomness")
    low, high = channel["train_snr_range"]
    device = next(model.base.parameters()).device
    if paired:
        generator = torch.Generator(device=device).manual_seed(
            derived_seed(paired["seed"], f"train:{step}:{microbatch}:snr"))
        return torch.empty((batch_size, 1, 1), device=device).uniform_(low, high, generator=generator)
    from .runtime import sample_snr_db
    return sample_snr_db(model, low, high, batch_size)


def run_local(config: dict, cache_dir: str | Path, *, deadline: float | None = None) -> Path:
    """Execute a bounded B stage from a completed cache; terminal state is evaluator-ready."""
    started = time.monotonic()
    if deadline is not None and started >= deadline:
        raise TimeoutError("local phase wall deadline reached before setup")
    manifest = load_manifest(cache_dir, config)
    settings = config["training"]
    if settings.get("phase_transfer"):
        raise ValueError("B must start from fresh communication initialization")
    if settings.get("loss_weights"):
        raise ValueError("local stage uses the legacy reconstruction coefficient")
    if config["codec"].get("snr_film"):
        raise ValueError("local pilot requires FiLM disabled")
    paired = settings.get("paired_randomness")
    if paired and (paired.get("policy") != "step-microbatch-stream-v1" or config["channel"]["type"] != "awgn"):
        raise ValueError("paired replay requires the AWGN step-microbatch policy")
    configure_training_determinism(settings)
    seed_everything(config["seed"])
    from .models.split_model import build_model
    from .config import save_config
    _, model = build_model(config)
    prepare_trainable_parameters(model)
    model.train()
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("local phase wall deadline reached during setup")
    run = new_run(config["run"]["output_dir"], config["run"]["name"])
    initialization = initialization_evidence(model, run)
    initial_sha256 = file_digest(run / "initial_communication.pt")
    save_config(config, run / "config.yaml")
    (run / "data_ids.json").write_text(json.dumps(manifest["data_ids"], indent=2) + "\n")
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=settings["lr"], weight_decay=settings["weight_decay"])
    scheduler = make_scheduler(optimizer, settings)
    scaler = GradScaler("cuda", enabled=config["model"]["dtype"] == "float16" and
                        config["model"]["device"] == "cuda")
    local_settings = {**settings, "kl_weight": 0.0}
    if settings.get("streamed_backward", False) is False:
        raise ValueError("local stage requires streamed_backward for bounded activation memory")
    record = {"phase": "B-local-reconstruction", "cache_path": str(Path(cache_dir).resolve()),
              "cache_identity_sha256": manifest["identity_sha256"],
              "source_sha256": training_source_digest(),
              "data_ids_sha256": canonical_digest(manifest["data_ids"]),
              "presentations": manifest["presentations"], "updates": settings["max_steps"],
              "initialization_seed": config["seed"], "terminal_step": settings["max_steps"],
              "initial_communication_sha256": initial_sha256,
              "initialization": initialization,
              "objective": "masked per-sample encoder nMSE; no KL/backbone suffix/decoder",
              "optimizer_policy": "fresh AdamW/scaler/scheduler",
              "setup_seconds": time.monotonic() - started,
              "cache_bytes": manifest.get("cache_bytes"),
              "valid_vectors": manifest["valid_vectors"],
              "allocated_vectors": manifest["allocated_vectors"]}
    (run / "phase.json").write_text(json.dumps(record, indent=2) + "\n")
    completed_updates = 0
    try:
        for step in range(1, settings["max_steps"] + 1):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("local phase wall deadline reached before next update")
            step_started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            first = (step - 1) * settings["gradient_accumulation"]
            batches = [load_shard(cache_dir, manifest, first + micro)
                       for micro in range(settings["gradient_accumulation"])]
            io_seconds = time.monotonic() - step_started
            denominator = effective_batch_denominators(model, batches, next(model.base.parameters()).device)
            results = []
            for micro, batch in enumerate(batches):
                snr = paired_snr(model, config, step, micro, len(batch["row_ids"]))
                context = (model.channel.replay(paired["seed"], f"train:{step}:{micro}")
                           if paired and isinstance(model.channel, AWGNChannel) else nullcontext())
                with context:
                    values = local_batch_values(model, batch, snr, local_settings)
                scaler.scale(scaled_batch_loss(values, local_settings, denominator)).backward()
                results.append(detached_batch_values(values))
            totals = aggregate_batch_losses(results, local_settings)
            if not bool(torch.isfinite(totals["loss"])):
                raise FloatingPointError(f"nonfinite local loss at step {step}")
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, settings["grad_clip"])
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError(f"nonfinite local gradient at step {step}")
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < old_scale:
                raise FloatingPointError("skipped local optimizer update")
            scheduler.step()
            completed_updates = step
            append_metrics(run / "metrics.jsonl", {"phase": "local_train", "step": step,
                       "loss": float(totals["loss"]), "nmse": float(totals["nmse"]),
                       "step_seconds": time.monotonic() - step_started,
                       "elapsed_seconds": time.monotonic() - started,
                       "cache_read_seconds": io_seconds,
                           "valid_vectors": sum(int(batch["attention_mask"].sum()) for batch in batches),
                           "presentations": step * settings["batch_size"] * settings["gradient_accumulation"],
                           "lr": optimizer.param_groups[0]["lr"],
                           "gradient_norm": float(grad_norm),
                           "snr_policy": "paired step/microbatch" if paired else "ordinary"})
    except Exception as exc:
        (run / "completion.json").write_text(json.dumps({"status": "FAILED_OR_PARTIAL",
            "reason": type(exc).__name__, "message": str(exc), "completed_updates": completed_updates,
            "completed_presentations": completed_updates * settings["batch_size"] * settings["gradient_accumulation"],
            "elapsed_seconds": time.monotonic() - started}, indent=2) + "\n")
        raise
    if deadline is not None and time.monotonic() >= deadline:
        (run / "completion.json").write_text(json.dumps({"status": "FAILED_OR_PARTIAL", "reason": "deadline-before-checkpoint", "completed_updates": completed_updates, "elapsed_seconds": time.monotonic() - started}, indent=2) + "\n")
        raise TimeoutError("local phase wall deadline reached before terminal checkpoint")
    temporary_checkpoint = run / "last.partial.pt"
    final_checkpoint = run / "last.pt"
    try:
        save_checkpoint(temporary_checkpoint, model, optimizer, scheduler, scaler, config,
                        manifest["data_ids"], settings["max_steps"], float("inf"), 0,
                        phase_record=record)
        checkpoint_sha256 = file_digest(temporary_checkpoint)
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("local phase wall deadline exceeded during terminal serialization")
        temporary_checkpoint.replace(final_checkpoint)
        receipt = {"status": "complete", "phase": "B",
                   "checkpoint": "last.pt", "checkpoint_sha256": checkpoint_sha256,
                   "presentations": manifest["presentations"], "updates": settings["max_steps"],
                   "elapsed_seconds": time.monotonic() - started,
                   "cache_bytes": manifest.get("cache_bytes")}
        temporary_receipt = run / "completion.tmp"
        temporary_receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("local phase wall deadline reached before completion publication")
        temporary_receipt.replace(run / "completion.json")
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("local phase wall deadline exceeded during completion publication")
    except Exception as exc:
        if final_checkpoint.exists():
            final_checkpoint.replace(temporary_checkpoint)
        (run / "completion.tmp").unlink(missing_ok=True)
        (run / "completion.json").write_text(json.dumps({"status": "FAILED_OR_PARTIAL",
            "reason": type(exc).__name__, "message": str(exc), "completed_updates": completed_updates,
            "elapsed_seconds": time.monotonic() - started}, indent=2) + "\n")
        raise
    return run
