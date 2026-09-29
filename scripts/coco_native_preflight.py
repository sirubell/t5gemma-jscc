"""Bounded real-image COCO acceptance; disposable updates, never quality training.

Run as a module. Output must be new. Slurm provides the external hard timeout;
this script checks a monotonic deadline between bounded operations and writes
partial evidence continuously. Memory results cover observed fixtures only.
"""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import resource
import time
from typing import cast

import torch
from torch.utils.data import DataLoader, Dataset, default_collate

from jscc import training
from jscc.config import load_config, save_config
from jscc.evaluation import evaluate_coco
from jscc.models.channel import AWGNChannel
from jscc.data.coco import caption_prompt, load_data
from jscc.runtime import (autocast_for, model_inputs, prepare_trainable_parameters,
                          precision_telemetry, seed_everything, isolated_rng)
from scripts.five_shot_preflight import check_gradients
from scripts.coco_cache_diagnostic import compare_outputs, generation_output as cache_generation_output


def write(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def batch_plan(candidates, effective_batch):
    if effective_batch < 1 or not candidates or candidates != sorted(set(candidates)):
        raise ValueError("Require positive effective batch and unique ascending candidates")
    if any(x not in (1, 2, 4, 8, 16, 32, 64) or effective_batch % x for x in candidates):
        raise ValueError("Microbatches must be 1/2/4/8/16/32/64 and divide effective batch")
    return [(x, effective_batch // x) for x in candidates]


def assert_visual_tokens(full_ids, actual_ids, image_token_id):
    if image_token_id is None:
        raise ValueError("Model has no image token identity")
    expected = int((full_ids == image_token_id).sum())
    actual = int((actual_ids == image_token_id).sum())
    if expected == 0 or actual != expected:
        raise ValueError(f"Visual token truncation/routing: {actual} vs {expected}")
    return expected



def communication_snapshot(model):
    return {name: ({key: value.detach().cpu().clone() for key, value in component.state_dict().items()}
                   if component is not None else None)
            for name in ("codec", "channel", "memory_codec")
            for component in [getattr(model, name)]}


def communication_digest(state):
    digest = hashlib.sha256()
    for component, values in sorted(state.items()):
        digest.update(component.encode())
        if values is None:
            digest.update(b"NONE")
            continue
        for name, value in sorted(values.items()):
            digest.update(name.encode())
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def perturb_and_reload(model, path):
    """Prove reload restores changed values, including receiver-memory weights."""
    saved = communication_snapshot(model)
    before = communication_digest(saved)
    torch.save(saved, path)
    mutated = []
    with torch.no_grad():
        for name in ("codec", "channel", "memory_codec"):
            component = getattr(model, name)
            if component is None:
                continue
            parameter = next(component.parameters(), None)
            if parameter is not None and parameter.numel():
                parameter.reshape(-1)[0].add_(1.0)
                mutated.append(name)
    changed = communication_digest(communication_snapshot(model))
    if changed == before:
        raise AssertionError("Reload control failed to change communication state")
    model.load_communication_state(torch.load(path, map_location="cpu", weights_only=True))
    restored = communication_snapshot(model)
    after = communication_digest(restored)
    if after != before:
        raise AssertionError("Reload did not restore exact communication state")
    return {"before_sha256": before, "perturbed_sha256": changed,
            "restored_sha256": after, "perturbed_components": mutated, "state_exact": True}


def record_production_update(record, effective_batch, counter, intervals):
    record["updates"] += 1
    record["presentations"] += effective_batch
    record["production_progress"] = {"optimizer_updates_observed": counter,
        "completed_update_presentations": counter * effective_batch,
        "interval_boundaries": dict(intervals),
        "accounting_scope": "optimizer calls returned; no per-update CUDA synchronization"}



class IndexedCaptionEvidence(Dataset):
    """Attach image identity without changing caption sampling or image processing."""
    def __init__(self, dataset, image_ids):
        self.dataset, self.image_ids = dataset, image_ids
        if len(dataset) != len(image_ids):
            raise ValueError("Caption evidence IDs do not match dataset")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return {**self.dataset[index], "evidence_image_id": self.image_ids[index]}



def fp32_math_reference(config, model, fixture, prefix_path, output):
    # Local import avoids coco_route_smoke's import of our state helpers.
    from scripts.coco_route_smoke import prefix_replay, load_forced_prefix
    from torch.nn.attention import sdpa_kernel, SDPBackend
    prefix = load_forced_prefix(prefix_path)
    original_dtypes = {name: str(value.dtype) for name, value in model.base.named_buffers()}
    original_state = communication_digest(communication_snapshot(model))
    control_config = copy.deepcopy(config)
    control_config["model"]["sdpa_backend_policy"] = "auto"  # Preserve the outer math-only reference context.
    with isolated_rng(config["seed"]):
        _, control = training.build_model(control_config)
    try:
        control.float().eval()
        control.load_communication_state(communication_snapshot(model))
        kwargs, _ = model_inputs(fixture, control)
        with sdpa_kernel(SDPBackend.MATH):
            rows = prefix_replay(control, kwargs, output, "fp32-math", 1e-4, 1e-4, forced_ids=prefix)
            if not all(row["allclose"] for row in rows):
                raise AssertionError("FP32 math same-prefix cache reference failed")
            generation = {k: v for k, v in kwargs.items() if k not in ("decoder_input_ids", "use_cache")}
            with torch.no_grad(), control.transmission(None):
                ids = control.generate(**generation, max_new_tokens=64, do_sample=False, num_beams=1, use_cache=True)
        record = {"pass": True, "prefix_sha256": hashlib.sha256(prefix_path.read_bytes()).hexdigest(),
            "prefix_ids": prefix, "same_prefix": rows, "cached_generation_ids": ids.tolist(),
            "cached_max_new_tokens": 64, "initialization": "independent fresh BF16-loaded backbone cast once to FP32; current communication state loaded",
            "backend": "SDPA math", "production_backend_policy": config["model"].get("sdpa_backend_policy", "auto"), "atol": 1e-4, "rtol": 1e-4,
            "original_backbone_buffer_dtypes": original_dtypes}
        (output / "forced-prefix-source.json").write_bytes(prefix_path.read_bytes())
    finally:
        del control
        gc.collect()
        torch.cuda.empty_cache()
    if original_dtypes != {name: str(value.dtype) for name, value in model.base.named_buffers()}:
        raise AssertionError("FP32 control modified original backbone buffers")
    if original_state != communication_digest(communication_snapshot(model)):
        raise AssertionError("FP32 control modified original communication weights")
    return record


def memory():
    free, total = torch.cuda.mem_get_info()
    return {"peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "sampled_free_gib": free / 2**30, "total_gib": total / 2**30,
            "cpu_maxrss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def run(args):
    plan = batch_plan(args.batches, args.effective_batch)
    if not 1 <= args.updates_per_batch <= 4 or not 1 <= args.max_minutes <= 30:
        raise ValueError("Bounded to 1..4 updates per batch and 1..30 minutes")
    if not 4 <= args.production_steps <= 32:
        raise ValueError("Production segment bounded to 4..32 updates")
    if not 2 <= args.inventory_size <= 64:
        raise ValueError("Inventory must contain 2..64 distinct images")
    cache_policy = getattr(args, "cache_policy", "strict_tokens")
    if cache_policy not in ("strict_tokens", "fp32_math_reference"):
        raise ValueError("Unknown cache acceptance policy")
    if cache_policy == "fp32_math_reference" and getattr(args, "prefix_json", None) is None:
        raise ValueError("FP32 math reference requires fixed prefix provenance")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    deadline = start + 60 * args.max_minutes
    record = {"status": "RUNNING", "quality_evidence": False, "updates": 0,
              "presentations": 0, "batch_results": [], "checks": {},
              "timing_scope": "synchronized bounded preflight, not production throughput",
              "limitations": ["Observed image inventory only; not a proof of all dataset shapes",
                              "Generation/scorer combined wall time is separate from training timing",
                              "Teacher/student/vision/codec are combined in forward_loss timing"],
              "setup_seconds": {}, "cache_policy": cache_policy,
              "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest()}

    def checkpoint():
        record["elapsed_seconds"] = time.perf_counter() - start
        write(output / "results.json", record)
        if time.perf_counter() >= deadline:
            raise TimeoutError("Preflight cooperative time budget exhausted")

    try:
        config = load_config(args.config)
        if cache_policy == "fp32_math_reference" and config["model"].get("sdpa_backend_policy") != "flash_math":
            raise ValueError("New COCO cache policy requires flash_math production attention backend")
        if config["task"] != "coco" or not config["channel"]["normalize_power"]:
            raise ValueError("Require COCO with power constraint")
        if config["codec"]["snr_film"]:
            raise ValueError("FiLM must remain off")
        seed_everything(config["seed"])
        config["data"]["num_workers"] = 0
        config["training"].update(streamed_backward=True, valid_only_kl=True)
        save_config(config, output / "executed-config.yaml")
        stage_started = time.perf_counter()
        processor, model = training.build_model(config)
        prepare_trainable_parameters(model)
        record["setup_seconds"]["model_load_and_precision"] = sync() - stage_started
        checkpoint()
        stage_started = time.perf_counter()
        data = load_data(config, processor)
        record["setup_seconds"]["data_load"] = time.perf_counter() - stage_started
        checkpoint()
        write(output / "data-ids.json", data.ids)
        for key, expected_count in (("train_ids", config["data"]["num_train"]),
            ("demo_ids", config["data"]["num_demos"]), ("selection_ids", config["data"]["num_validation"]),
            ("report_ids", config["data"]["num_report"])):
            if len(data.ids[key]) != expected_count or len(set(data.ids[key])) != expected_count:
                raise ValueError(f"Unexpected COCO cardinality/duplicates in {key}")
        if len(data.demo_images) != 4:
            raise ValueError("Approved COCO protocol requires four demo images")
        dataset = data.train.dataset
        token_id = getattr(model.base.config, "image_token_index", None)
        if token_id is None:
            token_id = getattr(model.base.config, "image_token_id", None)
        if token_id is None:
            token_id = processor.tokenizer.convert_tokens_to_ids("<image_soft_token>")
        inventory, items = [], []
        inventory_started = time.perf_counter()
        for index in range(min(args.inventory_size, len(dataset))):
            checkpoint()
            row = dataset.rows[index]
            item_started = time.perf_counter()
            item = dataset[index]
            item_seconds = time.perf_counter() - item_started
            images = data.demo_images + [row["image"].convert("RGB")]
            processor_started = time.perf_counter()
            full = processor(images=images, text=caption_prompt(data.demo_captions), return_tensors="pt")
            processor_seconds = time.perf_counter() - processor_started
            visual = assert_visual_tokens(full["input_ids"], item["input_ids"], token_id)
            ids = item["labels"][item["labels"] != -100].tolist()
            inventory.append({"index": index, "image_id": data.ids["train_ids"][index],
                "image_sha256_rgb": hashlib.sha256(images[-1].tobytes()).hexdigest(),
                "image_sizes": [list(im.size) for im in images], "num_images": len(images),
                "pixel_shape": list(item["pixel_values"].shape), "visual_tokens": visual,
                "dataset_item_seconds": item_seconds, "untruncated_processor_seconds": processor_seconds,
                "source_full": full["input_ids"].shape[-1],
                "source_valid": int(item["attention_mask"].sum()),
                "target_ids": ids, "all_reference_captions": row["answer"]})
            items.append(item)
        record["setup_seconds"]["fixture_inventory"] = time.perf_counter() - inventory_started
        write(output / "fixture-inventory.json", inventory)
        if len(items) < 2:
            raise ValueError("Need two real images")
        # Stress proxy deliberately remains restricted to actual sampled shapes.
        order = sorted(range(len(items)), key=lambda i: (
            items[i]["pixel_values"].numel(), inventory[i]["source_valid"] * len(inventory[i]["target_ids"])), reverse=True)
        fixture = default_collate([items[order[0]]])
        kwargs, labels = model_inputs(fixture, model)
        expected = model.base.prepare_decoder_input_ids_from_labels(labels=labels)
        if not torch.equal(expected, kwargs["decoder_input_ids"]):
            raise AssertionError("Native decoder preparation mismatch")
        record["checks"]["native_decoder"] = {"pass": True, "first_ids": expected[:, :8].tolist()}
        model.eval()
        with torch.no_grad(), autocast_for(model):
            with model.transmission(bypass=True):
                first = model(**kwargs).encoder_last_hidden_state.detach().float()
            changed = dict(kwargs)
            changed["pixel_values"] = model_inputs(default_collate([items[order[1]]]), model)[0]["pixel_values"]
            # Keep text/token shape fixed; only swap image pixels.
            if changed["pixel_values"].shape != kwargs["pixel_values"].shape:
                raise ValueError("Image sensitivity fixture pixel shapes differ")
            with model.transmission(bypass=True):
                second = model(**changed).encoder_last_hidden_state.detach().float()
            delta = float((first - second).abs().max())
        if not delta > 0:
            raise AssertionError("Real-image swap did not change encoder representation")
        record["checks"]["image_sensitivity"] = {"pass": True, "max_abs_delta": delta}
        del first, second, changed
        model.train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                    lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])
        # Task-only gradients are necessary to detect detached receiver paths.
        values = training.batch_losses(model, fixture, config["training"], None, return_stats=True, valid_only_kl=True)
        values["kl"].backward()
        record["checks"]["kl_only_gradients"] = check_gradients(model)
        optimizer.zero_grad(set_to_none=True)
        del values
        for microbatch, accumulation in plan:
            checkpoint()
            if microbatch == 16 and record["batch_results"] and record["batch_results"][-1].get("memory", {}).get("sampled_free_gib", 0) < 12:
                record["batch_results"].append({"microbatch": 16, "status": "SKIPPED_HEADROOM"})
                break
            result = {"microbatch": microbatch, "accumulation": accumulation, "steps": [], "status": "RUNNING"}
            record["batch_results"].append(result)
            torch.cuda.reset_peak_memory_stats()
            try:
                for update_index in range(args.updates_per_batch):
                    checkpoint()
                    begin = sync()
                    batches = [default_collate([items[order[(j * microbatch + k) % len(order)]]
                               for k in range(microbatch)]) for j in range(accumulation)]
                    prepared = sync()
                    denom = training.effective_batch_denominators(model, batches, next(model.base.parameters()).device)
                    optimizer.zero_grad(set_to_none=True)
                    forward_seconds = backward_seconds = 0.0
                    losses = []
                    for batch in batches:
                        checkpoint()
                        a = sync()
                        snr = torch.empty((microbatch, 1, 1), device=next(model.base.parameters()).device).uniform_(*config["channel"]["train_snr_range"])
                        values = training.batch_losses(model, batch, config["training"], snr,
                                                       return_stats=True, valid_only_kl=True)
                        loss = training.scaled_batch_loss(values, config["training"], denom)
                        b = sync()
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite loss")
                        loss.backward()
                        c = sync()
                        losses.append(float(loss.detach()))
                        forward_seconds += b - a
                        backward_seconds += c - b
                        values = loss = None
                    grads = check_gradients(model)
                    a = sync()
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config["training"]["grad_clip"])
                    optimizer.step()
                    end = sync()
                    record["updates"] += 1
                    record["presentations"] += args.effective_batch
                    result["steps"].append({"step": update_index + 1, "seconds": end - begin,
                        "data_collate_seconds": prepared - begin, "forward_loss_seconds": forward_seconds,
                        "backward_seconds": backward_seconds, "optimizer_seconds": end - a,
                        "loss": sum(losses), "gradient_norms": grads,
                        "payload_last_microbatch": dict(model.valid_payload_counts)})
                result["memory"] = memory()
                result["precision"] = precision_telemetry(model, optimizer)
                result["status"] = "PASS"
            except torch.OutOfMemoryError as exc:
                result.update(status="OOM", error=str(exc), memory=memory())
                optimizer.zero_grad(set_to_none=True)
                for name in ("activation", "reconstruction", "memory_activation", "memory_reconstruction"):
                    setattr(model, name, None)
                values = loss = None
                gc.collect()
                torch.cuda.empty_cache()
                break
        checkpoint()
        model.eval()
        torch.cuda.reset_peak_memory_stats()
        generation = {k: v for k, v in kwargs.items() if k not in ("decoder_input_ids", "use_cache")}
        # Persist the exact post-capacity state BEFORE the potentially failing gate.
        frozen_state = communication_snapshot(model)
        torch.save(frozen_state, output / "cache-fixture-communication.pt")
        torch.save({k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
                    for k, v in fixture.items()}, output / "cache-fixture.pt")
        record["checks"]["cache_fixture"] = {
            "communication_sha256": communication_digest(frozen_state),
            "inventory_index": order[0], "image_id": inventory[order[0]]["image_id"],
            "updates_before_snapshot": record["updates"],
            "presentations_before_snapshot": record["presentations"]}
        checkpoint()
        a = sync()
        cached_output = cache_generation_output(model, generation, config["evaluation"]["max_new_tokens"], cached=True)
        payload = dict(model.valid_payload_counts)
        cached_seconds = sync() - a
        uncached_output = cache_generation_output(model, generation, config["evaluation"]["max_new_tokens"], cached=False)
        record["checks"]["cache_comparison"] = compare_outputs(cached_output, uncached_output, output, "cache-comparison")
        cached = cached_output.sequences.to(next(model.base.parameters()).device)
        uncached = uncached_output.sequences.to(cached.device)
        del cached_output, uncached_output, frozen_state
        checkpoint()
        tokens_equal = torch.equal(cached, uncached)
        if cache_policy == "strict_tokens" and not tokens_equal:
            raise AssertionError("Cached/uncached generated tokens differ; inspect numerical margins before widening")
        if cache_policy == "fp32_math_reference":
            record["checks"]["bf16_free_running"] = {"tokens_equal": tokens_equal, "diagnostic_only": True}
            checkpoint()
            record["checks"]["fp32_math_reference"] = fp32_math_reference(
                config, model, fixture, args.prefix_json, output)
            checkpoint()
        if getattr(args, "cache_capture_only", False):
            record["status"] = "DIAGNOSTIC_CAPTURED_NOT_ACCEPTANCE"
            checkpoint()
            return
        a = sync()
        reload_receipt = perturb_and_reload(model, output / "disposable-communication.pt")
        save_reload_seconds = sync() - a
        with torch.no_grad(), autocast_for(model), model.transmission(None):
            replay = model.generate(**generation, max_new_tokens=config["evaluation"]["max_new_tokens"], do_sample=False, num_beams=1, use_cache=True)
        if not torch.equal(cached, replay):
            raise AssertionError("Reload generation replay differs")
        noisy_outputs = []
        for _ in range(2):
            checkpoint()
            with torch.no_grad(), autocast_for(model), cast(AWGNChannel, model.channel).replay(config["seed"], "coco-preflight-noisy-generation"), model.transmission(-6.0):
                noisy_outputs.append(model.generate(**generation, max_new_tokens=config["evaluation"]["max_new_tokens"],
                    do_sample=False, num_beams=1, use_cache=True))
        if not torch.equal(noisy_outputs[0], noisy_outputs[1]):
            raise AssertionError("Fixed-noise generation replay differs")
        record["checks"]["noisy_generation_replay"] = {"pass": True, "snr_db": -6.0,
            "seed": config["seed"], "namespace": "coco-preflight-noisy-generation",
            "token_ids": noisy_outputs[0].tolist()}
        if payload["hidden"] <= 0 or (model.memory_codec is not None and payload["memory"] <= 0):
            raise AssertionError("Generation did not traverse required coded streams")
        record["checks"]["generation"] = {"pass": True, "batch_size": 1, "cached_seconds": cached_seconds,
            "token_ids": cached.tolist(), "reload_exact": True, "cache_exact": tokens_equal, "cache_policy": cache_policy,
            "save_reload_seconds": save_reload_seconds, "reload_state": reload_receipt,
            "payload": payload, "memory": memory()}
        record["safe_microbatches"] = [r["microbatch"] for r in record["batch_results"] if r["status"] == "PASS"]
        if not record["safe_microbatches"]:
            raise RuntimeError("No safe training microbatch")
        # Continuous production loop: synchronization only at interval boundaries.
        # Disposable existing codec initialization is explicitly not fresh quality evidence.
        selected = min((r for r in record["batch_results"] if r["status"] == "PASS"),
                       key=lambda r: sum(x["seconds"] for x in r["steps"]) / len(r["steps"]))
        short = copy.deepcopy(config)
        short["run"] = {"name": "coco-preflight-production", "output_dir": str(output / "production"), "wandb_project": None}
        short["training"].update(batch_size=selected["microbatch"], gradient_accumulation=selected["accumulation"],
            max_steps=args.production_steps, schedule_steps=config["training"].get("schedule_steps", config["training"]["max_steps"]),
            eval_every=args.production_steps, validation_batches=1, validation_snrs=[0],
            save_steps=[args.production_steps], feature_summary=None, log_every=4,
            patience=None, max_minutes=max(0.1, (deadline - time.perf_counter()) / 60))
        short["training"].pop("presentation_stream", None)
        short["training"].pop("paired_randomness", None)
        original_build, original_step = training.build_model, torch.optim.AdamW.step
        original_validate, original_save = training.validate, training.save_checkpoint
        original_load_data, original_losses = training.load_data, training.batch_losses
        intervals, stage_seconds = {}, {"validation": [], "checkpoint": []}
        counter = 0
        def optimizer_step(self, closure=None):
            nonlocal counter
            result = original_step(self, closure)
            counter += 1
            if counter == 2:
                intervals["start"] = sync()
            if counter == args.production_steps:
                intervals["end"] = sync()
            record_production_update(record, args.effective_batch, counter, intervals)
            checkpoint()
            return result
        def timed_validate(*a, **kw):
            begin = sync()
            value = original_validate(*a, **kw)
            stage_seconds["validation"].append(sync() - begin)
            return value
        def timed_save(*a, **kw):
            begin = sync()
            value = original_save(*a, **kw)
            stage_seconds["checkpoint"].append(sync() - begin)
            return value
        def evidence_load_data(c, proc, saved_ids=None):
            loaded = original_load_data(c, proc, saved_ids)
            for name, key, shuffle in (("train", "train_ids", True), ("validation", "selection_ids", False)):
                old = getattr(loaded, name)
                wrapped = IndexedCaptionEvidence(old.dataset, loaded.ids[key])
                setattr(loaded, name, DataLoader(wrapped, batch_size=old.batch_size,
                    shuffle=shuffle, num_workers=0, pin_memory=old.pin_memory))
            return loaded
        def evidence_losses(model, batch, settings, snr, **kw):
            # CPU metadata only: no GPU copy/synchronization or changes to channel draws.
            event = {"phase": "train" if model.training else "validation",
                "image_ids": batch["evidence_image_id"].tolist(),
                "target_ids": [row[row != -100].tolist() for row in batch["labels"]],
                "source_ids_sha256": hashlib.sha256(batch["input_ids"].contiguous().numpy().tobytes()).hexdigest(),
                "source_valid_lengths": batch["attention_mask"].sum(-1).tolist()}
            with (output / "production-inputs.jsonl").open("a") as stream:
                stream.write(json.dumps(event) + "\n")
            return original_losses(model, batch, settings, snr, **kw)
        optimizer.zero_grad(set_to_none=True)
        del optimizer, kwargs, labels, expected, generation
        gc.collect()
        torch.cuda.empty_cache()
        training.build_model = lambda _: (processor, model)
        torch.optim.AdamW.step = optimizer_step
        training.validate, training.save_checkpoint = timed_validate, timed_save
        training.load_data, training.batch_losses = evidence_load_data, evidence_losses
        checkpoint()
        try:
            production_path = training.train(short)
        finally:
            training.build_model, torch.optim.AdamW.step = original_build, original_step
            training.validate, training.save_checkpoint = original_validate, original_save
            training.load_data, training.batch_losses = original_load_data, original_losses
        completion = json.loads((production_path / "completion.json").read_text())
        elapsed = intervals.get("end", 0) - intervals.get("start", 0)
        record["production"] = {"run_path": str(production_path), "completion": completion,
            "microbatch": selected["microbatch"], "accumulation": selected["accumulation"],
            "optimizer_updates_observed": counter, "warmup_updates": 2,
            "measured_updates": args.production_steps - 2,
            "interval_seconds": elapsed if "end" in intervals else None,
            "seconds_per_update": elapsed / (args.production_steps - 2) if "end" in intervals else None,
            "stage_seconds": stage_seconds,
            "timing_scope": "normal training data/AWGN/backward/optimizer/logging plus CPU input evidence writes after update2 through final optimizer; synchronized interval boundaries only; final validation/save separate",
            "initialization": "disposable codec after capacity probes; not a fresh quality result"}
        if counter != args.production_steps or "end" not in intervals:
            raise RuntimeError("Production timing segment did not complete bounded update budget")
        # Real scorer and generation batch sweep use identical report images.
        model.eval()
        report = copy.copy(data)
        report.report = data.report.select(range(min(8, len(data.report))))
        record["generation_batches"] = []
        for generation_batch in (1, 2, 4, 8):
            checkpoint()
            generation_output = output / f"generation-batch-{generation_batch}"
            generation_output.mkdir()
            settings = dict(config["evaluation"], batch_size=generation_batch)
            torch.cuda.reset_peak_memory_stats()
            begin = sync()
            try:
                with autocast_for(model), model.transmission(None):
                    scores = evaluate_coco(model, processor, report, settings, generation_output, "no_noise")
                seconds = sync() - begin
                record["generation_batches"].append({"batch_size": generation_batch, "status": "PASS",
                    "images": len(report.report), "seconds_including_scorer": seconds,
                    "seconds_per_image_including_scorer": seconds / len(report.report),
                    "metrics": scores, "memory": memory()})
            except torch.OutOfMemoryError as exc:
                record["generation_batches"].append({"batch_size": generation_batch, "status": "OOM", "error": str(exc)})
                for name in ("activation", "reconstruction", "memory_activation", "memory_reconstruction"):
                    setattr(model, name, None)
                gc.collect()
                torch.cuda.empty_cache()
                break
        if not any(row["status"] == "PASS" for row in record["generation_batches"]):
            raise RuntimeError("No safe generation batch/scorer completion")
        record["status"] = "PASS_WITH_LIMITATIONS"
        checkpoint()
    except BaseException as exc:
        record.update(status="BLOCKED", error={"type": type(exc).__name__, "message": str(exc)})
        write(output / "results.json", record)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-policy", choices=["strict_tokens", "fp32_math_reference"], default="strict_tokens")
    parser.add_argument("--prefix-json", type=Path)
    parser.add_argument("--cache-capture-only", action="store_true",
                        help="Stop after saving cache comparison; no production segment or acceptance")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--effective-batch", type=int, default=32)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--updates-per-batch", type=int, default=2)
    parser.add_argument("--production-steps", type=int, default=12)
    parser.add_argument("--inventory-size", type=int, default=16)
    parser.add_argument("--max-minutes", type=float, default=25)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
