"""Evaluate the saved model configuration; optional YAML overrides evaluation only."""
import copy
from contextlib import nullcontext
import json
import time
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from tqdm import tqdm

from .evaluation_policy import (
    EVALUATION_OPTIONS, evaluation_conditions, evaluation_identity, file_digest,
    record_fewshots, scoring_kwargs, write_compact_evidence,
)
from .config import save_config
from .data import load_data
from .data.coco import caption_prompt
from .models.channel import build_channel
from .models.split_model import build_model
from .runtime import (
    append_metrics, configure_training_determinism, isolated_rng, new_run,
    prepare_trainable_parameters, source_state,
)


def clean_caption(text):
    text = text.split("\n")[0].split("\r")[0].strip()
    return text.split(". ")[0] + "." if ". " in text else text


def coco_generation_inputs(processor, data, batch, device, dtype):
    """Native processor batching preserves variable image crops and pads text only."""
    prompt = caption_prompt(data.demo_captions)
    inputs = processor(
        images=[data.demo_images + [row["image"].convert("RGB")] for row in batch],
        text=[prompt] * len(batch), return_tensors="pt", padding=True,
        truncation=False,
    )
    # Keep the original production input contract; processor owns image crop
    # expansion and its correspondence to the expanded image tokens.
    return {key: inputs[key].to(device=device, dtype=dtype if key == "pixel_values" else None)
            for key in ("input_ids", "attention_mask", "pixel_values")}


def coco_payload_records(events, emitted_lengths, stack):
    """Separate exact execution counts from EOS-aware useful decoder payload."""
    records: list[dict[str, Any]] = [{"allocated": {"hidden": 0, "memory": 0},
                "mask_valid": {"hidden": 0, "memory": 0},
                "generation_valid": {"hidden": 0, "memory": 0}}
               for _ in emitted_lengths]
    decoder_step = 0
    semantic_supported = True
    for event in events:
        if len(event["allocated"]) != len(records):
            raise ValueError("Payload observer batch differs from generated image batch")
        decoder_hidden = event["stream"] == "hidden" and stack == "dec"
        if decoder_hidden and event["shape"][1] != 1:
            semantic_supported = False
        for index, record in enumerate(records):
            stream = event["stream"]
            record["allocated"][stream] += event["allocated"][index]
            record["mask_valid"][stream] += event["mask_valid"][index]
            if not decoder_hidden or decoder_step < emitted_lengths[index]:
                record["generation_valid"][stream] += event["mask_valid"][index]
        if decoder_hidden:
            decoder_step += 1
    for record in records:
        record["policy"] = "per-row-observed-latents-v1"
        record["generation_valid_policy"] = "one-token-cached-decoder-through-first-eos-v1"
        if not semantic_supported:
            record["generation_valid"] = None
            record["generation_valid_policy"] = "unavailable-non-single-token-decoder-events"
    return records


def score_coco_captions(references, captions):
    """Return CIDEr plus frozen PTB tokens, allowing recomputation without Java."""
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    tokenizer = PTBTokenizer()
    reference_tokens, caption_tokens = tokenizer.tokenize(references), tokenizer.tokenize(captions)
    score, per_image = Cider().compute_score(reference_tokens, caption_tokens)
    return score, per_image, reference_tokens, caption_tokens


@torch.no_grad()
def evaluate_coco(model, processor, data, settings, output, condition, data_settings=None):
    from importlib.metadata import version
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    parameter = next(model.base.parameters())
    references, captions, records, batch_records = {}, {}, [], []
    for start in tqdm(range(0, len(data.report), settings["batch_size"]), desc=str(condition)):
        batch = [data.report[i] for i in range(start, min(start + settings["batch_size"], len(data.report)))]
        kwargs = coco_generation_inputs(processor, data, batch, parameter.device, parameter.dtype)
        with model.observe_payload() as events:
            generated = model.generate(**kwargs, max_new_tokens=settings["max_new_tokens"],
                                       do_sample=False, num_beams=1, use_cache=True)
        batch_ids, emitted_lengths = [], []
        for index, (row, token_ids) in enumerate(zip(batch, generated)):
            image_id = int(row["file_name"].rsplit("_", 1)[1].split(".")[0])
            if image_id in references:
                raise ValueError(f"Duplicate COCO report image ID: {image_id}")
            batch_ids.append(image_id)
            raw = processor.tokenizer.decode(token_ids, skip_special_tokens=True)
            caption = clean_caption(raw)
            reference_texts = list(row["answer"])
            references[image_id] = [{"caption": text.lower().strip()} for text in reference_texts]
            captions[image_id] = [{"caption": caption.lower().strip()}]
            # Seq2seq output starts with decoder-start; EOS and padded tail are
            # separated from generated content for latency and payload evidence.
            emitted = token_ids.tolist()[1:]
            eos_id = processor.tokenizer.eos_token_id
            eos = eos_id in emitted
            emitted_length = emitted.index(eos_id) + 1 if eos else len(emitted)
            emitted_lengths.append(emitted_length)
            reference_lengths = [len(processor.tokenizer(text, add_special_tokens=True)["input_ids"])
                                 for text in reference_texts]
            records.append({"image_id": image_id, "file_name": row["file_name"],
                            "references": reference_texts, "reference_token_lengths": reference_lengths,
                            "raw_caption": raw, "caption": caption, "batch_index": len(batch_records),
                            "source_valid_tokens": int(kwargs["attention_mask"][index].sum().item()),
                            "source_allocated_tokens": int(kwargs["input_ids"].shape[1]),
                            "source_truncation": False,
                            "token_ids": token_ids.tolist(), "generated_tokens_through_eos": emitted_length,
                            "eos": eos, "empty": not bool(caption),
                            "truncated": not eos and emitted_length >= settings["max_new_tokens"]})
        payloads = coco_payload_records(events, emitted_lengths, model.split["stack"])
        for record, payload in zip(records[-len(batch):], payloads):
            record["payload"] = payload
        batch_records.append({"image_ids": batch_ids, "payload_events": events,
                              "input_shapes": {key: list(value.shape) for key, value in kwargs.items()}})
    if not records:
        raise ValueError("COCO report set is empty")
    score, per_image, reference_tokens, caption_tokens = score_coco_captions(references, captions)
    for record, value in zip(records, per_image):
        image_id = record["image_id"]
        append_metrics(output / f"captions_{condition}.jsonl", {
            **record, "ptb_references": reference_tokens[image_id],
            "ptb_caption": caption_tokens[image_id], "cider": float(value)})
    identity = {"processor_class": type(processor).__name__,
                "tokenizer_class": type(processor.tokenizer).__name__,
                "tokenizer_name_or_path": getattr(processor.tokenizer, "name_or_path", None),
                "model_name_or_path": getattr(model.base.config, "_name_or_path", None),
                "model_commit_hash": getattr(model.base.config, "_commit_hash", None),
                "packages": {name: version(name) for name in ("transformers", "torch", "pycocoevalcap")},
                "scorer": "pycocoevalcap.Cider(n=4,sigma=6.0); PTBTokenizer",
                "caption_policy": "first-line-first-period-space; lowercase-strip-before-PTB",
                "data_settings": data_settings, "data_ids": data.ids,
                "prompt": caption_prompt(data.demo_captions), "demo_captions": data.demo_captions,
                "generation": {"max_new_tokens": settings["max_new_tokens"], "do_sample": False,
                               "num_beams": 1, "use_cache": True}, "batches": batch_records}
    (output / f"coco_evidence_{condition}.json").write_text(json.dumps(identity, indent=2) + "\n")
    generation_valid = (None if any(row["payload"]["generation_valid"] is None for row in records)
                        else {stream: sum(row["payload"]["generation_valid"][stream] for row in records)
                              for stream in ("hidden", "memory")})
    return {"cider": float(score), "num_samples": len(records),
            "channel_uses_generation_valid": generation_valid,
            "generation_valid_policy": "per-row-observed-cached-decoder-through-first-eos-v1",
            "mask_valid_limitation": "Execution mask counts can include already-finished generation rows",
            "evidence": f"coco_evidence_{condition}.json",
            **{f"{key}_rate": sum(record[key] for record in records) / len(records)
               for key in ("eos", "empty", "truncated")}}


def evaluate_hellaswag(model, tokenizer, settings, output=None, condition: str | float = "no_noise", data_settings=None):
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.api.registry import get_model
    from lm_eval.tasks import TaskManager
    from lm_eval.tasks._yaml_loader import load_yaml
    from lm_eval.utils import handle_non_serializable
    # Hooks live on the backbone. Passing it directly keeps HFLM's usual HF interface.
    # The registry is typed as base LM; its HF implementation accepts these HF kwargs.
    harness_class = cast(type[Any], get_model("hf"))
    from .harness_payload import payload_harness_class
    compact = settings.get("evidence_mode") == "compact-v1"
    if settings.get("evidence_mode") not in (None, "legacy", "compact-v1"):
        raise ValueError("Unknown evaluation evidence mode")
    adapter_kwargs = scoring_kwargs(settings, model)
    if model.split["stack"] == "dec" or compact or settings.get("scoring_policy") == "fp32-v1":
        adapter = payload_harness_class(harness_class)(
            communication_model=model, pretrained=model.base, tokenizer=tokenizer,
            record_requests=compact, backend="seq2seq", batch_size=settings["batch_size"],
            **adapter_kwargs)
    else:
        adapter = harness_class(pretrained=model.base, tokenizer=tokenizer, backend="seq2seq",
                                batch_size=settings["batch_size"], **adapter_kwargs)
    task_manager = TaskManager()
    # Resolve !function entries before passing an inline recipe back to the factory.
    # The index .cfg deliberately contains strings, not executable preprocessing.
    recipe_path = task_manager.task_index["hellaswag"].yaml_path
    if recipe_path is None:
        raise RuntimeError("Installed harness has no HellaSwag YAML recipe")
    task_config = copy.deepcopy(load_yaml(recipe_path, resolve_func=True))
    if data_settings:
        task_config["dataset_path"] = data_settings["name"]
        if data_settings.get("revision") is not None:
            task_config.setdefault("dataset_kwargs", {})["revision"] = data_settings["revision"]
    fewshots = None
    task = task_config
    if compact:
        from lm_eval.api.task import ConfigurableTask
        task = ConfigurableTask(config=task_config)
        fewshots = record_fewshots(task)
    if compact and isinstance(settings.get("num_samples"), int):
        adapter.forward_request_limit = 4 * settings["num_samples"]
    harness_options: dict[str, Any] = {"bootstrap_iters": 0} if compact else {}
    result = simple_evaluate(
        model=adapter, tasks=[task], task_manager=task_manager, log_samples=True, num_fewshot=settings["num_fewshot"],
        limit=settings["num_samples"], random_seed=0, numpy_random_seed=1234,
        torch_random_seed=0, fewshot_random_seed=1234,
        **harness_options,
    )
    if result is None or "results" not in result:
        raise RuntimeError("lm-eval returned no task results")
    evidence = {}
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        # Keep all harness provenance, but avoid duplicating the large per-example payload.
        metadata = {key: value for key, value in result.items() if key != "samples"}
        (output / f"harness_{condition}.json").write_text(
            json.dumps(metadata, indent=2, default=handle_non_serializable) + "\n")
        if compact:
            identity = evaluation_identity(model, tokenizer, settings, data_settings, metadata, adapter)
            evidence = write_compact_evidence(
                output, result, adapter, identity, condition, model,
                {"algorithm": "awgn-independent-streams-shared-epsilon-v1", "seed": settings["noise_seed"],
                 "namespace": settings.get("noise_namespace", "evaluation-v1")}
                if "noise_seed" in settings else "legacy-global-rng",
                settings.get("run_identity"), fewshots=fewshots)
        else:
            with (output / f"samples_{condition}.jsonl").open("w") as stream:
                for task, samples in result.get("samples", {}).items():
                    for sample in samples:
                        stream.write(json.dumps({"task": task, **sample},
                                                default=handle_non_serializable) + "\n")
    metrics = result["results"]["hellaswag"]
    return {**evidence, **({"candidate_forward_requests": adapter.forward_request_count} if compact else {}),
            "acc": metrics["acc,none"], "acc_norm": metrics["acc_norm,none"],
            "num_fewshot": settings["num_fewshot"],
            "num_samples": result.get("n-samples", {}).get("hellaswag", {})}


def evaluate(run_path, checkpoint_name=None, overrides_path=None, expected_step=None):
    if expected_step is not None and not checkpoint_name:
        raise ValueError("expected-step requires an explicit checkpoint")
    checkpoint_name = checkpoint_name or "best.pt"
    run_path = Path(run_path).resolve()
    checkpoint_path = run_path / checkpoint_name
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    from .checkpoint_policy import validate_checkpoint_step
    validate_checkpoint_step(state, expected_step)
    if state.get("schema") == "experiment-state-v2" or state.get("kind") in ("single_site_v2", "shared_encoder_v2"):
        raise ValueError("Versioned checkpoints require evaluate_checkpoint and its metadata gate")
    config = copy.deepcopy(state["config"])
    settings = config["evaluation"]
    if overrides_path:
        overrides = yaml.safe_load(Path(overrides_path).read_text())
        unknown = set(overrides) - (set(settings) | EVALUATION_OPTIONS | {"channel", "device"})
        if unknown:
            raise ValueError(f"Unknown evaluation options: {sorted(unknown)}")
        settings.update(overrides)
    if "device" in settings:
        config["model"]["device"] = settings["device"]
    conditions = evaluation_conditions(settings)
    if "deterministic_algorithms" in settings:
        configure_training_determinism(settings)
    if settings.get("scoring_policy") == "fp32-v1":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    processor, model = build_model(config)
    if config["task"] == "coco" or settings.get("scoring_policy") == "fp32-v1":
        prepare_trainable_parameters(model)
    if conditions != ["vanilla"]:
        model.load_communication_state(state)
    if settings.get("attention_backend"):
        actual_backend = getattr(getattr(model.base.config, "decoder", model.base.config),
                                 "_attn_implementation", None)
        if actual_backend != settings["attention_backend"]:
            raise ValueError(f"Attention backend mismatch: {actual_backend}")
    if "channel" in settings:
        parameter = next(model.base.parameters())
        model.channel = build_channel(settings["channel"]).to(device=parameter.device, dtype=parameter.dtype)
    model.eval()
    data = (load_data(config, processor, state["data_ids"], for_training=False)
            if config["task"] == "coco" else None)
    if data is not None and settings["num_samples"]:
        data.report = data.report.select(range(min(settings["num_samples"], len(data.report))))
    output = new_run(run_path / "evaluations", "eval")
    save_config(settings, output / "config.yaml")
    results = []
    checkpoint_sha256 = file_digest(checkpoint_path)
    settings["run_identity"] = str(run_path)
    if "noise_seed" in settings and not hasattr(model.channel, "replay"):
        raise ValueError("Paired noise requires a replay-capable channel")
    for condition in conditions:
        snr = None if condition in ("no_noise", "vanilla") else float(condition)
        started = time.perf_counter()
        noise_scope = (cast(Any, model.channel).replay(settings["noise_seed"], "evaluation-v1")
                       if "noise_seed" in settings else nullcontext())
        with isolated_rng(config["seed"]), noise_scope, model.transmission(snr, bypass=condition == "vanilla"):
            if config["task"] == "coco":
                metrics = evaluate_coco(model, processor, data, settings, output, condition, config["data"])
            else:
                metrics = evaluate_hellaswag(model, processor, settings, output, condition, config["data"])
        allocated = dict(getattr(model, "channel_uses_allocated", model.channel_uses))
        valid = getattr(model, "valid_payload_counts", None)
        results.append({"condition": condition, **metrics,
                        # Keep the historical key as an allocated execution
                        # count, and add the corrected valid-payload count.
                        "channel_uses_real": allocated,
                        "channel_uses_allocated": allocated,
                        "channel_uses_valid": dict(valid) if valid is not None else None,
                        "payload_count_policy": ("actual-harness-continuation-lengths-v2"
                                                 if config["task"] == "hellaswag"
                                                 else "stream-masks-v1"),
                        "elapsed_seconds": time.perf_counter() - started})
        print(results[-1], flush=True)
        # Write after each condition so a later interruption preserves completed points.
        (output / "results.json").write_text(json.dumps({
            "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha256,
            "optimizer_step": state["step"], "evaluation_mode": settings.get("mode", "all"),
            "codec_checkpoint_loaded": conditions != ["vanilla"],
            "task": config["task"], "split": config["split"], "source": source_state(),
            "torch_version": str(torch.__version__), "conditions": results,
        }, indent=2) + "\n")
    print(f"Evaluation: {output}", flush=True)
    return output


BASELINE_CONDITIONS = ("no_noise", -6, 0, 6, 12, 18)


def codec_observation_identity(request):
    """Bind result reuse to the entire execution request, unlike input comparability."""
    required = {"schema", "checkpoint_sha256", "target_site", "target_role", "task",
                "input_ids", "source_family_ids", "prompt_policy", "noise", "layout",
                "backend", "precision", "scorer", "reference_corpus", "source",
                "expected_items", "settings", "data_settings", "conditions", "config", "data",
                "parent", "panel", "protocol", "learner_kind", "step"}
    missing = required - request.keys()
    if missing:
        raise ValueError(f"Observation request missing fields: {sorted(missing)}")
    if request["schema"] != "codec-observation-v1":
        raise ValueError("Unsupported observation schema")
    if tuple(request["conditions"]) != BASELINE_CONDITIONS:
        raise ValueError("Baseline observation requires the exact six conditions")
    count = request["expected_items"]
    if type(count) is not int or count < 1:
        raise ValueError("expected_items must be a positive integer")
    if (len(request["input_ids"]) != count or len(request["source_family_ids"]) != count
            or len(set(request["input_ids"])) != count
            or any(value is None or value == "" for value in request["source_family_ids"])):
        raise ValueError("Observation requires unique input IDs and actual source-family IDs")
    if request["task"] not in ("hellaswag", "coco"):
        raise ValueError("Unsupported observation task")
    if (type(request["noise"].get("seed")) is not int
            or not request["noise"].get("namespace")):
        raise ValueError("Observation requires explicit noise seed and namespace")
    for key in ("checkpoint_sha256", "target_site", "target_role", "prompt_policy", "layout",
                "backend", "precision", "scorer", "reference_corpus", "source"):
        if not request[key]:
            raise ValueError(f"Observation identity requires {key}")
    import hashlib
    projected = observation_record_request(request)
    encoded = json.dumps(projected, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(b"codec-observation-v1\0" + encoded).hexdigest()


def _observation_items(output, task, condition):
    """Read the actual adapter evidence, never infer completion from requested count."""
    name = f"captions_{condition}.jsonl" if task == "coco" else f"compact_{condition}.jsonl"
    path = output / name
    if not path.exists():
        raise ValueError(f"Missing per-item evidence: {name}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if task == "coco":
        ids = [row["image_id"] for row in rows]
        families = ids  # Each COCO image is the source family for its captions.
    else:
        document_path = output / "documents.jsonl"
        if document_path.exists():
            documents = {row["sample_id"]: row for row in
                         (json.loads(line) for line in document_path.read_text().splitlines() if line.strip())}
            for row in rows:
                document = documents[row["sample_id"]]
                prediction = row["normalized_prediction"]
                row["tokens"] = document["requests"][prediction]["continuation_token_ids"]
                row["tokens_policy"] = "normalized-prediction-continuation-v1"
        ids = [row["sample_id"] for row in rows]
        families = [row["source_id"] for row in rows]
    return rows, ids, families


def evaluate_checkpoint(validated_state, request, *, model, processor, data=None, output):
    """Run a gated v2 six-condition observation through the existing task adapters.

    The caller opens the checkpoint through experiment_state first. Each condition
    has a fresh directory; partial evidence and failures remain in the receipt.
    No count here certifies scientific quality or a validation threshold.
    """
    from .experiment_state import isolated_rng as state_isolated_rng, open_state
    identity = codec_observation_identity(request)
    reopened = open_state(validated_state.reference, expected=validated_state.payload["metadata"])
    if not _state_payload_equal(validated_state.payload, reopened.payload):
        raise ValueError("Validated state payload was modified after checkpoint validation")
    validated_state = reopened
    site, authorization = validate_evaluation_target(validated_state, request, model)
    settings = copy.deepcopy(request["settings"])
    settings["noise_seed"] = request["noise"]["seed"]
    settings["noise_namespace"] = request["noise"]["namespace"]
    if request["task"] == "hellaswag":
        settings["evidence_mode"] = "compact-v1"
    if settings.get("num_samples") != request["expected_items"]:
        raise ValueError("Adapter sample limit differs from observation request")
    if request["task"] == "coco" and (data is None or len(data.report) != request["expected_items"]):
        raise ValueError("COCO report membership must be prepared before evaluation")
    if not hasattr(model.channel, "replay"):
        raise ValueError("Versioned evaluation requires replay-capable channel")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "codec-observation-v1", "identity": identity,
               "request": copy.deepcopy(request), "status": "incomplete", "conditions": [], "reuse": None}

    def persist():
        temporary = output / "observation.json.tmp"
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "observation.json")

    persist()
    model.codec.load_state_dict(validated_state.payload["model"])
    previous_training = model.training
    model.eval()
    try:
        for condition in BASELINE_CONDITIONS:
            condition_output = output / str(condition)
            condition_output.mkdir()
            started = time.perf_counter()
            row = {"condition": condition, "requested": request["expected_items"],
                   "completed": 0, "failed": 0, "unobserved": request["expected_items"],
                   "denominator": 0, "status": "incomplete", "metrics": {}, "items": []}
            try:
                snr = None if condition == "no_noise" else float(condition)
                with state_isolated_rng(settings["noise_seed"]), model.channel.replay(
                        settings["noise_seed"], request["noise"]["namespace"]), \
                        model.at_site(site, purpose="evaluate", authorization=authorization), model.transmission(snr):
                    if request["task"] == "coco":
                        metrics = evaluate_coco(model, processor, data, settings, condition_output,
                                                condition, request["data_settings"])
                    else:
                        metrics = evaluate_hellaswag(model, processor, settings, condition_output,
                                                     condition, request["data_settings"])
                items, ids, families = _observation_items(condition_output, request["task"], condition)
                row.update(completed=len(items), denominator=len(items), items=items, metrics=metrics,
                           unobserved=max(0, request["expected_items"] - len(items)))
                if ids != request["input_ids"] or families != request["source_family_ids"]:
                    raise ValueError("Observed item/source-family order differs from request")
                row.update(completed=len(items), denominator=len(items), unobserved=0,
                           metrics=metrics, items=items, status="complete",
                           channel_uses_allocated=dict(getattr(model, "channel_uses_allocated", model.channel_uses)),
                           channel_uses_valid=(dict(model.valid_payload_counts)
                                               if model.valid_payload_counts is not None else None))
            except Exception as error:
                # Adapters may have durably emitted a strict subset before failing.
                if not row["items"]:
                    try:
                        partial, _, _ = _observation_items(condition_output, request["task"], condition)
                        row.update(items=partial, completed=len(partial), denominator=len(partial),
                                   unobserved=max(0, request["expected_items"] - len(partial)))
                    except (OSError, ValueError, KeyError):
                        pass
                row.update(error={"type": type(error).__name__, "message": str(error)},
                           failure_scope="condition; per-item failure count unavailable", failed=None)
            row["elapsed_seconds"] = time.perf_counter() - started
            row["output_hashes"] = {path.name: file_digest(path) for path in sorted(condition_output.iterdir())
                                    if path.is_file()}
            receipt["conditions"].append(row)
            persist()
    finally:
        model.train(previous_training)
    receipt["status"] = ("complete" if all(row["status"] == "complete" for row in receipt["conditions"])
                         else "incomplete")
    persist()
    return receipt


def validate_evaluation_target(validated_state, request, model):
    """Authorize exact checkpoint/site identities before any codec weight load."""
    from dataclasses import asdict
    from .experiment_state import ValidatedState
    from .models.split_model import ResolvedSite, resolve_encoder_site
    if not isinstance(validated_state, ValidatedState):
        raise ValueError("Evaluation requires a validated versioned state")
    payload, reference = validated_state.payload, validated_state.reference
    if request["checkpoint_sha256"] != reference.sha256:
        raise ValueError("Observation checkpoint hash mismatch")
    metadata = payload["metadata"]
    if request["learner_kind"] != payload["kind"]:
        raise ValueError("Observation learner kind mismatch")
    if "data_identity" in metadata and request["data"] != metadata["data_identity"]:
        raise ValueError("Observation data identity mismatch")
    if metadata.get("model_state_contract") != "codec-only-stateless-channel-v1":
        raise ValueError("Unsupported checkpoint codec/channel state contract")
    if any(True for _ in model.channel.parameters()):
        raise ValueError("Versioned codec-only evaluation requires a stateless channel")
    role = metadata.get("snapshot_role")
    step = metadata["completed_updates"]
    if role not in ("initialization", "trained") or (role == "initialization") != (step == 0):
        raise ValueError("Invalid initialization/trained snapshot role")
    if request["step"] != step:
        raise ValueError("Observation completed update count mismatch")
    for field, key in (("config", "config_identity"), ("source", "source_identity"),
                       ("protocol", "protocol_identity"), ("parent", "parent_identity")):
        if request[field] != metadata[key]:
            raise ValueError(f"Observation {field} identity mismatch")
    site = ResolvedSite(**request["target_site"])
    base_config = getattr(model.base, "config", None)
    backend = getattr(getattr(base_config, "decoder", base_config), "_attn_implementation", None)
    if backend is not None and request["backend"] != backend:
        raise ValueError("Observation attention backend differs from model")
    if hasattr(model.base, "parameters"):
        parameter = next(model.base.parameters(), None)
        dtype_names = {torch.float32: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16"}
        if parameter is not None and request["precision"] not in (
                str(parameter.dtype), dtype_names.get(parameter.dtype)):
            raise ValueError("Observation precision differs from frozen backbone")
    resolved = resolve_encoder_site(model.base, site.split, site.model_revision)
    if asdict(resolved) != asdict(site):
        raise ValueError("Target site does not match actual backbone metadata")
    if payload["kind"] == "single_site_v2":
        if request["target_site"] != metadata.get("site") or request["target_role"] != "trained":
            raise ValueError("Single-site checkpoint cannot override its saved site")
        return site, {"sites": {site.site_id: "trained"}}
    if payload["kind"] != "shared_encoder_v2":
        raise ValueError("Unsupported evaluation checkpoint kind")
    declaration = metadata.get("evaluation_sites", {}).get(site.site_id)
    if not declaration or declaration.get("site") != request["target_site"]:
        raise ValueError("Shared checkpoint does not declare the requested exact site")
    if declaration.get("role") != request["target_role"]:
        raise ValueError("Shared checkpoint site role mismatch")
    if request["target_role"] == "trained":
        return site, {"sites": {site.site_id: "trained"}}
    if request["target_role"] != "heldout_after_freeze" or site.site_id != "enc_l14":
        raise ValueError("Unsupported shared heldout request")
    freeze = request.get("freeze_receipt", {})
    required = ("recipe_identity", "architecture_decision", "trained_run_inventory", "observation_plan")
    if (freeze.get("schema") != "study-freeze-v1" or freeze.get("status") != "complete"
            or any(not freeze.get(field) for field in required)
            or reference.sha256 not in freeze.get("checkpoint_hashes", [])
            or freeze.get("protocol_identity") != metadata["protocol_identity"]
            or freeze.get("site_policy") != metadata.get("evaluation_sites")
            or request["panel"] not in freeze.get("observation_plan", [])):
        raise ValueError("Heldout evaluation requires a complete bound study freeze receipt")
    from .evaluation_policy import digest
    freeze_reference = request.get("freeze_reference", {})
    freeze_path = Path(freeze_reference.get("path", ""))
    if (not freeze_path.is_file()
            or file_digest(freeze_path) != freeze_reference.get("sha256")
            or json.loads(freeze_path.read_text()) != freeze):
        raise ValueError("Freeze receipt differs from its immutable file reference")
    return site, {"sites": {site.site_id: "heldout"}, "heldout_freeze_receipt": digest(freeze)}



def observation_record_request(request):
    """Project execution inputs into the model-free records request schema.

    The complete execution request is retained inline in details.outputs, so no
    input, scorer, source-family, freeze, or runtime binding disappears in this
    projection. This is a reference to planned output evidence, not its results.
    """
    from .evaluation_policy import digest
    def reference(value):
        return value if isinstance(value, str) and value else "sha256:" + digest(value)

    comparison = request.get("comparison", {})
    controls = {"architecture", "initialization", "training_data", "objective", "exposure", "schedule"}
    if (set(comparison) != controls or any(not isinstance(value, str) or not value.strip()
                                           for value in comparison.values())):
        raise ValueError("Observation requires explicit nonempty comparison control references")
    site = request["target_site"]
    site_id = "enc_fn" if site["where"] == "after_final_norm" else f"enc_l{site['index']}"
    kind = ("initialization" if request["step"] == 0 else
            "shared" if request["learner_kind"] == "shared_encoder_v2" else "specialist")
    return {
        **{field: reference(request[field]) for field in
           ("source", "config", "data", "panel", "task", "protocol", "noise", "scorer",
            "layout", "precision", "backend")},
        "parent": None if request["parent"] is None else reference(request["parent"]),
        "checkpoint": request["checkpoint_sha256"], "site": site_id,
        "role": "heldout" if request["target_role"] == "heldout_after_freeze" else request["target_role"],
        "conditions": list(request["conditions"]), "expected_items": request["expected_items"],
        "learner_kind": kind, "step": request["step"],
        "site_step": request.get("site_step", request["step"]), "purpose": "task",
        "objective_kind": None, "comparison": copy.deepcopy(comparison),
        "details": {"prompt": reference(request["prompt_policy"]),
                    "native_policy": reference(request["settings"]),
                    "draws": reference(request["noise"]),
                    "outputs": "inline-execution-request-v1:" + json.dumps(
                        request, sort_keys=True, separators=(",", ":"), allow_nan=False)},
    }


def observation_event_payload(receipt):
    """Return experiment-records-v1 observation payload without a records import.

    Non-numeric adapter metadata and errors remain in the execution receipt;
    all per-item data is retained here with normalized ID names. Unknown failed
    item counts remain null with a reason, and unobserved work stays explicit.
    """
    import math
    if receipt["identity"] != codec_observation_identity(receipt["request"]):
        raise ValueError("Execution receipt observation identity mismatch")
    conditions = []
    for row in receipt["conditions"]:
        metrics = {name: {"value": value, "reason": None}
                   for name, value in row["metrics"].items()
                   if type(value) in (int, float) and math.isfinite(value)}
        if not metrics:
            metrics["task_score"] = {"value": None, "reason": row.get("error", {}).get(
                "message", "No finite task score available")}
        metrics["unobserved_items"] = {"value": row.get(
            "unobserved", max(0, row["requested"] - row["completed"])), "reason": None}
        if "elapsed_seconds" in row:
            metrics["elapsed_seconds"] = {"value": row["elapsed_seconds"], "reason": None}
        items = []
        for item in row["items"]:
            item_id = item.get("sample_id", item.get("image_id"))
            source_id = item.get("source_id", item.get("image_id"))
            if item_id is None or source_id is None:
                raise ValueError("Cannot project item with unavailable item/source identity")
            normalized = {**item, "item_id": str(item_id), "source_id": str(source_id)}
            if receipt["request"]["task"] == "hellaswag":
                required = ("tokens", "raw_correct", "normalized_correct", "normalized_prediction", "normalized_scores")
                if any(key not in item for key in required):
                    raise ValueError("HellaSwag item lacks actual token/prediction/correctness evidence")
                prediction = item["normalized_prediction"]
                normalized.update(raw_correct=int(item["raw_correct"]),
                                  normalized_correct=int(item["normalized_correct"]),
                                  prediction=str(prediction), score=item["normalized_scores"][prediction])
            else:
                required = ("token_ids", "raw_caption", "caption", "cider", "eos", "truncated")
                if any(key not in item for key in required):
                    raise ValueError("COCO item lacks actual caption/token/generation evidence")
                normalized.update(tokens=item["token_ids"], caption_raw=item["raw_caption"],
                                  caption_clean=item["caption"], cap_hit=item["truncated"])
            items.append(normalized)
        task_fields = ({"raw_accuracy": "raw_correct", "normalized_accuracy": "normalized_correct"}
                       if receipt["request"]["task"] == "hellaswag" else {"cider": "cider"})
        for metric, field in task_fields.items():
            metrics[metric] = ({"value": sum(item[field] for item in items) / len(items), "reason": None}
                               if items else {"value": None, "reason": "No completed per-item task evidence"})
        conditions.append({"condition": row["condition"], "requested": row["requested"],
                           "completed": row["completed"], "failed": row["failed"],
                           "failure_reason": (row.get("failure_scope", "Per-item failure count unavailable")
                                              if row["failed"] is None else
                                              row.get("error", {}).get("message")),
                           "denominator": row["denominator"], "metrics": metrics,
                           "items": items, "objectives": {}})
    return {"request": observation_record_request(receipt["request"]), "identity": receipt["identity"],
            "status": receipt["status"], "conditions": conditions, "reuse": receipt.get("reuse")}



def _state_payload_equal(left, right):
    """Compare a caller's mutable state against freshly verified artifact bytes."""
    import numpy as np
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left.cpu(), right.cpu()))
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys()
                and all(_state_payload_equal(value, right[key]) for key, value in left.items()))
    if isinstance(left, (tuple, list)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_state_payload_equal(a, b) for a, b in zip(left, right)))
    return type(left) is type(right) and left == right
