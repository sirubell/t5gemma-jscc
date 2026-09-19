"""Evaluate the saved model configuration; optional YAML overrides evaluation only."""
import copy
import json
import time
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from tqdm import tqdm

from .config import save_config
from .data import load_data
from .data.coco import caption_prompt
from .models.channel import build_channel
from .models.split_model import build_model
from .runtime import append_metrics, isolated_rng, new_run, source_state


def clean_caption(text):
    text = text.split("\n")[0].split("\r")[0].strip()
    return text.split(". ")[0] + "." if ". " in text else text


@torch.no_grad()
def evaluate_coco(model, processor, data, settings, output, condition):
    prompt = caption_prompt(data.demo_captions)
    parameter = next(model.base.parameters())
    references, captions, records = {}, {}, []
    for start in tqdm(range(0, len(data.report), settings["batch_size"]), desc=str(condition)):
        batch = [data.report[i] for i in range(start, min(start + settings["batch_size"], len(data.report)))]
        inputs = [processor(images=data.demo_images + [row["image"].convert("RGB")], text=prompt,
                            return_tensors="pt") for row in batch]
        kwargs = {key: torch.cat([item[key] for item in inputs], dim=0).to(parameter.device)
                  for key in ("input_ids", "attention_mask", "pixel_values")}
        kwargs["pixel_values"] = kwargs["pixel_values"].to(parameter.dtype)
        generated = model.generate(**kwargs, max_new_tokens=settings["max_new_tokens"],
                                   do_sample=False, num_beams=1)
        for row, token_ids in zip(batch, generated):
            image_id = int(row["file_name"].rsplit("_", 1)[1].split(".")[0])
            raw = processor.tokenizer.decode(token_ids, skip_special_tokens=True)
            caption = clean_caption(raw)
            references[image_id] = [{"caption": caption.lower().strip()} for caption in row["answer"]]
            captions[image_id] = [{"caption": caption.lower().strip()}]
            # Seq2seq output begins with the decoder start token; do not count it as EOS.
            emitted = token_ids.tolist()[1:]
            eos = processor.tokenizer.eos_token_id in emitted
            records.append({"image_id": image_id, "raw_caption": raw, "caption": caption,
                            "token_ids": token_ids.tolist(), "eos": eos, "empty": not bool(caption),
                            "truncated": not eos and len(emitted) >= settings["max_new_tokens"]})
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    tokenizer = PTBTokenizer()  # Bundled COCO tokenizer requires Java.
    score, per_image = Cider().compute_score(tokenizer.tokenize(references), tokenizer.tokenize(captions))
    for record, value in zip(records, per_image):
        append_metrics(output / f"captions_{condition}.jsonl", {**record, "cider": float(value)})
    return {"cider": float(score), "num_samples": len(records),
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
    if model.split["stack"] == "dec":
        from .harness_payload import payload_harness_class
        adapter = payload_harness_class(harness_class)(
            communication_model=model, pretrained=model.base, tokenizer=tokenizer,
            backend="seq2seq", batch_size=settings["batch_size"])
    else:
        adapter = harness_class(pretrained=model.base, tokenizer=tokenizer, backend="seq2seq",
                                batch_size=settings["batch_size"])
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
    result = simple_evaluate(
        model=adapter, tasks=[task_config], task_manager=task_manager, log_samples=True, num_fewshot=settings["num_fewshot"],
        limit=settings["num_samples"], random_seed=0, numpy_random_seed=1234,
        torch_random_seed=0, fewshot_random_seed=1234,
    )
    if result is None or "results" not in result:
        raise RuntimeError("lm-eval returned no task results")
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        # Keep all harness provenance, but avoid duplicating the large per-example payload.
        metadata = {key: value for key, value in result.items() if key != "samples"}
        (output / f"harness_{condition}.json").write_text(
            json.dumps(metadata, indent=2, default=handle_non_serializable) + "\n")
        with (output / f"samples_{condition}.jsonl").open("w") as stream:
            for task, samples in result.get("samples", {}).items():
                for sample in samples:
                    stream.write(json.dumps({"task": task, **sample},
                                            default=handle_non_serializable) + "\n")
    metrics = result["results"]["hellaswag"]
    return {"acc": metrics["acc,none"], "acc_norm": metrics["acc_norm,none"],
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
    config = copy.deepcopy(state["config"])
    settings = config["evaluation"]
    if overrides_path:
        overrides = yaml.safe_load(Path(overrides_path).read_text())
        unknown = set(overrides) - (set(settings) | {"channel", "device"})
        if unknown:
            raise ValueError(f"Unknown evaluation options: {sorted(unknown)}")
        settings.update(overrides)
    if "device" in settings:
        config["model"]["device"] = settings["device"]
    processor, model = build_model(config)
    model.load_communication_state(state)
    if "channel" in settings:
        parameter = next(model.base.parameters())
        model.channel = build_channel(settings["channel"]).to(device=parameter.device, dtype=parameter.dtype)
    model.eval()
    data = load_data(config, processor, state["data_ids"], for_training=False)
    if config["task"] == "coco" and settings["num_samples"]:
        data.report = data.report.select(range(min(settings["num_samples"], len(data.report))))
    output = new_run(run_path / "evaluations", "eval")
    save_config(settings, output / "config.yaml")
    results = []
    conditions = list(settings["snrs"])
    if settings["vanilla"]:
        conditions.append("vanilla")
    for condition in conditions:
        snr = None if condition in ("no_noise", "vanilla") else float(condition)
        started = time.perf_counter()
        with isolated_rng(config["seed"]), model.transmission(snr, bypass=condition == "vanilla"):
            if config["task"] == "coco":
                metrics = evaluate_coco(model, processor, data, settings, output, condition)
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
            "checkpoint": str(checkpoint_path), "optimizer_step": state["step"],
            "task": config["task"], "split": config["split"], "source": source_state(),
            "torch_version": str(torch.__version__), "conditions": results,
        }, indent=2) + "\n")
    print(f"Evaluation: {output}", flush=True)
    return output
