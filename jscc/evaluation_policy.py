"""Versioned HellaSwag scoring and compact, independently recomputable evidence."""
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import torch


EVALUATION_OPTIONS = {"scoring_policy", "evidence_mode", "mode", "noise_seed",
                      "deterministic_algorithms", "attention_backend"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def scoring_kwargs(settings, model):
    policy = settings.get("scoring_policy", "legacy")
    if policy == "legacy":
        return {}
    if policy != "fp32-v1":
        raise ValueError(f"Unknown scoring policy: {policy}")
    dtype = next(model.base.parameters()).dtype
    return {"softmax_dtype": torch.float32,
            "mixed_precision_dtype": dtype if dtype in (torch.bfloat16, torch.float16) else None}


def evaluation_conditions(settings):
    mode = settings.get("mode", "all")
    if mode == "vanilla_only":
        return ["vanilla"]
    if mode not in {"all", "codec_only"}:
        raise ValueError(f"Unknown evaluation mode: {mode}")
    conditions = list(settings["snrs"])
    if mode == "all" and settings["vanilla"]:
        conditions.append("vanilla")
    if not conditions:
        raise ValueError("Evaluation requires at least one condition")
    return conditions


def evaluation_identity(model, tokenizer, settings, data_settings, metadata, adapter):
    """Intentionally exclude codec checkpoint/split/norm and noise condition."""
    parameter = next(model.base.parameters())
    config = model.base.config
    decoder_config = getattr(config, "decoder", config)
    policy = settings.get("scoring_policy", "legacy")
    source_files = ["evaluation.py", "evaluation_policy.py", "harness_payload.py",
                    "models/split_model.py", "models/channel.py"]
    root = Path(__file__).parent
    identity = {
        "version": "hellaswag-evaluation-identity-v1",
        "frozen_model": {"name": getattr(config, "_name_or_path", None),
                         "revision": getattr(config, "_commit_hash", None),
                         "config": config.to_dict()},
        "tokenizer": {"name": getattr(tokenizer, "name_or_path", None),
                      "class": type(tokenizer).__name__,
                      "init_kwargs": getattr(tokenizer, "init_kwargs", {}),
                      "special_tokens": tokenizer.special_tokens_map},
        "dataset": data_settings,
        "task_config": metadata.get("configs"), "task_versions": metadata.get("versions"),
        "num_samples": settings["num_samples"], "num_fewshot": settings["num_fewshot"],
        "seeds": {"random": 0, "numpy": 1234, "torch": 0, "fewshot": 1234},
        "batch_size_candidate_requests": settings["batch_size"],
        "scoring_policy": policy, "weight_dtype": str(parameter.dtype),
        "forward_logits_dtypes": sorted(adapter.forward_logits_dtypes),
        "log_softmax_dtype": "float32" if policy == "fp32-v1" else "forward_logits_dtype",
        "gather_dtype": "float32" if policy == "fp32-v1" else "forward_logits_dtype",
        "sum_dtype": "float32" if policy == "fp32-v1" else "forward_logits_dtype",
        "full_vocabulary": True, "normalization": "processed-choice-python-character-length",
        "backend": "seq2seq", "attention_backend": getattr(decoder_config, "_attn_implementation", None),
        "max_length": adapter.max_length, "truncation": "hflm-left-context;assert-continuation-fits",
        "device": str(parameter.device),
        "hardware": torch.cuda.get_device_name(parameter.device) if parameter.is_cuda else "cpu",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "packages": {name: importlib.metadata.version(name)
                     for name in ("lm_eval", "torch", "transformers", "datasets")},
        "source_sha256": {name: file_digest(root / name) for name in source_files},
    }
    # Tokenizer/config objects occasionally carry AddedToken or dtype objects.
    return json.loads(json.dumps(identity, default=str))


def compact_sample(sample, condition, identity_id, noise_policy_id, run_id=None):
    doc = sample["doc"]
    choices = doc["choices"]
    raw = [float(value[0]) for value in sample["filtered_resps"]]
    denominators = [len(choice) for choice in choices]
    if len(raw) != 4 or len(denominators) != 4 or not all(denominators):
        raise ValueError("HellaSwag evidence requires four nonempty choices")
    if not all(math.isfinite(value) for value in raw):
        raise ValueError("Non-finite HellaSwag score")
    normalized = [value / denominator for value, denominator in zip(raw, denominators, strict=True)]
    gold = int(doc["gold"])
    if gold not in range(4):
        raise ValueError("Invalid HellaSwag gold index")
    rankings = {"raw": sorted(range(4), key=lambda i: (-raw[i], i)),
                "normalized": sorted(range(4), key=lambda i: (-normalized[i], i))}
    row = {"run": run_id, "condition": condition, "sample_id": sample["doc_id"],
           "source_id": doc.get("source_id"), "gold": gold,
           "raw_scores": raw, "denominators": denominators, "normalized_scores": normalized,
           "evaluation_identity": identity_id, "noise_policy_id": noise_policy_id}
    for key, scores in (("raw", raw), ("normalized", normalized)):
        ranking = rankings[key]
        row.update({f"{key}_ranking": ranking, f"{key}_prediction": ranking[0],
                    f"{key}_correct": ranking[0] == gold,
                    f"{key}_gold_margin": scores[gold] - max(scores[i] for i in range(4) if i != gold),
                    f"{key}_top1_top2_gap": scores[ranking[0]] - scores[ranking[1]]})
    return row


def record_fewshots(task):
    """Observe the harness sampler without changing its pool, seed, or ordering."""
    by_prompt = {}
    original_sample, original_context = task.sampler.sample, task.fewshot_context
    sampled = []

    def sample(*args, **kwargs):
        selected = original_sample(*args, **kwargs)
        sampled[:] = selected
        return selected

    def context(*args, **kwargs):
        sampled.clear()
        prompt = original_context(*args, **kwargs)
        if not isinstance(prompt, str):
            raise ValueError("HellaSwag evidence expects plain text prompts")
        by_prompt[prompt] = [dict(document) for document in sampled]
        return prompt

    task.sampler.sample, task.fewshot_context = sample, context
    return by_prompt


def write_compact_evidence(output, result, adapter, identity, condition, model, noise_policy_id, run_id=None, fewshots=None):
    """Save full documents/prompts once per invocation, scores once per condition."""
    samples = result.get("samples", {}).get("hellaswag", [])
    if not samples:
        raise ValueError("Compact evidence requires actual logged HellaSwag samples")
    if identity["num_samples"] is not None and len(samples) != identity["num_samples"]:
        raise ValueError("Actual HellaSwag document count does not match requested panel")
    if len({sample["doc_id"] for sample in samples}) != len(samples):
        raise ValueError("Duplicate HellaSwag document identity")
    documents, prompts, rows, examples = [], {}, [], {}
    for sample in samples:
        requests = []
        for context, continuation in sample["arguments"]:
            request = adapter.request_evidence[(context, continuation)]
            prompt_id = digest(context)
            selected = (fewshots or {}).get(context)
            if selected is None:
                raise ValueError("Missing actual few-shot evidence for prompt")
            fewshot_ids = [digest(example) for example in selected]
            examples.update({digest(example): example for example in selected})
            prompts[prompt_id] = {"prompt_id": prompt_id, "text": context,
                                  "fewshot_ids": fewshot_ids,
                                  "context_token_ids": request["context_token_ids"]}
            requests.append({"prompt_id": prompt_id, "continuation": continuation,
                             "continuation_token_ids": request["continuation_token_ids"]})
        documents.append({"sample_id": sample["doc_id"], "document": sample["doc"],
                          "requests": requests, "doc_hash": sample.get("doc_hash")})
    identity["documents_digest"] = digest(documents)
    identity["prompts_digest"] = digest(prompts)
    identity["fewshot_examples_digest"] = digest(examples)
    identity_id = digest(identity)
    for sample, document in zip(samples, documents, strict=True):
        row = compact_sample(sample, condition, identity_id, noise_policy_id, run_id)
        payload = []
        for request in document["requests"]:
            context_len = len(prompts[request["prompt_id"]]["context_token_ids"])
            continuation_len = len(request["continuation_token_ids"])
            width = model.codec.config["bottleneck_dim"]
            memory_codec = getattr(model, "memory_codec", None)
            payload.append({"hidden": 0 if condition == "vanilla" else width * (
                continuation_len if model.split["stack"] == "dec" else context_len),
                "memory": 0 if condition == "vanilla" or memory_codec is None
                else memory_codec.config["bottleneck_dim"] * context_len})
        row["valid_payload_coordinates"] = payload
        rows.append(row)
    for field, metric in (("raw_correct", "acc,none"), ("normalized_correct", "acc_norm,none")):
        observed = sum(row[field] for row in rows) / len(rows)
        if not math.isclose(observed, result["results"]["hellaswag"][metric], abs_tol=1e-12):
            raise ValueError(f"Compact scores disagree with harness aggregate: {metric}")
    expected_counts = {stream: sum(candidate[stream] for row in rows
                                   for candidate in row["valid_payload_coordinates"])
                       for stream in ("hidden", "memory")}
    if expected_counts != dict(model.valid_payload_counts):
        raise ValueError(f"Per-item valid payload disagrees with measured counters: {expected_counts}")
    for name, values in (("documents", documents), ("prompts", list(prompts.values())),
                         ("fewshots", [{"fewshot_id": key, "document": value} for key, value in examples.items()])):
        path = output / f"{name}.jsonl"
        content = "".join(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n" for value in values)
        if path.exists() and path.read_text() != content:
            raise ValueError(f"Evaluation document/prompt identity changed between conditions: {name}")
        if not path.exists():
            path.write_text(content)
    (output / f"compact_{condition}.jsonl").write_text("".join(
        json.dumps(row, allow_nan=False) + "\n" for row in rows))
    (output / f"identity_{condition}.json").write_text(json.dumps(identity, indent=2) + "\n")
    return {"evaluation_identity": identity_id, "vanilla_key": identity_id,
            "finite_scores": True, "compact_num_samples": len(rows)}
