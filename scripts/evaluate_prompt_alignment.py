"""Inference-only paired training evaluation against the frozen 256-document fixture."""
import argparse
import copy
import gc
import json
from pathlib import Path
import sys
import time
from types import MethodType, SimpleNamespace
from typing import Any, cast

# Support both `python scripts/...py` and package imports in CPU tests.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.frozen_prompt_gap import (
    RequestBudget, check_replay, read, rows, score_batch, sha, tensor_batch, write,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def split_name(config):
    split = config["split"]
    require(split["stack"] == "enc", "Only encoder splits are authorized")
    if split["where"] == "after_final_norm":
        return "enc_fn"
    require(split["where"] == "after_layer" and split["index"] in (9, 19),
            "Only enc_l9, enc_l19, enc_fn are authorized")
    return f"enc_l{split['index']}"


def validate_run(record, current_source_digest):
    """Validate saved provenance before constructing a pretrained model."""
    import torch
    import yaml

    run = Path(record["run_path"])
    checkpoint = Path(record["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = run / checkpoint
    expected = record["expected_step"]
    require(type(expected) is int and expected > 0, "expected_step must be a positive integer")
    config_path = run / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    metadata = read(run / "run.json")
    require(metadata["resolved_config_sha256"] == sha(config_path), "Run config hash mismatch")
    require(metadata["training_source_sha256"] == current_source_digest, "Training source mismatch")
    completion = read(run / "completion.json")
    require(completion["status"] == "FULL_BUDGET_COMPLETED", "Training did not complete its full budget")
    require(completion["step"] == completion["optimizer_updates"] == expected
            and config["training"]["max_steps"] == expected,
            "Training completion step/update count differs from expected_step")
    require(completion["source_verified"] is True and completion["presentation_budget_verified"] is True,
            "Training completion source/presentation budget not verified")
    final = completion["final_checkpoint"]
    require(final["step"] == expected and (run / final["file"]).resolve() == checkpoint.resolve(),
            "Expected checkpoint is not the completed final checkpoint")
    checkpoint_sha = sha(checkpoint)
    require(final["sha256"] == checkpoint_sha, "Final checkpoint completion hash mismatch")
    if "checkpoint_sha256" in record:
        require(checkpoint_sha == record["checkpoint_sha256"], "Checkpoint hash mismatch")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(state["step"] == expected, "Checkpoint optimizer step does not match expected_step")
    require(state["config"] == config, "Checkpoint config differs from saved run config")
    require(config["task"] == "hellaswag", "Expected HellaSwag")
    require(config["model"]["dtype"] == "bfloat16", "Expected BF16 backbone")
    require(config["channel"]["normalize_power"] is True, "Power normalization must remain enabled")
    require(config["codec"]["bottleneck_dim"] == 512, "Expected 512 latent channels")
    split = split_name(config)
    return config, state, {"run_id": record["run_id"], "checkpoint": str(checkpoint),
                           "checkpoint_sha256": checkpoint_sha, "expected_step": expected,
                           "source": metadata["source"], "training_source_sha256": current_source_digest,
                           "resolved_config_sha256": metadata["resolved_config_sha256"],
                           "split": split}


def validate_prepared(prepared):
    import torch

    receipt = read(prepared / "completion.json")
    require(receipt["status"] == "PASS_CPU_NO_MODEL", "Frozen CPU preparation did not pass")
    for field, name in (("encoded_requests_sha256", "encoded-requests.jsonl"),
                        ("batch_manifest_sha256", "batch-manifest.json")):
        require(receipt[field] == sha(prepared / name), f"Frozen {name} hash mismatch")
    require(receipt["encoded_requests_sha256"] ==
            "5422c0cb564ec405efcbfb1a2f265087286748418f803fa97fde15d9b346dd74",
            "Not the authorized FROZEN_PROMPT_GAP_V1 request fixture")
    require(receipt["batch_manifest_sha256"] ==
            "d485e728e29a9125a5558f15c37bf70a495e423e753323297a46effdad812b1b",
            "Not the authorized FROZEN_PROMPT_GAP_V1 batch fixture")
    encoded = rows(prepared / "encoded-requests.jsonl")
    ids = sorted({r["sample_id"] for r in encoded})
    require(len(ids) == 256, "Require exactly 256 frozen sample IDs")
    require([(r["shot"], r["sample_id"], r["candidate"]) for r in encoded]
            == [(s, i, c) for s in (0, 5) for i in ids for c in range(4)],
            "Frozen request order mismatch")
    manifest = read(prepared / "batch-manifest.json")
    require(len(manifest) == 32, "Require exactly 32 frozen batches")
    for n, entry in enumerate(manifest):
        batch = encoded[n * 64:(n + 1) * 64]
        require((entry["shot"], entry["index"]) == (batch[0]["shot"], n % 16), "Batch order mismatch")
        require(entry["sample_candidate_order"] == [[r["sample_id"], r["candidate"]] for r in batch],
                "Batch companions mismatch")
        for key, tensor in tensor_batch(batch, torch.device("cpu")).items():
            require(entry[key] == tensor.tolist(), f"Frozen tensor/mask mismatch: {key}")
    return encoded, receipt


def verify_tokenization(tokenizer, encoded):
    from lm_eval.models.huggingface import HFLM

    only = SimpleNamespace(tokenizer=tokenizer, backend="seq2seq", add_bos_token=None,
                           prefix_token_id=tokenizer.bos_token_id or tokenizer.eos_token_id)
    harness = cast(Any, HFLM)
    only.tok_encode = MethodType(harness.tok_encode, only)
    for row in encoded:
        context, target = harness._encode_pair(only, row["context"], row["continuation"])
        require(context[-2048:] == row["context_token_ids"] and target == row["continuation_token_ids"],
                "Current tokenizer differs from frozen request tokens")


def compact_row(group, scores, route, shot, receipt, frozen):
    row = group[0]
    denominators = [r["denominator"] for r in group]
    require(all(d > 0 for d in denominators), "Invalid normalization denominator")
    norm = [s / d for s, d in zip(scores, denominators)]
    result = {**receipt, "route": route, "num_fewshot": shot, "sample_id": row["sample_id"],
              "source_id": row["source_id"], "gold": row["gold"], "raw_scores": scores,
              "denominators": denominators, "normalized_scores": norm,
              "prompt_sha256": row["prompt_sha256"], "encoded_requests_sha256": frozen}
    for name, values in (("raw", scores), ("normalized", norm)):
        ranking = sorted(range(4), key=lambda i: (-values[i], i))
        result[f"{name}_ranking"] = ranking
        result[f"{name}_prediction"] = ranking[0]
        result[f"{name}_correct"] = ranking[0] == row["gold"]
        result[f"{name}_margin"] = values[row["gold"]] - max(values[i] for i in range(4) if i != row["gold"])
    return result


def run(runs_json, prepared, output):
    import torch
    from lm_eval.models.huggingface import HFLM
    from jscc.evaluation_policy import scoring_kwargs
    from jscc.harness_payload import payload_harness_class
    from jscc.models.split_model import build_model
    from jscc.presentation import training_source_digest
    from jscc.runtime import configure_training_determinism, isolated_rng, prepare_trainable_parameters

    output.mkdir(parents=True, exist_ok=False)
    budget = RequestBudget(output / "request-ledger.jsonl")
    completed = []
    started = time.monotonic()
    try:
        records = read(runs_json)
        require(isinstance(records, list) and len(records) == 2, "Provide exactly one raw/five_shot pair per GPU job")
        require(len({r["run_id"] for r in records}) == 2, "Run IDs must be distinct")
        encoded, frozen = validate_prepared(prepared)
        source_digest = training_source_digest()
        validated = [validate_run(r, source_digest) for r in records]
        require(len({r[2]["split"] for r in validated}) == 1, "Pair must use the same split")
        require(len({r[2]["expected_step"] for r in validated}) == 1, "Pair must use the same optimizer step")
        validated.sort(key=lambda item: item[0]["data"]["prompt_policy"]["mode"] == "raw")
        arms = [c["data"]["prompt_policy"]["mode"] for c, _, _ in validated]
        require(set(arms) == {"raw", "five_shot"}, "Pair must contain raw and five_shot arms")
        require(validated[0][0]["model"] == validated[1][0]["model"], "Pair backbone configurations differ")
        configure_training_determinism({"deterministic_algorithms": True})
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        adapter_class = payload_harness_class(HFLM)
        for index, (config, state, receipt) in enumerate(validated):
            route = arms[index]
            tokenizer, model = build_model(copy.deepcopy(config))
            prepare_trainable_parameters(model)
            model.load_communication_state(state)
            # Release CPU optimizer tensors before inference; no optimizer is constructed.
            state.clear()
            verify_tokenization(tokenizer, encoded)
            model.eval().requires_grad_(False)
            require(model.memory_codec is None and model.codec.film is None, "Unexpected memory codec/FiLM")
            require(next(model.base.parameters()).dtype == torch.bfloat16, "Backbone precision mismatch")
            require(all(p.dtype == torch.float32 for p in model.codec.parameters()), "Codec precision mismatch")
            require(getattr(model.base.config.decoder, "_attn_implementation", None) == "sdpa", "Expected SDPA")
            device = next(model.base.parameters()).device
            require(device.type == "cuda", "Evaluation requires its assigned GPU")
            adapter = adapter_class(communication_model=model, pretrained=model.base, tokenizer=tokenizer,
                                    backend="seq2seq", batch_size=64, max_length=2048,
                                    **scoring_kwargs({"scoring_policy": "fp32-v1"}, model))
            for active_route in ([route, "vanilla"] if index == 1 else [route]):
                active_receipt = dict(receipt)
                if active_route == "vanilla":
                    active_receipt.update(run_id="vanilla", checkpoint_sha256=None, checkpoint=None,
                                          expected_step=None, base_model=config["model"])
                for shot in (0, 5):
                    cell = f"{active_route}_{shot}shot"
                    selected = [r for r in encoded if r["shot"] == shot]
                    first = selected[:64]
                    with isolated_rng(0):
                        warm, _ = score_batch(adapter, model, first, budget, cell, "warmup", device)
                        scores, _ = score_batch(adapter, model, first, budget, cell, "core", device)
                        check_replay(warm, scores, first)
                        replay, _ = score_batch(adapter, model, first, budget, cell, "self_replay", device)
                        check_replay(scores, replay, first)
                        for start in range(64, 1024, 64):
                            values, _ = score_batch(adapter, model, selected[start:start + 64], budget, cell, "core", device)
                            scores.extend(values)
                    with (output / "compact_samples.jsonl").open("a") as stream:
                        for start in range(0, 1024, 4):
                            item = compact_row(selected[start:start + 4], scores[start:start + 4], active_route,
                                               shot, active_receipt, frozen["encoded_requests_sha256"])
                            stream.write(json.dumps(item, allow_nan=False) + "\n")
                    completed.append({"cell": cell, "samples": 256, "core_requests": 1024,
                                      "total_requests_including_checks": 1152, "self_replay": "PASS_ATOL_RTOL_1e-5"})
                    write(output / "completion.json", {"status": "RUNNING", "completed_cells": completed,
                                                        "requests": budget.used})
                    print(f"{cell} COMPLETE requests={budget.used}", flush=True)
            del adapter, model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
        require(budget.used == 6912 and len(completed) == 6, "Incomplete evaluation")
        write(output / "completion.json", {"status": "COMPLETED", "completed_cells": completed,
              "requests": budget.used, "optimizer_updates": 0, "elapsed_seconds": time.monotonic() - started,
              "runner_sha256": sha(__file__), "runs_json_sha256": sha(runs_json),
              "frozen_preparation_sha256": sha(prepared / "completion.json")})
    except BaseException as exc:
        write(output / "failure.json", {"type": type(exc).__name__, "message": str(exc),
              "requests_attempted": budget.used, "completed_cells": completed})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-json", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.runs_json, args.prepared, args.output)


if __name__ == "__main__":
    main()
