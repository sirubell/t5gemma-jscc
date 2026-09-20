"""Bounded frozen-checkpoint six-cell inference, preserving exact request companions.

prepare: CPU tokenizer/formatter/batch contract only, no model construction.
run: one authorized GPU allocation, no optimizer/backward; fresh output required.
Numerical model and payload adapter are imported from an immutable source release.
"""

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
from types import MethodType, SimpleNamespace
from typing import Any, cast


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def bind_inputs(inputs, release):
    spec = read(inputs / "AUTHORIZED-SPEC.json")
    assert spec["execution_authorized"] and not spec["training_authorized"]
    assert spec["version"] == "FROZEN_PROMPT_GAP_V1"
    expected_source = read(inputs / "execution-source.json")
    assert expected_source == read(release / "outer-input/source.json")
    for filename, expected in expected_source["files"].items():
        assert sha(release / filename) == expected, filename
    ids = read(inputs / "fixtures/subset-ids.json")["actual_sample_ids"]
    assert len(ids) == len(set(ids)) == 256 and ids == sorted(ids)
    assert digest(ids) == spec["frozen_subset_sha256"]
    for name, expected in spec["archived_policy_binding"]["source_sha256"].items():
        assert sha(release / "jscc" / name) == expected, name
    for name, expected in spec["archived_policy_binding"]["packages"].items():
        assert importlib.metadata.version(name).split("+")[0] == expected, name
    for receipt in spec["frozen_checkpoints"].values():
        path = Path(receipt["path"])
        assert (
            path.stat().st_size == receipt["bytes"] and sha(path) == receipt["sha256"]
        )
    return spec


def prepare(inputs, release, output):
    """Use installed pinned HFLM tokenizer methods without building any model."""
    from transformers import AutoTokenizer
    from lm_eval.models.huggingface import HFLM
    from lm_eval.api.task import ConfigurableTask
    import torch

    spec = bind_inputs(inputs, release)
    tokenizer = AutoTokenizer.from_pretrained(
        "google/t5gemma-2-1b-1b",
        revision=spec["archived_policy_binding"]["model_revision"],
        local_files_only=True,
    )
    tokenizer_only: Any = SimpleNamespace(
        tokenizer=tokenizer,
        backend="seq2seq",
        add_bos_token=None,
        prefix_token_id=tokenizer.bos_token_id or tokenizer.eos_token_id,
    )
    harness: Any = HFLM
    tokenizer_only.tok_encode = MethodType(harness.tok_encode, tokenizer_only)
    task_only = SimpleNamespace(
        prompt=None,
        features=[],
        config=SimpleNamespace(doc_to_text="{{query}}"),
        _config=SimpleNamespace(doc_to_choice="choices"),
    )
    docs = {
        d["sample_id"]: d["document"] for d in rows(inputs / "fixtures/documents.jsonl")
    }
    source = rows(inputs / "fixtures/ordered-request-text.jsonl")
    ids = read(inputs / "fixtures/subset-ids.json")["actual_sample_ids"]
    expected_order = [(s, i, c) for s in (0, 5) for i in ids for c in range(4)]
    assert [
        (r["shot"], r["sample_id"], r["candidate"]) for r in source
    ] == expected_order
    encoded = []
    for item in source:
        doc = docs[item["sample_id"]]
        assert item["gold"] == doc["gold"] and item["denominator"] == len(
            doc["choices"][item["candidate"]]
        )
        if item["shot"] == 0:
            assert (
                ConfigurableTask.doc_to_text(cast(Any, task_only), doc)
                == item["context"]
            )
        assert item["continuation"] == " " + doc["choices"][item["candidate"]]
        context, target = harness._encode_pair(
            tokenizer_only, item["context"], item["continuation"]
        )
        assert context and 0 < len(target) <= 2048
        context = context[-2048:]
        if item["shot"] == 5:
            assert context == item["archived_context_token_ids"]
            assert target == item["archived_continuation_token_ids"]
        encoded.append(
            {**item, "context_token_ids": context, "continuation_token_ids": target}
        )
    output.mkdir(parents=True, exist_ok=False)
    path = output / "encoded-requests.jsonl"
    path.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in encoded))
    batches = []
    for shot in (0, 5):
        selected = [r for r in encoded if r["shot"] == shot]
        for start in range(0, len(selected), 64):
            batch = selected[start : start + 64]
            tensors = tensor_batch(batch, torch.device("cpu"))
            batches.append(
                {
                    "shot": shot,
                    "index": start // 64,
                    "sample_candidate_order": [
                        [r["sample_id"], r["candidate"]] for r in batch
                    ],
                    "input_ids": tensors["input_ids"].tolist(),
                    "attention_mask": tensors["attention_mask"].tolist(),
                    "labels": tensors["labels"].tolist(),
                    "decoder_payload_mask": tensors["decoder_payload_mask"].tolist(),
                }
            )
    write(output / "batch-manifest.json", batches)
    write(
        output / "completion.json",
        {
            "status": "PASS_CPU_NO_MODEL",
            "encoded_requests_sha256": sha(path),
            "batch_manifest_sha256": sha(output / "batch-manifest.json"),
            "spec_sha256": sha(inputs / "AUTHORIZED-SPEC.json"),
            "formatter": "pinned ConfigurableTask.doc_to_text",
            "tokenization": "pinned HFLM._encode_pair + HFLM.tok_encode; 5shot equality verified",
            "requests": 2048,
            "batches": 32,
            "model_loaded": False,
            "gpu_requests": 0,
            "input_files": {
                str(p.relative_to(inputs)): sha(p)
                for p in inputs.rglob("*")
                if p.is_file()
            },
            "runner_sha256": sha(__file__),
        },
    )
    print("PASS_CPU_NO_MODEL 2048 requests / 32 exact batches", flush=True)


def tensor_batch(batch, device):
    import torch

    if len(batch) != 64:
        raise ValueError("Require exact64 requests; never silently regroup companions")
    lc = max(len(r["context_token_ids"]) for r in batch)
    lt = max(len(r["continuation_token_ids"]) for r in batch)
    inp = torch.zeros((64, lc), dtype=torch.long, device=device)
    attn = torch.zeros_like(inp)
    labels = torch.zeros((64, lt), dtype=torch.long, device=device)
    valid = torch.zeros_like(labels, dtype=torch.bool)
    for i, r in enumerate(batch):
        c, t = r["context_token_ids"], r["continuation_token_ids"]
        inp[i, : len(c)] = torch.tensor(c, device=device)
        attn[i, : len(c)] = 1
        labels[i, : len(t)] = torch.tensor(t, device=device)
        valid[i, : len(t)] = True
    return {
        "input_ids": inp,
        "attention_mask": attn,
        "labels": labels,
        "decoder_payload_mask": valid,
    }


class RequestBudget:
    def __init__(self, path, maximum=8192):
        self.path, self.maximum, self.used = Path(path), maximum, 0

    def reserve(self, count, cell, phase):
        if count != 64 or self.used + count > self.maximum:
            raise RuntimeError("Candidate request budget exhausted or wrong batch size")
        self.used += count
        # Record before forward: failures still consume their attempted batch.
        with self.path.open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "cell": cell,
                        "phase": phase,
                        "count": count,
                        "cumulative": self.used,
                    }
                )
                + "\n"
            )


def score_batch(adapter, model, batch, budget, cell, phase, device):
    import torch
    from jscc.harness_payload import ContinuationLengths

    tensors = tensor_batch(batch, device)
    encoded = [
        (
            (r["context"], r["continuation"]),
            r["context_token_ids"],
            r["continuation_token_ids"],
        )
        for r in batch
    ]
    adapter._payload_lengths = ContinuationLengths(encoded, 2048)
    budget.reserve(len(batch), cell, phase)
    with (
        torch.inference_mode(),
        model.transmission(None, bypass=cell.startswith("vanilla_")),
    ):
        logits = adapter._model_call(
            tensors["input_ids"],
            attn_mask=tensors["attention_mask"],
            labels=tensors["labels"],
        )
        assert logits.dtype == torch.bfloat16
        # Same full-vocabulary FP32 normalization and valid-target sum as HFLM.
        logp = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
        values = []
        for i, r in enumerate(batch):
            n = len(r["continuation_token_ids"])
            value = (
                logp[i, :n]
                .gather(-1, tensors["labels"][i, :n, None])
                .sum(dtype=torch.float32)
            )
            assert torch.isfinite(value)
            values.append(float(value))
        payload = dict(model.valid_payload_counts)
    with (budget.path.parent / "batch-scores.jsonl").open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "cell": cell,
                    "phase": phase,
                    "sample_candidates": [
                        [r["sample_id"], r["candidate"]] for r in batch
                    ],
                    "scores": values,
                    "payload": payload,
                    "input_shape": list(tensors["input_ids"].shape),
                    "target_shape": list(tensors["labels"].shape),
                }
            )
            + "\n"
        )
    wanted = (
        0
        if cell.startswith("vanilla_")
        else 512 * sum(len(r["context_token_ids"]) for r in batch)
    )
    assert payload == {"hidden": wanted, "memory": 0}
    return values, payload


def check_replay(reference, actual, batch):
    import numpy as np

    np.testing.assert_allclose(actual, reference, atol=1e-5, rtol=1e-5)
    d = np.array([r["denominator"] for r in batch])
    np.testing.assert_allclose(
        np.array(actual) / d, np.array(reference) / d, atol=1e-5, rtol=1e-5
    )


def run(inputs, release, prepared, output):
    import torch
    import yaml
    from lm_eval.models.huggingface import HFLM
    from jscc.evaluation_policy import scoring_kwargs
    from jscc.harness_payload import payload_harness_class
    from jscc.models.split_model import build_model
    from jscc.runtime import (
        configure_training_determinism,
        isolated_rng,
        prepare_trainable_parameters,
    )

    started = time.monotonic()
    spec = bind_inputs(inputs, release)
    cpu = read(prepared / "completion.json")
    assert cpu["status"] == "PASS_CPU_NO_MODEL" and cpu["spec_sha256"] == sha(
        inputs / "AUTHORIZED-SPEC.json"
    )
    assert cpu["encoded_requests_sha256"] == sha(prepared / "encoded-requests.jsonl")
    assert cpu["batch_manifest_sha256"] == sha(prepared / "batch-manifest.json")
    assert cpu["runner_sha256"] == sha(__file__)
    for name, expected in cpu["input_files"].items():
        assert sha(inputs / name) == expected, name
    output.mkdir(parents=True, exist_ok=False)
    budget = RequestBudget(output / "request-ledger.jsonl")
    encoded = rows(prepared / "encoded-requests.jsonl")
    configs = {
        r: yaml.safe_load((inputs / f"{r}.yaml").read_text())
        for r in ("ref_enc_fn", "ref_enc_l9")
    }
    configure_training_determinism({"deterministic_algorithms": True})
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    adapter_class = payload_harness_class(HFLM)
    complete = []
    try:
        for route in ("vanilla", "ref_enc_l9", "ref_enc_fn"):
            name = "ref_enc_fn" if route == "vanilla" else route
            receipt = spec["frozen_checkpoints"][name]
            config = copy.deepcopy(configs[name])
            state = None
            if route != "vanilla":
                state = torch.load(
                    receipt["path"], map_location="cpu", weights_only=True
                )
                assert state["step"] == 5000 and state["config"] == config
            tokenizer, model = build_model(config)
            prepare_trainable_parameters(model)
            if route != "vanilla":
                assert state is not None
                model.load_communication_state(state)
                del state
            model.eval()
            assert model.memory_codec is None and model.codec.film is None
            assert next(model.base.parameters()).dtype == torch.bfloat16
            assert all(p.dtype == torch.float32 for p in model.codec.parameters())
            assert (
                getattr(model.base.config.decoder, "_attn_implementation", None)
                == "sdpa"
            )
            model.requires_grad_(False)
            device = next(model.base.parameters()).device
            adapter = adapter_class(
                communication_model=model,
                pretrained=model.base,
                tokenizer=tokenizer,
                backend="seq2seq",
                batch_size=64,
                max_length=2048,
                **scoring_kwargs({"scoring_policy": "fp32-v1"}, model),
            )
            for shot in (0, 5):
                cell = f"{route}_{shot}shot"
                selected = [r for r in encoded if r["shot"] == shot]
                first = selected[:64]
                cell_started = time.monotonic()
                with isolated_rng(0):
                    warm, _ = score_batch(
                        adapter, model, first, budget, cell, "warmup", device
                    )
                    first_scores, _ = score_batch(
                        adapter, model, first, budget, cell, "core", device
                    )
                    check_replay(warm, first_scores, first)
                    replay, _ = score_batch(
                        adapter, model, first, budget, cell, "self_replay", device
                    )
                    check_replay(first_scores, replay, first)
                    scores = list(first_scores)
                    for start in range(64, len(selected), 64):
                        if time.monotonic() - started > 3300:
                            raise TimeoutError(
                                "Stop before allocation cap; preserve partial evidence"
                            )
                        values, _ = score_batch(
                            adapter,
                            model,
                            selected[start : start + 64],
                            budget,
                            cell,
                            "core",
                            device,
                        )
                        scores.extend(values)
                cell_rows = []
                for start in range(0, 1024, 4):
                    group = selected[start : start + 4]
                    row = group[0]
                    cell_rows.append(
                        {
                            "cell": cell,
                            "route": route,
                            "num_fewshot": shot,
                            "sample_id": row["sample_id"],
                            "source_id": row["source_id"],
                            "gold": row["gold"],
                            "raw_scores": scores[start : start + 4],
                            "denominators": [r["denominator"] for r in group],
                            "prompt_sha256": row["prompt_sha256"],
                            "checkpoint_sha256": None
                            if route == "vanilla"
                            else receipt["sha256"],
                            "encoded_requests_sha256": cpu["encoded_requests_sha256"],
                            "execution_source": str(release),
                        }
                    )
                with (output / "compact_samples.jsonl").open("a") as stream:
                    stream.write(
                        "".join(
                            json.dumps(r, allow_nan=False) + "\n" for r in cell_rows
                        )
                    )
                complete.append(
                    {
                        "cell": cell,
                        "num_samples": 256,
                        "core_requests": 1024,
                        "total_requests_including_checks": 1152,
                        "elapsed_seconds": time.monotonic() - cell_started,
                        "self_replay": "PASS_ATOL_RTOL_1e-5",
                        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    }
                )
                write(
                    output / "completion.json",
                    {
                        "status": "RUNNING",
                        "completed_cells": complete,
                        "requests": budget.used,
                        "elapsed_seconds": time.monotonic() - started,
                    },
                )
                print(f"{cell} COMPLETE requests={budget.used}", flush=True)
            del adapter, model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
        assert budget.used == 6912 and len(complete) == 6
        write(
            output / "completion.json",
            {
                "status": "COMPLETED",
                "completed_cells": complete,
                "requests": budget.used,
                "optimizer_updates": 0,
                "elapsed_seconds": time.monotonic() - started,
                "input_spec_sha256": sha(inputs / "AUTHORIZED-SPEC.json"),
                "cpu_preflight_sha256": sha(prepared / "completion.json"),
                "runner_sha256": sha(__file__),
                "optional_feature_metrics": "NOT_MEASURED",
            },
        )
    except BaseException as exc:
        write(
            output / "failure.json",
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "requests_attempted": budget.used,
                "completed_cells": complete,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "run"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepared", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_release.resolve()))
    if args.phase == "prepare":
        prepare(args.input, args.source_release, args.output)
    else:
        if args.prepared is None:
            parser.error("run requires --prepared CPU fixture")
        run(args.input, args.source_release, args.prepared, args.output)


if __name__ == "__main__":
    main()
