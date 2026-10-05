"""Replay existing complete request ordering; diagnose selected original batch companions."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
import torch.nn.functional as F

from jscc.models.split_model import build_model
from jscc.runtime import isolated_rng


def digest(value):
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--repeat", type=int, default=0)
    args = p.parse_args()
    from lm_eval.models.huggingface import HFLM

    rows = [json.loads(line) for line in args.samples.read_text().splitlines()]
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tokenizer, model = build_model(state["config"])
    model.load_communication_state(state)
    model.eval()
    adapter = cast(Any, HFLM)(
        pretrained=model.base,
        tokenizer=tokenizer,
        backend="seq2seq",
        batch_size=args.batch,
    )
    targets = set([343, 392, 464, *range(8)])
    requests, request_meta, lookup = [], [], {}
    for row in rows:
        for choice, pair in enumerate(row["arguments"]):
            req = SimpleNamespace(args=tuple(pair), task_name="hellaswag")
            requests.append(req)
            # This is the installed adapter's actual seq2seq encoding contract.
            context, continuation = adapter._encode_pair(*pair)
            item = {
                "doc_id": row["doc_id"],
                "choice": choice,
                "context": context[-adapter.max_length :],
                "continuation": continuation[-adapter.max_length :],
                "prompt_hash": digest(pair[0]),
            }
            request_meta.append(item)
            lookup.setdefault(tuple(item["context"]), []).append(item)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "request-manifest.json").write_text(json.dumps(request_meta))
    original = adapter._model_call
    for condition in ["no_noise", "vanilla"]:
        captures, batches, order = [], [], []

        def instrument(inps, attn_mask=None, labels=None):
            assert attn_mask is not None and labels is not None
            logits = original(inps, attn_mask=attn_mask, labels=labels)
            companions = []
            for i in range(len(inps)):
                context = tuple(inps[i][attn_mask[i].bool()].tolist())
                label = labels[i].tolist()
                matches = [
                    v
                    for v in lookup.get(context, [])
                    if label[: len(v["continuation"])] == v["continuation"]
                ]
                if len(matches) != 1:
                    raise RuntimeError(f"request mapping ambiguous: {len(matches)}")
                item = matches[0]
                companions.append(
                    {
                        "doc_id": item["doc_id"],
                        "choice": item["choice"],
                        "input_hash": digest(item["context"]),
                        "target_hash": digest(item["continuation"]),
                    }
                )
                order.append([item["doc_id"], item["choice"]])
                if item["doc_id"] in targets:
                    target = torch.tensor(item["continuation"], device=logits.device)
                    selected = logits[i, : len(target)]
                    old = (
                        F.log_softmax(selected, dim=-1, dtype=adapter.softmax_dtype)
                        .gather(1, target[:, None])
                        .squeeze(1)
                    )
                    fp32 = (
                        F.log_softmax(selected.float(), dim=-1)
                        .gather(1, target[:, None])
                        .squeeze(1)
                    )
                    captures.append(
                        {
                            **companions[-1],
                            "batch_index": len(batches),
                            "context_ids": item["context"],
                            "target_ids": item["continuation"],
                            "selected_logits": selected.gather(1, target[:, None])
                            .squeeze(1)
                            .float()
                            .tolist(),
                            "old_log_probs": old.float().tolist(),
                            "fp32_log_probs": fp32.tolist(),
                            "old_sum": float(old.sum()),
                            "fp32_sum": float(fp32.sum()),
                            "logits_dtype": str(logits.dtype),
                            "softmax_dtype": str(old.dtype),
                            "decoder_start_token_id": getattr(
                                model.base.config, "decoder_start_token_id", None
                            ),
                            "labels_padding_value": 0,
                            "cache_policy": "model default, fresh forward; no score cache",
                        }
                    )
            # Preserve actual companions and padded inputs for target-containing batches.
            record = {
                "shape": list(inps.shape),
                "label_shape": list(labels.shape),
                "companions": companions,
            }
            if any(x["doc_id"] in targets for x in companions):
                record.update(
                    input_ids=inps.tolist(),
                    attention_mask=attn_mask.tolist(),
                    labels=labels.tolist(),
                )
            batches.append(record)
            return logits

        adapter._model_call = instrument
        with isolated_rng(0), model.transmission(None, bypass=condition == "vanilla"):
            scores = adapter.loglikelihood(requests, disable_tqdm=True)
        compact = []
        for index, row in enumerate(rows):
            raw = [scores[index * 4 + j][0] for j in range(4)]
            denominators = [len(c) for c in row["doc"]["choices"]]
            norm = [a / b for a, b in zip(raw, denominators)]
            gold = int(row["doc"]["gold"])
            compact.append(
                {
                    "doc_id": row["doc_id"],
                    "source_id": row["doc"]["source_id"],
                    "gold": gold,
                    "raw_scores": raw,
                    "denominators": denominators,
                    "normalized_scores": norm,
                    "prediction": max(range(4), key=lambda x: raw[x]),
                    "prediction_norm": max(range(4), key=lambda x: norm[x]),
                    "gold_margin": norm[gold]
                    - max(norm[j] for j in range(4) if j != gold),
                    "top1_top2_margin": sorted(norm, reverse=True)[0]
                    - sorted(norm, reverse=True)[1],
                }
            )
        output = {
            "condition": condition,
            "batch": args.batch,
            "repeat": args.repeat,
            "request_count": len(requests),
            "document_count": len(rows),
            "forward_count": len(batches),
            "request_order_hash": digest(order),
            "softmax_dtype_option": str(adapter.softmax_dtype),
            "backend": adapter.backend,
            "max_length": adapter.max_length,
            "captures": captures,
            "batches": batches,
            "compact": compact,
            "panel_equivalents": len(rows) / 512,
        }
        (args.output / f"{condition}.json").write_text(
            json.dumps(output, allow_nan=False)
        )
        print(condition, len(captures), flush=True)


if __name__ == "__main__":
    main()
