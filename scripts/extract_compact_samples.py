#!/usr/bin/env python3
"""Reduce lm-eval HellaSwag samples to auditable per-example scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def extract(source: Path, output: Path, *, run_id: str, checkpoint_sha256: str,
            checkpoint_step: int, condition: str, batch_size: int,
            num_fewshot: int = 5):
    output.parent.mkdir(parents=True, exist_ok=True)
    documents = output.with_name("documents.jsonl")
    seen_docs = set()
    with source.open() as src, output.open("w") as dst, documents.open("a") as docs:
        for line in src:
            row = json.loads(line)
            doc = row["doc"]
            choices = list(doc["choices"])
            raw_scores = [float(item[0]) for item in row["filtered_resps"]]
            denominators = [len(choice) for choice in choices]
            normalized = [score / denominator for score, denominator in zip(raw_scores, denominators)]
            gold = int(row["target"])
            pred_raw = max(range(len(raw_scores)), key=raw_scores.__getitem__)
            pred_norm = max(range(len(normalized)), key=normalized.__getitem__)
            source_id = doc.get("source_id", str(doc.get("ind", row["doc_id"])))
            compact = {
                "run_id": run_id,
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_step": checkpoint_step,
                "condition": condition,
                "batch_size": batch_size,
                "num_fewshot": num_fewshot,
                "sample_id": str(row["doc_id"]),
                "source_id": source_id,
                "doc_hash": row.get("doc_hash"),
                "prompt_hash": row.get("prompt_hash"),
                "gold_index": gold,
                "candidate_loglikelihoods": raw_scores,
                "normalization_denominators": denominators,
                "normalization_denominator_unit": "choice characters",
                "normalized_scores": normalized,
                "prediction_index": pred_raw,
                "prediction_norm_index": pred_norm,
                "prediction_acc": int(pred_raw == gold),
                "prediction_acc_norm": int(pred_norm == gold),
                "correct_acc": pred_raw == gold,
                "correct_acc_norm": pred_norm == gold,
                "noise_replay_id": None,
                "harness_definition": "lm_eval.api.task.MultipleChoiceTask.process_results",
            }
            dst.write(json.dumps(compact, allow_nan=False) + "\n")
            if row["doc_id"] not in seen_docs:
                docs.write(json.dumps({
                    "sample_id": str(row["doc_id"]),
                    "source_id": source_id,
                    "doc_hash": row.get("doc_hash"),
                    "prompt_hash": row.get("prompt_hash"),
                    "query": doc.get("query"),
                    "choices": choices,
                    "gold_index": gold,
                    "dataset_split": doc.get("split"),
                    "dataset_split_type": doc.get("split_type"),
                }, ensure_ascii=False, allow_nan=False) + "\n")
                seen_docs.add(row["doc_id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-step", required=True, type=int)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--num-fewshot", type=int, default=5)
    args = parser.parse_args()
    extract(args.source, args.output, run_id=args.run_id,
            checkpoint_sha256=args.checkpoint_sha256,
            checkpoint_step=args.checkpoint_step, condition=args.condition,
            batch_size=args.batch_size, num_fewshot=args.num_fewshot)


if __name__ == "__main__":
    main()
