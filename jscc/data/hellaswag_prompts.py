"""Versioned prompt-alignment diagnostic; targets retain the training contract."""
import hashlib
import inspect
import json
import random


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def validate_policy(policy, target_max_length):
    if policy.get("version") != "prompt-alignment-v1":
        raise ValueError("unknown HellaSwag prompt policy version")
    if policy.get("mode") not in ("raw", "five_shot"):
        raise ValueError("prompt policy mode must be raw or five_shot")
    if policy.get("source_max_length") != 2048 or target_max_length != 512:
        raise ValueError("prompt-alignment-v1 requires source cap 2048 and target cap 512")
    if not isinstance(policy.get("seed"), int):
        raise ValueError("prompt policy requires an integer seed")


class PromptBuilder:
    def __init__(self, raw_train, ids, policy):
        from lm_eval.tasks.hellaswag.utils import preprocess
        if ids.get("validation_split") != "train":
            raise ValueError("prompt alignment requires the train selection holdout")
        train_ids = list(ids["train_rows"])
        if len(set(train_ids)) != len(train_ids) or not set(train_ids).isdisjoint(ids["validation_rows"]):
            raise ValueError("optimization IDs must be unique and exclude selection IDs")
        self.policy = policy
        self.train_ids = train_ids
        self.preprocess = preprocess
        self.helper_digest = digest(inspect.getsource(preprocess))
        self.documents = dict(zip(train_ids, raw_train.select(train_ids)))
        self.formatted = {row_id: self.format(doc) for row_id, doc in self.documents.items()}

    def format(self, doc):
        query = self.preprocess(doc["activity_label"] + ": " + doc["ctx_a"] + " " + doc["ctx_b"].capitalize())
        return query, [self.preprocess(ending) for ending in doc["endings"]]

    def build(self, doc, row_id):
        if self.policy["mode"] == "raw":
            return doc["ctx"], []
        # Private, per-document randomness does not consume training/channel RNG.
        rng = random.Random(digest([self.policy["version"], self.policy["seed"], row_id]))
        selected = []
        source_id = doc.get("source_id")
        # Sampling a permutation lazily avoids a full-pool scan for every query.
        remaining = len(self.train_ids)
        swaps = {}
        while remaining and len(selected) < 5:
            slot = rng.randrange(remaining)
            candidate = self.train_ids[swaps.get(slot, slot)]
            remaining -= 1
            swaps[slot] = swaps.get(remaining, remaining)
            if candidate == row_id:
                continue
            if source_id is not None and self.documents[candidate].get("source_id") == source_id:
                continue
            selected.append(candidate)
        if len(selected) != 5:
            raise ValueError("five-shot prompt needs five eligible optimization demonstrations")
        demos = [self.formatted[key][0] + " " + self.formatted[key][1][int(self.documents[key]["label"])]
                 for key in selected]
        return "\n\n".join([*demos, self.format(doc)[0]]), selected

    def tokenize(self, rows, row_ids, tokenizer):
        documents = [dict(zip(rows, values)) for values in zip(*rows.values())]
        prompts = [self.build(doc, row_id) for doc, row_id in zip(documents, row_ids)]
        # Slice actual token sequences on the left without changing tokenizer state
        # or its special-token defaults (including the native target BOS contract).
        inputs = tokenizer([text for text, _ in prompts], truncation=False)
        targets = [doc["endings"][int(doc["label"])] for doc in documents]
        labels = tokenizer(targets, truncation=True, max_length=512)
        cap = self.policy["source_max_length"]
        token_ids = [tokens[-cap:] for tokens in inputs["input_ids"]]
        return {"input_ids": token_ids,
                "attention_mask": [mask[-cap:] for mask in inputs["attention_mask"]],
                "label_ids": labels["input_ids"], "prompt_row_id": row_ids,
                "demo_ids": [demos for _, demos in prompts],
                "source_id": [doc.get("source_id") for doc in documents],
                "demo_source_ids": [[self.documents[key].get("source_id") for key in demos] for _, demos in prompts],
                "source_length": [len(tokens) for tokens in inputs["input_ids"]],
                "input_length": [len(tokens) for tokens in token_ids],
                "source_hash": [digest(doc) for doc in documents],
                "text_hash": [digest(text) for text, _ in prompts],
                "input_hash": [digest(tokens) for tokens in token_ids],
                "target_hash": [digest(tokens) for tokens in labels["input_ids"]]}


def evidence(dataset):
    columns = ("prompt_row_id", "demo_ids", "source_id", "demo_source_ids", "source_length", "input_length",
               "source_hash", "text_hash", "input_hash", "target_hash")
    rows = list(dataset.select_columns(columns))
    lengths = [row["source_length"] for row in rows]
    return {"count": len(rows), "truncated_count": sum(row["source_length"] > row["input_length"] for row in rows),
            "source_length_min": min(lengths), "source_length_max": max(lengths),
            "source_length_mean": sum(lengths) / len(lengths), "rows_digest": digest(rows), "rows": rows}
