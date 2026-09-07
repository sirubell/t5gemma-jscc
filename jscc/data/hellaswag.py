"""HellaSwag: distill context -> correct ending; dynamically pad each batch."""
from typing import cast

import torch
from torch.utils.data import DataLoader, Dataset

from . import TaskData


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, rows):
        input_length = max(len(row["input_ids"]) for row in rows)
        label_length = max(len(row["label_ids"]) for row in rows)
        return {
            "input_ids": torch.tensor([row["input_ids"] + [self.pad_id] * (input_length - len(row["input_ids"])) for row in rows]),
            "attention_mask": torch.tensor([row["attention_mask"] + [0] * (input_length - len(row["attention_mask"])) for row in rows]),
            "labels": torch.tensor([row["label_ids"] + [-100] * (label_length - len(row["label_ids"])) for row in rows]),
        }


def load_data(config, tokenizer, saved_ids=None, *, for_training=True):
    if not for_training:
        return TaskData(ids=saved_ids)
    from datasets import load_dataset
    data = config["data"]
    raw = load_dataset(data["name"], revision=data["revision"])
    ids = saved_ids or {
        "train_rows": list(range(min(data["num_train"] or len(raw["train"]), len(raw["train"])))),
        "validation_rows": list(range(min(data["num_validation"], len(raw["validation"])))),
    }
    def tokenize(rows):
        targets = [endings[int(label)] for endings, label in zip(rows["endings"], rows["label"])]
        inputs = tokenizer(rows["ctx"], truncation=True, max_length=data["max_length"])
        labels = tokenizer(targets, truncation=True, max_length=data["max_length"])
        return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"],
                "label_ids": labels["input_ids"]}
    def loader(split, indices, training):
        rows = raw[split].select(indices)
        dataset = rows.map(tokenize, batched=True, remove_columns=rows.column_names)
        # HF Dataset implements the map-style protocol but does not inherit torch Dataset.
        return DataLoader(cast(Dataset, dataset), batch_size=config["training"]["batch_size"], shuffle=training,
                          num_workers=data["num_workers"], collate_fn=Collator(tokenizer.pad_token_id),
                          pin_memory=config["model"]["device"] == "cuda")
    return TaskData(train=loader("train", ids["train_rows"], True),
                    validation=loader("validation", ids["validation_rows"], False), ids=ids)
