"""Task data: HellaSwag five-shot distillation and COCO four-shot captions.

Both tasks expose the same interface to the trainer:
    train_batch(k)        k-th training batch (deterministic in seed and k)
    selection_batches()   held-out selection batches for validation
    model_inputs(batch, model) -> (kwargs, labels)
    autocast(model)
Fixed data splits live in configs/data/*.json.
"""
import hashlib
import json
import random
from pathlib import Path

import torch


def model_inputs(batch, model):
    """Native decoder inputs: BOS-prefixed shift of labels, -100 -> PAD."""
    param = next(model.base.parameters())
    labels = batch["labels"].to(param.device)
    kwargs = {"input_ids": batch["input_ids"].to(param.device),
              "attention_mask": batch["attention_mask"].to(param.device),
              "decoder_input_ids": model.base.prepare_decoder_input_ids_from_labels(labels=labels),
              "use_cache": False}
    if "pixel_values" in batch:
        pixels = batch["pixel_values"]
        kwargs["pixel_values"] = (pixels.flatten(0, 1) if pixels.ndim == 5 else pixels).to(param.device, param.dtype)
    return kwargs, labels


def autocast(model):
    param = next(model.base.parameters())
    return torch.autocast(device_type=param.device.type, dtype=param.dtype,
                          enabled=param.dtype in (torch.bfloat16, torch.float16))


class EpochOrder:
    """Concatenated random permutations of ``n`` items (one per epoch), seekable by position."""

    def __init__(self, n, seed):
        self.n, self.generator, self.epochs = n, torch.Generator().manual_seed(seed), []

    def __getitem__(self, position):
        epoch, offset = divmod(position, self.n)
        while len(self.epochs) <= epoch:
            self.epochs.append(torch.randperm(self.n, generator=self.generator).tolist())
        return self.epochs[epoch][offset]


class TaskData:
    def __init__(self, config):
        self.config = config
        self.data = config["data"]
        self.batch_size = config["training"]["batch_size"]
        self.device = torch.device(config["model"]["device"])
        self.ids = json.loads(Path(self.data["ids"]).read_text())
        self._selection = None

    model_inputs = staticmethod(model_inputs)

    def selection_rows(self) -> list:
        raise NotImplementedError

    def collate(self, rows) -> dict:
        raise NotImplementedError

    def train_batch(self, k) -> dict:
        raise NotImplementedError
    autocast = staticmethod(autocast)

    def selection_batches(self):
        if self._selection is None:
            rows = self.selection_rows()[: self.config["training"]["validation"].get("items", 512)]
            self._selection = [self.collate(rows[i:i + self.batch_size])
                               for i in range(0, len(rows), self.batch_size)]
        return self._selection


# ---- HellaSwag --------------------------------------------------------
def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


class HellaSwagData(TaskData):
    """Training target: the raw correct ending (right-truncated to 512 tokens).
    Source: five demonstrations from the training pool + the query, left-truncated to 2048 tokens.
    """

    def __init__(self, config, tokenizer):
        super().__init__(config)
        from datasets import load_dataset
        from lm_eval.tasks.hellaswag.utils import preprocess
        self.tokenizer, self.preprocess = tokenizer, preprocess
        self.raw = load_dataset(self.data["name"], revision=self.data["revision"], split="train")
        self.pool = self.ids["train_rows"]
        self.documents = {i: self.raw[i] for i in self.pool + self.ids["validation_rows"]}
        self.formatted = {i: self.format(self.documents[i]) for i in self.pool}
        self.order = EpochOrder(len(self.pool), self.data["order_seed"])

    def format(self, doc):
        query = self.preprocess(doc["activity_label"] + ": " + doc["ctx_a"] + " " + doc["ctx_b"].capitalize())
        return query, [self.preprocess(e) for e in doc["endings"]]

    def prompt(self, row):
        doc = self.documents[row]
        rng = random.Random(_digest(["prompt-alignment-v1", self.data["prompt_seed"], row]))
        selected, remaining, swaps = [], len(self.pool), {}
        while remaining and len(selected) < 5:  # lazy Fisher-Yates over the pool
            slot = rng.randrange(remaining)
            candidate = self.pool[swaps.get(slot, slot)]
            remaining -= 1
            swaps[slot] = swaps.get(remaining, remaining)
            if candidate == row or self.documents[candidate].get("source_id") == doc.get("source_id"):
                continue
            selected.append(candidate)
        demos = [self.formatted[c][0] + " " + self.formatted[c][1][int(self.documents[c]["label"])] for c in selected]
        return "\n\n".join([*demos, self.format(doc)[0]])

    def collate(self, rows):
        docs = [self.documents[r] for r in rows]
        inputs = self.tokenizer([self.prompt(r) for r in rows], truncation=False)
        labels = self.tokenizer([d["endings"][int(d["label"])] for d in docs], truncation=True, max_length=512)
        src = [ids[-2048:] for ids in inputs["input_ids"]]
        pad = self.tokenizer.pad_token_id
        n, m = max(map(len, src)), max(map(len, labels["input_ids"]))
        return {"input_ids": torch.tensor([s + [pad] * (n - len(s)) for s in src]),
                "attention_mask": torch.tensor([[1] * len(s) + [0] * (n - len(s)) for s in src]),
                "labels": torch.tensor([t + [-100] * (m - len(t)) for t in labels["input_ids"]])}

    def train_batch(self, k):
        start = k * self.batch_size
        return self.collate([self.pool[self.order[start + j]] for j in range(self.batch_size)])

    def selection_rows(self):
        return self.ids["validation_rows"]


# ---- COCO -------------------------------------------------------------
def caption_prompt(captions):
    return "\n".join([p for c in captions for p in ("<start_of_image>", c)] + ["<start_of_image>"])


def image_id(row):
    return int(row["file_name"].rsplit("_", 1)[1].split(".")[0])


class CocoData(TaskData):
    """Prompt: four demo images with their first caption, then the query image.
    Training target: one reference caption chosen by a hash of (seed, presentation, image).
    """

    def __init__(self, config, processor):
        super().__init__(config)
        from datasets import Dataset
        self.processor = processor
        self.raw = Dataset.from_file(self.data["arrow"])
        self.position = {image_id(r): i for i, r in enumerate(self.raw.select_columns(["file_name"]))}
        demos = [self.raw[self.position[i]] for i in self.ids["demo_ids"]]
        self.demo_images = [d["image"].convert("RGB") for d in demos]
        self.demo_captions = [d["answer"][0] for d in demos]
        self.prompt = caption_prompt(self.demo_captions)
        self.order = EpochOrder(len(self.ids["train_ids"]), self.data["order_seed"])

    def row(self, image):
        return self.raw[self.position[image]]

    def item(self, image, caption_index):
        row = self.row(image)
        captions = [c for c in row["answer"] if c]
        inputs = self.processor(images=self.demo_images + [row["image"].convert("RGB")], text=self.prompt,
                                return_tensors="pt", padding="max_length", truncation=True,
                                max_length=self.data["max_prompt_length"])
        target = self.processor.tokenizer(captions[caption_index % len(captions)], return_tensors="pt",
                                          padding="max_length", truncation=True,
                                          max_length=self.data["max_target_length"])
        labels = target["input_ids"][0].clone()
        labels[target["attention_mask"][0] == 0] = -100
        return {"input_ids": inputs["input_ids"][0], "attention_mask": inputs["attention_mask"][0],
                "pixel_values": inputs["pixel_values"], "labels": labels}

    def collate(self, rows):
        items = rows
        if items and not isinstance(items[0], dict):  # selection rows: image ids, first caption
            items = [self.item(i, 0) for i in items]
        return {"input_ids": torch.stack([i["input_ids"] for i in items]),
                "attention_mask": torch.stack([i["attention_mask"] for i in items]),
                "pixel_values": torch.cat([i["pixel_values"] for i in items]),
                "labels": torch.stack([i["labels"] for i in items])}

    def train_batch(self, k):
        items = []
        for j in range(self.batch_size):
            presentation = k * self.batch_size + j
            image = self.ids["train_ids"][self.order[presentation]]
            digest = hashlib.sha256(f"{self.data['caption_seed']}:{presentation}:{image}".encode()).digest()
            items.append(self.item(image, int.from_bytes(digest, "big")))
        return self.collate(items)

    def selection_rows(self):
        return self.ids["selection_ids"]


def load_task_data(config, processor):
    return (CocoData if config["task"] == "coco" else HellaSwagData)(config, processor)
