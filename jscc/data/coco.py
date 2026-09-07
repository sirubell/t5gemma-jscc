"""COCO 4-shot captions, with disjoint train, demos, validation and report IDs."""
import random
from typing import Any, cast

from torch.utils.data import DataLoader, Dataset

from . import TaskData


def caption_prompt(captions):
    return "\n".join([part for caption in captions for part in ("<start_of_image>", caption)]
                     + ["<start_of_image>"])


class CaptionDataset(Dataset):
    def __init__(self, rows, processor, demo_images, demo_captions, config, training):
        self.rows, self.processor = rows, processor
        self.demo_images = demo_images
        self.prompt = caption_prompt(demo_captions)
        self.config, self.training = config, training

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        captions = [caption for caption in row["answer"] if caption]
        target = random.choice(captions) if self.training else captions[0]
        inputs = self.processor(
            images=self.demo_images + [row["image"].convert("RGB")], text=self.prompt,
            return_tensors="pt", padding="max_length", truncation=True,
            max_length=self.config["max_prompt_length"],
        )
        tokens = self.processor.tokenizer(
            target, return_tensors="pt", padding="max_length", truncation=True,
            max_length=self.config["max_target_length"],
        )
        labels = tokens["input_ids"][0].clone()
        labels[tokens["attention_mask"][0] == 0] = -100
        return {"input_ids": inputs["input_ids"][0],
                "attention_mask": inputs["attention_mask"][0],
                "pixel_values": inputs["pixel_values"], "labels": labels}


def partition_ids(raw, val_ids, test_ids, restval_ids, config):
    """Choose demos/train from restval and two disjoint subsets of Karpathy val."""
    seed, data = config["seed"], config["data"]
    positions = {int(name.rsplit("_", 1)[1].split(".")[0]): i
                 for i, name in enumerate(raw["file_name"])}
    def shuffled(ids):
        subset = raw.select([position for image_id, position in positions.items() if image_id in ids])
        return [int(name.rsplit("_", 1)[1].split(".")[0])
                for name in subset.shuffle(seed=seed)["file_name"]]
    demos = shuffled(restval_ids)[:data["num_demos"]]
    train_pool = set(positions) - val_ids - test_ids - set(demos)
    train = shuffled(train_pool)[:data["num_train"]]
    heldout = shuffled(val_ids)
    count = data["num_validation"]
    report_count = data["num_report"]
    if len(heldout) < count + report_count:
        raise ValueError("num_validation + num_report exceeds available Karpathy validation images")
    return {"train_ids": train, "demo_ids": demos, "selection_ids": heldout[:count],
            "report_ids": heldout[count:count + report_count]}


def validate_ids(ids, val_ids, test_ids):
    groups = [ids[key] for key in ("train_ids", "demo_ids", "selection_ids", "report_ids")]
    flat = [image_id for group in groups for image_id in group]
    if len(flat) != len(set(flat)):
        raise ValueError("COCO train/demo/validation/report IDs must be disjoint and unique")
    if (set(ids["train_ids"]) | set(ids["demo_ids"])) & (val_ids | test_ids):
        raise ValueError("COCO training/demos overlap Karpathy validation/test")
    if not (set(ids["selection_ids"]) | set(ids["report_ids"])) <= val_ids:
        raise ValueError("COCO validation/report IDs must belong to Karpathy validation")


def load_data(config, processor, saved_ids=None, *, for_training=True):
    from datasets import load_dataset
    data = config["data"]
    raw = load_dataset(data["name"], split="val", revision=data["revision"])
    def karpathy(split):
        return set(load_dataset(data["karpathy_name"], split=split,
                                revision=data["karpathy_revision"])["cocoid"])
    val_ids, test_ids = karpathy("validation"), karpathy("test")
    ids = saved_ids
    if ids is None:
        ids = partition_ids(raw, val_ids, test_ids, karpathy("restval"), config)
    validate_ids(ids, val_ids, test_ids)
    positions = {int(name.rsplit("_", 1)[1].split(".")[0]): i
                 for i, name in enumerate(raw["file_name"])}
    def rows(key):
        return raw.select([positions[image_id] for image_id in ids[key]])
    demos = rows("demo_ids")
    # The default HF row format is a dict; alternate batch formats are unused here.
    demo_rows = [cast(dict[str, Any], demos[index]) for index in range(len(demos))]
    demo_images = [row["image"].convert("RGB") for row in demo_rows]
    demo_captions = [row["answer"][0] for row in demo_rows]
    result = TaskData(ids=ids, report=rows("report_ids"), demo_images=demo_images,
                      demo_captions=demo_captions)
    if for_training:
        def loader(key, training):
            dataset = CaptionDataset(rows(key), processor, demo_images, demo_captions, data, training)
            return DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=training,
                              num_workers=data["num_workers"], pin_memory=config["model"]["device"] == "cuda")
        result.train = loader("train_ids", True)
        result.validation = loader("selection_ids", False)
    return result
