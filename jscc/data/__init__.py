"""Task-specific data loading; both tasks produce the same training batch keys."""
from dataclasses import dataclass
from typing import Any


@dataclass
class TaskData:
    train: Any = None
    validation: Any = None
    ids: Any = None
    report: Any = None
    demo_images: Any = None
    demo_captions: Any = None


def load_data(config, processor, saved_ids=None, *, for_training=True):
    if config["task"] == "coco":
        from .coco import load_data as load
    else:
        from .hellaswag import load_data as load
    return load(config, processor, saved_ids, for_training=for_training)
