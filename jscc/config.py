"""Compose one model design with task settings; save a complete resolved config."""
from pathlib import Path

import yaml


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    model_file = config.pop("model_config", None)
    if model_file is not None:
        model_path = (path.parent / Path(model_file).expanduser()).resolve()
        design = yaml.safe_load(model_path.read_text())
        sections = {"model", "split", "codec", "channel"}
        if set(design) != sections:
            raise ValueError("model_config must contain model, split, codec and channel sections only")
        if sections & config.keys():
            raise ValueError("Put model/split/codec/channel settings in model_config, not in the task YAML")
        config.update(design)
    runtime = config.pop("runtime", {})
    if set(runtime) - {"device", "dtype"}:
        raise ValueError("runtime overrides are limited to device and dtype")
    if runtime:
        config["model"].update(runtime)
    for section, key in (("run", "output_dir"),):
        value = config[section].get(key)
        if value:
            config[section][key] = str((path.parent / Path(value).expanduser()).resolve())
    validate_config(config)
    return config


def validate_config(config):
    """Checks shared by file loading and study expansion."""
    if config["task"] not in ("coco", "hellaswag"):
        raise ValueError("task must be coco or hellaswag")
    training = config["training"]
    for key in ("max_steps", "batch_size", "gradient_accumulation", "eval_every"):
        if training[key] < 1:
            raise ValueError(f"training.{key} must be positive")
    if training["monitor"] not in ("kl", "loss"):
        raise ValueError("training.monitor must be kl or loss")
    if not training["validation_snrs"]:
        raise ValueError("training.validation_snrs must contain at least one condition")


def save_config(config, path):
    Path(path).write_text(yaml.safe_dump(config, sort_keys=False))
