"""Compose one model design with task settings; save a complete resolved config."""
import copy
from pathlib import Path

import yaml


def resolve_codec_configs(codec):
    """Return the main and receiver-memory codec configurations.

    A flat codec configuration is the historical format: both communication
    streams use the same settings.  ``codec.memory`` may provide overrides for
    the receiver-memory stream while inheriting every unspecified setting from
    the main codec.  Neither returned mapping aliases the input mapping.
    """
    if not isinstance(codec, dict):
        raise TypeError("codec must be a mapping")

    main = copy.deepcopy(codec)
    memory = main.pop("memory", None)
    if memory is None:
        return main, copy.deepcopy(main)
    if not isinstance(memory, dict):
        raise TypeError("codec.memory must be a mapping of codec overrides")

    resolved_memory = copy.deepcopy(main)
    resolved_memory.update(copy.deepcopy(memory))
    # A nested override is an override mapping, not another configuration
    # level.  Removing this key also prevents accidental recursive expansion.
    resolved_memory.pop("memory", None)
    return main, resolved_memory


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
    schedule_steps = training.get("schedule_steps", training["max_steps"])
    if type(schedule_steps) is not int or schedule_steps < training["max_steps"]:
        raise ValueError("training.schedule_steps must be an integer >= training.max_steps")
    if training["monitor"] not in ("kl", "loss"):
        raise ValueError("training.monitor must be kl or loss")
    if not training["validation_snrs"]:
        raise ValueError("training.validation_snrs must contain at least one condition")
    # Keep the historical flat codec schema valid, while rejecting malformed
    # nested memory overrides before a model is constructed.
    resolve_codec_configs(config["codec"])


def save_config(config, path):
    Path(path).write_text(yaml.safe_dump(config, sort_keys=False))
