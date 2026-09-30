"""Compose one model design with task settings; save a complete resolved config."""
import copy
import math
from pathlib import Path

import yaml


def validate_codec_architecture(codec):
    """Validate explicit designs without rewriting historical saved configs."""
    architecture = codec.get("architecture", "residual_mlp")
    if architecture not in ("residual_mlp", "direct_affine", "direct_outer_ln"):
        raise ValueError("codec.architecture must be residual_mlp, direct_affine, or direct_outer_ln")
    if architecture in ("direct_affine", "direct_outer_ln"):
        required = {"n_res_blocks": 0, "layernorm": ("both" if architecture == "direct_outer_ln" else "none"),
                    "snr_film": False}
        for key, value in required.items():
            if codec.get(key) != value:
                raise ValueError(f"{architecture} requires codec.{key}={value!r}")
        if codec.get("dropout", 0.0) != 0.0:
            raise ValueError(f"{architecture} requires codec.dropout=0")


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
    if set(runtime) - {"device", "dtype", "sdpa_backend_policy", "numerical_policy"}:
        raise ValueError("runtime overrides are limited to device, dtype, sdpa_backend_policy and numerical_policy")
    if runtime:
        config["model"].update(runtime)
    for section, key in (("run", "output_dir"),):
        value = config[section].get(key)
        if value:
            config[section][key] = str((path.parent / Path(value).expanduser()).resolve())
    validate_config(config)
    return config


def validate_sdpa_backend_policy(policy):
    if policy not in ("auto", "flash_math"):
        raise ValueError("model.sdpa_backend_policy must be auto or flash_math")


def validate_numerical_policy(model, split):
    policy = model.get("numerical_policy", "native")
    if policy not in ("native", "codec_receiver_fp32"):
        raise ValueError("model.numerical_policy must be native or codec_receiver_fp32")
    if policy == "codec_receiver_fp32":
        if model.get("dtype") != "bfloat16":
            raise ValueError("codec_receiver_fp32 requires model.dtype=bfloat16")
        if model.get("sdpa_backend_policy", "auto") != "auto":
            raise ValueError("codec_receiver_fp32 requires sdpa_backend_policy=auto")
        if split != {"stack": "enc", "where": "after_final_norm"}:
            raise ValueError("codec_receiver_fp32 requires encoder after_final_norm without index")


def validate_config(config):
    """Checks shared by file loading and study expansion."""
    if config["task"] not in ("coco", "hellaswag"):
        raise ValueError("task must be coco or hellaswag")
    validate_sdpa_backend_policy(config.get("model", {}).get("sdpa_backend_policy", "auto"))
    if config["task"] != "coco" and config.get("model", {}).get("sdpa_backend_policy", "auto") != "auto":
        raise ValueError("flash_math is currently supported only for task=coco")
    validate_numerical_policy(config.get("model", {}), config["split"])
    training = config["training"]
    if "loss_weights" in training:
        weights = training["loss_weights"]
        if not isinstance(weights, dict) or set(weights) != {"kl", "hidden", "memory"}:
            raise ValueError("training.loss_weights requires exactly kl, hidden, memory")
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in weights.values()):
            raise ValueError("loss_weights must be finite nonnegative numbers")
        # Explicit weights take precedence; legacy values are retained for provenance.

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
    for codec in resolve_codec_configs(config["codec"]):
        validate_codec_architecture(codec)


def save_config(config, path):
    Path(path).write_text(yaml.safe_dump(config, sort_keys=False))
