"""Transferred concrete site routing responsibility."""
from __future__ import annotations

from copy import deepcopy
import math
import torch

# Transferred from original13_routing; serialized meanings remain unchanged.
def canonical_codec(model):
    if not model.universal_codec or model.memory_codec is not model.codec:
        raise ValueError("model does not have one original13 canonical parameter owner")
    if any(p.requires_grad for p in model.base.parameters()):
        raise ValueError("backbone must remain frozen")
    parameters = [p for p in model.parameters() if p.requires_grad]
    if {id(p) for p in parameters} != {id(p) for p in model.codec.parameters()}:
        raise ValueError("trainable parameters differ from the canonical codec")
    return model.codec


def export_codec(model):
    """Flat codec state_dict: encoder.0.{weight,bias}, decoder.0.{weight,bias}."""
    return {key: tensor.detach().clone() for key, tensor in canonical_codec(model).state_dict().items()}


def load_codec(model, weights):
    codec = canonical_codec(model)
    expected = codec.state_dict()
    if not isinstance(weights, dict) or set(weights) != set(expected):
        raise ValueError("wrong-kind or duplicate body; expected a flat canonical codec state")
    for key, value in weights.items():
        if (not torch.is_tensor(value) or value.shape != expected[key].shape
                or value.dtype != expected[key].dtype or not bool(torch.isfinite(value).all())):
            raise ValueError("invalid canonical codec tensor: " + key)
    codec.load_state_dict(weights, strict=True)
    canonical_codec(model)


_RUNTIME_FIELDS = ("snr_db", "bypass", "_encoder_valid_mask", "_decoder_valid_mask",
    "_active_encoder_valid_mask", "_encoder_mask_representation", "_decoder_mask_representation",
    "_active_encoder_mask_representation", "channel_uses", "channel_uses_valid")


def _require_idle(model):
    canonical_codec(model)
    if (model._routing_active or model._forward_active or model._transmission_active
            or model._payload_observer is not None
            or model.channel._replay is not None):
        raise ValueError("runtime-state export/restore requires an idle route and noise context")


def observation_state(model):
    """Detached idle state, including modes/masks/noise audits/payload counters."""
    _require_idle(model)
    state = deepcopy({"schema": "original13-idle-runtime-v1", "split": model.split,
        "fields": {key: getattr(model, key) for key in _RUNTIME_FIELDS},
        "modes": {name: module.training for name, module in model.named_modules()},
        "channel_stream": model.channel._stream, "draw_summaries": model.channel.draw_summaries,
        "memory_hook_count": len(model._memory_handles)})
    _validate_runtime_state(model, state)
    return state


def _validate_runtime_state(model, state):
    """Validate the complete idle-state boundary before mutating the model."""
    modules = dict(model.named_modules())
    keys = {"schema", "split", "fields", "modes", "channel_stream", "draw_summaries", "memory_hook_count"}
    if (not isinstance(state, dict) or set(state) != keys
            or not isinstance(state["schema"], str) or state["schema"] != "original13-idle-runtime-v1"
            or not isinstance(state["split"], dict) or set(state["split"]) != set(model.split)
            or any(type(state["split"][key]) is not type(value) or state["split"][key] != value
                   for key, value in model.split.items())
            or type(state["memory_hook_count"]) is not int
            or state["memory_hook_count"] != len(model._memory_handles)
            or not isinstance(state["fields"], dict) or set(state["fields"]) != set(_RUNTIME_FIELDS)
            or not isinstance(state["modes"], dict) or set(state["modes"]) != set(modules)
            or any(type(mode) is not bool for mode in state["modes"].values())
            or any(state["modes"][name] for name in modules if name == "base" or name.startswith("base."))
            or not isinstance(state["channel_stream"], str) or state["channel_stream"] not in ("hidden", "memory")):
        raise ValueError("unsupported original13 idle runtime state")
    fields = state["fields"]
    if type(fields["bypass"]) is not bool:
        raise ValueError("runtime bypass must be boolean")
    snr = fields["snr_db"]
    if snr is not None:
        scalar = type(snr) in (int, float) and math.isfinite(snr)
        tensor = (torch.is_tensor(snr) and snr.is_floating_point() and not snr.requires_grad
                  and (snr.ndim == 0 or (snr.ndim == 3 and snr.shape[0] > 0 and snr.shape[1:] == (1, 1)))
                  and bool(torch.isfinite(snr).all()))
        if not scalar and not tensor:
            raise ValueError("runtime SNR must be finite scalar or per-sequence tensor")
    for prefix in ("_encoder", "_decoder", "_active_encoder"):
        mask, representation = fields[prefix + "_valid_mask"], fields[prefix + "_mask_representation"]
        if (representation is not None and not isinstance(representation, str)) or representation not in (None, "binary", "additive"):
            raise ValueError("unsupported runtime mask representation")
        if mask is None:
            continue
        if (not torch.is_tensor(mask) or mask.requires_grad or mask.is_complex() or mask.ndim not in (2, 4)
                or any(dim <= 0 for dim in mask.shape) or representation is None):
            raise ValueError("unsupported runtime validity mask")
        if representation == "binary":
            if not bool(((mask == 0) | (mask == 1)).all()):
                raise ValueError("runtime binary mask contains invalid values")
        elif (not mask.is_floating_point() or mask.ndim != 4
              or not bool((torch.isfinite(mask) | torch.isneginf(mask)).all())):
            raise ValueError("unsupported runtime additive mask")
    for name in ("channel_uses", "channel_uses_valid"):
        counts = fields[name]
        if (not isinstance(counts, dict) or set(counts) != {"hidden", "memory"}
                or any(type(value) is not int or value < 0 for value in counts.values())):
            raise ValueError("runtime counters require complete nonnegative integer role maps")
    if any(fields["channel_uses_valid"][role] > fields["channel_uses"][role] for role in ("hidden", "memory")):
        raise ValueError("runtime valid coordinates exceed allocated coordinates")
    draws = state["draw_summaries"]
    if not isinstance(draws, list):
        raise ValueError("unsupported runtime noise audit")
    for draw in draws:
        if (not isinstance(draw, dict) or set(draw) != {"stream", "shape", "dtype", "epsilon_sha256"}
                or not isinstance(draw["stream"], str) or draw["stream"] not in ("hidden", "memory") or not isinstance(draw["shape"], list)
                or len(draw["shape"]) < 2 or any(type(n) is not int or n <= 0 for n in draw["shape"])
                or not isinstance(draw["dtype"], str) or draw["dtype"] not in ("torch.float32", "torch.bfloat16", "torch.float16", "torch.float64")
                or not isinstance(draw["epsilon_sha256"], str) or len(draw["epsilon_sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in draw["epsilon_sha256"])):
            raise ValueError("unsupported runtime noise audit draw")


def restore_runtime_state(model, state):
    _require_idle(model)
    _validate_runtime_state(model, state)
    for key, value in state["fields"].items():
        setattr(model, key, deepcopy(value))
    for name, module in model.named_modules():
        module.training = state["modes"][name]
    model.channel._stream = state["channel_stream"]
    model.channel.draw_summaries = deepcopy(state["draw_summaries"])
    model.activation = model.reconstruction = None
    model.memory_activation = model.memory_reconstruction = None

