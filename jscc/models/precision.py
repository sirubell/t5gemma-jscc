"""Explicit frozen FP32 receiver arithmetic over unchanged BF16 weight storage."""
from typing import Any, cast

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel


class FrozenFP32(nn.Module):
    """Forward adapter retaining the original module's parameter hierarchy.

    The adapter owns no new parameters or persistent buffers. Detached compute
    copies are rebuilt after device moves, weight replacement or in-place loads.
    Functional substitution is scoped to the receiver, so encoder ties retain
    their original BF16 objects. Like functional_call itself this is not a
    concurrent-forward interface on a shared module.
    """

    _original: nn.Module

    def __init__(self, original):
        super().__init__()
        object.__setattr__(self, "_original", original)
        self._modules = original._modules
        self._parameters = original._parameters
        self._buffers = original._buffers
        self._non_persistent_buffers_set = original._non_persistent_buffers_set
        self._signature = None
        self._copies = ({}, {})
        self.cache_bytes = 0
        self.train(original.training)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._original, name)

    def train(self, mode=True):
        super().train(mode)
        self._original.training = mode
        return self

    def _compute_copies(self):
        parameters = dict(self._original.named_parameters(remove_duplicate=False))
        buffers = dict(self._original.named_buffers(remove_duplicate=False))
        values = [*parameters.values(), *buffers.values()]
        if any(p.requires_grad for p in parameters.values()):
            raise ValueError("FP32 receiver storage must remain frozen")
        if any(p.is_floating_point() and p.dtype != torch.bfloat16 for p in parameters.values()):
            raise ValueError("FP32 receiver requires BF16 stored backbone parameters")
        signature = tuple((id(t), t._version, t.device, t.dtype) for t in values)
        if signature != self._signature:
            copies = {}
            # Evaluation may first warm the cache under inference_mode; the
            # same copies must subsequently support gradients with respect to
            # receiver inputs during training.
            with torch.inference_mode(False):
                for t in values:
                    if id(t) not in copies:
                        copies[id(t)] = t.detach().float() if t.is_floating_point() else t.detach()
            self._copies = tuple({key: copies[id(t)] for key, t in group.items()}
                                 for group in (parameters, buffers))
            self.cache_bytes = sum(t.numel() * t.element_size() for t in copies.values())
            self._signature = signature
        return self._copies

    def forward(self, *args, **kwargs):
        # Cast tensor inputs only. In particular, KV Cache objects retain their
        # identity and are populated by the FP32 receiver on the first step.
        def widen(value):
            return value.float() if torch.is_tensor(value) and value.is_floating_point() else value

        copies = self._compute_copies()
        device = next(self._original.parameters()).device.type
        with torch.autocast(device_type=device, enabled=False), sdpa_kernel(SDPBackend.MATH):
            return torch.func.functional_call(
                self._original, copies, tuple(widen(v) for v in args),
                {key: widen(v) for key, v in kwargs.items()}, tie_weights=False, strict=True,
            )


def install_fp32_receiver(base):
    """Install on the supported T5Gemma2 receiver without changing storage keys."""
    from transformers import T5Gemma2ForConditionalGeneration

    if not isinstance(base, T5Gemma2ForConditionalGeneration):
        raise ValueError("codec_receiver_fp32 currently requires T5Gemma2")
    if any(p.dtype != torch.bfloat16 for p in base.parameters() if p.is_floating_point()):
        raise ValueError("codec_receiver_fp32 requires actual BF16 stored backbone parameters")
    cast(Any, base.model).decoder = FrozenFP32(base.model.decoder)
    cast(Any, base).lm_head = FrozenFP32(base.lm_head)


def compute_provenance(model, *, include_cache=False):
    policy = model.numerical_policy
    result = {"policy": policy}
    if policy == "codec_receiver_fp32":
        result.update({
            "encoder_compute": "bfloat16", "codec_compute": "float32",
            "power_statistics": "float32", "receiver_compute": "float32",
            "head_compute": "float32", "receiver_attention": "math",
            "k_memory": "float32", "r_target": "bfloat16", "r_prediction": "bfloat16",
            "compute_cache": "detached_nonpersistent_lazy",
        })
    if include_cache and policy == "codec_receiver_fp32":
        result["compute_cache_bytes"] = model.base.get_decoder().cache_bytes + model.base.lm_head.cache_bytes
    return result
