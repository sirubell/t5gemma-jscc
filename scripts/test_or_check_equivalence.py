#!/usr/bin/env python3
"""Report deterministic reference/streamed objective differences on CPU toys."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jscc import training
from jscc.models.channel import IdentityChannel
from tests.test_core import toy_model


def _fixture(stack):
    first = {"input_ids": torch.tensor([[1, 2, 3]]),
             "attention_mask": torch.tensor([[1, 1, 1]]),
             "labels": torch.tensor([[6, 7, -100]])}
    second = {"input_ids": torch.tensor([[4, 5, 0]]),
              "attention_mask": torch.tensor([[1, 1, 0]]),
              "labels": torch.tensor([[9, 10, 11]])}
    settings = {"temperature": 1.0, "kl_weight": 1.0, "mse_weight": 0.1}
    torch.manual_seed(100 + (stack == "dec"))
    reference = toy_model(stack, channel=IdentityChannel()).train()
    streamed = toy_model(stack, channel=IdentityChannel()).train()
    streamed.load_state_dict(reference.state_dict())
    reference_values = [training.batch_losses(reference, item, settings, None, return_stats=True)
                        for item in (first, second)]
    reference_loss = training.aggregate_batch_losses(reference_values, settings)["loss"]
    reference_loss.backward()
    denominators = training.effective_batch_denominators(streamed, [first, second], torch.device("cpu"))
    streamed_values = []
    for item in (first, second):
        value = training.batch_losses(streamed, item, settings, None, return_stats=True)
        training.scaled_batch_loss(value, settings, denominators).backward()
        streamed_values.append(training.detached_batch_values(value))
    streamed_loss = training.aggregate_batch_losses(streamed_values, settings)["loss"]
    reference_memory = list(reference.memory_codec.parameters()) if reference.memory_codec is not None else []
    streamed_memory = list(streamed.memory_codec.parameters()) if streamed.memory_codec is not None else []
    reference_parameters = list(reference.codec.parameters()) + reference_memory
    streamed_parameters = list(streamed.codec.parameters()) + streamed_memory
    ref_grads = [p.grad.detach() for p in reference_parameters]
    new_grads = [p.grad.detach() for p in streamed_parameters]
    differences = [new.float() - old.float() for new, old in zip(new_grads, ref_grads)]
    numerator = torch.sqrt(sum(diff.square().sum() for diff in differences))
    denominator = torch.sqrt(sum(old.float().square().sum() for old in ref_grads)).clamp_min(1e-12)
    dot = sum(new.float().flatten().dot(old.float().flatten()) for new, old in zip(new_grads, ref_grads))
    new_norm = torch.sqrt(sum(new.float().square().sum() for new in new_grads)).clamp_min(1e-12)
    old_norm = torch.sqrt(sum(old.float().square().sum() for old in ref_grads)).clamp_min(1e-12)
    return {
        "stack": stack,
        "loss_abs": float((streamed_loss - reference_loss.detach()).abs()),
        "gradient_max_abs": max(float(diff.abs().max()) for diff in differences),
        "gradient_relative_l2": float(numerator / denominator),
        "gradient_cosine": float(dot / (new_norm * old_norm)),
        "main_gradient_parameters": len(list(reference.codec.parameters())),
        "memory_gradient_parameters": len(list(reference.memory_codec.parameters())) if reference.memory_codec is not None else 0,
    }


def main():
    results = [_fixture(stack) for stack in ("enc", "dec")]
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
