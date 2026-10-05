"""Tests for provenance and fixed-horizon diagnostic plumbing."""
import math

import pytest
import torch

from jscc.runtime import source_state
from jscc.training import make_scheduler


def test_scheduler_can_stop_early_without_compressing_horizon():
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=2e-4)
    scheduler = make_scheduler(optimizer, {
        "max_steps": 4,
        "schedule_steps": 20,
        "warmup_ratio": 0.05,
        "min_lr_ratio": 0.0,
    })
    for _ in range(4):
        optimizer.step()
        scheduler.step()
    expected = 0.5 * (1.0 + math.cos(math.pi * (4 - 1) / (20 - 1)))
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2e-4 * expected)
    assert optimizer.param_groups[0]["lr"] > 0


def test_source_state_is_explicit_when_recorded():
    state = source_state()
    assert set(state) == {"revision", "dirty"}
    assert state["revision"] is None or isinstance(state["revision"], str)
    assert state["dirty"] is None or isinstance(state["dirty"], bool)

