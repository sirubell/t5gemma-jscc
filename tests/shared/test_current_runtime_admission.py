"""Current execution guard checks; CPU mocks never allocate or query a GPU."""
from types import SimpleNamespace

import pytest
from jscc import shared_runs


@pytest.fixture
def assigned_guard(monkeypatch):
    environment = {"CUDA_VISIBLE_DEVICES": "GPU-assigned"}
    clock = [10.0]
    monkeypatch.setattr(shared_runs, "os", SimpleNamespace(environ=environment))
    monkeypatch.setattr(shared_runs, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(shared_runs.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(shared_runs.torch.cuda, "is_bf16_supported", lambda: True)
    return environment, clock


def test_current_guard_rejects_uuid_mismatch_before_execution(assigned_guard):
    with pytest.raises(ValueError, match="assigned GPU differs"):
        shared_runs._guard({"model": {"device": "cuda"}}, synthetic=False,
                           timeout_seconds=20, device_uuid="GPU-other")


def test_current_guard_rechecks_assignment_and_deadline(assigned_guard):
    environment, clock = assigned_guard
    guard = shared_runs._guard({"model": {"device": "cuda"}}, synthetic=False,
                               timeout_seconds=20, device_uuid="GPU-assigned")
    guard()
    environment["CUDA_VISIBLE_DEVICES"] = "GPU-changed"
    with pytest.raises(ValueError, match="assigned GPU changed"):
        guard()
    environment["CUDA_VISIBLE_DEVICES"] = "GPU-assigned"
    clock[0] = 30.0
    with pytest.raises(TimeoutError, match="execution timeout"):
        guard()


@pytest.mark.parametrize("count,bf16", [(0, True), (2, True), (1, False)])
def test_current_guard_requires_one_bf16_device(assigned_guard, monkeypatch, count, bf16):
    monkeypatch.setattr(shared_runs.torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(shared_runs.torch.cuda, "is_bf16_supported", lambda: bf16)
    with pytest.raises(ValueError, match="one assigned BF16-capable GPU"):
        shared_runs._guard({"model": {"device": "cuda"}}, synthetic=False,
                           timeout_seconds=20, device_uuid="GPU-assigned")
