"""CPU fake-clock tests for durable whole-campaign allocation limits."""

import json

import pytest

from jscc.experiment_schedule import AllocationLedger, AllocationStopped


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def start(ledger, name="setup", **overrides):
    arguments = dict(
        stage=name,
        devices=1,
        max_duration_seconds=20,
        mandatory_reserve_device_seconds=10,
        measurement_receipt="measured-preparation.json",
        identities={
            "source": "source-sha",
            "config": "config-sha",
            "input": "data-sha",
        },
    )
    arguments.update(overrides)
    return ledger.start(name, **arguments)


def test_all_allocation_wall_time_and_prior_attempts_count(tmp_path):
    clock = Clock()
    ledger = AllocationLedger(
        tmp_path / "ledger.json", "baseline", 100, prior_device_seconds=12, clock=clock
    )
    start(ledger, "setup")
    clock.now = 3
    ledger.stop("setup")
    start(ledger, "idle")
    clock.now = 8
    ledger.stop("idle")
    start(ledger, "failed-training", devices=2)
    clock.now = 12
    result = ledger.stop("failed-training", status="failed")
    assert result["charged_device_seconds"] == 12 + 3 + 5 + 2 * 4
    assert result["remaining_device_seconds"] == 72
    assert json.loads(ledger.path.read_text())["terminal_reason"] == "failed"
    with pytest.raises(AllocationStopped):
        start(ledger, "replacement")
    assert len(ledger.snapshot()["allocations"]) == 3


def test_parallel_devices_and_concurrent_reservations(tmp_path):
    clock = Clock()
    ledger = AllocationLedger(tmp_path / "ledger.json", "baseline", 100, clock=clock)
    start(ledger, "a", devices=2)
    clock.now = 4
    start(ledger, "b", devices=1)
    clock.now = 9
    ledger.stop("a")
    clock.now = 12
    ledger.stop("b")
    assert ledger.snapshot()["charged_device_seconds"] == 2 * 9 + 8
    start(ledger, "c", devices=2)
    with pytest.raises(AllocationStopped, match="reserve"):
        start(ledger, "d", devices=2)
    assert len(ledger.snapshot()["allocations"]) == 3


def test_measured_reserve_blocks_before_attempt(tmp_path):
    ledger = AllocationLedger(
        tmp_path / "ledger.json", "baseline", 50, prior_device_seconds=25, clock=Clock()
    )
    with pytest.raises(ValueError, match="measured"):
        start(ledger, measurement_receipt="")
    with pytest.raises(AllocationStopped, match="reserve"):
        start(ledger)
    assert ledger.snapshot()["allocations"] == []
    assert (
        json.loads(ledger.path.read_text())["terminal_reason"]
        == "insufficient_measured_reserve"
    )


def test_deadline_retains_overrun_charge_and_requires_release(tmp_path):
    clock = Clock()
    ledger = AllocationLedger(tmp_path / "ledger.json", "baseline", 100, clock=clock)
    start(ledger, devices=2)
    clock.now = 21
    with pytest.raises(AllocationStopped, match="deadline"):
        ledger.check()
    assert (
        json.loads(ledger.path.read_text())["allocations"][0]["charged_device_seconds"]
        == 42
    )
    clock.now = 23
    with pytest.raises(AllocationStopped, match="deadline"):
        ledger.stop("setup")
    assert ledger.snapshot()["charged_device_seconds"] == 46
    assert ledger.snapshot()["allocations"][0]["status"] == "interrupted"
    with pytest.raises(AllocationStopped):
        start(ledger, "retry")


def test_duplicate_or_existing_ledger_never_replays(tmp_path):
    clock = Clock()
    path = tmp_path / "ledger.json"
    ledger = AllocationLedger(path, "baseline", 100, clock=clock)
    start(ledger)
    clock.now = 1
    ledger.stop("setup")
    with pytest.raises(AllocationStopped, match="duplicate"):
        start(ledger)
    with pytest.raises(FileExistsError):
        AllocationLedger(path, "baseline", 100, clock=clock)
    assert len(json.loads(path.read_text())["allocations"]) == 1


def test_record_write_failure_is_terminal(tmp_path, monkeypatch):
    ledger = AllocationLedger(tmp_path / "ledger.json", "baseline", 100, clock=Clock())
    original = ledger._write

    def broken(output):
        raise OSError("disk full")

    monkeypatch.setattr(ledger, "_write", broken)
    with pytest.raises(OSError, match="disk full"):
        start(ledger)
    monkeypatch.setattr(ledger, "_write", original)
    with pytest.raises(AllocationStopped, match="record_write_failure"):
        start(ledger, "retry")
    assert ledger.snapshot()["allocations"] == []


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True])
def test_invalid_resource_measurements_rejected(tmp_path, value):
    ledger = AllocationLedger(tmp_path / "ledger.json", "baseline", 100, clock=Clock())
    with pytest.raises(ValueError):
        start(ledger, mandatory_reserve_device_seconds=value)


def test_unallocated_cpu_and_queue_time_do_not_consume_device_budget(tmp_path):
    clock = Clock()
    ledger = AllocationLedger(tmp_path / "ledger.json", "baseline", 100, clock=clock)
    clock.now = 1000
    start(ledger)
    clock.now = 1002
    ledger.stop("setup")
    clock.now = 2000
    assert ledger.check()["charged_device_seconds"] == 2


def test_concurrent_start_preserves_larger_existing_reserve(tmp_path):
    ledger = AllocationLedger(tmp_path / "ledger.json", "baseline", 100, clock=Clock())
    start(ledger, "a", mandatory_reserve_device_seconds=70)
    with pytest.raises(AllocationStopped, match="reserve"):
        start(ledger, "b", mandatory_reserve_device_seconds=0)
    assert len(ledger.snapshot()["allocations"]) == 1


def test_start_requires_exact_identity_references(tmp_path):
    ledger = AllocationLedger(tmp_path / "ledger.json", "baseline", 100, clock=Clock())
    with pytest.raises(ValueError, match="identity"):
        start(ledger, identities={"source": "source-sha"})
    assert ledger.snapshot()["allocations"] == []


@pytest.mark.parametrize("operation", ["start", "stop"])
def test_ambiguous_allocation_write_never_automatically_retries(
    tmp_path, monkeypatch, operation
):
    clock = Clock()
    ledger = AllocationLedger(tmp_path / "ledger.json", "baseline", 100, clock=clock)
    if operation == "stop":
        start(ledger)
        clock.now = 5
    original = ledger._write
    calls = 0

    def fail_second_write(output):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("ambiguous record write")
        original(output)

    monkeypatch.setattr(ledger, "_write", fail_second_write)
    with pytest.raises(OSError, match="ambiguous"):
        if operation == "start":
            start(ledger)
        else:
            ledger.stop("setup")
    monkeypatch.setattr(ledger, "_write", original)
    with pytest.raises(AllocationStopped, match="record_write_failure"):
        start(ledger, "replacement")
    assert len(ledger.snapshot()["allocations"]) == 1
    if operation == "stop":
        assert ledger.snapshot()["charged_device_seconds"] == 5
