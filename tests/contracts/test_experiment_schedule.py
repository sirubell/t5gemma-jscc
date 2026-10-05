import pytest
import torch

from jscc.experiment_schedule import (
    CELLS,
    ExposureLedger,
    NoiseKey,
    baseline_lr_factor,
    compile_baseline_plan,
    paired_noise,
)
from jscc.experiment_state import isolated_rng


def test_finite_initial_and_owner_gated_conditional_plan():
    first = compile_baseline_plan()
    assert (first.physical_updates, first.task_assessments, first.condition_panels) == (
        1600,
        12,
        72,
    )
    with pytest.raises(ValueError, match="owner"):
        compile_baseline_plan(CELLS[:2])
    complete = compile_baseline_plan(CELLS[:2], "owner-decision-123")
    assert (
        complete.physical_updates,
        complete.task_assessments,
        complete.condition_panels,
        complete.development_item_assessments,
    ) == (2800, 22, 132, 33792)
    assert all(s.stop <= 400 for s in complete.segments)
    prefix = complete.segments[4]
    assert prefix.checkpoints == (50, 100, 150, 200)
    assert complete.segments[5].parent == complete.segments[6].parent
    with pytest.raises(ValueError):
        compile_baseline_plan(CELLS[:3], "approved")


def test_exact_lr_indexing_and_no_extension():
    assert baseline_lr_factor(0) == 0
    assert baseline_lr_factor(20) == 1
    assert baseline_lr_factor(400) == 0
    assert baseline_lr_factor(200) > baseline_lr_factor(201)
    with pytest.raises(ValueError):
        baseline_lr_factor(401)


def test_paired_stream_does_not_consume_global_rng():
    before = torch.get_rng_state()
    key = NoiseKey("study", "training", "enc_fn", 200, "sequence-view", "64x1")
    snr, noise, receipt = paired_noise(key, (64, 3, 2))
    again = paired_noise(key, (64, 3, 2))
    assert torch.equal(noise, again[1]) and receipt == again[2]
    assert snr.min() >= -6 and snr.max() < 18
    assert torch.equal(before, torch.get_rng_state())
    with pytest.raises(RuntimeError), isolated_rng(receipt["seed"]):
        torch.randn(7)
        raise RuntimeError("validation failed")
    assert torch.equal(before, torch.get_rng_state())
    other = paired_noise(
        NoiseKey(
            "study", "validation", "enc_fn", 200, "sequence-view", "64x1", condition="0"
        ),
        (64, 3, 2),
    )
    assert not torch.equal(noise, other[1])


def test_failed_attempts_preserve_actual_exposure_without_advancing_completed():
    ledger = ExposureLedger()
    for status in ("completed", "nonfinite", "skipped"):
        ledger.record(
            status=status,
            sequences=64,
            source_tokens=160,
            target_tokens=96,
            receipt=status,
        )
    assert (ledger.requested, ledger.completed, ledger.failed, ledger.skipped) == (
        3,
        1,
        1,
        1,
    )
    assert (
        ledger.sequence_presentations,
        ledger.source_valid_tokens,
        ledger.target_valid_tokens,
    ) == (192, 480, 288)
    with pytest.raises(ValueError):
        ledger.record(
            status="completed",
            sequences=64,
            source_tokens=1,
            target_tokens=1,
            receipt="completed",
        )
