from typing import Any

import pytest
import torch

from scripts.coco_native_preflight import assert_visual_tokens, batch_plan


def test_effective_batch_is_preserved():
    assert batch_plan([1, 2, 4, 8, 16], 32) == [(1, 32), (2, 16), (4, 8), (8, 4), (16, 2)]


@pytest.mark.parametrize('candidates,effective', [([2, 1], 16), ([1, 1], 16), ([3], 16), ([8], 4), ([], 16), ([1], 0)])
def test_invalid_or_unbounded_batch_plan(candidates, effective):
    with pytest.raises(ValueError):
        batch_plan(candidates, effective)


def test_visual_token_truncation_is_a_blocker():
    full = torch.tensor([[1, 9, 9, 9, 3]])
    assert assert_visual_tokens(full, torch.tensor([1, 9, 9, 9, 3, 0]), 9) == 3
    with pytest.raises(ValueError, match='Visual token truncation'):
        assert_visual_tokens(full, torch.tensor([1, 9, 9]), 9)
    with pytest.raises(ValueError):
        assert_visual_tokens(full, full, 99)
    with pytest.raises(ValueError):
        assert_visual_tokens(full, full, None)


@pytest.mark.parametrize('updates,minutes,production', [(0, 30, 12), (5, 30, 12), (2, 46, 12), (2, 0, 12), (2, 30, 33)])
def test_budget_rejected_before_output_or_gpu(tmp_path, updates, minutes, production):
    from types import SimpleNamespace
    from scripts.coco_native_preflight import run
    destination = tmp_path / 'must-not-exist'
    args = SimpleNamespace(batches=[1], effective_batch=32,
                           updates_per_batch=updates, max_minutes=minutes,
                           production_steps=production, inventory_size=16,
                           output=destination)
    with pytest.raises(ValueError):
        run(args)
    assert not destination.exists()


class FakeCommunication:
    def __init__(self, noop=False):
        self.codec = torch.nn.Linear(2, 2)
        self.channel = torch.nn.Identity()
        self.memory_codec = torch.nn.Linear(2, 2)
        self.noop = noop

    def load_communication_state(self, state):
        if not self.noop:
            for name in ('codec', 'channel', 'memory_codec'):
                getattr(self, name).load_state_dict(state[name])


def test_reload_requires_restoration_after_mutation(tmp_path):
    from scripts.coco_native_preflight import perturb_and_reload
    receipt = perturb_and_reload(FakeCommunication(), tmp_path / 'state.pt')
    assert receipt['before_sha256'] == receipt['restored_sha256']
    assert receipt['perturbed_sha256'] != receipt['restored_sha256']
    assert receipt['perturbed_components'] == ['codec', 'memory_codec']
    with pytest.raises(AssertionError, match='did not restore'):
        perturb_and_reload(FakeCommunication(noop=True), tmp_path / 'noop.pt')


def test_production_progress_counts_completed_calls_once():
    from scripts.coco_native_preflight import record_production_update
    record: dict[str, Any] = {'updates': 8, 'presentations': 256}
    record_production_update(record, 32, 1, {})
    assert record['updates'] == 9
    assert record['presentations'] == 288
    record_production_update(record, 32, 2, {'start': 123.0})
    assert record['updates'] == 10
    assert record['production_progress']['optimizer_updates_observed'] == 2
    assert record['production_progress']['interval_boundaries'] == {'start': 123.0}


def test_indexed_caption_evidence_preserves_sample_and_ids():
    from scripts.coco_native_preflight import IndexedCaptionEvidence
    item = {'labels': torch.tensor([2, 3, -100]), 'input_ids': torch.tensor([7, 8])}
    wrapped = IndexedCaptionEvidence([item], [123])
    assert len(wrapped) == 1
    assert wrapped[0]['evidence_image_id'] == 123
    assert wrapped[0]['labels'] is item['labels']
    assert 'evidence_image_id' not in item
    with pytest.raises(ValueError):
        IndexedCaptionEvidence([item], [])



def test_fp32_reference_requires_frozen_prefix_before_gpu(tmp_path):
    from types import SimpleNamespace
    from scripts.coco_native_preflight import run
    args = SimpleNamespace(batches=[8, 16], effective_batch=32,
        updates_per_batch=2, max_minutes=28, production_steps=32,
        inventory_size=16, cache_policy='fp32_math_reference', prefix_json=None,
        output=tmp_path / 'not-created')
    with pytest.raises(ValueError, match='fixed prefix provenance'):
        run(args)
    assert not args.output.exists()


def test_h200_large_capacity_batches_preserve_effective_batch():
    assert batch_plan([8, 16, 32, 64], 64) == [(8, 8), (16, 4), (32, 2), (64, 1)]
    with pytest.raises(ValueError, match='divide effective batch'):
        batch_plan([8, 16, 32, 64], 32)
    with pytest.raises(ValueError):
        batch_plan([128], 128)
