from types import SimpleNamespace

import pytest
import torch

from scripts.coco_capacity_check import replicated_batch, run


def fixture():
    return {'pixel_values': torch.zeros(1, 5, 3, 896, 896, device='meta'),
            'input_ids': torch.tensor([[2, 10, 0]]),
            'labels': torch.tensor([[4, 7] + [-100] * 62]),
            'attention_mask': torch.tensor([[1, 1, 0]])}


def test_replication_preserves_original_and_masks():
    original = fixture()
    result = replicated_batch(original, 16)
    assert result['pixel_values'].shape == (16, 5, 3, 896, 896)
    assert torch.equal(result['labels'], original['labels'].expand(16, -1))
    assert result['attention_mask'].tolist() == [[1, 1, 0]] * 16
    result['labels'][0, 0] = 99
    assert original['labels'][0, 0] == 4


def test_stress_changes_only_ignored_target_slots():
    result = replicated_batch(fixture(), 16, True)
    assert result['labels'].tolist() == [[4] + [7] * 63] * 16
    assert (result['labels'] != -100).sum() == 1024


def test_invalid_fixture_rejected():
    value = fixture()
    value['pixel_values'] = torch.zeros(1, 3, 896, 896, device='meta')
    with pytest.raises(ValueError, match='five-image'):
        replicated_batch(value, 16)
    value = fixture()
    value['labels'].fill_(-100)
    with pytest.raises(ValueError, match='No valid'):
        replicated_batch(value, 16, True)


@pytest.mark.parametrize('batch,minutes', [(0, 3), (17, 3), (16, 4), (16, 0)])
def test_bounds_before_any_loading(batch, minutes):
    with pytest.raises(ValueError, match='Bounded'):
        run(SimpleNamespace(batch_size=batch, max_minutes=minutes))
