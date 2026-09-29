import pytest
import torch
from scripts.coco_long_prefix_diagnostic import fixed_prefix, compare_logits, cache_metadata


def test_prefix_exactly_64_native_tokens_and_discards_last():
    ids = [2] + list(range(3, 70))
    assert fixed_prefix({'cached_ids': ids}).tolist() == [ids[:64]]
    for bad in ([2] * 64, [1] * 65, [2] + [-1] * 64):
        with pytest.raises(ValueError):
            fixed_prefix({'cached_ids': bad})


def test_fixed_full_vs_cache_failure_localization_and_shapes():
    full = torch.zeros(1, 64, 5)
    other = full.clone()
    other[0, 32, 3] = 1
    result = compare_logits(full, other)
    assert result['first_failure'] == 32
    assert result['maximum_absolute_delta'] == 1
    assert not result['all_argmax_equal']
    with pytest.raises(ValueError):
        compare_logits(full[:, :63], other[:, :63])
    with pytest.raises(ValueError):
        compare_logits(full, torch.full_like(full, float('nan')))
    assert cache_metadata(None) is None


def test_backend_choice_rejected_before_model_or_config():
    from types import SimpleNamespace
    from scripts.coco_long_prefix_diagnostic import run
    with pytest.raises(ValueError, match='Backend'):
        run(SimpleNamespace(max_minutes=10, sdpa_backend='unsupported'))


def test_query_last_matches_corresponding_prefix_position():
    from scripts.coco_long_prefix_diagnostic import compare_query_last
    reference = torch.arange(320).reshape(1, 64, 5).float()
    result = compare_query_last(reference[:, :33], reference, reference, 33)
    assert result['max_abs_vs_cached'] == 0
    assert result['max_abs_vs_full64'] == 0
    assert result['returned_logit_positions'] == 33
    assert compare_query_last(reference[:, 32:33], reference, reference, 33)['max_abs_vs_cached'] == 0
    with pytest.raises(ValueError):
        compare_query_last(reference[:, :33], reference, reference, 65)
    with pytest.raises(ValueError):
        compare_query_last(reference[:, :33, :4], reference, reference, 33)


def test_source_padding_preserves_valid_tokens_and_pixels():
    from scripts.coco_long_prefix_diagnostic import source_padding_inputs
    kwargs = {'input_ids': torch.tensor([[9, 7, 0, 0]]),
              'attention_mask': torch.tensor([[1, 1, 0, 0]]), 'pixel_values': torch.ones(1, 3, 2, 2)}
    trimmed, receipt = source_padding_inputs(kwargs, 'trim')
    assert trimmed['input_ids'].tolist() == [[9, 7]]
    padded, other = source_padding_inputs(kwargs, 'plus32')
    assert padded['input_ids'].shape == (1, 36)
    assert padded['attention_mask'].sum() == 2
    assert receipt['pixels_sha256'] == other['pixels_sha256']
    assert padded['pixel_values'] is kwargs['pixel_values']
    assert kwargs['input_ids'].shape == (1, 4)


def test_tensor_digest_supports_bfloat16_without_casting_values():
    from scripts.coco_long_prefix_diagnostic import tensor_digest
    values = torch.tensor([1., 2.], dtype=torch.bfloat16)
    assert tensor_digest(values) == tensor_digest(values.clone())
    assert tensor_digest(values) != tensor_digest(values.float())


def test_precision_rejected_before_loading_model():
    from types import SimpleNamespace
    from scripts.coco_long_prefix_diagnostic import run
    with pytest.raises(ValueError, match='Precision'):
        run(SimpleNamespace(max_minutes=10, sdpa_backend='auto', precision='invalid'))


def test_double_precision_differences_are_not_downcast():
    from scripts.coco_long_prefix_diagnostic import compare_query_last
    full = torch.full((1, 64, 5), 1024., dtype=torch.float64)
    cached = full.clone()
    cached[0, 32, 0] += 1e-6
    assert compare_logits(full, cached)['positions'][32]['max_abs_logit_delta'] > 0
    row = compare_query_last(full[:, 32:33], cached, full, 33)
    assert row['max_abs_vs_cached'] > 0
    assert row['max_abs_vs_full64'] == 0
