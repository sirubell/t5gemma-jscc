from types import SimpleNamespace

import pytest
import torch

from scripts.coco_cache_diagnostic import compare_outputs, run


def output(ids, scores):
    values = tuple(torch.tensor([x], dtype=torch.float32) for x in scores)
    return SimpleNamespace(sequences=torch.tensor([ids]), scores=values, logits=values)


def test_divergence_compares_same_prefix_only_and_saves_raw_logits(tmp_path):
    cached = output([2, 1, 0, 1], [[0, 2, 1], [2, 1, 0], [0, 2, 1]])
    other = output([2, 1, 2, 0], [[0, 2.01, 1], [1.99, 1, 2], [2, 0, 1]])
    result = compare_outputs(cached, other, tmp_path, 'test')
    assert not result['tokens_equal']
    assert result['first_divergence']['step'] == 1
    assert result['first_divergence']['same_prefix_ids'] == [2, 1]
    assert len(result['common_prefix_comparisons']) == 2
    raw = torch.load(tmp_path / 'test-first-divergence.pt', weights_only=True)
    assert torch.equal(raw['cached_logits'], cached.logits[1][0])


def test_masked_scores_json_and_exact_replay(tmp_path):
    a = output([2, 1], [[float('-inf'), 2, 1]])
    result = compare_outputs(a, a, tmp_path, 'equal')
    assert result['tokens_equal']
    assert result['first_divergence'] is None
    assert result['common_prefix_comparisons'][0]['max_abs_finite_score_delta'] == 0
    assert not (tmp_path / 'equal-first-divergence.pt').exists()


def test_batched_input_rejected(tmp_path):
    a = output([2, 1], [[0, 2, 1]])
    a.sequences = a.sequences.repeat(2, 1)
    with pytest.raises(ValueError, match='single fixed'):
        compare_outputs(a, a, tmp_path, 'no')


@pytest.mark.parametrize('minutes', [0, -1, 11])
def test_budget_rejected_before_gpu_or_files(minutes):
    with pytest.raises(ValueError, match='maximum'):
        run(SimpleNamespace(max_minutes=minutes))


def test_forced_eos_with_one_finite_score_has_undefined_margin(tmp_path):
    a = output([2, 1], [[float('-inf'), 0, float('-inf')]])
    a.logits = (torch.tensor([[1., 2., 3.]]),)
    result = compare_outputs(a, a, tmp_path, 'forced-eos')
    step = result['common_prefix_comparisons'][0]
    assert step['cached_finite_score_count'] == 1
    assert step['uncached_finite_score_count'] == 1
    assert step['cached_top2_margin'] is None
    assert step['uncached_top2_margin'] is None
    assert result['tokens_equal']
    import json
    assert json.loads((tmp_path / 'forced-eos.json').read_text()) == result


def test_generation_cache_overrides_only_apply_to_cached_calls():
    from scripts.coco_cache_diagnostic import cache_generation_options
    assert cache_generation_options(cached=True) == {}
    assert cache_generation_options(cached=False, cache_implementation='hybrid', disable_compile=True) == {}
    assert cache_generation_options(cached=True, cache_implementation='dynamic', disable_compile=True) == {
        'cache_implementation': 'dynamic', 'disable_compile': True}
    with pytest.raises(ValueError):
        cache_generation_options(cached=True, cache_implementation='bogus')
