from types import SimpleNamespace

import pytest
import torch

from scripts.coco_route_smoke import logit_comparison, run


def test_logit_check_distinguishes_tolerance_from_argmax():
    a = torch.tensor([[1.0, 1.000001]])
    b = torch.tensor([[1.000001, 1.0]])
    result = logit_comparison(a, b)
    assert result['allclose']
    assert not result['argmax_equal']
    assert not logit_comparison(a, b, atol=0, rtol=0)['allclose']


@pytest.mark.parametrize('a,b', [(torch.zeros(2), torch.zeros(3)),
                                (torch.tensor([float('nan')]), torch.zeros(1))])
def test_nonfinite_or_mismatch_rejected(a, b):
    with pytest.raises(ValueError):
        logit_comparison(a, b)


@pytest.mark.parametrize('updates,minutes', [(0, 1), (3, 1), (1, 6), (1, 0)])
def test_resource_bound_checked_before_loading(updates, minutes):
    with pytest.raises(ValueError, match='Bounded'):
        run(SimpleNamespace(updates=updates, max_minutes=minutes))


def test_prefix_replay_compares_native_prefix_without_free_running(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from scripts import coco_route_smoke as smoke

    class Model:
        def __init__(self):
            self.calls = []

        def transmission(self, *args, **kwargs):
            return nullcontext()

        def __call__(self, **kwargs):
            ids = kwargs['decoder_input_ids']
            self.calls.append(kwargs)
            position = ids[0, -1].item()
            return SimpleNamespace(logits=torch.tensor([[[position, 0.0]]]),
                                   past_key_values=object(), encoder_last_hidden_state=torch.zeros(1, 4, 2))

    monkeypatch.setattr(smoke, 'autocast_for', lambda model: nullcontext())
    model = Model()
    kwargs = {'decoder_input_ids': torch.tensor([[2, 8, 9, 10]]),
              'input_ids': torch.ones(1, 4), 'attention_mask': torch.ones(1, 4),
              'pixel_values': torch.zeros(1, 3, 2, 2)}
    result = smoke.prefix_replay(model, kwargs, tmp_path, 'fp32', 1e-4, 1e-4)
    assert all(row['allclose'] for row in result)
    assert [row['prefix'] for row in result] == [[2], [2, 8], [2, 8, 9]]
    for index in (3, 5):
        assert 'encoder_outputs' in model.calls[index]
        assert 'pixel_values' not in model.calls[index]
        assert model.calls[index]['decoder_input_ids'].shape[1] == 1
    assert (tmp_path / 'fp32-same-prefix-logits.pt').exists()


@pytest.mark.parametrize('counts,decoder', [({'hidden': 0, 'memory': 0}, False),
    ({'hidden': 512, 'memory': 0}, True), ({'hidden': 512, 'memory': 512}, False)])
def test_stream_payload_must_match_route(counts, decoder):
    from scripts.coco_route_smoke import check_payload
    with pytest.raises(AssertionError):
        check_payload(counts, decoder)


def test_frozen_prefix_requires_bos_and_enough_ids(tmp_path):
    import json
    from scripts.coco_route_smoke import load_forced_prefix
    path = tmp_path / 'prefix.json'
    ids = [2] + list(range(64))
    path.write_text(json.dumps({'cached_ids': ids}))
    assert load_forced_prefix(path) == ids[:64]
    path.write_text(json.dumps({'cached_ids': ids[:64]}))
    with pytest.raises(ValueError):
        load_forced_prefix(path)


def test_long_prefix_replay_steps_all_positions_and_compares_thresholds(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from scripts import coco_route_smoke as smoke

    class Model:
        def __init__(self):
            self.calls = []

        def transmission(self, *args, **kwargs):
            return nullcontext()

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            ids = kwargs['decoder_input_ids']
            return SimpleNamespace(logits=torch.tensor([[[float(ids[0, -1]), 0.0]]]),
                                   past_key_values=object(), encoder_last_hidden_state=torch.zeros(1, 4, 2))

    monkeypatch.setattr(smoke, 'autocast_for', lambda model: nullcontext())
    model = Model()
    kwargs = {'decoder_input_ids': torch.tensor([[2, 8, 9]]), 'input_ids': torch.ones(1, 4),
              'attention_mask': torch.ones(1, 4), 'pixel_values': torch.zeros(1, 3, 2, 2)}
    forced_ids = [2] + list(range(3, 66))
    rows = smoke.prefix_replay(model, kwargs, tmp_path, 'fp32', 1e-4, 1e-4, forced_ids=forced_ids)
    assert [row['prefix_length'] for row in rows] == [1, 2, 3, 31, 32, 33, 34, 64]
    cached = [call for call in model.calls if call['use_cache']]
    assert len(cached) == 64
    assert [int(call['decoder_input_ids'][0, -1]) for call in cached] == forced_ids
    assert all(row['allclose'] for row in rows)
    assert len(model.calls) == 64 + 8
