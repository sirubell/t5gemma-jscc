from types import SimpleNamespace

import pytest
import torch

from scripts.five_shot_preflight import check_gradients


def test_both_stream_gradients_are_required():
    a, b = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    a(torch.ones(1, 2)).sum().backward()
    model = SimpleNamespace(codec=a, memory_codec=b)
    with pytest.raises(ValueError, match='memory gradients'):
        check_gradients(model)
    b(torch.ones(1, 2)).sum().backward()
    assert set(check_gradients(model)) == {'hidden', 'memory'}
    assert b.weight.grad is not None
    b.weight.grad.fill_(float('nan'))
    with pytest.raises(ValueError, match='memory gradients'):
        check_gradients(model)


def test_encoder_gradient_check_does_not_require_memory():
    a = torch.nn.Linear(2, 2)
    a(torch.ones(1, 2)).sum().backward()
    assert set(check_gradients(SimpleNamespace(codec=a, memory_codec=None))) == {'hidden'}


def test_group_rejects_incomplete_roster(tmp_path, monkeypatch):
    from scripts.five_shot_preflight_group import main
    monkeypatch.setattr('sys.argv', ['preflight', '--root', str(tmp_path), '--family', 'enc'])
    with pytest.raises(ValueError, match='split roster'):
        main()
