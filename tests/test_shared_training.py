import pytest
import torch
from sharing_fixtures import sharing_model, sharing_batches, sharing_plan
from jscc.sharing_protocol import SharingLearner


def learner(name='shared', cell='D-LN'):
    return SharingLearner(sharing_model(cell), schedule=sharing_plan().learner(name), run_id=name, identity={'source': 'source', 'config': 'config', 'data': 'data', 'parent': None}, model_revision='tiny-cpu')


@pytest.mark.parametrize('cell', ['D-none', 'D-LN', 'R-none', 'R-LN'])
def test_rotating_actual_updates(cell):
    item = learner(cell=cell)
    batches = sharing_batches()
    initial = {k: v.clone() for k, v in item.model.codec.state_dict().items()}
    original = dict(item.model.split)
    for index in range(12):
        update = item.schedule.update_at(index)
        item.update([batches[update.site_local_index]])
        assert item.model.split == original
        assert all(p.grad is None for p in item.model.base.parameters())
        if index == 0:
            assert all(torch.equal(initial[k], v) for k, v in item.model.codec.state_dict().items())
    assert item.completed == 12
    assert item.scheduler.last_epoch == 12
    assert any(not torch.equal(initial[k], v) for k, v in item.model.codec.state_dict().items())
    item.exposure.finalize()
    with pytest.raises(RuntimeError, match='terminal'):
        item.update([batches[0]])


def test_reject_wrong_view_before_update():
    item = learner()
    with pytest.raises(ValueError, match='view/layout'):
        item.update([sharing_batches()[1]])
    assert item.completed == 0


def test_heldout_requires_freeze():
    item = learner()
    with pytest.raises(ValueError, match='freeze'):
        item.validate_site(sharing_batches()[:1], 'enc_l14')


def test_pairing_uses_site_local_index_in_actual_draws():
    shared = learner()
    specialist = learner('specialist_enc_l19')
    shared.audit_policy = specialist.audit_policy = 'full'
    batch = sharing_batches()[0]
    shared.update([batch])  # l9, global 0
    shared.update([batch])  # l19, global 1, site-local 0
    specialist.update([batch])  # l19, global 0, site-local 0
    assert shared.last_draws == specialist.last_draws
    assert shared.completed == 2 and specialist.completed == 1


def test_skipped_optimizer_is_terminal_and_route_restores(monkeypatch):
    item = learner()
    original = dict(item.model.split)
    monkeypatch.setattr(item.scaler, 'step', lambda optimizer: None)
    with pytest.raises(FloatingPointError, match='skipped'):
        item.update([sharing_batches()[0]])
    assert item.failed and item.completed == 0
    assert item.exposure.completed == 0
    assert item.model.split == original
    with pytest.raises(RuntimeError, match='terminal'):
        item.update([sharing_batches()[0]])
