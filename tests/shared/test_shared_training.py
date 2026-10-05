"""Actual current native16 training; historical SharingLearner front removed."""
import pytest
import torch
from jscc.shared_inputs import bind_batch_identity
from shared.test_recipe_training import setup_recipe, reset_fixture


def test_rotating_actual_updates(tmp_path):
    item, batch, _ = setup_recipe(tmp_path, locations=['enc_l2', 'enc_fn'], q=8)
    initial = {k: v.clone() for k, v in item.model.codec.state_dict().items()}
    original = dict(item.model.split)
    for index in range(item.schedule.horizon):
        if item.schedule.stage_at(index).start != item.phase_start_step:
            reset_fixture(item)
        event = item.update([batch])['payload']
        assert event['site_id'] == item.schedule.trained_sites[index % 2]
        assert item.model.split == original
        assert all(p.grad is None for p in item.model.base.parameters())
    assert item.completed == item.scheduler.last_epoch + item.phase_start_step == item.schedule.horizon
    assert any(not torch.equal(initial[k], v) for k, v in item.model.codec.state_dict().items())
    item.exposure.finalize()
    with pytest.raises(RuntimeError, match='terminal'):
        item.update([batch])


def test_reject_wrong_view_before_update(tmp_path):
    item, batch, _ = setup_recipe(tmp_path, locations=['enc_fn'], q=8)
    batch['input_ids'] = batch['input_ids'] + 1
    with pytest.raises(ValueError, match='view/layout'):
        item.update([batch], batch_identities=[bind_batch_identity(batch)])
    assert item.completed == item.attempted == 0


def test_pairing_uses_site_local_index_in_actual_draws(tmp_path):
    (tmp_path/'shared').mkdir()
    (tmp_path/'single').mkdir()
    shared, batch, _ = setup_recipe(tmp_path/'shared', locations=['enc_l2', 'enc_fn'], q=8)
    specialist, single_batch, _ = setup_recipe(tmp_path/'single', locations=['enc_fn'], q=8)
    shared.audit_policy = specialist.audit_policy = 'full'
    shared.update([batch])
    shared.update([batch])
    specialist.update([single_batch])
    assert shared.last_draws == specialist.last_draws
    assert shared.completed == 2 and specialist.completed == 1


def test_skipped_optimizer_is_terminal_and_route_restores(tmp_path, monkeypatch):
    item, batch, _ = setup_recipe(tmp_path, locations=['enc_fn'], q=8)
    original = dict(item.model.split)
    monkeypatch.setattr(item.scaler, 'step', lambda optimizer: None)
    with pytest.raises(FloatingPointError, match='skipped'):
        item.update([batch])
    assert item.failed and item.completed == 0
    assert item.exposure.completed == 0
    assert item.model.split == original
    with pytest.raises(RuntimeError, match='terminal'):
        item.update([batch])
