"""Current shared ownership, input rejection and fail-stop numerical behavior."""
import pytest
import torch

from jscc.shared_inputs import bind_batch_identity
from jscc.shared_training import RecipeLearner
from shared.test_recipe_training import setup_recipe


def test_current_learner_owns_real_update_without_historical_ancestry(tmp_path):
    learner, data, events = setup_recipe(tmp_path, locations=['dec_l3', 'enc_fn'])
    assert type(learner) is RecipeLearner
    assert RecipeLearner.__bases__ == (object,)
    owner = {id(p) for p in learner.model.codec.parameters()}
    assert {id(p) for p in learner.parameters} == owner
    assert {id(p) for g in learner.optimizer.param_groups for p in g['params']} == owner
    event = learner.update([data])
    assert events == [event]
    assert event['task'] == learner.schedule.description['config']['task']
    assert event['payload']['site_id'] == 'dec_l3'
    assert event['payload']['objective']['kind'] == 'reconstruction'
    assert event['payload']['objective']['components']['memory']['denominator'] == 16
    assert any(p.grad is not None and bool(torch.isfinite(p.grad).all())
               and bool(p.grad.abs().sum() > 0) for p in learner.parameters)
    assert learner.model.memory_codec is learner.model.codec
    assert all(not p.requires_grad for p in learner.model.base.parameters())
    assert not learner.model._routing_active and learner.pending_update is None


def test_changed_complete_view_rejects_before_attempt_or_exposure(tmp_path):
    learner, data, _ = setup_recipe(tmp_path, locations=['enc_fn'])
    changed = {k: v.clone() if torch.is_tensor(v) else v for k, v in data.items()}
    changed['input_ids'][0, 0] += 1
    with pytest.raises(ValueError, match='view/layout'):
        learner.update([changed], batch_identities=[bind_batch_identity(changed)])
    assert (learner.completed, learner.attempted, learner.offset, learner.valid_tokens) == (0, 0, 0, 0)
    assert learner.exposure.receipts == []
    assert learner.pending_update is None and not learner.model._routing_active


def test_skipped_optimizer_poison_is_terminal_and_restores_route(tmp_path, monkeypatch):
    learner, data, events = setup_recipe(tmp_path, locations=['dec_l3'])
    monkeypatch.setattr(learner.scaler, 'step', lambda optimizer: None)
    with pytest.raises(FloatingPointError, match='skipped optimizer update'):
        learner.update([data])
    assert learner.failed
    assert (learner.completed, learner.attempted, learner.offset) == (0, 1, 16)
    assert learner.exposure.receipts == []
    assert learner.pending_update is learner.active_site is learner.active_local is None
    assert not learner.model._routing_active
    assert events[-1]['event_type'] == 'failure'
    with pytest.raises(RuntimeError, match='terminal'):
        learner.update([data])
