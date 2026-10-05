
import copy
import pytest
import torch
from jscc.checkpoint_readers import objective_settings
from jscc.training import batch_losses
from jscc.training_objectives import aggregate_batch_losses
from contracts.test_core import batch, toy_model
from shared.test_recipe_training import setup_recipe








def test_unequal_microbatch_denominators_and_gradients():
    from jscc.models.channel import IdentityChannel
    model = toy_model(where="after_final_norm", channel=IdentityChannel())
    model.codec.film = None
    other = copy.deepcopy(model)
    first = {k: v[:1] for k, v in batch().items()}
    second = batch()
    joined = {k: torch.cat((first[k], second[k])) for k in first}
    settings = objective_settings("combined")
    values = [batch_losses(model, b, settings, None, return_stats=True) for b in [first, second]]
    aggregate = aggregate_batch_losses(values, settings)
    aggregate["loss"].backward()
    full = batch_losses(other, joined, settings, None, return_stats=True)
    full["loss"].backward()
    assert int(aggregate["kl_denominator"]) == 8
    assert int(aggregate["hidden_denominator"]) == 3
    torch.testing.assert_close(aggregate["loss"], full["loss"])
    for p, q in zip(model.codec.parameters(), other.codec.parameters()):
        torch.testing.assert_close(p.grad, q.grad)

def test_site_provenance_does_not_change_paired_draws():
    from jscc.shared_inputs import batch_identity
    b = batch()
    assert batch_identity(b) == batch_identity({**b, "site_id": "enc_fn", "activation": torch.randn(2, 3, 8)})

def test_task_reuse_requires_exact_complete_observation(monkeypatch, tmp_path):
    from jscc.checkpoint_readers import validate_task_receipt
    CONDITIONS = ["no_noise", -6, 0, 6, 12, 18]
    from jscc.evaluation import codec_observation_identity
    from ordinary.test_evaluation_evidence import observation_fixture
    _, request, _, _ = observation_fixture(monkeypatch, tmp_path)
    receipt = {"status": "complete", "request": request, "identity": codec_observation_identity(request),
               "conditions": [{"condition": c, "status": "complete", "requested": 1, "completed": 1,
                               "failed": 0, "denominator": 1, "metrics": {"acc": 1.0},
                               "items": [{"sample_id": 7, "source_id": "source-7"}]} for c in CONDITIONS]}
    validate_task_receipt(receipt, request)
    for field, value in [("panel", "other-panel"), ("scorer", "other-scorer"),
                         ("noise", {"seed": 9, "namespace": "other"})]:
        changed = {**request, field: value}
        previous = {**receipt, "request": changed, "identity": codec_observation_identity(changed)}
        with pytest.raises(ValueError, match="identity differs"):
            validate_task_receipt(previous, request)
    with pytest.raises(ValueError, match="conditions incomplete"):
        validate_task_receipt({**receipt, "conditions": []}, request)
    for field, value in [("failed", None), ("completed", 0), ("status", "incomplete")]:
        rows = copy.deepcopy(receipt["conditions"])
        rows[0][field] = value
        with pytest.raises(ValueError, match="accounting incomplete"):
            validate_task_receipt({**receipt, "conditions": rows}, request)




def test_current_recipe_real_updates_frozen_teacher_and_recorded_denominators(tmp_path):
    obj, data, events = setup_recipe(tmp_path, locations=['enc_fn'], q=8)
    initial = copy.deepcopy(obj.model.codec.state_dict())
    rng = torch.get_rng_state().clone()
    event = obj.update([data])
    assert torch.equal(rng, torch.get_rng_state())
    assert any(not torch.equal(initial[k], v) for k, v in obj.model.codec.state_dict().items())
    assert all(p.grad is None for p in obj.model.base.parameters())
    assert not obj.model.base.training
    assert all(torch.isfinite(p.grad).all() for p in obj.parameters if p.grad is not None)
    assert all(state['step'] == 1 for state in obj.optimizer.state.values())
    assert event['payload']['objective']['components']['hidden']['denominator'] == 16
    assert event['payload']['objective']['components']['kl']['denominator'] == 0
    assert events == [event]


@pytest.mark.parametrize('failure', ['skip', 'nonfinite'])
def test_current_recipe_failure_preserves_attempted_exposure_and_disallows_retry(tmp_path, monkeypatch, failure):
    obj, data, events = setup_recipe(tmp_path, locations=['enc_fn'], q=8)
    if failure == 'skip':
        monkeypatch.setattr(obj.scaler, 'step', lambda optimizer: None)
    else:
        with torch.no_grad():
            next(obj.model.codec.parameters()).fill_(float('nan'))
    with pytest.raises(FloatingPointError, match='skipped|nonfinite'):
        obj.update([data])
    assert obj.attempted == 1 and obj.completed == 0 and obj.offset == 16
    assert obj.scheduler.last_epoch == 0
    assert obj.failed
    assert not obj.exposure.receipts
    with pytest.raises(RuntimeError, match='terminal'):
        obj.update([data])


def test_current_recipe_sparse_capture_does_not_change_science(tmp_path):
    first = tmp_path/'first'
    first.mkdir()
    second = tmp_path/'second'
    second.mkdir()
    full, data, _ = setup_recipe(first, locations=['enc_fn'], q=8)
    sparse, other, _ = setup_recipe(second, locations=['enc_fn'], q=8)
    sparse.model.load_state_dict(copy.deepcopy(full.model.state_dict()))
    full.audit_policy = 'full'
    for step in range(2):
        a, b = full.update([data]), sparse.update([other])
        assert a['payload']['objective'] == b['payload']['objective']
        assert a['payload']['gradient'] == b['payload']['gradient']
        assert full.last_draws[0]['key'] == sparse.last_draws[0]['key']
        assert bool(sparse.last_draws[0]['draws']) == (step == 0)
        for p, q in zip(full.parameters, sparse.parameters):
            torch.testing.assert_close(p, q, atol=0, rtol=0)
            for key, value in full.optimizer.state[p].items():
                torch.testing.assert_close(value, sparse.optimizer.state[q][key], atol=0, rtol=0)
