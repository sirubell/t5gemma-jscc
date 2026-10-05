"""Objective parity, real learner receipts and M1 canonical-state continuation."""

from copy import deepcopy
from typing import cast
import pytest
import torch
from jscc import site_routing as routing
from jscc.shared_inputs import bind_batch_identity
from jscc.reconstruction import reconstruction_batch_values
from jscc.checkpoint_readers import ORIGINAL13_SITES
from jscc.models.split_model import resolve_original13_site
from jscc.asset_identity import state_dict_identity
from jscc.qualification import state_exact
from jscc.training import batch_losses
from jscc.training_objectives import BatchValues, aggregate_batch_losses, effective_batch_denominators, scaled_batch_loss
from shared.test_shared_case_routes import batch, make_model, settings





def summed_stat(records, name):
    tensors = []
    for item in records:
        value = item.get(name)
        assert torch.is_tensor(value)
        tensors.append(value)
    return sum(tensors)


def unequal_batches():
    first = batch()
    first['attention_mask'][1] = 0
    second = {key: value[:1].clone() for key, value in batch().items()}
    second['input_ids'] = torch.tensor([[2, 5]])
    second['attention_mask'] = torch.tensor([[1, 0]])
    second['labels'] = torch.tensor([[7, -100]])
    return [first, second]


@pytest.mark.parametrize('site_id', ('enc_l9', 'dec_l0', 'dec_l24'))
@pytest.mark.parametrize('functional', (False, True))
@pytest.mark.parametrize('coefficients', (None, {"hidden": .3, "memory": .4, "kl": .7}))
def test_weighted_forward_aggregate_streamed_gradients_unequal_masks(site_id, functional, coefficients):
    model = make_model()
    site = resolve_original13_site(model.base, site_id, 'tiny-original13')
    objective = settings(site_id, functional)
    if coefficients is not None:
        objective["loss_weights"] = {**coefficients, "kl": coefficients["kl"] if functional else 0.,
                                     "memory": coefficients["memory"] if site_id.startswith("dec_") else 0.}
    batches = unequal_batches()
    with model.at_site(site, purpose='train', authorization={'sites': {site_id: 'trained'}}):
        def values(data):
            return (batch_losses(model, data, objective, None, return_stats=True, valid_only_kl=True)
                    if functional else reconstruction_batch_values(model, data, None, objective, site=site))
        records = [values(data) for data in batches]
        totals = aggregate_batch_losses(records, objective)
        assert int(totals['hidden_denominator']) == (3 if site_id.startswith('dec_') else 2)
        if site_id.startswith('dec_'):
            assert int(totals['memory_denominator']) == 2
        assert int(totals['kl_denominator']) == (6 if functional else 0)
        # Independently apply the adopted coefficients to each stream's own
        # valid-sample/token denominator, never average microbatch means.
        expected = summed_stat(records, 'hidden_numerator') / summed_stat(records, 'hidden_denominator')
        expected = objective['loss_weights']['hidden'] * expected
        if objective['requires_memory']:
            expected = expected + objective["loss_weights"]["memory"] * summed_stat(records, 'memory_numerator') / summed_stat(records, 'memory_denominator')
        if functional:
            expected = expected + objective["loss_weights"]["kl"] * summed_stat(records, 'kl_numerator') / summed_stat(records, 'kl_denominator')
        torch.testing.assert_close(totals['loss'], expected)
        totals['loss'].backward()
        aggregate_grads = [cast(torch.Tensor, p.grad).clone() for p in model.codec.parameters()]
        model.zero_grad(set_to_none=True)
        denominators = effective_batch_denominators(model, batches, 'cpu')
        streamed = torch.zeros(())
        for data in batches:
            value = values(data)
            loss = scaled_batch_loss(value, objective, denominators)
            streamed = streamed + loss.detach()
            loss.backward()
        torch.testing.assert_close(streamed, totals['loss'])
        for p, grad in zip(model.codec.parameters(), aggregate_grads):
            torch.testing.assert_close(p.grad, grad, rtol=1e-5, atol=1e-6)


def real_setup(tmp_path, ratio=50):
    from jscc.activation_replay import canonical_digest
    from jscc.shared_state import RecipeTrainingState
    from shared.test_recipe_training import setup_recipe
    tmp_path.mkdir(parents=True, exist_ok=True)
    learner, data, _ = setup_recipe(tmp_path, q=400 if ratio == 90 else 40, ratio=ratio, retain_defaults=True)
    plan = learner.schedule
    metadata = {'source_identity': 'tiny-source',
        'config_identity': canonical_digest(plan.description['config']),
        'data_identity': canonical_digest({'rows': 'tiny'}), 'prepared_identity': 'indexed-tiny',
        'observation_identity': canonical_digest(plan.description['observations']),
        'initialization_identity': state_dict_identity(routing.export_codec(learner.model)),
        'model_revision': 'tiny-original13'}
    learner.identity.update(config=metadata['config_identity'], data=metadata['data_identity'])
    state = RecipeTrainingState(learner, plan=plan, metadata=metadata)
    return learner, state, [data] * plan.q


def advance(learner, batches):
    data = batches[learner.completed // 13]
    return learner.update([data], batch_identities=[bind_batch_identity(data)])


def test_real_native16_cycle_receipts_snr_pairing_and_lr(tmp_path):
    learner, _, batches = real_setup(tmp_path/'fixture')
    assert len(learner.parameters) == len({id(p) for p in learner.parameters}) == 4
    initial = routing.export_codec(learner.model)
    for i, site in enumerate(ORIGINAL13_SITES):
        lr = learner.optimizer.param_groups[0]['lr']
        event = advance(learner, batches)
        payload = event['payload']
        objective = payload['objective']
        assert objective['kind'] == 'reconstruction'
        assert objective['resolved_weights'] == learner.schedule.objective_at(i)
        assert payload['site_id'] == site and payload['lr_used'] == [lr]
        assert objective['components']['kl']['denominator'] == 0
        assert objective['components']['hidden']['denominator'] == 16
        assert objective['components']['memory']['denominator'] == (16 if site.startswith('dec_') else 0)
        roles = [draw['stream'] for draw in learner.last_draws[0]['draws']]
        if i == 0:
            assert roles == ['hidden']
        elif i == 6:
            # Sparse capture ordinarily skips this step; explicit probe below
            # verifies role-specific epsilon namespaces on the same SNR.
            assert roles == []
        assert payload['exposure']['hidden_coordinates'] == 40 * 8
        assert payload['exposure']['hidden_coordinates_allocated'] == (64 if site.startswith('dec_') else 48) * 8
        assert learner.completed == learner.exposure.completed == i + 1
    assert any(not torch.equal(initial[key], value) for key, value in routing.export_codec(learner.model).items())
    assert all(p.grad is None for p in learner.model.base.parameters())
    model = learner.model
    site = learner.sites['dec_l0']
    with model.at_site(site, purpose='train', authorization={'sites': {site.site_id: 'trained'}}):
        snr = torch.full((16, 1, 1), 0.)
        with model.channel.replay(0, 'role-test', capture=True):
            reconstruction_batch_values(model, batches[0], snr, settings(site.site_id), site=site)
        draws = model.channel.draw_summaries
        assert [draw['stream'] for draw in draws] == ['hidden', 'memory']
        assert draws[0]['epsilon_sha256'] != draws[1]['epsilon_sha256']
    paired, _, _ = real_setup(tmp_path/'paired', ratio=90)
    assert paired.schedule.update_at(6).noise_key == learner.schedule.update_at(6).noise_key


def test_real_decoder_interior_canonical_checkpoint_continuation(tmp_path):
    learner, state, batches = real_setup(tmp_path/'fixture1')
    for _ in range(8):
        advance(learner, batches)
    checkpoint = state.save(tmp_path/'interior.pt')
    assert set(checkpoint.payload['model']) == set(routing.export_codec(learner.model))
    assert learner.model.memory_codec is learner.model.codec
    expected_event = advance(learner, batches)
    expected_weights = routing.export_codec(learner.model)
    expected_optimizer = deepcopy(learner.optimizer.state_dict())
    resumed, continuation, resumed_batches = real_setup(tmp_path/'fixture2')
    continuation.restore(checkpoint.reference)
    actual = advance(resumed, resumed_batches)
    assert actual['payload']['site_id'] == 'dec_l8'
    assert actual['payload']['objective'] == expected_event['payload']['objective']
    assert state_exact(routing.export_codec(resumed.model), expected_weights)
    assert state_exact(resumed.optimizer.state_dict(), expected_optimizer)
    assert state_exact(resumed.exposure.per_site, learner.exposure.per_site)
    assert resumed.model.memory_codec is resumed.model.codec


def test_missing_memory_fails_forward_streaming_and_aggregate(monkeypatch):
    model = make_model()
    site = resolve_original13_site(model.base, 'dec_l0', 'tiny-original13')
    objective = settings('dec_l0', True)
    data = batch()
    with model.at_site(site, purpose='train', authorization={'sites': {site.site_id: 'trained'}}):
        denominators = effective_batch_denominators(model, [data], 'cpu')
        value = batch_losses(model, data, objective, None, return_stats=True, valid_only_kl=True)
        missing = cast(BatchValues, {**value, 'memory_numerator': None, 'memory_denominator': None})
        with pytest.raises(RuntimeError, match='missing required memory'):
            scaled_batch_loss(missing, objective, denominators)
        with pytest.raises(RuntimeError, match='missing required memory'):
            aggregate_batch_losses([missing], objective)
        with pytest.raises(RuntimeError, match='inconsistent memory'):
            aggregate_batch_losses([value, missing], objective)
        monkeypatch.setattr(model, '_receiver_memory', lambda *_args, **_kwargs: None)
        model._install_memory_hooks()
        with pytest.raises(RuntimeError, match='missing required memory'):
            batch_losses(model, data, objective, None, return_stats=True, valid_only_kl=True)


def test_nonfinite_and_changed_bound_batch_fail_stop(monkeypatch, tmp_path):
    learner, _, batches = real_setup(tmp_path/'fixture3')
    binding = bind_batch_identity(batches[0])
    batches[0]['input_ids'][0, 0] = 3
    with pytest.raises(ValueError, match='changed after identity binding'):
        learner.update([batches[0]], batch_identities=[binding])
    assert learner.completed == 0
    learner, _, batches = real_setup(tmp_path/'fixture4')
    with torch.no_grad():
        learner.model.codec.decoder[0].bias[0] = float('nan')
    with pytest.raises(FloatingPointError, match='nonfinite'):
        advance(learner, batches)
    assert learner.failed and learner.completed == learner.exposure.completed == 0
    with pytest.raises(RuntimeError, match='terminal'):
        advance(learner, batches)


def test_real_pre_post_reset_and_functional_interior_continuation(tmp_path):
    learner, state, batches = real_setup(tmp_path/'fixture5')
    while learner.completed < learner.schedule.stages[1].start:
        advance(learner, batches)
    pre = state.save(tmp_path/'pre.pt')
    initial_optimizer = deepcopy(learner.optimizer.state_dict())
    state.reset(pre)
    assert not learner.optimizer.state and initial_optimizer['state']
    post = state.save(tmp_path/'post.pt')
    assert state_exact(pre.payload['model'], post.payload['model'])
    expected_first = advance(learner, batches)
    first_weights = routing.export_codec(learner.model)
    assert expected_first['payload']['objective']['kind'] == 'combined'
    assert expected_first['payload']['objective']['resolved_weights']['kl'] == 1.
    assert expected_first['payload']['lr_used'] == [0.]
    for reference in (pre, post):
        resumed, restored, resumed_batches = real_setup(tmp_path/'fixture6')
        restored.restore(reference.reference)
        if resumed.phase_start_step == 0:
            restored.reset(reference)
        event = advance(resumed, resumed_batches)
        assert event['payload']['objective'] == expected_first['payload']['objective']
        assert state_exact(routing.export_codec(resumed.model), first_weights)
    while learner.completed < learner.schedule.stages[1].start + 8:
        advance(learner, batches)
    interior = state.save(tmp_path/'functional-interior.pt')
    expected = advance(learner, batches)
    weights = routing.export_codec(learner.model)
    optimizer = deepcopy(learner.optimizer.state_dict())
    resumed, restored, resumed_batches = real_setup(tmp_path/'fixture7')
    restored.restore(interior.reference)
    actual = advance(resumed, resumed_batches)
    assert actual['payload']['site_id'] == 'dec_l8'
    assert actual['payload']['objective'] == expected['payload']['objective']
    assert actual['payload']['objective']['resolved_weights'] == {'kl': 1., 'hidden': .05, 'memory': .05}
    assert state_exact(routing.export_codec(resumed.model), weights)
    assert state_exact(resumed.optimizer.state_dict(), optimizer)
