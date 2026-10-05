"""Actual six-site update/state/scoring lifecycle, without pretrained weights."""

from copy import deepcopy
from dataclasses import replace
import json
import pytest
import torch
from jscc.checkpoint_readers import EncoderPilotPlan, PILOT_SITE_SPECS as SITE_SPECS, PILOT_SITES as SITES, build_pilot_state, final_panels
from jscc.activation_replay import canonical_digest
from jscc.qualification import readiness_view_indices
from jscc.models.split_model import resolve_encoder_site
from jscc.shared_inputs import batch_view
from jscc.checkpoint_readers import ENCODER_PILOT_PROTOCOL
from jscc.checkpoint_readers import validate_pilot_state
from sharing_fixtures import sharing_model, sharing_batches





def fixture():
    torch.set_num_threads(1)
    batches = native_batches()
    for batch in batches:
        batch['source_family_ids'] = [f'train-{int(i)}' for i in batch['row_ids']]
    plan = EncoderPilotPlan(tuple(batch_view(b) for b in batches), 'six-pilot-cpu', synthetic=True)
    metadata = {'source_identity': 'source', 'config_identity': 'config', 'data_identity': 'data',
                'parent_identity': None, 'initialization_identity': 'init', 'stream_identity': 'stream',
                'protocol_identity': ENCODER_PILOT_PROTOCOL, 'prepared_identity': 'prepared',
                'model_revision': 'tiny-cpu', 'final_task_identity': 'task', 'lineage': [], 'phase': 'both'}
    return sharing_model('D-N'), plan, batches, metadata

def test_fixed_schedule_and_balanced_exposure():
    _, small, _, _ = fixture()
    views = tuple(replace(small.views[i % 4], view_sha256=canonical_digest(i)) for i in range(400))
    plan = EncoderPilotPlan(views, 'production-plan')
    assert plan.horizon == 2400 and plan.checkpoint_steps == (0, 1200, 2400)
    for step in (0, 1, 5, 6, 1200, 2400):
        sharing, stream = build_pilot_state(plan, step)
        assert sum(sharing['completed_per_site'].values()) == step
        assert stream['offset'] == step * 16
        assert max(sharing['completed_per_site'].values()) - min(sharing['completed_per_site'].values()) <= 1
    assert build_pilot_state(plan, 2400)[0]['completed_per_site'] == dict.fromkeys(SITES, 400)
    assert [plan.update_at(i).site for i in range(12)] == list(SITES) * 2
    assert plan.update_at(2399).lr_factor > 0
    with pytest.raises(ValueError, match='horizon'):
        plan.update_at(2400)
    with pytest.raises(ValueError, match='original six'):
        EncoderPilotPlan(views, 'x', trained_sites=tuple(reversed(SITES)))

def test_long_schedule_balances_same_stream_at_all_retained_states():
    _, small, _, _ = fixture()
    views = tuple(replace(small.views[i % 4], view_sha256=canonical_digest(i)) for i in range(6400))
    plan = EncoderPilotPlan(views, 'production-long')
    assert plan.horizon == 38400
    assert plan.checkpoint_steps == (0, 9600, 19200, 28800, 38400)
    assert plan.objective_steps == (0, 19200, 38400)
    assert plan.update_at(1920).lr_factor == 1
    assert plan.update_at(38399).lr_factor > 0
    assert [plan.update_at(i).site for i in range(12)] == list(SITES) * 2
    for step in plan.checkpoint_steps:
        sharing, stream = build_pilot_state(plan, step)
        assert sharing['completed_per_site'] == dict.fromkeys(SITES, step // 6)
        assert stream['offset'] == step * 16
        assert sharing['source_valid_per_site'] == dict.fromkeys(
            SITES, sum(v.source_tokens for v in views[:step // 6]))
    with pytest.raises(ValueError, match='horizon'):
        plan.update_at(38400)
    with pytest.raises(ValueError, match='original six'):
        EncoderPilotPlan(views[:-1], 'unapproved-count')

def test_readiness_ties_use_larger_companion_length():
    batches = [{'input_ids': torch.empty((16, source)), 'labels': torch.empty((16, target))}
               for source, target in [(684, 53), (684, 63), (544, 96), (600, 96)]]
    assert readiness_view_indices(batches) == {1, 3}

def test_embedding_insertion_is_actual_first_layer_input_and_restores():
    model, _, batches, _ = fixture()
    site = resolve_encoder_site(model.base, SITE_SPECS['enc_emb'], 'tiny-cpu')
    assert site.site_id == 'enc_emb' and site.module_path == 'encoder.layers.0.input'
    captured = []
    def capture(module, args, kwargs):
        captured.append((args[0] if args else kwargs['hidden_states']).detach().clone())
    first = model.base.get_encoder().text_model.layers[0]
    handle = first.register_forward_pre_hook(capture, with_kwargs=True, prepend=True)
    original = dict(model.split)
    try:
        with model.at_site(site, purpose='train', authorization={'sites': {'enc_emb': 'trained'}}):
            batch = batches[0]
            with model.transmission(None, encoder_mask=batch['attention_mask']):
                model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            assert model.activation is not None
            assert torch.equal(model.activation, captured[-1])
            raise RuntimeError('exercise finally')
    except RuntimeError as error:
        assert str(error) == 'exercise finally'
    finally:
        handle.remove()
    assert model.split == original and not model._routing_active
    with pytest.raises(ValueError, match='no layer index'):
        resolve_encoder_site(model.base, {**SITE_SPECS['enc_emb'], 'index': 0}, 'tiny-cpu')

@pytest.mark.parametrize('mutation', [
    lambda p: p['metadata']['sharing']['completed_per_site'].update(enc_emb=0),
    lambda p: p['metadata']['sharing'].update(next_site='enc_fn'),
    lambda p: p['metadata']['evaluation_sites']['enc_l14'].update(role='heldout_after_freeze'),
    lambda p: p['metadata']['evaluation_sites']['enc_emb']['site'].update(module_path='encoder.embed_tokens'),
    lambda p: p['metadata']['sharing'].update(schema='sharing-state-v1'),
])
def test_saved_pilot_drift_rejected(tmp_path, mutation):
    from dataclasses import asdict
    from jscc.models.split_model import resolve_encoder_site
    model, plan, batches, metadata = fixture()
    sharing, stream = build_pilot_state(plan, 1)
    metadata = {**metadata, 'completed_updates': 1, 'sharing': sharing,
        'evaluation_sites': {name: {'site': asdict(resolve_encoder_site(model.base, SITE_SPECS[name], 'tiny-cpu')),
                                   'role': 'trained'} for name in SITES}}
    payload = {'schema': 'experiment-state-v2', 'kind': 'shared_encoder_v2',
               'metadata': metadata, 'stream': stream, 'scheduler': {'last_epoch': 1}}
    validate_pilot_state(payload)
    damaged = deepcopy(payload)
    mutation(damaged)
    with pytest.raises(ValueError):
        validate_pilot_state(damaged)

def test_full_validation_scope_rejects_subset_or_extra_condition():
    from jscc.checkpoint_readers import validate_task_template
    template = task_template()
    template.update(expected_items=10042, input_ids=list(range(10042)),
                    source_family_ids=[f'family-{i}' for i in range(10042)], conditions=['no_noise'])
    template['settings'].update(num_samples=10042, batch_size=16, num_fewshot=5, scoring_policy='fp32-v1')
    config = {'model': {'dtype': 'float32', 'attn_implementation': 'eager'}}
    validate_task_template(template, config, synthetic=False, final=True, no_noise_only=True)
    assert final_panels(6400) == tuple((site, 'no_noise') for site in SITES)
    assert len(final_panels(400)) == 36
    with pytest.raises(ValueError, match='conditions differ'):
        validate_task_template({**template, 'conditions': ['no_noise', 6]}, config,
                               synthetic=False, final=True, no_noise_only=True)
    with pytest.raises(ValueError, match='full ordered'):
        validate_task_template({**template, 'input_ids': list(reversed(template['input_ids']))}, config,
                               synthetic=False, final=True, no_noise_only=True)
    with pytest.raises(ValueError, match='fixed HellaSwag panel'):
        validate_task_template({**template, 'expected_items': 256}, config,
                               synthetic=False, final=True, no_noise_only=True)


def test_formal_layer14_still_requires_freeze():
    model, _, _, _ = fixture()
    site = resolve_encoder_site(model.base, SITE_SPECS['enc_l14'], 'tiny-cpu')
    with pytest.raises(ValueError, match='freeze'):
        with model.at_site(site, purpose='evaluate', authorization={'sites': {'enc_l14':'heldout'}}):
            pass


def task_template():
    return {'task': 'hellaswag', 'input_ids': [80, 81], 'source_family_ids': ['dev80', 'dev81'],
            'prompt_policy': 'synthetic-native', 'noise': {'seed': 0, 'namespace': 'same-eval'},
            'layout': 'tiny-2x3', 'backend': 'eager', 'precision': 'fp32', 'scorer': 'fixture',
            'reference_corpus': 'fixture-dev', 'expected_items': 2,
            'settings': {'num_samples': 2, 'numerical_policy': 'native'}, 'data_settings': {},
            'conditions': ['no_noise', -6, 0, 6, 12, 18]}

def native_batches():
    result = []
    for i, original in enumerate(sharing_batches()):
        result.append({key: value.repeat(8, 1) for key, value in original.items()
                       if key in ('input_ids', 'attention_mask', 'labels') and torch.is_tensor(value)})
        result[-1]['row_ids'] = torch.arange(i * 16, (i + 1) * 16)
    return result

def test_readiness_discards_cycles_and_restores_exact_initial_state(tmp_path):
    from jscc.shared_state import recipe_observation_snapshot, restore_recipe_observation
    from jscc.qualification import state_exact
    from shared.test_panel_evaluation import current_stages, update
    machine, data = current_stages(tmp_path/'setup')
    initial = recipe_observation_snapshot(machine)
    for _ in range(2):
        sites = []
        for _ in SITES:
            sites.append(update(machine,data)['payload']['site_id'])
        assert sites == list(SITES)
        restore_recipe_observation(machine,initial)
        assert state_exact(recipe_observation_snapshot(machine),initial)
    assert machine.learner.completed == machine.learner.exposure.completed == 0
    assert all(torch.equal(machine.learner.model.codec.state_dict()[k], v) for k,v in initial['weights'].items())


def test_deadline_cannot_publish_training_or_final_success(tmp_path):
    from shared.test_panel_evaluation import current_stages
    from jscc.shared_training import run_recipe_training
    machine,data = current_stages(tmp_path/'setup')
    model=machine.learner.model
    model.memory_codec=None
    model.universal_codec=False
    def expired():
        raise TimeoutError('test deadline')
    with pytest.raises(TimeoutError):
        run_recipe_training(model,None,plan=machine.plan,batches=[data]*machine.plan.q,
            metadata=machine.metadata,output=tmp_path/'late',guard=expired,synthetic=True,
            observe=lambda **kw:None)
    assert json.loads((tmp_path/'late/campaign.json').read_text())['status']=='incomplete'
    assert not (tmp_path/'late/freeze.json').exists()
    assert not torch.cuda.is_initialized()


def test_no_noise_only_lifecycle_and_evaluation_failure_preserves_freeze(tmp_path, monkeypatch):
    from shared.test_panel_evaluation import current_stages
    from shared.test_shared_case_evaluation import template, score_stub
    from jscc.shared_training import run_recipe_training
    from jscc import panel_evaluation as panels
    from jscc.activation_replay import canonical_digest
    machine,data=current_stages(tmp_path/'setup',conditions=['no_noise'])
    model=machine.learner.model
    model.memory_codec=None
    model.universal_codec=False
    output=tmp_path/'train'
    def no_score(*args,**kwargs):
        return {'status':'complete','request_identity':canonical_digest(kwargs['request'])}
    result=run_recipe_training(model,None,plan=machine.plan,batches=[data]*machine.plan.q,
        metadata=machine.metadata,output=output,guard=lambda:None,synthetic=True,observe=no_score)
    assert result['status']=='training_frozen'
    before=(output/'freeze.json').read_bytes()
    calls=[]
    def score(*args,**kwargs):
        calls.append(kwargs['condition'])
        return score_stub(*args,**kwargs)
    monkeypatch.setattr(panels,'_score_panel',score)
    request=panels.terminal_recipe_request(output,template(2))
    receipt=panels.run_recipe_evaluation(model,None,request=request,output=tmp_path/'evaluate',guard=lambda:None)
    assert receipt['status']=='complete' and receipt['completed_panels']==6
    assert calls==['no_noise']*6
    assert not (tmp_path/'evaluate/vanilla').exists()
    def expired():
        raise TimeoutError('evaluation deadline')
    with pytest.raises(TimeoutError):
        panels.run_recipe_evaluation(model,None,request=request,output=tmp_path/'failed-evaluate',guard=expired)
    assert json.loads((tmp_path/'failed-evaluate/result.json').read_text())['status']=='incomplete'
    assert (output/'freeze.json').read_bytes()==before
