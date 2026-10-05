"""Current YAML stages retain optimizer-reset, exposure, noise and terminal science."""

from copy import deepcopy
import pytest
import torch
import yaml
from jscc.activation_replay import canonical_digest
from jscc.asset_identity import state_dict_identity
from jscc.checkpoint_readers import PILOT_SITES as SITES, EncoderStagesPlan, ENCODER_STAGES_PROTOCOL
from jscc.shared_config import resolve_recipe
from jscc.shared_plan import RecipePlan
from jscc.shared_inputs import batch_view, bind_batch_identity
from jscc.shared_training import create_recipe_learner, run_recipe_training
from jscc.shared_state import RecipeTrainingState
from jscc import panel_evaluation as panels
from jscc.observation_policy import recipe_observation_identity
from jscc import site_routing
from shared.test_encoder_shared_pilot import fixture
from shared.test_shared_case_evaluation import template, score_stub
from shared.test_recipe_training import setup_recipe


CONDITIONS = ['no_noise', -6, 0, 6, 18]


def stages_fixture(arm):
    model, old, batches, metadata = fixture()
    plan = EncoderStagesPlan(tuple(batch_view(b) for b in batches), old.study_pairing_id,
        arm, synthetic=True, width=int(model.codec.config['bottleneck_dim']))
    metadata.update(protocol_identity=ENCODER_STAGES_PROTOCOL, phase='phase1')
    return model, plan, batches, metadata


def current_stages(path, hidden=.2, conditions=CONDITIONS):
    path.mkdir(parents=True, exist_ok=True)
    initial, data, _ = setup_recipe(path, locations=list(SITES), q=8)
    recipe = yaml.safe_load((path/'recipe.yaml').read_text())
    recipe['overrides']['training'].update(warmup_ratio=.1, min_lr_ratio=0.)
    recipe['objectives']['encoder']['hidden'] = hidden
    recipe['evaluation'].update(items=2, conditions=conditions, supplemental_conditions=[], fewshot=5)
    (path/'recipe.yaml').write_text(yaml.safe_dump(recipe))
    description = resolve_recipe(path/'recipe.yaml', model_config_path=path/'config.json')['trajectories'][0]
    plan = RecipePlan(description, initial.schedule.views, initial.schedule.study_pairing_id)
    model = initial.model
    initialization = state_dict_identity(site_routing.export_codec(model))
    model.memory_codec = None
    model.universal_codec = False
    metadata = {'source_identity':'tiny-source', 'config_identity':canonical_digest(description['config']),
        'data_identity':'tiny-data','prepared_identity':'tiny-prepared',
        'initialization_identity':initialization,
        'observation_identity':recipe_observation_identity(description, template(256)), 'model_revision':'tiny-original13'}
    learner = create_recipe_learner(model, plan=plan, metadata=metadata, synthetic=True)
    return RecipeTrainingState(learner, plan=plan, metadata=metadata), data


def update(machine, data):
    return machine.learner.update([data], batch_identities=[bind_batch_identity(data)])


def test_reset_retains_global_stream_and_only_resets_optimizer(tmp_path):
    machine, data = current_stages(tmp_path/'setup')
    learner, plan, model = machine.learner, machine.plan, machine.learner.model
    for _ in range(plan.stages[1].start):
        update(machine, data)
    state = machine.save(tmp_path/'phase1.pt')
    codec = state_dict_identity(model.codec.state_dict())
    progress = (learner.completed, learner.attempted, learner.offset, learner.valid_tokens)
    exposure = deepcopy(learner.exposure.per_site)
    rng = torch.get_rng_state().clone()
    next_update = plan.update_at(learner.completed)
    machine.reset(state)
    assert state_dict_identity(model.codec.state_dict()) == codec
    assert (learner.completed, learner.attempted, learner.offset, learner.valid_tokens) == progress
    assert learner.exposure.per_site == exposure and torch.equal(torch.get_rng_state(), rng)
    assert learner.optimizer.state == {} and learner.scheduler.last_epoch == 0
    assert learner.optimizer.param_groups[0]['lr'] == 0
    reset = machine.save(tmp_path/'reset.pt')
    assert reset.payload['metadata']['parent_identity'] == state.reference.sha256
    assert reset.payload['metadata']['lineage'][-1]['parent_checkpoint']['sha256'] == state.reference.sha256
    assert plan.update_at(learner.completed) == next_update
    update(machine, data)
    assert learner.completed == plan.stages[1].start+1 and learner.scheduler.last_epoch == 1
    with pytest.raises(ValueError, match='exactly one'):
        machine.reset(state)


def test_e_r_phase_two_matches_actual_noise_draws(tmp_path):
    draws = []
    for hidden in (.1,.2):
        machine, data = current_stages(tmp_path/str(hidden), hidden)
        for _ in range(machine.plan.stages[1].start):
            update(machine, data)
        saved = machine.save(tmp_path/f'{hidden}.pt')
        machine.reset(saved)
        update(machine, data)
        draws.append(machine.learner.last_draws)
    assert draws[0] == draws[1]


def test_terminal_thirty_panels_use_one_frozen_staged_checkpoint(tmp_path, monkeypatch):
    machine, data = current_stages(tmp_path/'setup')
    model, plan = machine.learner.model, machine.plan
    model.memory_codec = None
    model.universal_codec = False
    def no_score(*args, **kwargs):
        return {'status':'complete','request_identity':canonical_digest(kwargs['request'])}
    result = run_recipe_training(model, None, plan=plan, batches=[data]*plan.q,
        metadata=machine.metadata, output=tmp_path/'train', guard=lambda:None, synthetic=True, observe=no_score)
    assert result['status'] == 'training_frozen'
    state, _ = panels.open_recipe_terminal(tmp_path/'train')
    assert state.payload['stream']['completed_per_site'] == dict.fromkeys(SITES,8)
    calls, checkpoints = [], []
    def score(*args, **kwargs):
        calls.append((kwargs['site_id'],kwargs['condition']))
        checkpoints.append(kwargs['state'].reference.sha256)
        return score_stub(*args, **kwargs)
    monkeypatch.setattr(panels,'_score_panel',score)
    request = panels.terminal_recipe_request(tmp_path/'train',template(2))
    receipt = panels.run_recipe_evaluation(model,None,request=request,output=tmp_path/'evaluate',guard=lambda:None)
    assert receipt['status'] == 'complete' and len(calls) == 30
    assert [p['condition'] for p in receipt['panels']] == [c for c in CONDITIONS for _ in range(6)]
    assert set(checkpoints) == {state.reference.sha256}
    assert receipt['checkpoint']['sha256'] == state.reference.sha256
    assert all(p['completed_items'] == 2 for p in receipt['panels'])
