"""YAML-selected real tiny model updates through the owned runtime seam."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
import yaml

from jscc.experiment_recipe import resolve_recipe
from jscc.config import load_config
from jscc.recipe_plan import RecipePlan
from jscc.original13_routing import create_recipe_learner, build_recipe_scheduler, export_codec
from jscc.models.channel import AWGNChannel, IdentityChannel
from jscc.models.codec import Codec
from jscc.models.split_model import SplitModel, resolve_training_site, resolve_original13_site
from jscc.sharing_protocol import batch_view
from jscc.baseline_protocol import bind_batch_identity
from model_helpers import tiny_backbone
from test_original13_routes import batch

ROOT = Path(__file__).resolve().parents[1]


def setup_recipe(tmp_path, *, locations=None, q=8, width=8, ratio=50, channel=None, retain_defaults=False):
    torch.set_num_threads(1)
    torch.manual_seed(41)
    recipe = yaml.safe_load((ROOT/'configs/recipes/original13.yaml').read_text())
    task_config = load_config((ROOT/'configs/recipes'/recipe['task_config']).resolve())
    task_config['codec']['hidden_dim'] = 16
    task_config['model']['revision'] = 'tiny-original13'
    task = tmp_path/'task.yaml'
    task.write_text(yaml.safe_dump(task_config))
    recipe.update(task_config=str(task),
                  updates_per_location=q, widths=[width], reconstruction_percentages=[ratio])
    if locations is not None:
        recipe['locations'] = locations
    if not retain_defaults:
        recipe['save']['functional_observation_fraction'] = .5
        recipe['overrides']['training'].update(warmup_ratio=0., min_lr_ratio=.2, lr=.001,
                                             weight_decay=.02, betas=[.8, .95], eps=1e-7, temperature=.8)
        recipe['objectives'] = {'encoder': {'hidden': .2, 'memory': 0.},
                                'decoder': {'hidden': .3, 'memory': .4}, 'functional_kl': .7}
        recipe['overrides']['channel'] = channel or {'train_snr_range': [3., 3.]}
    path = tmp_path/'recipe.yaml'
    path.write_text(yaml.safe_dump(recipe))
    metadata_path = tmp_path/'config.json'
    metadata_path.write_text(json.dumps({'encoder': {'text_config': {'num_hidden_layers': 26, 'hidden_size': 16}},
                                         'decoder': {'num_hidden_layers': 26, 'hidden_size': 16}}))
    description = resolve_recipe(path, model_config_path=metadata_path)['trajectories'][0]
    # Repeated actual content is allowed at distinct indexed stream positions.
    data = batch(16)
    plan = RecipePlan(description, (batch_view(data),)*q, 'paired')
    config = description['config']
    codec_config = deepcopy(config['codec'])
    codec_config['hidden_dim'] = 16
    model = SplitModel(tiny_backbone(26), Codec(16, codec_config),
        AWGNChannel() if config['channel']['type'] == 'awgn' else IdentityChannel(),
        {'stack': 'enc', 'where': 'after_final_norm'}, config['channel'])
    metadata = {'source_identity': 'tiny-source', 'config_identity': plan.identity,
                'data_identity': 'tiny-data', 'model_revision': 'tiny-original13'}
    events = []
    learner = create_recipe_learner(model, plan=plan, metadata=metadata, event_sink=events.append, synthetic=True)
    return learner, data, events


def reset_fixture(learner):
    """Minimal state-owner hook fixture, not a checkpoint/resume implementation."""
    learner.phase_start_step = learner.completed
    training = learner.schedule.description['config']['training']
    learner.optimizer = torch.optim.AdamW(learner.parameters, lr=training['lr'],
        weight_decay=training['weight_decay'], betas=tuple(training['betas']), eps=training['eps'])
    learner.scheduler = build_recipe_scheduler(learner.optimizer, learner.schedule, learner.phase_start_step)


@pytest.mark.parametrize('locations,q,width,ratio', [
    (None, 40, 8, 50),
    (['dec_l3', 'enc_l2', 'enc_emb'], 16, 4, 75),
    (['enc_fn', 'dec_l8'], 8, 12, 50),
    (['dec_l8', 'enc_fn'], 8, 12, 50),
])
def test_yaml_changes_train_ordered_complete_views_and_durable_counts(tmp_path, locations, q, width, ratio):
    learner, data, events = setup_recipe(tmp_path, locations=locations, q=q, width=width, ratio=ratio,
                                         retain_defaults=locations is None)
    initial = export_codec(learner.model)
    plan = learner.schedule
    training = plan.description['config']['training']
    low, high = plan.description['config']['channel']['train_snr_range']
    binding = bind_batch_identity(data)
    for index in range(plan.horizon):
        stage = plan.stage_at(index)
        if stage.start != learner.phase_start_step:
            with pytest.raises(ValueError, match='reset'):
                learner.update([data], batch_identities=[binding])
            reset_fixture(learner)
        event = learner.update([data], batch_identities=[binding])
        p = event['payload']
        assert p['site_id'] == plan.trained_sites[index % len(plan.trained_sites)]
        assert p['site_step'] == index // len(plan.trained_sites) + 1
        assert p['objective']['resolved_weights'] == plan.objective_at(index)
        assert p['lr_used'] == pytest.approx([training['lr'] * plan.update_at(index).lr_factor])
        assert low <= p['snr']['min']['value'] <= p['snr']['max']['value'] <= high
        assert p['exposure']['latent_width'] == width
        assert p['objective']['components']['hidden']['denominator'] == 16
        assert p['objective']['components']['memory']['denominator'] == (16 if p['site_id'].startswith('dec_') else 0)
        assert p['channel'] == plan.description['config']['channel']
    counts = learner.exposure.finalize()
    assert all(c['updates'] == q and c['sequences'] == 16*q for c in counts.values())
    assert sum(c['sequences'] for c in counts.values()) == plan.description['counts']['site_presentations']
    assert events[-1]['payload']['configured_counts'] == plan.description['counts']
    assert events[-1]['payload']['recipe_progress']['base_stream_presentations'] == q * 16
    assert events[-1]['payload']['recipe_progress']['site_presentations'] == plan.horizon * 16
    assert events[-1]['payload']['recipe_progress']['next_view'] == q
    assert events[-1]['payload']['recipe_progress']['next_site'] is None
    assert events[-1]['payload']['recipe_progress']['completed_per_site'] == dict.fromkeys(plan.trained_sites, q)
    receipt = tmp_path/'updates.json'
    receipt.write_text(json.dumps(events))
    assert len(json.loads(receipt.read_text())) == len(learner.exposure.receipts) == plan.horizon
    assert any(not torch.equal(initial[k], v) for k, v in export_codec(learner.model).items())
    assert all(p.grad is None for p in learner.model.base.parameters())
    assert learner.model.memory_codec is learner.model.codec
    assert learner.optimizer.param_groups[0]['betas'] == tuple(training['betas'])
    assert learner.optimizer.param_groups[0]['eps'] == training['eps']
    with pytest.raises(RuntimeError, match='terminal'):
        learner.update([data])


@pytest.mark.parametrize('channel', [
    {'type': 'awgn', 'train_noise': False, 'normalize_power': False},
    {'type': 'identity', 'train_noise': True, 'normalize_power': True},
])
def test_channel_yaml_reaches_forward_and_receipts(tmp_path, channel, monkeypatch):
    learner, data, _ = setup_recipe(tmp_path, locations=['dec_l3', 'enc_l2'], channel=channel)
    seen = []
    original = learner.model.channel.forward
    def capture(z, snr):
        seen.append(snr)
        return original(z, snr)
    monkeypatch.setattr(learner.model.channel, 'forward', capture)
    event = learner.update([data])
    assert seen and all(snr is None for snr in seen)
    assert event['payload']['snr']['condition'] == 'no_noise'
    assert event['payload']['channel']['normalize_power'] == channel['normalize_power']
    assert learner.last_draws[0]['draws'] == []


@pytest.mark.parametrize('site', ['enc_l26', 'dec_l25', 'dec_l999', 'enc_l01', 'other'])
def test_loaded_model_rejects_invalid_boundaries(site):
    with pytest.raises(ValueError):
        resolve_training_site(tiny_backbone(26), site, 'tiny-original13')


def test_legacy_cohort_guard_is_retained():
    with pytest.raises(ValueError, match='cohort'):
        resolve_original13_site(tiny_backbone(26), 'dec_l3', 'tiny-original13')


def test_runtime_rechecks_recipe_locations_on_loaded_model(tmp_path):
    learner, data, _ = setup_recipe(tmp_path, locations=['dec_l24', 'enc_l19'])
    smaller = deepcopy(learner.model)
    smaller.universal_codec = False
    smaller.memory_codec = None
    smaller.base = tiny_backbone(4).requires_grad_(False).eval()
    with pytest.raises(ValueError, match='suffix'):
        create_recipe_learner(smaller, plan=learner.schedule,
            metadata={'source_identity': 's', 'config_identity': 'c', 'data_identity': 'd',
                      'model_revision': 'tiny-original13'}, synthetic=True)


def test_runtime_requires_recipe_model_revision(tmp_path):
    learner, _, _ = setup_recipe(tmp_path, locations=['enc_fn'])
    with pytest.raises(ValueError, match='model revision'):
        create_recipe_learner(learner.model, plan=learner.schedule,
            metadata={'source_identity': 's', 'config_identity': 'c', 'data_identity': 'd',
                      'model_revision': 'different'}, synthetic=True)
