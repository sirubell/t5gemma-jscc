"""Real tiny CPU recipe continuation; no target or scientific acceptance."""

from jscc import shared_state as _body_shared_state
from copy import deepcopy
from dataclasses import asdict
from typing import Any
import hashlib
import io
import json

import numpy as np
import pytest
import torch
from torch.amp.grad_scaler import GradScaler

from jscc.activation_replay import canonical_digest
from jscc.experiment_state import CheckpointRef, capture_rng, restore_rng
from jscc.site_routing import export_codec
from jscc.shared_state import RecipeTrainingState, recipe_runtime_state, recipe_observation_snapshot, restore_recipe_observation
from jscc.shared_training import run_recipe_training
from jscc.asset_identity import state_dict_identity
from jscc.qualification import state_exact
from jscc.shared_inputs import bind_batch_identity
from jscc.panel_evaluation import ObservationScoringError
from shared.test_recipe_training import setup_recipe


def setup(tmp_path, channel=None, *, ratio=50):
    tmp_path.mkdir(parents=True, exist_ok=True)
    learner, batch, events = setup_recipe(tmp_path, locations=['dec_l3', 'enc_l2'], channel=channel, ratio=ratio)
    plan = learner.schedule
    metadata = {**learner.identity} if hasattr(learner, 'identity') else {}
    metadata.update(source_identity='tiny-source', config_identity=canonical_digest(plan.description['config']),
        data_identity=canonical_digest({'rows': 'tiny'}), prepared_identity='indexed-tiny',
        observation_identity=canonical_digest(plan.description['observations']),
        initialization_identity=state_dict_identity(export_codec(learner.model)), model_revision='tiny-original13')
    learner.identity.update(config=metadata['config_identity'], data=metadata['data_identity'])
    state = RecipeTrainingState(learner, plan=plan, metadata=metadata)
    return state, batch, events


def advance(state, batch, target):
    binding = bind_batch_identity(batch)
    while state.learner.completed < target:
        before = [g['lr'] for g in state.learner.optimizer.param_groups]
        expected = state.plan.update_at(state.learner.completed)
        event = state.learner.update([batch], batch_identities=[binding])
        assert before == event['payload']['lr_used']
        assert before == [state.plan.description['config']['training']['lr'] * expected.lr_factor]


def compare_state(state):
    value = recipe_observation_snapshot(state)
    value.pop('ancestry')
    value['exposure'] = value['exposure'][:3]  # fresh receipts bind wall-clock events
    value.pop('reset_parent')  # equivalent weight-reset parents may have different byte paths
    return value


@pytest.mark.parametrize('channel', [None, {'type': 'identity', 'train_noise': True, 'normalize_power': True}])
@pytest.mark.parametrize('position', ['reconstruction', 'pre_reset', 'post_reset', 'functional'])
def test_real_tiny_exact_continuation(tmp_path, channel, position):
    state, batch, events = setup(tmp_path/'first', channel)
    target = 3 if position == 'reconstruction' else state.plan.stages[1].start
    advance(state, batch, target)
    if position in ('post_reset', 'functional'):
        parent = state.save(tmp_path/'parent.pt')
        weights, exposure = deepcopy(export_codec(state.learner.model)), deepcopy(state.learner.exposure.per_site)
        state.reset(parent)
        assert state_exact(weights, export_codec(state.learner.model))
        assert state.learner.exposure.per_site == exposure
        assert not state.learner.optimizer.state
        with pytest.raises(ValueError, match='exactly one'):
            state.reset(parent)
        if position == 'functional':
            advance(state, batch, target+3)
    saved = state.save(tmp_path/'saved.pt')
    if position == 'pre_reset':
        state.reset(saved)
    initial = state.learner.completed
    advance(state, batch, initial+2)
    expected = compare_state(state)
    next_events = deepcopy(events[-2:])
    resumed, _, resumed_events = setup(tmp_path/'second', channel)
    resumed.restore(saved.reference)
    assert resumed.learner.completed == initial
    assert resumed.ancestry == [asdict(saved.reference)]
    assert resumed.learner.exposure.receipts == saved.payload['metadata']['exposure_receipts']
    if position == 'pre_reset':
        resumed.reset(saved)
    advance(resumed, batch, initial+2)
    assert state_exact(compare_state(resumed), expected)
    for actual, uninterrupted in zip(resumed_events, next_events):
        for key in ('completed_step', 'site_id', 'site_step', 'lr_used', 'lr_next', 'recipe_progress', 'objective', 'snr', 'audit'):
            assert state_exact(actual['payload'][key], uninterrupted['payload'][key])
    resumed.save(tmp_path/'continued.pt')
    (tmp_path/'continuation.json').write_text(json.dumps({'position': position, 'channel': channel,
        'saved': asdict(saved.reference), 'completed': resumed.learner.completed,
        'ancestry': resumed.ancestry, 'events': resumed_events}, default=str))


def rewrite(tmp_path, saved, mutate):
    payload = deepcopy(saved.payload)
    mutate(payload)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    data = buffer.getvalue()
    path = tmp_path/'bad.pt'
    path.write_bytes(data)
    return CheckpointRef(str(path), hashlib.sha256(data).hexdigest(), len(data))


@pytest.mark.parametrize('mutation', [
    lambda p: p['rng'].pop('python'),
    lambda p: p['rng'].update(python=('bad',)),
    lambda p: p['metadata'].update(noise_identity='wrong'),
    lambda p: p['metadata'].update(config_identity='wrong'),
    lambda p: p['stream'].update(next_view=7),
    lambda p: p['metadata']['exposure_receipts'].append('replay'),
    lambda p: p['metadata']['route_state']['idle']['modes'].update(codec='bad'),
    lambda p: p['optimizer']['param_groups'][0].update(weight_decay=99),
    lambda p: p['model'][next(iter(p['model']))].fill_(float('nan')),
    lambda p: p['optimizer']['state'][0]['exp_avg'].fill_(float('inf')),
    lambda p: p.update(scaler=None),
])
def test_incompatible_incomplete_rejected_before_training_mutation(tmp_path, mutation):
    state, batch, _ = setup(tmp_path/'setup')
    advance(state, batch, 3)
    saved = state.save(tmp_path/'valid.pt')
    invalid = rewrite(tmp_path, saved, mutation)
    before = recipe_observation_snapshot(state)
    optimizer = state.learner.optimizer
    with pytest.raises((ValueError, TypeError)):
        state.restore(invalid)
    assert state.learner.optimizer is optimizer
    assert state_exact(recipe_observation_snapshot(state), before)


def complete_observer(model, processor, *, state, request, output, guard):
    torch.rand(3)
    np.random.rand(3)
    return {'status': 'complete', 'request_identity': canonical_digest(request)}


def run(state, batch, output, **kwargs):
    state.learner.model.memory_codec = None
    state.learner.model.universal_codec = False
    return run_recipe_training(state.learner.model, None, plan=state.plan,
        batches=[batch]*state.plan.q, metadata=state.metadata, output=output, guard=lambda: None,
        data_ids={'rows': 'tiny'}, synthetic=True, observe=complete_observer, **kwargs)


def test_runner_yaml_saves_reset_and_prior_segment_obligations(tmp_path):
    state, batch, _ = setup(tmp_path/'full')
    result = run(state, batch, tmp_path/'uninterrupted')
    expected = deepcopy(export_codec(state.learner.model))
    assert result['status'] == 'training_frozen'
    assert result['completed_updates'] == result['segment_updates'] == state.plan.horizon
    assert [(r['step'], r['state']) for r in result['observations']] == [
        (r['step'], r['state']) for r in state.plan.description['observations']]
    assert len(result['checkpoints']) == len(state.plan.description['checkpoint_states'])
    assert result['observations_status'] == 'complete'
    pre = next(c for c in result['checkpoints'] if c['role'] == 'pre_reset')
    resumed, _, _ = setup(tmp_path/'resume-setup')
    reference = CheckpointRef(**{k: pre[k] for k in ('path', 'sha256', 'size')})
    continued = run(resumed, batch, tmp_path/'continued', resume=reference)
    assert continued['segment_updates'] == state.plan.horizon-state.plan.stages[1].start
    assert state_exact(export_codec(resumed.learner.model), expected)
    assert continued['observations'][0]['status'] == 'pending_prior_segment'
    assert continued['observations_status'] == 'pending'
    assert sum(c['role'] == 'post_reset' for c in continued['checkpoints']) == 1
    terminal = next(c for c in continued['checkpoints'] if c['role'] == 'terminal')
    fresh, _, _ = setup(tmp_path/'terminal-setup')
    before = deepcopy(export_codec(fresh.learner.model))
    with pytest.raises(ValueError, match='requires evaluation'):
        run(fresh, batch, tmp_path/'terminal-resume',
            resume=CheckpointRef(**{k: terminal[k] for k in ('path', 'sha256', 'size')}))
    assert state_exact(export_codec(fresh.learner.model), before)


def test_observation_hook_restores_all_state_on_scoring_exception(tmp_path):
    state, batch, _ = setup(tmp_path/'setup', {'type': 'identity', 'normalize_power': True})
    advance(state, batch, 3)
    snapshot = recipe_observation_snapshot(state)
    with torch.no_grad():
        next(state.codec.parameters()).add_(1)
    torch.rand(9)
    state.learner.completed = 100
    state.learner.optimizer.param_groups[0]['lr'] = 2
    state.learner.model.codec.eval()
    restore_recipe_observation(state, snapshot)
    assert state_exact(recipe_observation_snapshot(state), snapshot)
    assert recipe_runtime_state(state.learner.model)['channel_type'] == 'identity'


def test_failed_save_nonfinite_and_checkpoint_damage_are_fatal(tmp_path, monkeypatch):
    import jscc.experiment_state as storage
    state, batch, _ = setup(tmp_path/'setup')
    advance(state, batch, 2)
    saved = state.save(tmp_path/'valid.pt')
    with pytest.raises(FileExistsError):
        state.save(tmp_path/'valid.pt')
    assert state.save(tmp_path/'second.pt').payload['metadata']['completed_updates'] == 2
    state.learner.failed = True
    with pytest.raises(ValueError, match='failed'):
        state.save(tmp_path/'failed.pt')
    state.learner.failed = False
    with torch.no_grad():
        next(state.codec.parameters()).fill_(float('nan'))
    with pytest.raises(ValueError, match='nonfinite'):
        state.save(tmp_path/'nonfinite.pt')
    assert not (tmp_path/'nonfinite.pt').exists()
    state.learner.model.codec.load_state_dict(saved.payload['model'])
    def fail(*args, **kwargs):
        raise OSError('injected save failure')
    monkeypatch.setattr(storage.os, 'link', fail)
    with pytest.raises(OSError, match='save failure'):
        state.save(tmp_path/'save-failure.pt')
    assert not (tmp_path/'save-failure.pt').exists()
    assert not list(tmp_path.glob('.incomplete-*'))


def test_runner_scoring_failure_restores_and_continues(tmp_path):
    state, batch, _ = setup(tmp_path/'setup')
    baseline = run(state, batch, tmp_path/'baseline')
    expected = deepcopy(export_codec(state.learner.model))
    resumed, _, _ = setup(tmp_path/'scoring-setup')
    def failing(model, processor, **kwargs):
        with torch.no_grad():
            next(model.codec.parameters()).add_(7)
        torch.rand(13)
        raise ObservationScoringError('injected isolated scorer failure')
    resumed.learner.model.memory_codec = None
    resumed.learner.model.universal_codec = False
    result = run_recipe_training(resumed.learner.model, None, plan=resumed.plan,
        batches=[batch]*resumed.plan.q, metadata=resumed.metadata, output=tmp_path/'scoring',
        guard=lambda: None, data_ids={'rows': 'tiny'}, synthetic=True, observe=failing)
    assert baseline['observations_status'] == 'complete'
    assert result['observations_status'] == 'pending'
    assert state_exact(export_codec(resumed.learner.model), expected)


def test_multiple_segments_preserve_ancestry(tmp_path):
    first, batch, _ = setup(tmp_path/'first')
    advance(first, batch, 3)
    a = first.save(tmp_path/'a.pt')
    second, _, _ = setup(tmp_path/'second')
    second.restore(a.reference)
    advance(second, batch, 5)
    b = second.save(tmp_path/'b.pt')
    third, _, _ = setup(tmp_path/'third')
    third.restore(b.reference)
    assert third.ancestry == [asdict(a.reference), asdict(b.reference)]
    assert third.learner.completed == len(third.learner.exposure.receipts) == 5
    assert third.learner.exposure.completed == 5
    assert state_exact(capture_rng(), b.payload['rng'])
    restore_rng(b.payload['rng'])


def configured_setup(tmp_path, training):
    import yaml
    from jscc.shared_config import resolve_recipe
    from jscc.shared_plan import RecipePlan
    from jscc.shared_training import create_recipe_learner
    initial, batch, _ = setup(tmp_path)
    recipe_path = tmp_path/'recipe.yaml'
    yaml_recipe = yaml.safe_load(recipe_path.read_text())
    yaml_recipe['overrides']['training'].update(training)
    recipe_path.write_text(yaml.safe_dump(yaml_recipe))
    description = resolve_recipe(recipe_path, model_config_path=tmp_path/'config.json')['trajectories'][0]
    plan = RecipePlan(description, initial.plan.views, initial.plan.study_pairing_id)
    model = initial.learner.model
    model.memory_codec = None
    model.universal_codec = False
    metadata = {k: v for k, v in initial.metadata.items() if k not in {'stream_identity', 'protocol_identity', 'noise_identity'}}
    metadata['config_identity'] = canonical_digest(description['config'])
    events = []
    learner = create_recipe_learner(model, plan=plan, metadata=metadata, event_sink=events.append, synthetic=True)
    return RecipeTrainingState(learner, plan=plan, metadata=metadata), batch, events


def test_changed_yaml_optimizer_warmup_decay_reach_real_both_phase_updates(tmp_path):
    outcomes = []
    variants: list[dict[str, Any]] = [
        {'lr': .001, 'weight_decay': .02, 'warmup_ratio': 0., 'min_lr_ratio': .2},
        {'lr': .003, 'weight_decay': .2, 'warmup_ratio': .5, 'min_lr_ratio': .6,
         'betas': [.7, .9], 'eps': 1e-6},
    ]
    for index, training in enumerate(variants):
        state, batch, events = configured_setup(tmp_path/str(index), training)
        boundary = state.plan.stages[1].start
        advance(state, batch, boundary)
        parent = state.save(tmp_path/f'pre-{index}.pt')
        reset_weights = deepcopy(export_codec(state.learner.model))
        reset_counts = deepcopy(state.learner.exposure.per_site)
        state.reset(parent)
        assert state_exact(reset_weights, export_codec(state.learner.model))
        assert reset_counts == state.learner.exposure.per_site
        advance(state, batch, state.plan.horizon)
        terminal = state.save(tmp_path/f'terminal-{index}.pt')
        assert terminal.payload['scheduler']['last_epoch'] == state.plan.horizon-boundary
        for event in events:
            step = event['payload']['completed_step']-1
            assert event['payload']['lr_used'] == [training['lr'] * state.plan.update_at(step).lr_factor]
        group = state.learner.optimizer.param_groups[0]
        assert group['weight_decay'] == training['weight_decay']
        assert group['betas'] == tuple(training.get('betas', [.8, .95]))
        assert group['eps'] == training.get('eps', 1e-7)
        outcomes.append((deepcopy(export_codec(state.learner.model)), [e['payload']['lr_used'] for e in events]))
    assert not state_exact(outcomes[0][0], outcomes[1][0])
    assert outcomes[0][1] != outcomes[1][1]
    assert outcomes[1][1][0] == outcomes[1][1][8] == [0.]
    assert outcomes[1][1][4] == outcomes[1][1][12] == [.003]


def test_indexed_data_resume_seeks_partial_location_cycle(tmp_path):
    from jscc.shared_plan import RecipePlan
    from jscc.shared_inputs import batch_view
    from jscc.shared_training import create_recipe_learner
    initial, batch, _ = setup(tmp_path/'setup')
    batches = []
    for index in range(initial.plan.q):
        item = deepcopy(batch)
        item['row_ids'] = torch.arange(16) + 16*index
        item['input_ids'][:, -1] = 3+index
        batches.append(item)
    plan = RecipePlan(initial.plan.description, [batch_view(b) for b in batches], 'paired')
    model = initial.learner.model
    model.memory_codec = None
    model.universal_codec = False
    metadata = {k: v for k, v in initial.metadata.items() if k not in {'stream_identity', 'protocol_identity', 'noise_identity'}}
    learner = create_recipe_learner(model, plan=plan, metadata=metadata, synthetic=True)
    state = RecipeTrainingState(learner, plan=plan, metadata=metadata)
    initial_weights = deepcopy(export_codec(model))
    for index in range(3):
        state.learner.update([batches[index//2]], batch_identities=[bind_batch_identity(batches[index//2])])
    saved = state.save(tmp_path/'partial.pt')
    expected = plan.progress(3, 0)
    assert expected['next_view'] == 1 and expected['next_site'] == 'enc_l2'
    reads = []
    class Indexed:
        def __len__(self):
            return len(batches)
        def __getitem__(self, index):
            reads.append(index)
            return deepcopy(batches[index])
    model.codec.load_state_dict(initial_weights)
    model.memory_codec = None
    model.universal_codec = False
    result = run_recipe_training(model, None, plan=plan, batches=Indexed(), metadata=metadata,
        output=tmp_path/'continued', guard=lambda: None, synthetic=True, data_ids={'rows': 'tiny'},
        observe=complete_observer, resume=saved.reference)
    assert reads == list(range(1, plan.q))
    assert result['segment_updates'] == plan.horizon-3
    rows = [json.loads(row) for row in (tmp_path/'continued/metrics.jsonl').read_text().splitlines()]
    assert rows[0]['payload']['site_id'] == 'enc_l2'
    assert rows[0]['payload']['site_step'] == 2
    assert rows[0]['schema'] == 'recipe-training-event-v1'
    assert len(rows) == plan.horizon-3


def test_real_nonfinite_gradient_stops_without_checkpoint(tmp_path):
    state, batch, _ = setup(tmp_path/'setup')
    handle = next(state.codec.parameters()).register_hook(lambda grad: torch.full_like(grad, float('inf')))
    with pytest.raises(FloatingPointError):
        advance(state, batch, 1)
    handle.remove()
    assert state.learner.failed and state.learner.completed == 0
    assert state.learner.attempted == 1
    assert state.learner.offset == 16
    with pytest.raises(ValueError, match='failed'):
        state.save(tmp_path/'bad.pt')
    with pytest.raises(RuntimeError, match='terminal'):
        state.learner.update([batch])
    assert not (tmp_path/'bad.pt').exists()


@pytest.mark.parametrize('mode', ['save_failure', 'checkpoint_damage', 'restoration_failure'])
def test_runner_fatal_failures_leave_incomplete_receipt(tmp_path, monkeypatch, mode):
    import jscc.experiment_state as storage
    state, batch, _ = setup(tmp_path/'setup')
    model = state.learner.model
    model.memory_codec = None
    model.universal_codec = False
    def fail(*args, **kwargs):
        raise OSError('injected save failure')
    if mode == 'save_failure':
        monkeypatch.setattr(storage.os, 'link', fail)
    if mode == 'restoration_failure':
        monkeypatch.setattr(_body_shared_state, 'restore_recipe_runtime', lambda *args: (_ for _ in ()).throw(
            ValueError('injected restoration failure')))
    def observe(model, processor, **kwargs):
        if mode == 'checkpoint_damage':
            from pathlib import Path
            Path(kwargs['state'].reference.path).write_bytes(b'corrupt')
        raise ObservationScoringError('scoring failure')
    with pytest.raises((ValueError, OSError)):
        run_recipe_training(model, None, plan=state.plan, batches=[batch]*state.plan.q,
            metadata=state.metadata, output=tmp_path/'run', guard=lambda: None,
            synthetic=True, data_ids={'rows': 'tiny'}, observe=observe)
    receipt = json.loads((tmp_path/'run/campaign.json').read_text())
    assert receipt['status'] == 'incomplete' and 'error' in receipt
    assert not (tmp_path/'run/freeze.json').exists()


def test_enabled_cpu_scaler_restores_same_next_updates(tmp_path):
    state, batch, _ = setup(tmp_path/'first')
    state.learner.scaler = GradScaler('cpu', enabled=True, init_scale=8., growth_interval=2)
    advance(state, batch, 3)
    saved = state.save(tmp_path/'scaled.pt')
    assert saved.payload['scaler']['scale'] == 16.
    advance(state, batch, 5)
    expected = compare_state(state)
    resumed, _, _ = setup(tmp_path/'second')
    resumed.learner.scaler = GradScaler('cpu', enabled=True)
    resumed.restore(saved.reference)
    assert resumed.learner.scaler.state_dict() == saved.payload['scaler']
    advance(resumed, batch, 5)
    assert state_exact(compare_state(resumed), expected)


@pytest.mark.parametrize('mutation', [
    lambda p: p.update(scaler={}),
    lambda p: p['scaler'].update(scale=float('nan')),
    lambda p: p['scaler'].update(growth_interval=0),
    lambda p: p['scaler'].update(_growth_tracker=-1),
    lambda p: p['scaler'].update(backoff_factor=2.),
    lambda p: p['optimizer']['state'][0].update(exp_avg=p['optimizer']['state'][0]['exp_avg'].to(torch.int64)),
    lambda p: p['optimizer']['state'][0].update(exp_avg_sq=p['optimizer']['state'][0]['exp_avg_sq'].to(torch.float64)),
])
def test_scaler_and_moment_incompatibility_rejected_before_mutation(tmp_path, mutation):
    source, batch, _ = setup(tmp_path/'source')
    source.learner.scaler = GradScaler('cpu', enabled=True, init_scale=8.)
    advance(source, batch, 3)
    saved = source.save(tmp_path/'source.pt')
    reference = rewrite(tmp_path, saved, mutation)
    destination, _, _ = setup(tmp_path/'destination')
    destination.learner.scaler = GradScaler('cpu', enabled=True)
    before = recipe_observation_snapshot(destination)
    optimizer = destination.learner.optimizer
    with pytest.raises(ValueError):
        destination.restore(reference)
    assert destination.learner.optimizer is optimizer
    assert state_exact(recipe_observation_snapshot(destination), before)


@pytest.mark.parametrize('source_enabled', [True, False])
def test_scaler_enabled_status_mismatch_rejected_before_mutation(tmp_path, source_enabled):
    source, batch, _ = setup(tmp_path/'source')
    source.learner.scaler = GradScaler('cpu', enabled=source_enabled)
    advance(source, batch, 3)
    saved = source.save(tmp_path/'source.pt')
    destination, _, _ = setup(tmp_path/'destination')
    destination.learner.scaler = GradScaler('cpu', enabled=not source_enabled)
    before = recipe_observation_snapshot(destination)
    with pytest.raises(ValueError, match='scaler enabled'):
        destination.restore(saved.reference)
    assert state_exact(recipe_observation_snapshot(destination), before)


@pytest.mark.parametrize('position', [0, 3, 16])
def test_r0_exact_continuation_and_terminal_finalize(tmp_path, position):
    from jscc.panel_evaluation import open_recipe_terminal
    state, batch, events = setup(tmp_path/'first', ratio=0)
    plan = state.plan
    assert len(plan.stages) == 1
    assert plan.stages[0].phase_id == 'functional'
    assert plan.stages[0].objective_kind == 'combined'
    assert not plan.stages[0].reset_optimizer
    assert plan.stages[0].horizon == plan.horizon
    assert all(role not in {'reconstruction', 'pre_reset', 'post_reset'}
               for _, role in plan.description['checkpoint_states'])
    assert all(row['state'] == 'functional_observation' for row in plan.description['observations'])
    advance(state, batch, position)
    saved = state.save(tmp_path/'saved.pt')
    assert saved.payload['metadata']['phase'] == 'functional'
    assert saved.payload['stream']['reset_count'] == 0
    assert saved.payload['metadata']['lineage'] == []
    with pytest.raises(ValueError, match='does not declare'):
        state.reset(saved)
    advance(state, batch, plan.horizon)
    expected = compare_state(state)
    next_events = deepcopy(events[position:])
    resumed, _, resumed_events = setup(tmp_path/'second', ratio=0)
    resumed.restore(saved.reference)
    advance(resumed, batch, plan.horizon)
    actual_state = compare_state(resumed)
    if position == plan.horizon:
        # Gradients are transient scratch state, absent from durable checkpoints.
        # Terminal restore performs no next update that would overwrite them.
        expected.pop('grads')
        actual_state.pop('grads')
    assert state_exact(actual_state, expected)
    for actual, uninterrupted in zip(resumed_events, next_events):
        for key in ('lr_used', 'lr_next', 'objective', 'snr', 'audit', 'recipe_progress'):
            assert state_exact(actual['payload'][key], uninterrupted['payload'][key])
    for index, event in enumerate(events):
        assert event['payload']['objective']['resolved_weights'] == plan.objective_at(index)
        assert plan.objective_at(index)['kl'] == .7
        assert event['payload']['lr_used'] == [.001 * plan.update_at(index).lr_factor]
    terminal = resumed.save(tmp_path/'terminal.pt')
    fresh, _, _ = setup(tmp_path/'final', ratio=0)
    result = run(fresh, batch, tmp_path/'finalized', resume=terminal.reference, finalize_terminal=True)
    assert result['segment_updates'] == 0
    endpoint, _ = open_recipe_terminal(tmp_path/'finalized')
    assert endpoint.payload['metadata']['phase_start_step'] == 0
    assert state_exact(endpoint.payload['model'], terminal.payload['model'])


def test_r0_runner_has_no_reset_and_initializer_resume(tmp_path):
    state, batch, _ = setup(tmp_path/'full', ratio=0)
    result = run(state, batch, tmp_path/'run')
    expected = deepcopy(export_codec(state.learner.model))
    assert result['observations_status'] == 'complete'
    assert not any('reset' in row['role'] or row['role'] == 'reconstruction' for row in result['checkpoints'])
    initial = next(row for row in result['checkpoints'] if row['role'] == 'initialization')
    fresh, _, _ = setup(tmp_path/'resume', ratio=0)
    continued = run(fresh, batch, tmp_path/'continued', resume=CheckpointRef(
        **{key: initial[key] for key in ('path', 'sha256', 'size')}))
    assert continued['segment_updates'] == state.plan.horizon
    assert state_exact(export_codec(fresh.learner.model), expected)
