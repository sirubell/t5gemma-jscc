"""Current recipe observation restoration retaining original13 fatal-state invariants."""

from copy import deepcopy
from pathlib import Path
import pytest
import torch
from jscc import panel_evaluation as panels, site_routing
from jscc.experiment_state import capture_rng
from jscc.qualification import state_exact
from jscc.observation_policy import recipe_observation_identity
from jscc.shared_state import RecipeTrainingState
from shared.test_shared_case_objectives import real_setup, advance
from shared.test_shared_case_evaluation import template, score_stub



def observation_state(tmp_path):
    learner, machine, batches = real_setup(tmp_path/'setup')
    plan = learner.schedule
    metadata = dict(machine.metadata)
    metadata['observation_identity'] = recipe_observation_identity(plan.description, template(256))
    machine = RecipeTrainingState(learner, plan=plan, metadata=metadata)
    while learner.completed < plan.stages[1].start:
        advance(learner, batches)
    saved = machine.save(tmp_path/'pre.pt')
    policy = plan.description['observations'][0]
    request = panels._recipe_observation_request(saved,
        {**policy, 'plan_identity': plan.identity, 'checkpoint_sha256': saved.reference.sha256}, template(256))
    return learner.model, plan, batches, machine, saved, request


@pytest.mark.parametrize('mode', ['wrong_panel', 'wrong_step', 'checkpoint', 'restoration', 'nonfinite', 'save'])
def test_identity_integrity_restoration_and_save_failures_are_fatal(tmp_path, monkeypatch, mode):
    model, plan, batches, machine, state, request = observation_state(tmp_path)
    def score(model, processor, **kwargs):
        torch.rand(7)
        if mode == 'checkpoint':
            Path(state.reference.path).write_bytes(b'corrupt')
        if mode == 'nonfinite':
            raise ValueError('Non-finite HellaSwag score')
        raise panels.ObservationScoringError('scorer failure')
    monkeypatch.setattr(panels, '_score_panel', score)
    if mode == 'wrong_panel':
        request['source_family_ids'][0] = 'other'
    if mode == 'wrong_step':
        request['step'] += 13
    if mode == 'restoration':
        monkeypatch.setattr(site_routing, 'restore_runtime_state', lambda *a: (_ for _ in ()).throw(ValueError('restoration')))
    if mode == 'save':
        original = panels.write_manifest
        def fail(path, value):
            if value.get('status') == 'failed':
                raise OSError('receipt save failure')
            original(path, value)
        monkeypatch.setattr(panels, 'write_manifest', fail)
    with pytest.raises((ValueError, OSError)) as error:
        panels.observe_recipe(model, None, state=state, request=request, output=tmp_path/'observe', guard=lambda: None)
    assert not isinstance(error.value, panels.ObservationScoringError)


def test_nested_runtime_and_exception_restore_next_update(tmp_path):
    model, plan, batches, machine, state, request = observation_state(tmp_path)
    before = deepcopy(capture_rng())
    weights = deepcopy(model.state_dict())
    with panels._panel_runtime(model, state):
        inside = deepcopy(capture_rng())
        with pytest.raises(panels.ObservationScoringError):
            with panels._panel_runtime(model, state):
                torch.rand(11)
                raise panels.ObservationScoringError('injected')
        assert state_exact(inside, capture_rng())
    assert state_exact(before, capture_rng()) and state_exact(weights, model.state_dict())
    machine.reset(state)
    advance(machine.learner, batches)
    expected = deepcopy(model.state_dict())
    learner, control, resumed_batches = real_setup(tmp_path/'control')
    control = RecipeTrainingState(learner, plan=learner.schedule, metadata=machine.metadata)
    control.restore(state.reference)
    control.reset(state)
    advance(learner, resumed_batches)
    assert state_exact(expected, control.learner.model.state_dict())


def test_integrity_export_failure_still_attempts_restoration(tmp_path, monkeypatch):
    model, plan, batches, machine, state, request = observation_state(tmp_path)
    before = deepcopy(capture_rng())
    calls = []
    original = site_routing.export_codec
    def export(model):
        calls.append('export')
        if len(calls) == 2:
            raise ValueError('integrity export failed')
        return original(model)
    restored = []
    original_restore = site_routing.restore_runtime_state
    def restore(model, value):
        restored.append(True)
        original_restore(model, value)
    monkeypatch.setattr(site_routing, 'export_codec', export)
    monkeypatch.setattr(site_routing, 'restore_runtime_state', restore)
    monkeypatch.setattr(panels, '_score_panel', score_stub)
    with pytest.raises(ValueError, match='integrity export failed'):
        panels.observe_recipe(model, None, state=state, request=request, output=tmp_path/'score', guard=lambda: None)
    assert restored and model.training
    assert state_exact(before, capture_rng())


def test_both_observations_exact_count_and_next_training_update(tmp_path, monkeypatch):
    from jscc.shared_training import run_recipe_training
    from jscc.activation_replay import canonical_digest
    weights = []
    for mode in ('no_scoring', 'complete', 'isolated_failure'):
        learner, machine, batches = real_setup(tmp_path/(mode+'-setup'))
        model, plan = learner.model, learner.schedule
        metadata = dict(machine.metadata)
        metadata['observation_identity'] = recipe_observation_identity(plan.description, template(256))
        model.memory_codec = None
        model.universal_codec = False
        def score(*args, **kwargs):
            value = score_stub(*args, **kwargs)
            if mode == 'isolated_failure':
                raise panels.ObservationScoringError('injected scorer failure')
            return value
        monkeypatch.setattr(panels, '_score_panel', score)
        def no_score(model, processor, *, state, request, output, guard):
            return {'status': 'complete', 'request_identity': canonical_digest(request)}
        result = run_recipe_training(model, None, plan=plan, batches=batches, metadata=metadata,
            output=tmp_path/mode, guard=lambda: None, synthetic=True,
            observe=no_score if mode == 'no_scoring' else panels.make_recipe_observer(template(256)))
        assert result['status'] == 'training_frozen'
        weights.append(deepcopy(model.state_dict()))
        assert result['observations_status'] == ('pending' if mode == 'isolated_failure' else 'complete')
        if mode == 'complete':
            import json
            receipts = [json.loads(p.read_text()) for p in (tmp_path/mode).glob('observation_*/result.json')]
            assert len(receipts) == 2
            assert sum(r['completed_panels'] for r in receipts) == 16
            assert sum(p['completed_items'] for r in receipts for p in r['panels']) == 4096
            assert {(r['request']['step'], r['request']['state_role']) for r in receipts} == {
                (r['step'], r['state']) for r in plan.description['observations']}
    assert state_exact(weights[0], weights[1]) and state_exact(weights[0], weights[2])


def test_report_rejects_other_run_observations_and_duplicate_endpoints(tmp_path, monkeypatch):
    import json
    from jscc.shared_training import run_recipe_training
    monkeypatch.setattr(panels, '_score_panel', score_stub)
    roots = []
    for label in ('a', 'b'):
        learner, machine, batches = real_setup(tmp_path/(label+'-setup'))
        model, plan = learner.model, learner.schedule
        metadata = dict(machine.metadata)
        metadata['observation_identity'] = recipe_observation_identity(plan.description, template(256))
        model.memory_codec = None
        model.universal_codec = False
        root = (tmp_path/label).resolve()
        run_recipe_training(model, None, plan=plan, batches=batches, metadata=metadata,
            output=root, guard=lambda: None, synthetic=True,
            observe=panels.make_recipe_observer(template(256)))
        roots.append(root)
    own = sorted(roots[0].glob('observation_*'))
    report = panels.recipe_report(roots[0], observations=own)
    assert sum(r['completed'] for r in report['observation_receipts']) == 16
    terminal, _ = panels.open_recipe_terminal(roots[1])
    ancestor_campaign = json.loads((roots[0]/'campaign.json').read_text())
    receipts = [json.loads((path/'result.json').read_text()) for path in own]
    pre = next(r for r in receipts if r['request']['state_role'] == 'pre_reset')
    for role, expected in (('reconstruction', False), ('post_reset', True), ('functional_observation', True)):
        saved = next(r for r in ancestor_campaign['checkpoints'] if r['role'] == role)
        terminal.payload['metadata']['segment_ancestry'] = [{k: saved[k] for k in ('path', 'sha256', 'size')}]
        assert panels._own_recipe_observation(terminal,
            json.loads((roots[1]/'campaign.json').read_text()), pre) is expected
    with pytest.raises(ValueError, match='duplicate observation endpoint'):
        panels.recipe_report(roots[0], observations=[own[0], own[0]])
    with pytest.raises(ValueError, match='recipe report observation trajectory differs'):
        panels.recipe_report(roots[0], observations=sorted(roots[1].glob('observation_*')))
