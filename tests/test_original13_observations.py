"""Observation restoration and runner continuation use the released real seam."""
from copy import deepcopy
import json

import pytest
import torch

from jscc import encoder_stages_evaluation as evaluation, original13_routing
from jscc.activation_replay import canonical_digest
from jscc.encoder_stages import run_original13_training
from jscc.experiment_state import capture_rng
from jscc.sharing_qualification_guard import state_exact
from test_original13_evaluation import score_stub, template
from test_original13_plan_state import MechanicalRouting, mechanics, panel, advance


def observation_template():
    p = panel()
    return {**template(256), 'schema': 'original13-observation-request-v1', 'panel': p,
            'noise': p['noise'], 'sites': ['enc_l9', 'enc_fn', 'dec_l0', 'dec_l24'],
            'conditions': ['no_noise', -6]}


@pytest.fixture
def routing(monkeypatch):
    for name in ('canonical_codec', 'export_codec', 'load_codec', 'observation_state', 'restore_runtime_state'):
        monkeypatch.setattr(original13_routing, name, getattr(MechanicalRouting, name))


def test_both_observations_exact_count_and_next_training_update(tmp_path, monkeypatch, routing):
    weights = []
    for mode in ('no_scoring', 'complete', 'isolated_failure'):
        model, plan, batches, metadata, _ = mechanics()
        def score(*args, **kwargs):
            value = score_stub(*args, **kwargs)
            if mode == 'isolated_failure':
                raise evaluation.ObservationScoringError('injected scorer failure')
            return value
        monkeypatch.setattr(evaluation, '_score_original13_panel', score)
        def no_score(model, processor, *, state, request, output, guard):
            return {'status': 'complete', 'request_identity': canonical_digest(request)}
        result = run_original13_training(model, None, plan=plan, batches=batches, metadata=metadata,
            observation_request=observation_template(), output=tmp_path/mode, guard=lambda: None,
            routing=MechanicalRouting, observe=no_score if mode == 'no_scoring' else evaluation.observe_original13)
        assert result['status'] == 'training_frozen'
        weights.append(deepcopy(model.state_dict()))
        assert result['observations_status'] == ('pending' if mode == 'isolated_failure' else 'complete')
        if mode == 'complete':
            receipts = [json.loads(p.read_text()) for p in (tmp_path/mode).glob('observation_*/result.json')]
            assert len(receipts) == 2
            assert sum(r['completed_panels'] for r in receipts) == 16
            assert sum(p['completed_items'] for r in receipts for p in r['panels']) == 4096
            assert {(r['request']['step'], r['request']['state_role']) for r in receipts} == set(plan.observations)
    assert state_exact(weights[0], weights[1]) and state_exact(weights[0], weights[2])


@pytest.mark.parametrize('mode', ['wrong_panel', 'wrong_step', 'checkpoint', 'restoration', 'nonfinite', 'save'])
def test_identity_integrity_restoration_and_save_failures_are_fatal(tmp_path, monkeypatch, routing, mode):
    model, plan, batches, metadata, state_machine = mechanics()
    advance(state_machine, batches, plan.reconstruction_stop)
    state = state_machine.save(tmp_path/'pre.pt')
    request = {**observation_template(), 'plan_identity': plan.identity, 'step': plan.reconstruction_stop,
               'state_role': 'pre_reset', 'checkpoint_sha256': state.reference.sha256}
    def score(model, processor, **kwargs):
        torch.rand(7)
        if mode == 'checkpoint':
            from pathlib import Path
            Path(state.reference.path).write_bytes(b'corrupt')
        if mode == 'nonfinite':
            raise ValueError('Non-finite HellaSwag score')
        raise evaluation.ObservationScoringError('scorer failure')
    monkeypatch.setattr(evaluation, '_score_original13_panel', score)
    if mode == 'wrong_panel':
        request['panel']['source_family_ids'][0] = 'other'
    if mode == 'wrong_step':
        request['step'] += 13
    if mode == 'restoration':
        monkeypatch.setattr(original13_routing, 'restore_runtime_state', lambda *a: (_ for _ in ()).throw(ValueError('restoration')))
    if mode == 'save':
        original = evaluation.write_manifest
        def fail(path, value):
            if value.get('status') == 'failed':
                raise OSError('receipt save failure')
            original(path, value)
        monkeypatch.setattr(evaluation, 'write_manifest', fail)
    with pytest.raises((ValueError, OSError)) as error:
        evaluation.observe_original13(model, None, state=state, request=request, output=tmp_path/'observe', guard=lambda: None)
    assert not isinstance(error.value, evaluation.ObservationScoringError)


def test_nested_runtime_and_exception_restore_next_update(tmp_path, monkeypatch, routing):
    model, plan, batches, _, machine = mechanics()
    advance(machine, batches, plan.reconstruction_stop)
    state = machine.save(tmp_path/'pre.pt')
    before = deepcopy(capture_rng())
    weights = deepcopy(model.state_dict())
    with evaluation._original13_runtime(model, state):
        inside = deepcopy(capture_rng())
        with pytest.raises(evaluation.ObservationScoringError):
            with evaluation._original13_runtime(model, state):
                torch.rand(11)
                raise evaluation.ObservationScoringError('injected')
        assert state_exact(inside, capture_rng())
    assert state_exact(before, capture_rng()) and state_exact(weights, model.state_dict())
    machine.reset(state)
    advance(machine, batches, plan.reconstruction_stop+1)
    expected = deepcopy(model.state_dict())
    _, _, _, _, control = mechanics()
    control.restore(state.reference)
    control.reset(state)
    advance(control, batches, plan.reconstruction_stop+1)
    assert state_exact(expected, control.learner.model.state_dict())


def test_integrity_export_failure_still_attempts_restoration(tmp_path, monkeypatch, routing):
    model, plan, batches, _, machine = mechanics()
    advance(machine, batches, plan.reconstruction_stop)
    state = machine.save(tmp_path/'pre.pt')
    request = {**observation_template(), 'plan_identity': plan.identity, 'step': plan.reconstruction_stop,
               'state_role': 'pre_reset', 'checkpoint_sha256': state.reference.sha256}
    before = deepcopy(capture_rng())
    calls = []
    original = original13_routing.export_codec
    def export(model):
        calls.append('export')
        if len(calls) == 2:
            raise ValueError('integrity export failed')
        return original(model)
    restored = []
    def restore(model, value):
        restored.append(True)
        MechanicalRouting.restore_runtime_state(model, value)
    monkeypatch.setattr(original13_routing, 'export_codec', export)
    monkeypatch.setattr(original13_routing, 'restore_runtime_state', restore)
    monkeypatch.setattr(evaluation, '_score_original13_panel', score_stub)
    with pytest.raises(ValueError, match='integrity export failed'):
        evaluation.observe_original13(model, None, state=state, request=request, output=tmp_path/'score', guard=lambda: None)
    assert restored and model.training
    assert state_exact(before, capture_rng())


def test_report_rejects_other_run_observations_and_duplicate_endpoints(tmp_path, monkeypatch, routing):
    monkeypatch.setattr(evaluation, '_score_original13_panel', score_stub)
    roots = []
    for label in ('a', 'b'):
        model, plan, batches, metadata, _ = mechanics()
        root = (tmp_path/label).resolve()
        run_original13_training(model, None, plan=plan, batches=batches, metadata=metadata,
            observation_request=observation_template(), output=root, guard=lambda: None,
            routing=MechanicalRouting, observe=evaluation.observe_original13)
        roots.append(root)
    own = sorted(roots[0].glob('observation_*'))
    report = evaluation.original13_report(roots[0], observations=own)
    assert sum(r['completed'] for r in report['observation_receipts']) == 16
    # An ancestor campaign may continue beyond the checkpoint used to branch.
    # Only evidence at/before the carried checkpoint is inherited.
    terminal, _ = evaluation.open_original13_terminal(roots[1])
    ancestor_campaign = json.loads((roots[0]/'campaign.json').read_text())
    # Select by endpoint rather than filename lexical order.
    receipts = [json.loads((path/'result.json').read_text()) for path in own]
    pre = next(r for r in receipts if r['request']['state_role'] == 'pre_reset')
    for role, expected in (('reconstruction', False), ('post_reset', True), ('functional_observation', True)):
        saved = next(r for r in ancestor_campaign['checkpoints'] if r['role'] == role)
        terminal.payload['metadata']['segment_ancestry'] = [{k: saved[k] for k in ('path', 'sha256', 'size')}]
        assert evaluation._own_observation_checkpoint(terminal, json.loads((roots[1]/'campaign.json').read_text()),
            pre['checkpoint'], pre['request']) is expected
    with pytest.raises(ValueError, match='duplicate observation endpoint'):
        evaluation.original13_report(roots[0], observations=[own[0], own[0]])
    with pytest.raises(ValueError, match='observation trajectory differs'):
        evaluation.original13_report(roots[0], observations=sorted(roots[1].glob('observation_*')))
