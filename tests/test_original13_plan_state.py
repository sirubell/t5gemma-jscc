"""Original13 CPU mechanics; toy route adapters do not establish real routing."""
from copy import deepcopy
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
import torch

from jscc.activation_replay import canonical_digest
from jscc.encoder_stages import Original13TrainingState, run_original13_training
from jscc.encoder_stages_evaluation import ObservationScoringError
from jscc.encoder_stages_plan import (ORIGINAL13_PROTOCOL, ORIGINAL13_SITES, Original13Plan,
                                      original13_progress)
from jscc.encoder_stages_preparation import resolve_original13_plans
from jscc.experiment_state import capture_rng, open_state
from jscc.sharing_preparation import state_dict_identity
from jscc.sharing_protocol import batch_view
from jscc.sharing_schedule import BatchView, SharingExposureLedger
from jscc.sharing_qualification_guard import state_exact
from test_experiment_records import update as event_template
from test_sharing_native16 import native_batches


def panel():
    return {'expected_items': 256, 'input_ids': list(range(256)),
            'source_family_ids': [f'family-{i}' for i in range(256)], 'split': 'validation',
            'panel': 'f3e5b8ffbaeaae3a775106937170d8f12cb8670b6ecb01f31efe3c7c33179bae',
            'noise': {'namespace': 'encfn-baseline-20260930-development-v2', 'seed': 0}}


def formal_views():
    return tuple(BatchView(canonical_digest(i), canonical_digest([i, 'mask']), canonical_digest([i, 'layout']),
                           16, 40, 32, 48) for i in range(10000))


@pytest.mark.parametrize('ratio,steps,warmup', [
    (50, [0, 16250, 32500, 48750, 65000, 71500, 81250, 97500, 113750, 130000], [3250, 3250]),
    (75, [0, 24375, 48750, 73125, 97500, 100750, 105625, 113750, 121875, 130000], [4875, 1625]),
    (90, [0, 29250, 58500, 87750, 117000, 118300, 120250, 123500, 126750, 130000], [5850, 650]),
])
@pytest.mark.parametrize('width', [512, 1152, 2304])
def test_all_nine_formal_plans(ratio, width, steps, warmup):
    plan = Original13Plan(formal_views(), 'paired', ratio, width, 'a'*64, 'b'*64)
    assert plan.horizon == 130000
    assert len(plan.checkpoint_states) == 11
    assert sorted({step for step, _ in plan.checkpoint_states}) == steps
    assert all(step % 13 == 0 for step, _ in plan.checkpoint_states)
    assert [s.warmup_updates for s in plan.stages] == warmup
    assert Original13Plan.from_resolved(plan.resolved()) == plan
    for stage in plan.stages:
        assert plan.update_at(stage.start).lr_factor == 0
        assert plan.update_at(stage.start + stage.warmup_updates).lr_factor == 1
        assert plan.update_at(stage.stop-1).lr_factor > 0
    assert original13_progress(plan, 130000, plan.reconstruction_stop)['completed_per_site'] == dict.fromkeys(ORIGINAL13_SITES, 10000)
    assert plan.objective_at(0) == {'kl': 0., 'hidden': .1, 'memory': 0.}
    assert plan.objective_at(6) == {'kl': 0., 'hidden': .05, 'memory': .05}
    assert plan.objective_at(plan.reconstruction_stop + 6) == {'kl': 1., 'hidden': .05, 'memory': .05}


def test_plan_rejects_changed_contract_and_ratio_noise_key():
    plan = Original13Plan(formal_views(), 'paired', 50, 512, 'a'*64, 'b'*64)
    value = plan.resolved()
    value['checkpoint_states'].pop()
    with pytest.raises(ValueError, match='changed'):
        Original13Plan.from_resolved(value)
    assert plan.update_at(6).noise_key == replace(plan, ratio=90).update_at(6).noise_key
    assert [plan.update_at(i).site for i in range(26)] == list(ORIGINAL13_SITES)*2
    with pytest.raises(ValueError):
        replace(plan, ratio=True)
    with pytest.raises(ValueError):
        plan.update_at(True)


def test_actual_panel_object_is_bound_and_never_reduced():
    inputs = {'data_ids': {'development': panel()}}
    views = formal_views()[:400]
    plans = resolve_original13_plans(views, inputs=inputs, input_identity='a'*64, synthetic=True)
    assert len(plans) == 9 and {p.observation_identity for p in plans} == {canonical_digest(panel())}
    inputs['data_ids']['development']['input_ids'][0] = 1
    with pytest.raises(ValueError, match='existing256'):
        resolve_original13_plans(views, inputs=inputs, input_identity='a'*64, synthetic=True)


class MechanicalLearner:
    optimizer: torch.optim.AdamW
    scheduler: torch.optim.lr_scheduler.LambdaLR
    def __init__(self, model, plan, event_sink=None):
        self.model, self.plan, self.event_sink = model, plan, event_sink
        self.parameters = list(model.parameters())
        self.completed = self.attempted = self.offset = self.valid_tokens = self.phase_start_step = 0
        self.failed = False
        self.scaler = None
        self.exposure = SharingExposureLedger(cast(Any, plan), sites=ORIGINAL13_SITES)

    def update(self, batches, *, batch_identities=None):
        expected = self.plan.update_at(self.completed)
        assert len(batches) == 1 and batch_view(batches[0]) == expected.view
        if batch_identities:
            batch_identities[0].validate(batches[0])
        used = [g['lr'] for g in self.optimizer.param_groups]
        self.optimizer.zero_grad(set_to_none=True)
        self.model(torch.randn(3, 2)).square().mean().backward()
        self.optimizer.step()
        self.scheduler.step()
        self.completed += 1
        self.attempted += 1
        self.offset += 16
        self.valid_tokens += expected.view.source_tokens
        event = event_template(self.completed, final=self.plan.horizon)
        event['phase_id'] = self.plan.stage_at(self.completed-1).phase_id
        event['payload'].update(lr_used=used, lr_next=[g['lr'] for g in self.optimizer.param_groups],
                                site_id=expected.site, phase_start_step=self.phase_start_step)
        if self.completed == self.phase_start_step + 1:
            event['payload']['update_l2'] = {'value': 0, 'reason': None}
        if self.event_sink:
            self.event_sink(event)
        self.exposure.record(expected, receipt=str(self.completed))
        return event


class MechanicalRouting:
    canonical_codec = staticmethod(lambda model: model)
    export_codec = staticmethod(lambda model: model.state_dict())
    load_codec = staticmethod(lambda model, weights: model.load_state_dict(weights, strict=True))
    observation_state = staticmethod(lambda model: {'training': model.training})
    restore_runtime_state = staticmethod(lambda model, state: model.train(state['training']))
    create_original13_learner = staticmethod(lambda model, *, plan, metadata, event_sink:
                                           MechanicalLearner(model, plan, event_sink))


def mechanics(ratio=90):
    torch.manual_seed(0)
    model = torch.nn.Linear(2, 1)
    originals = native_batches()
    batches = []
    for i in range(400):
        batch = deepcopy(originals[i % 4])
        batch['row_ids'] = torch.arange(i*16, (i+1)*16)
        batches.append(batch)
    plan = Original13Plan(tuple(batch_view(b) for b in batches), 'toy', ratio, 2,
                          'a'*64, canonical_digest(panel()), True)
    metadata = {'source_identity': 'source', 'config_identity': 'config', 'data_identity': 'data',
                'initialization_identity': state_dict_identity(model.state_dict()),
                'stream_identity': plan.input_identity, 'protocol_identity': ORIGINAL13_PROTOCOL,
                'prepared_identity': 'prepared', 'model_revision': 'toy'}
    learner = MechanicalLearner(model, plan)
    state = Original13TrainingState(learner, plan=plan, metadata=metadata, routing=MechanicalRouting)
    return model, plan, batches, metadata, state


def advance(state, batches, target):
    while state.learner.completed < target:
        index = state.learner.completed // 13
        expected_lr = 2e-4 * state.plan.update_at(state.learner.completed).lr_factor
        assert state.learner.optimizer.param_groups[0]['lr'] == expected_lr
        state.learner.update([batches[index]])


@pytest.mark.parametrize('position', ['interior', 'pre_reset', 'post_reset', 'functional'])
def test_exact_continuation_and_next_consumed_lr(tmp_path, position):
    _, plan, batches, metadata, state = mechanics()
    target = 17 if position == 'interior' else plan.reconstruction_stop
    advance(state, batches, target)
    parent = state.save(tmp_path/'pre.pt')
    if position in ('post_reset', 'functional'):
        state.reset(parent)
        if position == 'functional':
            advance(state, batches, target+17)
    saved = state.save(tmp_path/'saved.pt')
    rng = capture_rng()
    if position == 'pre_reset':
        state.reset(saved)
    advance(state, batches, state.learner.completed+3)
    expected = deepcopy(state.routing.export_codec(state.learner.model))
    expected_optimizer = deepcopy(state.learner.optimizer.state_dict())
    expected_exposure = deepcopy(state.learner.exposure.per_site)
    _, _, _, _, resumed = mechanics()
    resumed.restore(saved.reference)
    if position == 'pre_reset':
        resumed.reset(saved)
    assert state_exact(capture_rng(), rng)
    advance(resumed, batches, resumed.learner.completed+3)
    assert state_exact(resumed.routing.export_codec(resumed.learner.model), expected)
    assert state_exact(resumed.learner.optimizer.state_dict(), expected_optimizer)
    assert resumed.learner.exposure.per_site == expected_exposure
    assert resumed.ancestry[-1]['sha256'] == saved.reference.sha256
    with pytest.raises(ValueError):
        open_state(saved.reference, expected={**saved.payload['metadata'], 'source_identity': 'wrong'})
    with pytest.raises(ValueError):
        open_state(saved.reference, expected={**saved.payload['metadata'], 'input_identity': 'b'*64})
    if position != 'interior':
        with pytest.raises(ValueError, match='exactly one'):
            resumed.reset(parent)


def observe_complete(model, processor, *, state, request, output, guard):
    torch.rand(7)
    np.random.rand(3)
    return {'status': 'complete', 'request_identity': canonical_digest(request)}


def test_full_runner_eleven_saves_and_observation_failure_continues(tmp_path):
    results = []
    weights = []
    for mode in ('complete', 'scoring_failure'):
        model, plan, batches, metadata, _ = mechanics()
        def observer(*args, **kwargs):
            receipt = observe_complete(*args, **kwargs)
            if mode == 'scoring_failure':
                raise ObservationScoringError('injected scorer failure')
            return receipt
        result = run_original13_training(model, None, plan=plan, batches=batches, metadata=metadata,
            observation_request={'panel': panel()}, output=tmp_path/mode, guard=lambda: None,
            routing=MechanicalRouting, observe=observer)
        results.append(result)
        weights.append(deepcopy(model.state_dict()))
        assert result['status'] == 'training_frozen' and result['completed_updates'] == 5200
        assert len(result['checkpoints']) == 11
        assert len(list((tmp_path/mode).glob('*.pt'))) == 11
        assert len(result['observations']) == 2
    assert state_exact(weights[0], weights[1])
    assert results[0]['observations_status'] == 'complete'
    assert results[1]['observations_status'] == 'pending'
    assert all(r['requires_explicit_completion'] for r in results[1]['observations'])


def test_save_failure_and_mutated_observation_state_stop(tmp_path, monkeypatch):
    import jscc.encoder_stages as stages
    model, plan, batches, metadata, state = mechanics()
    advance(state, batches, plan.reconstruction_stop)
    state.save(tmp_path/'valid.pt')
    with pytest.raises(FileExistsError):
        state.save(tmp_path/'valid.pt')
    def failed_save(*args, **kwargs):
        raise OSError('injected save failure')
    monkeypatch.setattr(stages, 'save_state', failed_save)
    with pytest.raises(OSError, match='save failure'):
        run_original13_training(model, None, plan=plan, batches=batches, metadata={**metadata,
            'initialization_identity': state_dict_identity(model.state_dict())},
            observation_request={'panel': panel()}, output=tmp_path/'failure', guard=lambda: None,
            routing=MechanicalRouting, observe=observe_complete)
    assert not list((tmp_path/'failure').glob('*.pt'))


def test_real_factory_rejects_unsupported_plan_before_model_access():
    from jscc.original13_routing import create_original13_learner
    with pytest.raises(TypeError, match='released plan'):
        create_original13_learner(None, plan=None, metadata=None, event_sink=None)


def test_observation_mutation_and_checkpoint_damage_stop_training(tmp_path):
    for mode in ('optimizer_mutation', 'checkpoint_damage'):
        model, plan, batches, metadata, _ = mechanics()
        def observer(model, processor, *, state, request, output, guard):
            if mode == 'checkpoint_damage':
                from pathlib import Path
                Path(state.reference.path).write_bytes(b'corrupt')
            else:
                with torch.no_grad():
                    model.weight.add_(1)
            raise ObservationScoringError('scoring error cannot excuse state damage')
        with pytest.raises(ValueError):
            run_original13_training(model, None, plan=plan, batches=batches, metadata=metadata,
                observation_request={'panel': panel()}, output=tmp_path/mode, guard=lambda: None,
                routing=MechanicalRouting, observe=observer)
        import json
        campaign = json.loads((tmp_path/mode/'campaign.json').read_text())
        assert campaign['status'] == 'incomplete'
        assert campaign['completed_updates'] == plan.reconstruction_stop
        assert not (tmp_path/mode/'freeze.json').exists()


def test_partial_payload_and_duplicate_body_are_rejected(tmp_path):
    from dataclasses import asdict
    import hashlib
    import io
    from jscc.experiment_state import CheckpointRef
    model, _, batches, _, state = mechanics()
    advance(state, batches, 3)
    valid = state.save(tmp_path/'valid.pt')
    payload = deepcopy(valid.payload)
    payload['rng'].pop('python')
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    data = buffer.getvalue()
    path = tmp_path/'partial.pt'
    path.write_bytes(data)
    ref = CheckpointRef(str(path), hashlib.sha256(data).hexdigest(), len(data))
    with pytest.raises(ValueError, match='invalid checkpoint'):
        open_state(ref, expected=valid.payload['metadata'])
    original = state.routing
    cast(Any, state).routing = type('DuplicateBody', (), {
        'export_codec': staticmethod(lambda model: {**model.state_dict(), 'memory.weight': model.weight}),
        'observation_state': original.observation_state,
    })
    with pytest.raises(ValueError, match='exactly the communication'):
        state.save(tmp_path/'duplicate.pt')
    assert not (tmp_path/'duplicate.pt').exists()
    assert asdict(valid.reference)['sha256'] != ref.sha256


def test_malformed_ancestry_rejected_and_multiple_segments_retained(tmp_path):
    _, _, batches, _, state = mechanics()
    advance(state, batches, 3)
    state.ancestry = ['not-a-checkpoint-reference']
    with pytest.raises(ValueError, match='complete checkpoint references'):
        state.save(tmp_path/'bad.pt')
    state.ancestry = []
    first = state.save(tmp_path/'first.pt')
    _, _, _, _, second = mechanics()
    second.restore(first.reference)
    advance(second, batches, 6)
    second_saved = second.save(tmp_path/'second.pt')
    _, _, _, _, third = mechanics()
    third.restore(second_saved.reference)
    assert [r['sha256'] for r in third.ancestry] == [first.reference.sha256, second_saved.reference.sha256]
    assert third.learner.completed == 6
    assert third.learner.exposure.completed == 6
    assert len(third.learner.exposure.receipts) == 6


def test_runner_pre_reset_resume_preserves_diagnostics_and_obligations(tmp_path):
    import json
    from pathlib import Path
    from jscc.experiment_state import CheckpointRef, restore_rng
    model, plan, batches, metadata, state = mechanics()
    advance(state, batches, plan.reconstruction_stop)
    saved = state.save(tmp_path/'pre-reset.pt')
    rng = capture_rng()
    state.reset(saved)
    advance(state, batches, plan.horizon)
    expected = deepcopy(model.state_dict())
    fresh, _, _, _, _ = mechanics()
    restore_rng(rng)
    calls = []
    def observe(*args, **kwargs):
        calls.append(kwargs['request']['step'])
        return observe_complete(*args, **kwargs)
    result = run_original13_training(fresh, None, plan=plan, batches=batches, metadata=metadata,
        observation_request={'panel': panel()}, output=tmp_path/'resumed', guard=lambda: None,
        routing=MechanicalRouting, observe=observe, resume=saved.reference)
    assert state_exact(fresh.state_dict(), expected)
    assert result['segment_updates'] == plan.horizon-plan.reconstruction_stop
    assert len(result['checkpoints']) == 6
    assert calls == [plan.observations[1][0]]
    assert result['observations'][0]['status'] == 'pending_prior_segment'
    assert result['observations_status'] == 'pending'
    freeze = json.loads(Path(result['freeze_reference']['path']).read_text())
    assert freeze['segment_ancestry'][-1]['sha256'] == saved.reference.sha256
    fresh, _, _, _, _ = mechanics()
    bad = tmp_path/'bad-resume.pt'
    bad.write_bytes(b'bad')
    with pytest.raises(ValueError, match='checkpoint bytes differ'):
        run_original13_training(fresh, None, plan=plan, batches=batches, metadata=metadata,
            observation_request={'panel': panel()}, output=tmp_path/'bad-resumed', guard=lambda: None,
            routing=MechanicalRouting, observe=observe, resume=CheckpointRef(str(bad), 'a'*64, 99))
    failed = json.loads((tmp_path/'bad-resumed'/'campaign.json').read_text())
    assert failed['status'] == 'incomplete' and failed['error']['type'] == 'ValueError'


def test_factory_and_resumed_save_failures_record_campaign_errors(tmp_path, monkeypatch):
    import json
    import jscc.encoder_stages as stages
    class BrokenRouting(MechanicalRouting):
        @staticmethod
        def create_original13_learner(*args, **kwargs):
            raise RuntimeError('injected factory failure')
    fresh, plan, batches, metadata, state = mechanics()
    with pytest.raises(RuntimeError, match='factory failure'):
        run_original13_training(fresh, None, plan=plan, batches=batches, metadata=metadata,
            observation_request={'panel': panel()}, output=tmp_path/'factory', guard=lambda: None,
            routing=BrokenRouting, observe=observe_complete)
    assert json.loads((tmp_path/'factory'/'campaign.json').read_text())['error']['type'] == 'RuntimeError'
    advance(state, batches, plan.reconstruction_stop)
    saved = state.save(tmp_path/'pre-reset.pt')
    fresh, _, _, _, _ = mechanics()
    def failed_save(*args, **kwargs):
        raise OSError('injected resumed save failure')
    monkeypatch.setattr(stages, 'save_state', failed_save)
    with pytest.raises(OSError, match='resumed save failure'):
        run_original13_training(fresh, None, plan=plan, batches=batches, metadata=metadata,
            observation_request={'panel': panel()}, output=tmp_path/'resumed-save', guard=lambda: None,
            routing=MechanicalRouting, observe=observe_complete, resume=saved.reference)
    failed = json.loads((tmp_path/'resumed-save'/'campaign.json').read_text())
    assert failed['completed_updates'] == plan.reconstruction_stop
    assert failed['error']['type'] == 'OSError'
