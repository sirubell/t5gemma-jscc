"""Current shared checkpoint boundary checks preserving original13 state invariants."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import io
import pytest
import torch
from jscc.experiment_state import CheckpointRef, open_state
from shared.test_shared_case_objectives import real_setup, advance
from shared.test_shared_case_evaluation import frozen as frozen
from dataclasses import replace
from jscc.activation_replay import canonical_digest
from jscc.shared_inputs import BatchView
from jscc.checkpoint_readers import ORIGINAL13_SITES, Original13Plan, original13_progress, resolve_original13_plans



def test_partial_payload_and_duplicate_body_are_rejected(tmp_path):
    learner, state, batches = real_setup(tmp_path/'setup')
    for _ in range(3):
        advance(learner, batches)
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
    learner.model.memory_codec = deepcopy(learner.model.codec)
    with pytest.raises(ValueError):
        state.save(tmp_path/'duplicate.pt')
    assert not (tmp_path/'duplicate.pt').exists()
    assert asdict(valid.reference)['sha256'] != ref.sha256


def test_malformed_ancestry_rejected_and_multiple_segments_retained(tmp_path):
    learner, state, batches = real_setup(tmp_path/'first-setup')
    for _ in range(3):
        advance(learner, batches)
    state.ancestry = ['not-a-checkpoint-reference']
    with pytest.raises(ValueError, match='complete checkpoint references'):
        state.save(tmp_path/'bad.pt')
    state.ancestry = []
    first = state.save(tmp_path/'first.pt')
    resumed, second, second_batches = real_setup(tmp_path/'second-setup')
    second.restore(first.reference)
    for _ in range(3):
        advance(resumed, second_batches)
    second_saved = second.save(tmp_path/'second.pt')
    _, third, _ = real_setup(tmp_path/'third-setup')
    third.restore(second_saved.reference)
    assert [r['sha256'] for r in third.ancestry] == [first.reference.sha256, second_saved.reference.sha256]
    assert third.learner.completed == 6
    assert third.learner.exposure.completed == 6
    assert len(third.learner.exposure.receipts) == 6


def test_full_runner_eleven_saves_and_observation_failure_continues(frozen):
    import json
    model, plan, root = frozen
    campaign = json.loads((root/'campaign.json').read_text())
    assert campaign['status'] == 'training_frozen'
    assert len(campaign['checkpoints']) == 11
    from jscc.checkpoint_readers import _open_bound_reference
    assert [(_open_bound_reference(CheckpointRef(**{k:c[k] for k in ('path','sha256','size')})).payload['metadata']['completed_updates'], c['role'])
        for c in campaign['checkpoints']] == [tuple(c) for c in plan.description['checkpoint_states']]
    assert len(campaign['observations']) == 2
    assert campaign['completed_updates'] == plan.horizon

def panel():
    return {'expected_items': 256, 'input_ids': list(range(256)),
            'source_family_ids': [f'family-{i}' for i in range(256)], 'split': 'validation',
            'panel': 'f3e5b8ffbaeaae3a775106937170d8f12cb8670b6ecb01f31efe3c7c33179bae',
            'noise': {'namespace': 'encfn-baseline-20260930-development-v2', 'seed': 0}}


# Exact recorded-plan reader fixtures; these do not authorize a training loop.



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
