"""CPU schedule/state checks for balanced six-site matched A/E/R trajectories."""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
from typing import Any, cast

import pytest

from jscc.checkpoint_readers import PILOT_SITES as SITES, PILOT_SITE_SPECS as SITE_SPECS
from jscc.checkpoint_readers import ARMS, ENCODER_STAGES_PROTOCOL, EncoderStagesPlan, PEAK_LEARNING_RATE, TOTAL_SCIENTIFIC_UPDATES, WIDTHS, build_stages_state, validate_stages_state
from jscc.models.split_model import ResolvedSite
from jscc.shared_inputs import BatchView
from jscc.shared_plan import SharingExposureLedger


def plan(q=4, arm='A', *, synthetic=True, width=512):
    def digest(value):
        return hashlib.sha256(str(value).encode()).hexdigest()
    views = tuple(BatchView(digest(i), digest((i, 'mask')), digest((i, 'layout')),
                            16, 32 + i, 24 + i, 48 + i) for i in range(q))
    return EncoderStagesPlan(views, 'matched-stages', arm, synthetic=synthetic, width=width)


def policy():
    result = {}
    for name, split in SITE_SPECS.items():
        where = split['where']
        index = None if name in ('enc_emb', 'enc_fn') else int(name[5:])
        path = ('encoder.layers.0.input' if name == 'enc_emb' else 'encoder.norm' if name == 'enc_fn'
                else f'encoder.layers.{index}')
        result[name] = {'site': asdict(ResolvedSite(where, index, 'tiny-revision', 26, 8, path)), 'role': 'trained'}
    return result


def payload(p, completed, phase_start_step=None):
    sharing, stream = build_stages_state(p, completed, phase_start_step=phase_start_step)
    local = sharing['phase_local_completed']
    return {'schema': 'experiment-state-v2', 'kind': 'shared_encoder_v2',
            'metadata': {'protocol_identity': ENCODER_STAGES_PROTOCOL, 'phase': sharing['phase_id'],
                         'completed_updates': completed, 'sharing': sharing,
                         'model_revision': 'tiny-revision', 'evaluation_sites': policy()},
            'stream': stream, 'scheduler': {'last_epoch': local},
            'optimizer': {'state': {0: {'step': local}} if local else {}}}


@pytest.mark.parametrize('arm', ARMS)
def test_formal_horizons_boundaries_and_no_intermediate_observations(arm):
    p = plan(10000, arm, synthetic=False)
    assert p.plan is p and p.name == 'shared' and p.trained_sites == SITES
    assert p.batch_size == 16 and p.accumulation == 1
    assert p.horizon == 60000 and p.boundary_steps == (0, 30000, 60000)
    assert p.checkpoint_steps == tuple(range(0, 60001, 6000))
    assert p.objective_steps == p.observations == ()
    assert p.reset_steps == (() if arm == 'A' else (30000,))
    sharing, stream = build_stages_state(p, 60000)
    assert sharing['completed_per_site'] == dict.fromkeys(SITES, 10000)
    assert stream['offset'] == 960000 and stream['next_site'] is None
    assert TOTAL_SCIENTIFIC_UPDATES == len(WIDTHS) * len(ARMS) * p.horizon == 540000
    assert PEAK_LEARNING_RATE == 2e-4


@pytest.mark.parametrize('q', (2, 4, 8))
@pytest.mark.parametrize('arm', ARMS)
def test_synthetic_balanced_boundaries_and_matched_noise_stream(q, arm):
    p = plan(q, arm)
    ledger = SharingExposureLedger(cast(Any, p), sites=SITES)
    comparison = plan(q, 'E' if arm != 'E' else 'R')
    for index in range(p.horizon):
        update = p.update_at(index)
        assert update.site == SITES[index % 6] and update.site_local_index == index // 6
        assert (update.view, update.noise_key) == (comparison.update_at(index).view, comparison.update_at(index).noise_key)
        ledger.record(update, receipt=f'physical-{index}')
        if index + 1 in p.boundary_steps:
            expected = dict.fromkeys(SITES, (index + 1) // 6)
            assert build_stages_state(p, index + 1)[0]['completed_per_site'] == expected
    for site, counts in ledger.finalize().items():
        assert counts['updates'] == q and counts['sequences'] == 16 * q
        assert counts['source_tokens'] == sum(32 + i for i in range(q))
    assert all(step % 6 == 0 for step in p.checkpoint_steps)
    assert p.update_at(p.horizon - 1).lr_factor > 0
    with pytest.raises(ValueError, match='horizon'):
        p.update_at(p.horizon)


def test_fixed_lr_horizons_and_matched_second_stages():
    a = plan(10000, 'A', synthetic=False)
    e = replace(a, arm='E')
    r = replace(a, arm='R')
    assert a.scheduler_horizon_at(30000) == 60000
    assert a.scheduler_index_at(30000) == 30000
    assert a.update_at(3000).lr_factor == 1
    assert a.update_at(30000).lr_factor > 0
    for p in (e, r):
        assert p.scheduler_horizon_at(30000) == 30000
        assert p.scheduler_index_at(30000) == 0
        assert p.update_at(1500).lr_factor == 1
        assert p.update_at(30000).lr_factor == 0
        assert p.update_at(31500).lr_factor == 1
        assert p.stage_at(30000).reset_optimizer
    assert r.stage_at(0).objective_weights == {'kl': 0.0, 'hidden': 0.1, 'memory': 0.0}
    assert e.stage_at(0).objective_weights == {'kl': 1.0, 'hidden': 0.1, 'memory': 0.0}
    assert a.stage_at(0).warmup_updates == 3000 and r.stage_at(30000).warmup_updates == 1500
    assert r.objective_kind_at(29999) == 'reconstruction'
    assert r.objective_kind_at(30000) == e.objective_kind_at(30000) == 'combined'
    for index in (30000, 30001, 30005, 31500, 59999):
        assert r.update_at(index) == e.update_at(index)
        assert r.update_at(index).site_local_index == index // 6
    for p in (a, e, r):
        with pytest.raises(ValueError):
            p.stage_at(True)


def test_reset_retains_global_data_exposure_and_distinguishes_boundary_states():
    p = plan(4, 'R')
    before = payload(p, 12, 0)
    after = payload(p, 12, 12)
    validate_stages_state(before)
    validate_stages_state(after)
    assert before['stream'] == after['stream']
    assert before['metadata']['sharing']['completed_per_site'] == dict.fromkeys(SITES, 2)
    assert after['metadata']['sharing']['source_valid_per_site'] == dict.fromkeys(SITES, 65)
    assert before['scheduler']['last_epoch'] == 12 and after['scheduler']['last_epoch'] == 0
    assert before['optimizer']['state'] and not after['optimizer']['state']
    assert before['metadata']['phase'] == 'phase1' and after['metadata']['phase'] == 'phase2'
    assert after['metadata']['sharing']['optimizer_reset_step'] == 12
    assert after['stream']['offset'] == 192
    assert p.update_at(12).site == 'enc_emb' and p.update_at(12).site_local_index == 2
    validate_stages_state(payload(p, 13))
    validate_stages_state(payload(p, 24))
    with pytest.raises(ValueError, match='phase start'):
        build_stages_state(p, 13, phase_start_step=0)


@pytest.mark.parametrize('mutation', [
    lambda x: x['metadata'].update(phase='phase1'),
    lambda x: x['metadata']['sharing'].update(phase_start_step=0),
    lambda x: x['metadata']['sharing']['completed_per_site'].update(enc_emb=1),
    lambda x: x['metadata']['sharing']['source_valid_per_site'].update(enc_l14=0),
    lambda x: x['stream'].update(offset=16),
    lambda x: x['scheduler'].update(last_epoch=13),
    lambda x: x['optimizer']['state'][0].update(step=13),
    lambda x: x['optimizer'].update(state={}),
    lambda x: x['metadata']['evaluation_sites']['enc_l14'].update(role='heldout_after_freeze'),
    lambda x: x['metadata']['evaluation_sites']['enc_l4']['site'].update(hidden_dim=9),
])
def test_phase_local_and_global_drift_rejected(mutation):
    state = payload(plan(4, 'E'), 13)
    mutation(state)
    with pytest.raises(ValueError):
        validate_stages_state(state)


def test_zero_local_state_rejects_carried_moments_and_a_has_no_reset():
    p = plan(4, 'E')
    state = payload(p, 12)
    state['optimizer']['state'] = {0: {'step': 12}}
    with pytest.raises(ValueError, match='moments'):
        validate_stages_state(state)
    a = plan(4, 'A')
    validate_stages_state(payload(a, 12))
    assert build_stages_state(a, 12)[0]['phase_local_completed'] == 12
    with pytest.raises(ValueError, match='phase start'):
        build_stages_state(a, 12, phase_start_step=12)


@pytest.mark.parametrize('change', [
    {'arm': 'X'}, {'views': ()}, {'views': plan(4).views[:3]}, {'synthetic': False},
    {'protocol_id': 'old'}, {'trained_sites': SITES[:-1]}, {'width': True},
    {'views': (plan(4).views[0],) * 4},
    {'views': tuple(replace(v, sequences=64) for v in plan(4).views)},
])
def test_invalid_plans_rejected(change):
    with pytest.raises(ValueError, match='stages require'):
        replace(plan(), **change)
    state = deepcopy(payload(plan(4, 'A'), 1))
    state['metadata']['sharing']['ordered_views'][0]['source_tokens'] = 1
    with pytest.raises(ValueError):
        validate_stages_state(state)
