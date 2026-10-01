from dataclasses import replace
import hashlib
import json

import pytest

from jscc.sharing_schedule import (
    BatchView, SharingPlan, SharingExposureLedger, TRAINED_SITES, sharing_lr_factor,
)


def make_plan(q=4, batch=2, synthetic=True):
    def digest(value):
        return hashlib.sha256(str(value).encode()).hexdigest()
    views = tuple(BatchView(digest(i), digest((i, 'mask')), digest((i, 'layout')),
                            batch, batch * (i + 2), batch * 3, batch * (i + 3))
                  for i in range(q))
    return SharingPlan(views, 'paired-study', synthetic=synthetic)


def test_production_cadence_and_postfreeze_accounting():
    plan = make_plan(200, 64, False)
    assert plan.condition_panels == {'task': 108, 'objective': 192}
    assert plan.additional_vanilla_panels == 1
    assert len(plan.saved_states) == len(set(plan.saved_states)) == 17
    assert sum(plan.learner(n).horizon for n in plan.learners) == 1200
    assert sum(plan.learner(n).horizon * plan.batch_size for n in plan.learners) == 76800
    assert plan.learner('shared').checkpoint_steps == (150, 300, 450, 600)
    assert plan.learner('specialist_enc_fn').assessment_steps == (100, 200)
    assert plan.raw_final_diagnostic == 'enc_l25'
    assert all((o.site == 'enc_l14') == (o.stage == 'post_freeze') for o in plan.observations)
    assert all(o.learner in ('shared', 'common_initialization')
               for o in plan.observations if o.site == 'enc_l14')
    assert sum(o.purpose == 'task' and o.learner == 'common_initialization'
               for o in plan.observations) == 4


@pytest.mark.parametrize('q,batch,synthetic', [(4, 2, True), (200, 64, False)])
def test_exact_rotations_stream_noise_and_exposure(q, batch, synthetic):
    plan = make_plan(q, batch, synthetic)
    shared = plan.learner('shared')
    expected = ('enc_l9', 'enc_l19', 'enc_fn', 'enc_l19', 'enc_fn', 'enc_l9',
                'enc_fn', 'enc_l9', 'enc_l19')
    assert tuple(shared.update_at(i).site for i in range(9)) == expected
    shared_ledger = SharingExposureLedger(shared)
    for index in range(shared.horizon):
        update = shared.update_at(index)
        specialist = plan.learner('specialist_' + update.site).update_at(update.site_local_index)
        assert (update.view, update.noise_key) == (specialist.view, specialist.noise_key)
        assert update.noise_key.condition == 'uniform'
        assert update.noise_key.draw_schema == 'runtime-awgn-replay-v2'
        shared_ledger.record(update, receipt=f'shared-{index}')
    totals = shared_ledger.finalize()
    for site in TRAINED_SITES:
        schedule = plan.learner('specialist_' + site)
        ledger = SharingExposureLedger(schedule)
        for index in range(q):
            ledger.record(schedule.update_at(index), receipt=f'{site}-{index}')
        assert totals[site] == ledger.finalize()[site]
    json.dumps(totals)


def test_lr_indexing():
    for horizon, warmup in ((200, 10), (600, 30), (4, 1), (12, 1)):
        assert sharing_lr_factor(0, horizon) == 0
        assert sharing_lr_factor(warmup, horizon) == 1
        assert sharing_lr_factor(horizon, horizon) == 0
        assert sharing_lr_factor(horizon - 1, horizon) > 0
    assert sharing_lr_factor(100, 200) == sharing_lr_factor(300, 600)
    for index in (-1, 201, True):
        with pytest.raises(ValueError):
            sharing_lr_factor(index, 200)


@pytest.mark.parametrize('tamper', ['view', 'mask', 'layout', 'global_noise', 'heldout',
                                     'reorder', 'duplicate', 'skipped', 'nonfinite'])
def test_observed_mismatch_is_terminal(tamper):
    schedule = make_plan().learner('shared')
    ledger = SharingExposureLedger(schedule)
    ledger.record(schedule.update_at(0), receipt='first')
    update = schedule.update_at(1)
    status = 'completed'
    if tamper in ('view', 'mask', 'layout'):
        field = {'view': 'view_sha256', 'mask': 'mask_sha256', 'layout': 'layout_sha256'}[tamper]
        update = replace(update, view=replace(update.view, **{field: 'f' * 64}))
    elif tamper == 'global_noise':
        update = replace(update, noise_key=replace(update.noise_key, site_local_batch=update.global_index))
    elif tamper == 'heldout':
        update = replace(update, site='enc_l14')
    elif tamper == 'reorder':
        update = schedule.update_at(2)
    elif tamper == 'duplicate':
        update = schedule.update_at(0)
    else:
        status = tamper
    with pytest.raises(ValueError):
        ledger.record(update, receipt='second', status=status)
    assert ledger.completed == 1
    with pytest.raises(ValueError, match='terminal'):
        ledger.record(schedule.update_at(1), receipt='retry')
    with pytest.raises(ValueError):
        ledger.finalize()


def test_missing_duplicate_receipt_and_invalid_plans():
    plan = make_plan()
    schedule = plan.learner('shared')
    ledger = SharingExposureLedger(schedule)
    with pytest.raises(ValueError, match='incomplete'):
        ledger.finalize()
    ledger.record(schedule.update_at(0), receipt='same')
    with pytest.raises(ValueError, match='unique'):
        ledger.record(schedule.update_at(1), receipt='same')
    repeated = replace(plan, views=(plan.views[0],) * 4).learner('shared')
    repeated_ledger = SharingExposureLedger(repeated)
    for index in range(repeated.horizon):
        repeated_ledger.record(repeated.update_at(index), receipt=f'repeated-{index}')
    assert repeated_ledger.finalize()['enc_l9']['updates'] == 4
    reordered_ledger = SharingExposureLedger(repeated)
    reordered_ledger.record(repeated.update_at(0), receipt='first')
    with pytest.raises(ValueError, match='ordered'):
        reordered_ledger.record(repeated.update_at(4), receipt='wrong-position')
    with pytest.raises(ValueError, match='production'):
        replace(plan, synthetic=False)
    for name in ('specialist_enc_l14', 'specialist_enc_l25', 'unknown'):
        with pytest.raises(ValueError):
            plan.learner(name)
    for index in (-1, schedule.horizon, True):
        with pytest.raises(ValueError):
            schedule.update_at(index)
    with pytest.raises(ValueError):
        replace(plan, layer_count=19)
