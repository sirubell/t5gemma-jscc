"""One explicitly bound depth pilot; q200 remains the default finite protocol."""

import jscc
from dataclasses import replace
from pathlib import Path
import pytest
from jscc.config import load_config
from jscc.batch_partition import partition_policy
from jscc.checkpoint_readers import _config_contract
from jscc.checkpoint_readers import DEPTH_PROTOCOL, TRAINED_SITES
from jscc.shared_plan import SharingExposureLedger
from shared.test_shared_protocol_schedule import make_plan






def depth_plan():
    # Synthetic construction supplies unique cheap identities; the tested plan is production.
    return replace(make_plan(400, 64), synthetic=False, protocol_id=DEPTH_PROTOCOL)


def test_depth_finite_schedule_and_paired_exposure():
    plan = depth_plan()
    assert plan.learner('shared').checkpoint_steps == (300, 600, 900, 1200)
    assert plan.learner('specialist_enc_l9').checkpoint_steps == (100, 200, 300, 400)
    assert len(plan.saved_states) == 17
    assert plan.condition_panels == {'task': 108, 'objective': 192}
    assert sum(plan.learner(n).horizon for n in plan.learners) == 2400
    assert all((o.site == 'enc_l14') == (o.stage == 'post_freeze') for o in plan.observations)
    shared = plan.learner('shared')
    ledger = SharingExposureLedger(shared)
    for i in range(1200):
        update = shared.update_at(i)
        paired = plan.learner('specialist_' + update.site).update_at(update.site_local_index)
        assert update.view == paired.view and update.noise_key == paired.noise_key
        ledger.record(update, receipt=f'shared-{i}')
    counts = ledger.finalize()
    for site in TRAINED_SITES:
        assert counts[site]['updates'] == 400
        assert counts[site]['sequences'] == 25600
    with pytest.raises(ValueError, match='finite horizon'):
        shared.update_at(1200)
    assert make_plan(200, 64, False).learner('shared').horizon == 600


def test_depth_requires_explicit_protocol_and_no_cycled_views():
    plan = depth_plan()
    with pytest.raises(ValueError, match='production'):
        replace(plan, protocol_id=None)
    with pytest.raises(ValueError, match='production'):
        replace(plan, views=plan.views[:200])
    with pytest.raises(ValueError, match='cycling'):
        replace(plan, views=plan.views[:200] * 2)


def depth_config_manifest():
    config = load_config(Path(jscc.__file__).resolve().parents[1] / 'configs/experiments/ordinary/shared_q200_effective64.yaml')
    manifest = dict(synthetic_cpu=False, q=400, batch_size=64, cell='D-N', selected_bottleneck_dim=512,
                    execution_partition=partition_policy(16), protocol_id=DEPTH_PROTOCOL,
                    model_revision=config['model']['revision'], recipe_identity='K+0.1R-native-effective64-q400',
                    initialization_policy={'seed': 0, 'fresh_cpu_codec': True, 'baseline_state_reused': False})
    return config, manifest


@pytest.mark.parametrize('change', ['cell', 'width', 'partition', 'quota', 'seed', 'recipe', 'initialization'])
def test_depth_rejects_broader_scientific_scope(change):
    config, manifest = depth_config_manifest()
    _config_contract(config, manifest)
    if change == 'cell':
        manifest['cell'] = 'D-LN'
        config['codec'].update(architecture='direct_outer_ln', layernorm='both')
    elif change == 'width':
        manifest['selected_bottleneck_dim'] = config['codec']['bottleneck_dim'] = 1152
    elif change == 'partition':
        manifest['execution_partition'] = partition_policy(32)
        config['training'].update(batch_size=32, gradient_accumulation=2)
    elif change == 'quota':
        manifest['q'] = 200
    elif change == 'seed':
        config['seed'] = 1
    elif change == 'recipe':
        manifest['recipe_identity'] = 'arbitrary-depth'
    else:
        manifest['initialization_policy']['baseline_state_reused'] = True
    with pytest.raises(ValueError):
        _config_contract(config, manifest)
