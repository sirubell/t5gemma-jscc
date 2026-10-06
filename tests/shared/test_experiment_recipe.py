"""Recipe resolution through YAML/CLI, topology and ordered plan contracts."""

import jscc
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from jscc.shared_config import resolve_recipe, validate_retained_recipe, read_recipe
from jscc.shared_plan import RecipePlan, phase_lr_factor
from jscc.experiment_schedule import NoiseKey
from jscc.shared_plan import ScheduledUpdate
from jscc.checkpoint_readers import Original13Plan, ORIGINAL13_SITES
from jscc.checkpoint_readers import validate_original13_config
from shared.test_shared_case_joined import prepared as prepared
from shared.test_shared_case_plan_state import formal_views

ROOT = Path(jscc.__file__).resolve().parents[1]


@pytest.fixture
def recipe_files(tmp_path):
    recipe = read_recipe(ROOT/'configs/experiments/shared/shared_hellaswag_reference.yaml')
    recipe['task_config'] = str((ROOT/'configs/experiments/shared'/recipe['task_config']).resolve())
    path = tmp_path/'recipe.yaml'
    path.write_text(yaml.safe_dump(recipe))
    metadata = tmp_path/'config.json'
    metadata.write_text(json.dumps({'encoder': {'text_config': {'num_hidden_layers': 26, 'hidden_size': 1152}},
                                    'decoder': {'num_hidden_layers': 26, 'hidden_size': 1152}}))
    return path, metadata, recipe


def test_original13_yaml_counts_and_plan_parity(recipe_files):
    path, metadata, _ = recipe_files
    from request_helpers import invoke
    science = read_recipe(path)
    science.pop('schema')
    result = invoke(path.parent, 'shared', 'check', {'resolution': {'model_config': str(metadata),
                    'output': str(path.parent/'resolved.json')}}, science)
    assert json.loads((path.parent/'resolved.json').read_text()) == result
    assert len(result['trajectories']) == 9
    assert result['matrix_counts']['optimizer_updates'] == 1170000
    views = formal_views()
    for item in result['trajectories']:
        validate_original13_config(item['config'], item['width'])
        assert item['counts'] == {'locations': 13, 'optimizer_updates': 130000,
            'base_stream_presentations': 160000, 'presentations_per_location': 160000,
            'site_presentations': 2080000}
        retained = Original13Plan(views, 'paired', item['ratio'], item['width'], 'a'*64, 'b'*64)
        assert item['checkpoint_states'] == [list(s) for s in retained.checkpoint_states]
        assert [(o['step'], o['state']) for o in item['observations']] == list(retained.observations)
        plan = RecipePlan(item, views, 'paired')
        for index in (0, 6, 12, 13, retained.reconstruction_stop, retained.horizon-1):
            assert plan.update_at(index) == retained.update_at(index)
            assert plan.objective_at(index) == retained.objective_at(index)
    assert result['launch_authorized'] is False


def test_yaml_changes_resolve_to_complete_settings_and_order(recipe_files):
    path, metadata, recipe = recipe_files
    recipe.update(locations=['dec_l3', 'enc_l2', 'enc_emb'], updates_per_location=80,
                  widths=[32], reconstruction_percentages=[50])
    recipe['objectives'] = {'encoder': {'hidden': .2, 'memory': 0.},
                            'decoder': {'hidden': .3, 'memory': .4}, 'functional_kl': .7}
    recipe['overrides']['data'] = {'num_train': 12, 'selection_seed': 23}
    recipe['overrides']['training'].update(lr=.001, weight_decay=.02, warmup_ratio=.1,
                                         min_lr_ratio=.2, betas=[.8, .95], eps=1e-7)
    recipe['save']['functional_observation_fraction'] = .5
    recipe['observation']['items'] = 16
    path.write_text(yaml.safe_dump(recipe))
    item = resolve_recipe(path, model_config_path=metadata)['trajectories'][0]
    assert item['counts']['optimizer_updates'] == 240
    assert item['config']['data']['num_train'] == 12
    assert item['config']['training']['lr'] == .001
    assert item['config']['training']['betas'] == [.8, .95]
    assert item['observations'][1]['step'] == 180
    assert item['observations'][0]['items'] == 16
    # Identical content in several stream positions is valid, independent of Q.
    views = (formal_views()[0],) * 80
    plan = RecipePlan(item, views, 'paired')
    assert [plan.update_at(i).site for i in range(6)] == recipe['locations'] * 2
    assert [plan.update_at(i).site_local_index for i in range(6)] == [0, 0, 0, 1, 1, 1]
    for start, kind in ((0, 'reconstruction'), (120, 'combined')):
        # First/last cycles and partial-cycle indices retain the declared order.
        for index in (start, start+1, start+2, start+3, start+118, start+119):
            local, position = divmod(index, 3)
            site = recipe['locations'][position]
            role = 'decoder' if position == 0 else 'encoder'
            assert plan.stage_at(index).objective_kind == kind
            assert plan.objective_at(index) == {**recipe['objectives'][role],
                                                'kl': .7 if start else 0.}
            assert plan.update_at(index) == ScheduledUpdate('shared', index, site, local, views[local],
                NoiseKey('paired', kind, site, local, views[local].view_sha256, views[local].layout_sha256,
                         condition='uniform', draw_schema='original13-role-awgn-v1'),
                phase_lr_factor(index-start, 120, .1, .2))
    for index in (True, False, 1.0, '1', None, -1, plan.horizon, plan.horizon+1):
        for lookup in (plan.stage_at, plan.objective_at, plan.update_at):
            with pytest.raises(ValueError, match='update outside recipe horizon'):
                lookup(index)
    assert plan.progress(4, 0)['completed_per_site'] == {'dec_l3': 2, 'enc_l2': 1, 'enc_emb': 1}
    assert plan.progress(240, 120)['next_site'] is None
    with pytest.raises(ValueError, match='unrecorded reset'):
        plan.progress(121, 0)
    original = plan.update_at(0)
    item['locations'][0] = 'dec_l999'
    exposed = plan.description
    exposed['locations'][0] = 'dec_l999'
    exposed['config']['training']['lr'] = .1
    assert plan.update_at(0) == original
    assert plan.description['config']['training']['lr'] == .001


@pytest.mark.parametrize('change,match', [
    ({'updates_per_location': True}, 'positive integer'),
    ({'total_updates': 130000}, 'derived budgets'),
    ({'locations': []}, 'nonempty ordered'),
    ({'locations': ['enc_fn', 'enc_fn']}, 'duplicates'),
    ({'locations': ['enc_l26']}, 'actual model topology'),
    ({'locations': ['dec_l25']}, 'receiver suffix'),
    ({'locations': ['enc_l01']}, 'invalid model location'),
    ({'updates_per_location': 13}, 'integral per-location'),
    ({'widths': [False]}, 'positive integer'),
    ({'reconstruction_percentages': [100]}, 'phase ratios'),
    ({'reconstruction_percentages': [-1]}, 'phase ratios'),
    ({'reconstruction_percentages': [True]}, 'phase ratios'),
    ({'reconstruction_percentages': [0.0]}, 'phase ratios'),
    ({'reconstruction_percentages': [1.5]}, 'phase ratios'),
])
def test_invalid_recipe_fields(recipe_files, change, match):
    path, metadata, recipe = recipe_files
    recipe.update(change)
    path.write_text(yaml.safe_dump(recipe))
    with pytest.raises(ValueError, match=match):
        resolve_recipe(path, model_config_path=metadata)


@pytest.mark.parametrize('section,change,match', [
    ('data', {'num_train': 0}, 'positive integer'),
    ('data', {'row_ids': []}, 'nonempty'),
    ('training', {'max_steps': 20}, 'conflicting'),
    ('training', {'batch_size': 8}, 'native16'),
    ('training', {'gradient_accumulation': 2}, 'native16'),
    ('training', {'gradient_accumulation': True}, 'native16'),
    ('training', {'gradient_accumulation': 1.0}, 'native16'),
    ('training', {'lr': float('nan')}, 'finite'),
    ('training', {'betas': [1, .9]}, 'betas'),
    ('channel', {'train_snr_range': [float('nan'), 18]}, 'finite'),
    ('channel', {'train_snr_range': [18, -6]}, 'ordered'),
    ('channel', {'clean_film_snr': float('inf')}, 'finite'),
    ('channel', {'train_noise': 'true'}, 'boolean'),
])
def test_invalid_settings(recipe_files, section, change, match):
    path, metadata, recipe = recipe_files
    recipe['overrides'].setdefault(section, {}).update(change)
    path.write_text(yaml.safe_dump(recipe))
    with pytest.raises(ValueError, match=match):
        resolve_recipe(path, model_config_path=metadata)


SCORING_CONDITION_FIELDS = [
    ('observation', 'conditions'),
    ('evaluation', 'conditions'),
    ('evaluation', 'supplemental_conditions'),
]


@pytest.mark.parametrize('section,field', SCORING_CONDITION_FIELDS)
@pytest.mark.parametrize('conditions', [[6, 6], [6, 6.0]])
def test_duplicate_scoring_conditions_are_rejected(recipe_files, section, field, conditions):
    path, metadata, recipe = recipe_files
    recipe[section][field] = conditions
    path.write_text(yaml.safe_dump(recipe))
    with pytest.raises(ValueError, match=rf'{section}\.{field}.*duplicates'):
        resolve_recipe(path, model_config_path=metadata)


@pytest.mark.parametrize('section', ['observation', 'evaluation'])
def test_duplicate_no_noise_conditions_are_rejected(recipe_files, section):
    path, metadata, recipe = recipe_files
    recipe[section]['conditions'] = ['no_noise', 'no_noise']
    path.write_text(yaml.safe_dump(recipe))
    with pytest.raises(ValueError, match=rf'{section}\.conditions.*duplicates'):
        resolve_recipe(path, model_config_path=metadata)


@pytest.mark.parametrize('section,field', SCORING_CONDITION_FIELDS)
@pytest.mark.parametrize('conditions', [None, 6, 'no_noise', {6: None}])
def test_scoring_conditions_require_lists(recipe_files, section, field, conditions):
    path, metadata, recipe = recipe_files
    recipe[section][field] = conditions
    path.write_text(yaml.safe_dump(recipe))
    with pytest.raises(ValueError, match=rf'{section}\.{field}.*list'):
        resolve_recipe(path, model_config_path=metadata)


@pytest.mark.parametrize('section', ['observation', 'evaluation'])
def test_initial_scoring_conditions_must_be_nonempty(recipe_files, section):
    path, metadata, recipe = recipe_files
    recipe[section]['conditions'] = []
    path.write_text(yaml.safe_dump(recipe))
    with pytest.raises(ValueError, match=rf'{section}\.conditions.*nonempty'):
        resolve_recipe(path, model_config_path=metadata)


@pytest.mark.parametrize('supplemental', [[], [-6, 0, 6.0]])
def test_unique_scoring_conditions_preserve_order_and_scope_overlap(recipe_files, supplemental):
    path, metadata, recipe = recipe_files
    recipe['observation']['conditions'] = [6.0, 'no_noise', -6]
    recipe['evaluation']['conditions'] = ['no_noise', -6, 6]
    recipe['evaluation']['supplemental_conditions'] = supplemental
    path.write_text(yaml.safe_dump(recipe))
    resolved = resolve_recipe(path, model_config_path=metadata)
    for item in resolved['trajectories']:
        assert all(request['conditions'] == [6.0, 'no_noise', -6] for request in item['observations'])
        assert item['evaluation']['conditions'] == ['no_noise', -6, 6]
        assert item['evaluation']['supplemental_conditions'] == supplemental


@pytest.mark.parametrize('condition', ['no_noise', True, float('nan'), float('inf'), 1001])
def test_supplemental_scoring_conditions_retain_numeric_policy(recipe_files, condition):
    path, metadata, recipe = recipe_files
    recipe['evaluation']['supplemental_conditions'] = [condition]
    path.write_text(yaml.safe_dump(recipe))
    with pytest.raises(ValueError, match='supplemental SNR.*finite number'):
        resolve_recipe(path, model_config_path=metadata)


@pytest.fixture
def retained_resolution(prepared, tmp_path):
    path, manifest = prepared
    config = deepcopy(manifest['configs']['8'])
    task = tmp_path/'task.yaml'
    task.write_text(yaml.safe_dump(config))
    recipe = read_recipe(ROOT/'configs/experiments/shared/shared_hellaswag_reference.yaml')
    recipe.update(task_config=str(task), updates_per_location=400, widths=[8], reconstruction_percentages=[90])
    recipe_path = tmp_path/'recipe.yaml'
    recipe_path.write_text(yaml.safe_dump(recipe))
    result = resolve_recipe(recipe_path, model_config_path=tmp_path/'model/config.json')
    return path, manifest, config, recipe_path, result


def test_resolution_uses_real_tiny_config(retained_resolution):
    path, manifest, config, recipe_path, result = retained_resolution
    item = result['trajectories'][0]
    assert item['locations'] == list(ORIGINAL13_SITES)
    assert item['counts']['optimizer_updates'] == 5200
    assert item['checkpoint_states'] == manifest['plans'][0]['checkpoint_states']
    retained_plan = Original13Plan.from_resolved(manifest['plans'][0])
    validate_retained_recipe(item, retained_plan, config, manifest['observation_request'])
    for field in ('seed', 'deterministic_algorithms'):
        changed = deepcopy(item)
        if field == 'seed':
            changed['config']['seed'] = 999
        else:
            changed['config']['training'][field] = False
        with pytest.raises(ValueError, match='complete config'):
            validate_retained_recipe(changed, retained_plan, config, manifest['observation_request'])


def test_public_check_uses_resolved_small_recipe(retained_resolution, tmp_path):
    from request_helpers import invoke
    _, _, _, recipe_path, _ = retained_resolution
    science = read_recipe(recipe_path)
    science.pop('schema')
    science.update(locations=['enc_fn'], updates_per_location=8, widths=[8],
                   reconstruction_percentages=[50])
    science['save']['functional_observation_fraction'] = .5
    result = invoke(tmp_path, 'shared', 'check', {'resolution': {
        'model_config': str(tmp_path/'model/config.json')}}, science)
    item = result['trajectories'][0]
    assert item['locations'] == ['enc_fn'] and item['counts']['optimizer_updates'] == 8
    assert item['width'] == 8 and item['ratio'] == 50
    assert result['launch_authorized'] is False


def test_zero_warmup_uses_configured_peak_at_first_update():
    assert phase_lr_factor(0, 40, 0, 0) == 1
    assert phase_lr_factor(40, 40, 0, 0) == 0
