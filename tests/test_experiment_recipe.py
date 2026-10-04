"""Recipe resolution through YAML/CLI, topology and ordered plan contracts."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from jscc.experiment_recipe import resolve_recipe
from jscc.recipe_plan import RecipePlan
from jscc.encoder_stages_plan import Original13Plan, ORIGINAL13_SITES
from test_original13_joined import cli, prepared as prepared, execution_args, offline_dataset
from test_original13_plan_state import formal_views

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def recipe_files(tmp_path):
    recipe = yaml.safe_load((ROOT/'configs/recipes/original13.yaml').read_text())
    recipe['task_config'] = str(ROOT/'configs/tasks/hellaswag.yaml')
    path = tmp_path/'recipe.yaml'
    path.write_text(yaml.safe_dump(recipe))
    metadata = tmp_path/'config.json'
    metadata.write_text(json.dumps({'encoder': {'num_hidden_layers': 26, 'hidden_size': 1152},
                                    'decoder': {'num_hidden_layers': 26, 'hidden_size': 1152}}))
    return path, metadata, recipe


def test_original13_yaml_counts_and_plan_parity(recipe_files):
    path, metadata, _ = recipe_files
    result = cli('experiment_recipe.py')(['--config', str(path), '--model-config', str(metadata),
                                         '--output', str(path.parent/'resolved.json')])
    assert json.loads((path.parent/'resolved.json').read_text()) == result
    assert len(result['trajectories']) == 9
    assert result['matrix_counts']['optimizer_updates'] == 1170000
    views = formal_views()
    for item in result['trajectories']:
        assert item['counts'] == {'locations': 13, 'optimizer_updates': 130000,
            'base_stream_presentations': 160000, 'presentations_per_location': 160000,
            'site_presentations': 2080000}
        retained = Original13Plan(views, 'paired', item['ratio'], item['width'], 'a'*64, 'b'*64)
        assert item['checkpoint_states'] == [list(s) for s in retained.checkpoint_states]
        assert [(o['step'], o['state']) for o in item['observations']] == list(retained.observations)
        plan = RecipePlan(item, views, 'paired')
        for index in (0, 6, 12, 13, retained.reconstruction_stop, retained.horizon-1):
            assert plan.update_at(index) == retained.update_at(index)
    assert result['launch_authorized'] is False


def test_yaml_changes_resolve_to_complete_settings_and_order(recipe_files):
    path, metadata, recipe = recipe_files
    recipe.update(locations=['dec_l3', 'enc_l2', 'enc_emb'], updates_per_location=80,
                  widths=[32], reconstruction_percentages=[50])
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
    assert plan.progress(4, 0)['completed_per_site'] == {'dec_l3': 2, 'enc_l2': 1, 'enc_emb': 1}
    assert plan.progress(240, 120)['next_site'] is None
    with pytest.raises(ValueError, match='unrecorded reset'):
        plan.progress(121, 0)


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
    ('training', {'lr': float('nan')}, 'finite'),
    ('training', {'betas': [1, .9]}, 'betas'),
])
def test_invalid_settings(recipe_files, section, change, match):
    path, metadata, recipe = recipe_files
    recipe['overrides'].setdefault(section, {}).update(change)
    path.write_text(yaml.safe_dump(recipe))
    with pytest.raises(ValueError, match=match):
        resolve_recipe(path, model_config_path=metadata)


def test_resolution_uses_real_tiny_config_and_retained_entrypoint(prepared, tmp_path, monkeypatch):
    path, manifest = prepared
    config = deepcopy(manifest['configs']['8'])
    task = tmp_path/'task.yaml'
    task.write_text(yaml.safe_dump(config))
    recipe = yaml.safe_load((ROOT/'configs/recipes/original13.yaml').read_text())
    recipe.update(task_config=str(task), updates_per_location=400, widths=[8], reconstruction_percentages=[90])
    recipe_path = tmp_path/'recipe.yaml'
    recipe_path.write_text(yaml.safe_dump(recipe))
    result = cli('experiment_recipe.py')(['--config', str(recipe_path), '--model-config',
        str(tmp_path/'model/config.json'), '--output', str(tmp_path/'resolved.json')])
    item = result['trajectories'][0]
    assert item['locations'] == list(ORIGINAL13_SITES)
    assert item['counts']['optimizer_updates'] == 5200
    assert item['checkpoint_states'] == manifest['plans'][0]['checkpoint_states']
    offline_dataset(monkeypatch)
    # This ticket retains the production CLI; generalized recipe consumption
    # by the learner belongs to the downstream training/state owners.
    trained = cli('original13_training.py')(['train', *execution_args(path, manifest, tmp_path/'retained'),
                                            '--recipe', str(recipe_path)])
    assert trained['status'] == 'training_frozen' and trained['completed_updates'] == 5200
    assert len(trained['checkpoints']) == 11 and trained['observations_status'] == 'complete'
