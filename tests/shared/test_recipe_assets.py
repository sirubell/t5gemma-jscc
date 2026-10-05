"""Automatic preparation from explicit acquisition; real tiny local cache only."""
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

import pytest
import torch
import yaml

from jscc.activation_replay import canonical_digest, file_digest
from jscc.shared_assets import prepare_recipe_assets, discover_recipe_assets
from jscc.shared_preparation import validate_recipe_prepared, load_recipe_workflow
from request_helpers import invoke
from shared.test_recipe_workflow import workflow

pytest_plugins = ['shared.test_recipe_workflow']


def registered(assets):
    _, _, recipe, prepared, manifest = workflow(assets, 'acquired')
    directory = assets['root']/'registrations'
    directory.mkdir()
    invoke(assets['root'], 'shared', 'register', {'registration': {'from_prepared': str(prepared),
           'model_name': str(assets['model']), 'output': str(directory/'assets.json')}})
    return recipe, directory, manifest


def test_explicit_registration_automatic_prepare_and_repeated_cache_only(joined_assets, monkeypatch):
    recipe, directory, acquired = registered(joined_assets)
    from jscc import shared_assets as recipe_assets, asset_identity, shared_inputs
    hashes = []
    tensor_hashes = []
    real = file_digest
    def measured(path):
        path = Path(path)
        if path.suffix in ('.safetensors', '.bin', '.pt', '.arrow'):
            hashes.append(str(path))
        return real(path)
    for module in (recipe_assets, asset_identity, shared_inputs):
        monkeypatch.setattr(module, 'file_digest', measured)
    identity = recipe_assets.state_dict_identity
    monkeypatch.setattr(recipe_assets, 'state_dict_identity', lambda state: tensor_hashes.append(1) or identity(state))
    monkeypatch.setattr('transformers.AutoModelForSeq2SeqLM.from_pretrained', lambda *a, **k: pytest.fail('backbone loading forbidden'))
    monkeypatch.setattr('huggingface_hub.snapshot_download', lambda *a, **k: pytest.fail('download forbidden'))
    before = torch.get_rng_state()
    first = prepare_recipe_assets(recipe, registrations=directory, output=joined_assets['root']/'auto1')
    assert torch.equal(before, torch.get_rng_state())
    assert len(hashes) == 1 and hashes[0].endswith('initial.pt') and len(tensor_hashes) == 1
    hashes.clear()
    tensor_hashes.clear()
    changed = yaml.safe_load(recipe.read_text())
    changed['locations'].reverse()
    changed['updates_per_location'] = 16
    changed['overrides']['data'] = {'num_train': 20, 'selection_seed': 2}
    changed['overrides']['training']['lr'] = .003
    changed['observation']['items'] = 1
    recipe.write_text(yaml.safe_dump(changed))
    started = perf_counter()
    from jscc.shared_config import read_recipe
    science = read_recipe(recipe)
    science.pop('schema')
    science['task_config'] = str((recipe.parent/science['task_config']).resolve())
    second = invoke(joined_assets['root'], 'shared', 'prepare', {'preparation': {'registrations': str(directory),
                    'output': str(joined_assets['root']/'auto2')}}, science)
    validate_recipe_prepared(second)
    checked = validate_recipe_prepared(second)
    assert len(checked['trajectories']) == len(second['resolved']['trajectories'])
    assert checked['launch_authorized'] is False
    elapsed = perf_counter()-started
    assert hashes == [] and tensor_hashes == []
    assert first['initializations'] == second['initializations']
    assert second['cache'] == acquired['cache']
    assert second['resolved']['trajectories'][0]['counts']['optimizer_updates'] == 48
    observation = json.loads(Path(second['observation_template']['path']).read_text())
    assert observation['input_ids'] == [0]
    assert observation['noise'] == joined_assets['fixed']['development']['noise']
    # The full retained cache still excludes all acquired development families.
    assert json.loads(Path(second['cache']['path']).read_text())['fixed_data_ids'] == joined_assets['fixed']
    _, _, plan, stream, _, _, _ = load_recipe_workflow(joined_assets['root']/'auto2/prepared.json',
        checksum=file_digest(joined_assets['root']/'auto2/prepared.json'), width=8, ratio=50)
    assert plan is not None and stream is not None
    assert len(stream.data_ids['train_rows']) == 20
    assert plan.trained_sites == tuple(changed['locations'])
    (joined_assets['root']/'automatic-cost.json').write_text(json.dumps({
        'repeated_prepare_and_check_seconds': elapsed, 'large_file_hashes': hashes,
        'tensor_identity_hashes': tensor_hashes, 'selected_rows': 20, 'q': 16, 'locations': 3}))
    with pytest.raises(FileExistsError):
        prepare_recipe_assets(recipe, registrations=directory, output=joined_assets['root']/'auto2')


def test_missing_conflicting_assets_and_changed_initializer_fail(joined_assets):
    recipe, directory, _ = registered(joined_assets)
    registration = directory/'assets.json'
    record = json.loads(registration.read_text())
    record['model_revision'] = 'wrong-pin'
    record['identity'] = canonical_digest({k: v for k, v in record.items() if k != 'identity'})
    registration.write_text(json.dumps(record))
    with pytest.raises(FileNotFoundError, match='explicitly acquire/register'):
        discover_recipe_assets(recipe, directory)
    record['model_revision'] = 'tiny-workflow'
    record['identity'] = canonical_digest({k: v for k, v in record.items() if k != 'identity'})
    registration.write_text(json.dumps(record))
    conflicting = deepcopy(record)
    conflicting['synthetic_cpu'] = False
    conflicting['identity'] = canonical_digest({k: v for k, v in conflicting.items() if k != 'identity'})
    (directory/'conflict.json').write_text(json.dumps(conflicting))
    with pytest.raises(ValueError, match='ambiguous/conflicting'):
        discover_recipe_assets(recipe, directory)
    (directory/'conflict.json').unlink()
    first = prepare_recipe_assets(recipe, registrations=directory, output=joined_assets['root']/'auto1')
    initial = Path(first['initializations']['8']['path'])
    initial.write_bytes(initial.read_bytes())
    with pytest.raises(ValueError, match='trust signature'):
        prepare_recipe_assets(recipe, registrations=directory, output=joined_assets['root']/'auto2')


def test_width_seed_initializers_and_panel_limits(joined_assets):
    recipe, directory, _ = registered(joined_assets)
    changed = yaml.safe_load(recipe.read_text())
    changed['widths'] = [8, 12]
    recipe.write_text(yaml.safe_dump(changed))
    result = prepare_recipe_assets(recipe, registrations=directory, output=joined_assets['root']/'widths')
    assert set(result['initializations']) == {'8', '12'}
    assert len(result['resolved']['trajectories']) == 2
    task = yaml.safe_load(joined_assets['task'].read_text())
    task['seed'] = 23
    joined_assets['task'].write_text(yaml.safe_dump(task))
    seeded = prepare_recipe_assets(recipe, registrations=directory, output=joined_assets['root']/'seeded')
    assert seeded['initializations']['8']['seed'] == 23
    assert seeded['initializations']['8']['state_identity'] != result['initializations']['8']['state_identity']
    changed['evaluation']['items'] = 4
    recipe.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match='exceeds acquired fixed panel'):
        prepare_recipe_assets(recipe, registrations=directory, output=joined_assets['root']/'bad-panel')
    assert not (joined_assets['root']/'bad-panel').exists()


def test_direct_registration_and_hf_snapshot_blob_preparation(joined_assets):
    from jscc.asset_identity import bind_model_assets
    from jscc.shared_inputs import bind_recipe_data_cache
    recipe, _, _, _, acquired = workflow(joined_assets, 'existing')
    recipe = recipe/'recipe.yaml'
    repository = joined_assets['root']/'hf/models--fixture'
    snapshot = repository/'snapshots/tiny-workflow'
    blobs = repository/'blobs'
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    for original in joined_assets['model'].iterdir():
        if original.is_file():
            blob = blobs/file_digest(original)
            if not blob.exists():
                blob.write_bytes(original.read_bytes())
            (snapshot/original.name).symlink_to(blob)
    binding = bind_model_assets(snapshot, 'tiny-workflow')
    binding_path = joined_assets['root']/'snapshot-binding.json'
    binding_path.write_text(json.dumps(binding))
    cache = bind_recipe_data_cache(dataset_path=json.loads(joined_assets['cache'].read_text())['dataset']['path'],
        assets=snapshot, dataset_name='fixture/hellaswag', dataset_revision='fixed-data',
        model_revision='tiny-workflow', fixed_data_ids=joined_assets['fixed'], output=joined_assets['root']/'snapshot-cache.json')
    registrations = joined_assets['root']/'snapshot-registrations'
    registrations.mkdir()
    def ref(path):
        return {'path': str(path), 'sha256': file_digest(path)}
    invoke(joined_assets['root'], 'shared', 'register', {'registration': {'model_name': str(joined_assets['model']),
           'output': str(registrations/'pinned.json'), 'references': {'model_binding': ref(binding_path), 'cache': ref(cache),
           'observation_template': acquired['observation_template'], 'terminal_template': acquired['terminal_template'],
           'synthetic_cpu': True}}})

    result = prepare_recipe_assets(recipe, registrations=registrations, output=joined_assets['root']/'snapshot-prepared')
    assert Path(result['model_config']['path']).parent == blobs
    validate_recipe_prepared(result)
    assert result['model_assets']['path'] == str(snapshot)


def test_low_level_stat_reference_cannot_skip_tensor_setup_validation(joined_assets):
    from jscc.asset_paths import asset_stat
    from jscc.shared_preparation import bind_recipe_workflow
    _, _, recipe, _, acquired = workflow(joined_assets, 'caller-assets')
    initializations = deepcopy(acquired['initializations'])
    reference = initializations['8']
    reference['stat'] = asset_stat(reference['path'])
    reference['state_identity'] = 'incorrect-state'
    with pytest.raises(ValueError, match='initializer tensor identity differs'):
        bind_recipe_workflow(recipe, model_config_path=joined_assets['model']/'config.json',
            model_path=joined_assets['model'], model_binding=acquired['model_binding'],
            cache_manifest=acquired['cache']['path'], cache_sha256=acquired['cache']['sha256'],
            initializations=initializations, observation_template=acquired['observation_template'],
            terminal_template=acquired['terminal_template'], study_pairing_id='caller-assets',
            output=joined_assets['root']/'incorrect.json', synthetic=True)
    assert not (joined_assets['root']/'incorrect.json').exists()


def test_owned_initializer_metadata_requires_acquisition_evidence(joined_assets):
    recipe, directory, _ = registered(joined_assets)
    manifest = prepare_recipe_assets(recipe, registrations=directory, output=joined_assets['root']/'auto')
    manifest['initializations']['8']['state_identity'] = 'stale-state'
    manifest['manifest_identity'] = canonical_digest({k: v for k, v in manifest.items() if k != 'manifest_identity'})
    with pytest.raises(ValueError, match='acquisition evidence differs'):
        validate_recipe_prepared(manifest)


def test_simultaneous_first_use_waits_for_atomic_initializer_receipt(joined_assets, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time
    from jscc import shared_assets as recipe_assets
    from jscc.config import load_config
    config = load_config(joined_assets['task'])
    config['codec']['bottleneck_dim'] = 8
    directory = joined_assets['root']/'concurrent-initializers'
    entered = threading.Event()
    write = recipe_assets._write
    def slow_publish(path, value):
        entered.set()
        time.sleep(.1)
        return write(path, value)
    monkeypatch.setattr(recipe_assets, '_write', slow_publish)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(recipe_assets._initializer, config, directory)
        assert entered.wait(timeout=5)
        second = executor.submit(recipe_assets._initializer, config, directory)
        assert first.result(timeout=5) == second.result(timeout=5)
    assert len(list(directory.glob('*/initial.pt'))) == 1
    assert len(list(directory.glob('*/reference.json'))) == 1
