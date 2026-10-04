from pathlib import Path

import pytest

from jscc.asset_paths import asset_file_path, normalize_asset_path
from jscc.encoder_stages_preparation import bind_recipe_data_cache
from jscc.original13_inputs import validate_recipe_cache
from jscc.original13_provenance import bind_model_assets, verify_model_assets


def snapshot(tmp_path):
    repo = tmp_path / 'models--example'
    root = repo / 'snapshots' / 'revision'
    blobs = repo / 'blobs'
    root.mkdir(parents=True)
    blobs.mkdir()
    for name in ('config.json', 'tokenizer.json', 'model.safetensors'):
        blob = blobs / (name + '-blob')
        blob.write_text('{}')
        (root / name).symlink_to(Path('../../blobs') / blob.name)
    return root


def test_snapshot_equivalence_and_escape(tmp_path):
    root = snapshot(tmp_path)
    assert asset_file_path(root, 'config.json') == root.parent.parent / 'blobs/config.json-blob'
    for name in ('../blobs/config.json-blob', '/config.json', 'config.json/..'):
        with pytest.raises(ValueError):
            asset_file_path(root, name)
    outside = tmp_path / 'outside'
    outside.write_text('{}')
    (root / 'config.json').unlink()
    (root / 'config.json').symlink_to(outside)
    with pytest.raises(ValueError, match='symlink'):
        asset_file_path(root, 'config.json')


def test_model_signature_changes_and_legacy(tmp_path):
    root = snapshot(tmp_path)
    binding = bind_model_assets(root, 'revision', production=True)
    config = {'model': {'name': str(root), 'revision': 'revision', 'asset_binding': binding}}
    verify_model_assets(config)
    asset_file_path(root, 'model.safetensors').write_text('[]')
    with pytest.raises(ValueError, match='trust signature'):
        verify_model_assets(config)
    binding.pop('trust')
    verify_model_assets(config)
    with pytest.raises(ValueError, match='bytes differ'):
        verify_model_assets(config, verify_bytes=True)


def test_cache_signature_and_repoint(tmp_path):
    import json
    from jscc.activation_replay import canonical_digest
    root = snapshot(tmp_path)
    arrow = tmp_path / 'data.arrow'
    arrow.write_bytes(b'fixture')
    receipt_path = bind_recipe_data_cache(dataset_path=arrow, assets=root, dataset_name='fixture',
        dataset_revision='revision', model_revision='revision', fixed_data_ids={}, output=tmp_path / 'receipt.json')
    receipt = json.loads(receipt_path.read_text())
    assert validate_recipe_cache(receipt) == (arrow, root)
    replacement = root.parent.parent / 'blobs/replacement'
    replacement.write_text('{}')
    (root / 'config.json').unlink()
    (root / 'config.json').symlink_to('../../blobs/replacement')
    with pytest.raises(ValueError, match='trust signature'):
        validate_recipe_cache(receipt)
    receipt.pop('trust')
    receipt['identity'] = canonical_digest({k: v for k, v in receipt.items() if k != 'identity'})
    validate_recipe_cache(receipt)


def test_mount_alias():
    assert normalize_asset_path('/work/example') == Path('/raid/work/example')


def test_reuse_does_not_hash_immutable_assets(tmp_path, monkeypatch):
    root = snapshot(tmp_path)
    binding = bind_model_assets(root, 'revision', production=True)
    def unexpected_hash(_path):
        raise AssertionError('repeated verification hashed asset')
    monkeypatch.setattr('jscc.original13_provenance.file_digest', unexpected_hash)
    config = {'model': {'name': str(root), 'revision': 'revision', 'asset_binding': binding}}
    verify_model_assets(config)
    verify_model_assets(config)


def test_added_consumed_file_invalidates_acquisition(tmp_path):
    root = snapshot(tmp_path)
    binding = bind_model_assets(root, 'revision', production=True)
    (root/'tokenizer_config.json').write_text('{}')
    config = {'model': {'name': str(root), 'revision': 'revision', 'asset_binding': binding}}
    with pytest.raises(ValueError, match='layout changed'):
        verify_model_assets(config)
