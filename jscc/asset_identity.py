"""Immutable model, parameter and source asset bindings."""
from __future__ import annotations

import copy
from .asset_paths import asset_file_path, asset_trust_signature, normalize_asset_path, verify_asset_record
from .activation_replay import file_digest
from pathlib import Path
import torch
from .activation_replay import canonical_digest
from .presentation import tensor_digest

# Transferred from original13_provenance; serialized meanings remain unchanged.
def _model_asset_names(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
                  and p.suffix in {'.json', '.safetensors', '.bin', '.model', '.txt'}
                  and '.cache' not in p.relative_to(root).parts)


def model_asset_inventory(path):
    """Inventory consumed local HF configuration/tokenizer/weight files once."""
    root = normalize_asset_path(path)
    files = {name: {'bytes': asset_file_path(root, name).stat().st_size,
                    'sha256': file_digest(asset_file_path(root, name))}
             for name in _model_asset_names(root)}
    if ('config.json' not in files or not any(n.endswith(('.safetensors', '.bin')) for n in files)
            or not any(n.startswith('tokenizer') or n.startswith('vocab') for n in files)):
        raise ValueError('original13 complete local model/config/tokenizer assets required')
    return files


def bind_model_assets(path, revision, *, tokenizer_files=None, production=False):
    """Bind existing assets, or explicitly leave an unavailable target pending."""
    root = normalize_asset_path(path)
    reference = {'schema': 'original13-model-assets-v1', 'path': str(root), 'revision': revision,
                 'production': production, 'tokenizer_files': copy.deepcopy(tokenizer_files or {})}
    if not root.is_dir():
        return {**reference, 'status': 'pending', 'files': {}}
    if production and (root.name != revision or root.parent.name != 'snapshots'):
        raise ValueError('original13 production requires the pinned HF snapshot directory')
    files = model_asset_inventory(root)
    for name, expected in reference['tokenizer_files'].items():
        if name not in files or files[name]['sha256'] != expected['sha256'] or files[name]['bytes'] != expected['bytes']:
            raise ValueError('original13 cached tokenizer/config differs from retained inputs: ' + name)
    return {**reference, 'status': 'complete', 'files': files,
            'trust': {name: asset_trust_signature(asset_file_path(root, name)) for name in files}}


def verify_model_assets(config, *, require_complete=True, verify_bytes=False):
    """Check acquisition-bound immutable assets before model construction.

    Startup checks paths, presence, sizes and acquired stat signatures. Acquisition/transfer must hash
    bytes with bind_model_assets; use verify_bytes=True for an explicit audit.
    New receipts detect changed target/stat state; legacy receipts check size.
    """
    binding = config['model'].get('asset_binding')
    if (not isinstance(binding, dict) or binding.get('schema') != 'original13-model-assets-v1'
            or binding.get('revision') != config['model']['revision']
            or normalize_asset_path(binding.get('path', '')) != normalize_asset_path(config['model']['name'])
            or type(binding.get('production')) is not bool
            or not isinstance(binding.get('tokenizer_files'), dict)):
        raise ValueError('original13 model asset binding differs')
    if binding.get('status') == 'pending':
        if require_complete:
            raise ValueError('original13 model assets pending; bind the existing target snapshot before execution')
        if binding.get('files') != {}:
            raise ValueError('pending original13 assets cannot claim verified files')
        return binding
    if binding.get('status') != 'complete' or not isinstance(binding.get('files'), dict):
        raise ValueError('original13 complete asset binding required')
    root = normalize_asset_path(binding['path'])
    if binding['production'] and (root.name != binding['revision'] or root.parent.name != 'snapshots'):
        raise ValueError('original13 pinned snapshot provenance differs')
    for name, expected in binding['tokenizer_files'].items():
        if (name not in binding['files'] or binding['files'][name]['sha256'] != expected['sha256']
                or binding['files'][name]['bytes'] != expected['bytes']):
            raise ValueError('original13 tokenizer/input asset identity differs')
    files = binding['files']
    if ('config.json' not in files or not any(n.endswith(('.safetensors', '.bin')) for n in files)
            or not any(n.startswith('tokenizer') or n.startswith('vocab') for n in files)):
        raise ValueError('original13 complete local model/config/tokenizer assets required')
    if 'trust' in binding and set(_model_asset_names(root)) != set(files):
        raise ValueError('original13 model asset layout changed; reacquire assets')
    if verify_bytes and model_asset_inventory(root) != files:
        raise ValueError('original13 model/tokenizer asset bytes differ')
    trust = binding.get('trust')
    trust_inventory_differs = trust is not None and (not isinstance(trust, dict) or set(trust) != set(files))
    for name, record in files.items():
        try:
            asset = asset_file_path(root, name)
        except FileNotFoundError as error:
            raise ValueError('original13 model/tokenizer asset bytes differ (presence): ' + name) from error
        # Preserve path/presence rejection priority over malformed trust metadata.
        if trust_inventory_differs:
            raise ValueError('original13 asset trust inventory differs')
        if asset.stat().st_size != record['bytes']:
            raise ValueError('original13 model/tokenizer asset bytes differ (size): ' + name)
        verify_asset_record(asset, record, None if trust is None else trust[name])
    return binding


# Transferred from sharing_preparation; serialized meanings remain unchanged.
def state_dict_identity(state):
    if not isinstance(state, dict) or not state or any(
        not isinstance(key, str) or not torch.is_tensor(value)
        for key, value in state.items()
    ):
        raise ValueError("initialization must contain a nonempty codec tensor state")
    return canonical_digest({key: tensor_digest(value) for key, value in state.items()})


def saved_source_inventory(source_root):
    """Interpret the historical sharing inventory against explicit retained source.

    The fixed runner names and optional-file rules are saved-format semantics.
    Current preparation uses checkpoint_readers.current_source_inventory instead;
    absent historical runner bytes must fail rather than alter this inventory.
    """
    root = Path(source_root).resolve()
    names = {str(path.relative_to(root)) for path in (root / "jscc").rglob("*.py")}
    names |= {"scripts/shared_codec.py", "scripts/sharing_production.py", "scripts/sharing_qualification.py",
              "scripts/render_experiment_report.py", "scripts/prepare_sharing_native16.py",
              "uv.lock", "pyproject.toml"}
    if (root / "scripts/sharing_long_study.py").is_file():
        names.add("scripts/sharing_long_study.py")
    if not (root / "scripts/prepare_sharing_native16.py").is_file():
        names.discard("scripts/prepare_sharing_native16.py")
    return {name: file_digest(root / name) for name in sorted(names)}


def _reference(root, reference):
    if not isinstance(reference, dict) or not {"path", "sha256"} <= reference.keys():
        raise ValueError("immutable file reference required")
    name = reference["path"]
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ValueError("prepared references must be relative")
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("prepared reference escapes package or is missing")
    if file_digest(path) != reference["sha256"]:
        raise ValueError(f"prepared byte checksum mismatch: {name}")
    return path

