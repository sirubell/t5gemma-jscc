"""Original13 execution source, asset bindings and invocation guards."""
import copy
from pathlib import Path
import time

from .activation_replay import file_digest
from .sharing_preparation import source_inventory


def stages_source_inventory():
    root = Path(__file__).resolve().parents[1]
    return {**source_inventory(root), **{name: file_digest(root / name) for name in
            ('scripts/encoder_stages.py', 'scripts/render_encoder_stages.py',
             'scripts/original13_training.py', 'scripts/original13_evaluation.py')}}


def original13_execution_guard(config, *, synthetic, timeout_seconds, device_uuid=None):
    """Bound one direct invocation; GPU assignment is checked before model load."""
    import os
    import math
    if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0):
        raise ValueError('original13 finite positive execution timeout required')
    device = config['model']['device']
    if device == 'cpu':
        if not synthetic:
            raise ValueError('production original13 requires an assigned CUDA target')
    elif device == 'cuda':
        if not device_uuid or os.environ.get('CUDA_VISIBLE_DEVICES') != device_uuid:
            raise ValueError('original13 exact assigned GPU UUID required')
    else:
        raise ValueError('original13 requires explicit cpu or cuda device')
    started = time.monotonic()
    def guard():
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError('original13 finite invocation deadline reached')
    return guard


def verify_original13_execution_source(manifest, approved_source):
    import subprocess
    import os
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    root = Path(__file__).resolve().parents[1]
    current = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, env=env, text=True).strip()
    if current != approved_source or manifest['source_commit'] != current:
        raise ValueError('original13 exact approved source commit required')
    names = sorted(manifest['source_inventory'])
    subprocess.run(['git', 'ls-files', '--error-unmatch', '--', *names], cwd=root,
                   stdout=subprocess.DEVNULL, check=True, env=env)
    subprocess.run(['git', 'diff', '--exit-code', current, '--', *names], cwd=root,
                   stdout=subprocess.DEVNULL, check=True, env=env)


def model_asset_inventory(path):
    """Inventory consumed local HF configuration/tokenizer/weight files once."""
    root = Path(path).expanduser().resolve()
    files = {str(p.relative_to(root)): {'bytes': p.stat().st_size, 'sha256': file_digest(p)}
             for p in sorted(root.rglob('*')) if p.is_file()
             and p.suffix in {'.json', '.safetensors', '.bin', '.model', '.txt'}
             and '.cache' not in p.relative_to(root).parts}
    if ('config.json' not in files or not any(n.endswith(('.safetensors', '.bin')) for n in files)
            or not any(n.startswith('tokenizer') or n.startswith('vocab') for n in files)):
        raise ValueError('original13 complete local model/config/tokenizer assets required')
    return files


def bind_model_assets(path, revision, *, tokenizer_files=None, production=False):
    """Bind existing assets, or explicitly leave an unavailable target pending."""
    root = Path(path).expanduser().resolve()
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
    return {**reference, 'status': 'complete', 'files': files}


def verify_model_assets(config, *, require_complete=True, verify_bytes=True):
    """Fail closed before model construction; no checks in the training hot path."""
    binding = config['model'].get('asset_binding')
    if (not isinstance(binding, dict) or binding.get('schema') != 'original13-model-assets-v1'
            or binding.get('revision') != config['model']['revision']
            or binding.get('path') != str(Path(config['model']['name']).expanduser().resolve())
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
    root = Path(binding['path'])
    if binding['production'] and (root.name != binding['revision'] or root.parent.name != 'snapshots'):
        raise ValueError('original13 pinned snapshot provenance differs')
    for name, expected in binding['tokenizer_files'].items():
        if (name not in binding['files'] or binding['files'][name]['sha256'] != expected['sha256']
                or binding['files'][name]['bytes'] != expected['bytes']):
            raise ValueError('original13 tokenizer/input asset identity differs')
    if verify_bytes and model_asset_inventory(root) != binding['files']:
        raise ValueError('original13 model/tokenizer asset bytes differ')
    return binding

