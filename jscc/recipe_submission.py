"""YAML job preparation and durable, non-retrying Slurm submission.

Preparation reads small manifests/configs and selects an existing exact asset.
It never transfers data, loads model weights, or invokes a scheduler.
"""
from copy import deepcopy
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tempfile

import yaml

from .activation_replay import canonical_digest, file_digest
from .experiment_recipe import resolve_recipe

SCHEMA = 'shared-codec-submission-v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalize_path(value, root=None):
    """Canonicalize login/compute aliases and reject lexical escapes.

    This describes remote paths. Actual symlinks are checked again on the host
    when reading assets and launching; lexical confinement cannot prove them.
    """
    require(isinstance(value, str) and value.startswith('/') and
            not any(ord(c) < 32 or ord(c) == 127 for c in value), 'absolute clean path required')
    require(not any(p in ('.', '..') for p in value.split('/')), 'path escape rejected')
    path = PurePosixPath(value)
    if path == PurePosixPath('/work') or PurePosixPath('/work') in path.parents:
        path = PurePosixPath('/raid/work') / path.relative_to('/work')
    if root is not None:
        boundary = PurePosixPath(normalize_path(root))
        require(path != boundary and boundary in path.parents, 'path outside root or equal to root')
    return str(path)


def local_path(value, root):
    """Resolve an existing host path, including /work alias and symlinks."""
    normalized = normalize_path(value, root)
    path = Path(value) if Path(value).exists() else Path(normalized)
    resolved = path.resolve(strict=True)
    require(normalize_path(str(resolved), root) == normalized, 'symlink changes bound asset path')
    return path


def _digest(value, length=64):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{'+str(length)+'}', value),
            'full source SHA or SHA256 required')
    return value


def _integer(value, minimum, maximum, name):
    require(type(value) is int and minimum <= value <= maximum, 'invalid '+name)
    return value


def _keys(value, keys, name):
    require(isinstance(value, dict) and set(value) == set(keys), 'unsupported '+name+' fields')


def _read_yaml(path):
    return yaml.safe_load(Path(path).read_text())


def _relative(path, value):
    require(isinstance(value, str) and value, 'file reference required')
    return Path(path).resolve().parent / value


def prepare_job(job_path, limits_path):
    """Resolve recipe and target settings against separately approved limits."""
    job = _read_yaml(job_path)
    _keys(job, ('schema', 'recipe', 'root', 'source', 'model', 'prepared', 'python',
                'cache', 'output', 'resources', 'groups'), 'job')
    require(job['schema'] == SCHEMA, 'unsupported job schema')
    root = normalize_path(job['root'])
    limits = _read_yaml(limits_path)
    _keys(limits, ('root', 'resources', 'max_concurrency', 'max_array_tasks'), 'limits')
    require(normalize_path(limits['root']) == root, 'approved root differs')
    _keys(job['source'], ('commit', 'path'), 'source')
    source = {'commit': _digest(job['source']['commit'], 40),
              'path': normalize_path(job['source']['path'], root)}
    _keys(job['model'], ('path', 'config_sha256'), 'model')
    model_path = normalize_path(job['model']['path'], root)
    model_config = local_path(job['model']['path']+'/config.json', root)
    require(file_digest(model_config) == _digest(job['model']['config_sha256']), 'model config digest differs')
    resolved = resolve_recipe(_relative(job_path, job['recipe']), model_config_path=model_config)
    # Preserve the submitted YAML resolution, not a later edit of its source YAML.
    _keys(job['prepared'], ('sha256', 'candidates'), 'prepared')
    checksum = _digest(job['prepared']['sha256'])
    candidates = job['prepared']['candidates']
    require(isinstance(candidates, list) and candidates, 'existing prepared candidates required')
    selected = None
    inspected = []
    for candidate in candidates:
        canonical = normalize_path(candidate, root)
        try:
            path = local_path(candidate, root)
        except FileNotFoundError:
            inspected.append({'path': canonical, 'status': 'missing'})
            continue
        if file_digest(path) == checksum:
            selected = canonical
            inspected.append({'path': canonical, 'status': 'selected exact manifest'})
            break
        inspected.append({'path': canonical, 'status': 'digest differs'})
    require(selected is not None, 'no compatible existing prepared asset; acquisition required')
    resources = deepcopy(job['resources'])
    _keys(resources, ('partition', 'account', 'qos', 'node', 'gpus', 'cpus', 'memory_gb',
                      'walltime_minutes'), 'resources')
    _keys(limits['resources'], resources.keys(), 'resource limits')
    for field in ('partition', 'account', 'qos', 'node'):
        require(isinstance(resources[field], str) and re.fullmatch(r'[A-Za-z0-9_.-]+', resources[field]),
                'invalid Slurm '+field)
        require(resources[field] == limits['resources'][field], 'unapproved '+field)
    for field in ('gpus', 'cpus', 'memory_gb', 'walltime_minutes'):
        _integer(limits['resources'][field], 1, 100000, 'limit '+field)
        _integer(resources[field], 10 if field == 'walltime_minutes' else 1,
                 limits['resources'][field], field)
    require(resources['gpus'] == 1, 'runtime requires one assigned GPU')
    concurrency_limit = _integer(limits['max_concurrency'], 1, 10000, 'concurrency limit')
    array_limit = _integer(limits['max_array_tasks'], 1, 10000, 'array limit')
    groups = job['groups']
    require(isinstance(groups, list) and groups, 'array groups required')
    all_indices = []
    for index, group in enumerate(groups):
        _keys(group, ('indices', 'concurrency', 'afterok'), 'array group')
        indices = group['indices']
        require(isinstance(indices, list) and indices and len(indices) <= array_limit, 'invalid array size')
        for i in indices:
            _integer(i, 0, len(resolved['trajectories'])-1, 'trajectory index')
        _integer(group['concurrency'], 1, concurrency_limit, 'concurrency')
        dep = group['afterok']
        require(dep is None or (type(dep) is int and 0 <= dep < index), 'dependency requires an earlier group')
        all_indices.extend(indices)
    require(sorted(all_indices) == list(range(len(resolved['trajectories']))),
            'arrays must cover every trajectory exactly once')
    bundle = {'schema': SCHEMA, 'root': root, 'source': source,
              'model': {'path': model_path, 'config_sha256': job['model']['config_sha256']},
              'prepared': {'path': selected, 'sha256': checksum}, 'assets_inspected': inspected,
              'python': normalize_path(job['python'], root), 'cache': normalize_path(job['cache'], root),
              'output': normalize_path(job['output'], root), 'resources': resources, 'groups': groups,
              'resolved': resolved, 'launch_authorized': False}
    manifest = json.loads(local_path(selected, root).read_text())
    if manifest.get('schema') == 'recipe-prepared-v1':
        from .recipe_workflow import validate_recipe_prepared
        bundle['resolved'] = validate_recipe_prepared(manifest, resolved=resolved)
    # Existing prepared assets are accepted only if their recorded configs/plans
    # match. Tensor/model integrity stays at the runtime loader boundary.
    validate_prepared(bundle)
    bundle['identity'] = canonical_digest(bundle)
    return bundle


def validate_prepared(bundle):
    from .encoder_stages_plan import Original13Plan
    from .experiment_recipe import validate_retained_recipe

    model_config = local_path(bundle['model']['path']+'/config.json', bundle['root'])
    require(file_digest(model_config) == bundle['model']['config_sha256'], 'model config digest differs')
    asset = local_path(bundle['prepared']['path'], bundle['root'])
    require(file_digest(asset) == bundle['prepared']['sha256'], 'prepared digest differs')
    manifest = json.loads(asset.read_text())
    if manifest.get('schema') == 'recipe-prepared-v1':
        from .recipe_workflow import validate_recipe_prepared
        require(manifest['source_commit'] == bundle['source']['commit'], 'prepared source commit differs')
        require(normalize_path(manifest['model_assets']['path'], bundle['root']) == bundle['model']['path'],
                'prepared model path differs')
        bound = validate_recipe_prepared(manifest, resolved=bundle['resolved'])
        require(bound == bundle['resolved'], 'resolved acquired config differs')
        return
    require(manifest.get('schema') == 'original13-prepared-v1', 'prepared schema differs')
    require(manifest.get('source_commit') == bundle['source']['commit'], 'prepared source commit differs')
    for item in bundle['resolved']['trajectories']:
        matches = [p for p in manifest['plans'] if p['width'] == item['width'] and p['ratio'] == item['ratio']]
        require(len(matches) == 1, 'prepared asset lacks exact trajectory')
        plan = Original13Plan.from_resolved(matches[0])
        config = manifest['configs'][str(item['width'])]
        require(normalize_path(config['model']['name'], bundle['root']) == bundle['model']['path'],
                'prepared model path differs')
        # Alias spelling is operational, not a changed scientific config.
        description = deepcopy(item)
        description['config']['model']['name'] = config['model']['name']
        validate_retained_recipe(description, plan, config, manifest['observation_request'])


def validate_bundle(bundle):
    require(bundle.get('schema') == SCHEMA, 'unsupported bundle schema')
    raw = deepcopy(bundle)
    identity = raw.pop('identity', None)
    require(identity == canonical_digest(raw), 'bundle identity differs')
    return bundle


def write_json(path, value):
    """Atomically replace our receipt, retaining the reservation on any failure."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.'+path.name)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def render_payload(bundle, bundle_path):
    validate_bundle(bundle)
    return '\n'.join([
        '#!/bin/bash -l', 'set -euo pipefail',
        # Slurm copies the script; interpreter/helper paths must be absolute.
        'export HF_HOME='+shlex.quote(bundle['cache']),
        'export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1',
        'export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1',
        'export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false',
        'cd '+shlex.quote(bundle['source']['path']),
        'exec '+shlex.join([bundle['python'], bundle['source']['path']+'/scripts/submit_recipe.py',
                            'run', '--bundle', normalize_path(str(bundle_path), bundle['root']),
                            '--identity', bundle['identity']]), ''])


def scheduler_command(bundle, payload_path, group_index, dependency=None):
    group = bundle['groups'][group_index]
    resources = bundle['resources']
    command = ['sbatch', '--parsable', '--export=NONE', '--no-requeue',
               '--job-name=codec-recipe', '--partition='+resources['partition'],
               '--account='+resources['account'], '--qos='+resources['qos'], '--nodelist='+resources['node'],
               '--gres=gpu:'+str(resources['gpus']), '--cpus-per-task='+str(resources['cpus']),
               '--mem='+str(resources['memory_gb'])+'G', '--time='+str(resources['walltime_minutes']),
               '--array='+','.join(map(str, group['indices']))+'%'+str(group['concurrency']),
               '--output='+bundle['output']+'/logs/%A_%a.stdout',
               '--error='+bundle['output']+'/logs/%A_%a.stderr']
    if group['afterok'] is not None:
        require(isinstance(dependency, str) and re.fullmatch('[1-9][0-9]*', dependency), 'job ID required')
        command += ['--dependency=afterok:'+str(dependency), '--kill-on-invalid-dep=yes']
    return command+[normalize_path(str(payload_path), bundle['root'])]


def _slurm_submit(args):
    return subprocess.run(args, text=True, capture_output=True, check=False)


def submit_job(bundle, receipt_dir, *, scheduler=None, authorization=None):
    """Reserve first, persist each attempt before calling, never retry.

    Inject scheduler for offline tests. The real adapter requires explicit
    recorded human authority for the exact bundle, target and operation.
    """
    validate_bundle(bundle)
    validate_prepared(bundle)
    directory = Path(receipt_dir)
    require(normalize_path(str(directory), bundle['root']) == bundle['output'], 'receipt/output differs')
    require(normalize_path(str(directory.parent.resolve(strict=True))) == str(PurePosixPath(bundle['output']).parent),
            'output parent symlink differs')
    mocked = scheduler is not None
    if scheduler is None:
        require(isinstance(authorization, dict) and authorization.get('approved') is True
                and authorization.get('operation') == 'submit-recipe'
                and authorization.get('bundle_identity') == bundle['identity']
                and authorization.get('source_commit') == bundle['source']['commit']
                and authorization.get('output') == bundle['output']
                and isinstance(authorization.get('user_authorization_reference'), str)
                and authorization['user_authorization_reference'].strip(), 'explicit exact live submission approval required')
        scheduler = _slurm_submit
    directory.mkdir(exist_ok=False)
    receipt = {'schema': SCHEMA, 'bundle_identity': bundle['identity'], 'source_commit': bundle['source']['commit'],
               'status': 'reserved', 'jobs': [], 'mocked': mocked}
    receipt_path = directory/'SUBMISSION.json'
    try:
        (directory/'logs').mkdir()
        write_json(directory/'bundle.json', bundle)
        payload = render_payload(bundle, directory/'bundle.json')
        (directory/'job.sbatch').write_text(payload)
        receipt['payload_sha256'] = file_digest(directory/'job.sbatch')
        receipt['bundle_sha256'] = file_digest(directory/'bundle.json')
        write_json(receipt_path, receipt)
        for index, group in enumerate(bundle['groups']):
            dependency = None if group['afterok'] is None else receipt['jobs'][group['afterok']]['job_id']
            command = scheduler_command(bundle, directory/'job.sbatch', index, dependency)
            job = {'group': index, 'arguments': command, 'status': 'attempting'}
            receipt['jobs'].append(job)
            write_json(receipt_path, receipt)
            result = scheduler(command)
            job.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
            write_json(receipt_path, receipt)
            require(result.returncode == 0, 'scheduler failed; inspect receipt; do not retry')
            match = re.fullmatch(r'([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?\s*', result.stdout)
            if match is None:
                raise ValueError('ambiguous scheduler result; inspect receipt; do not retry')
            job.update(job_id=match[1], status='submitted')
            write_json(receipt_path, receipt)
        receipt['status'] = 'submitted'
        write_json(receipt_path, receipt)
    except BaseException:
        receipt['status'] = 'stopped-no-retry'
        write_json(receipt_path, receipt)
        raise
    return receipt


def run_job(bundle, identity):
    """Launch one bound array row through the retained CLI, without a scheduler."""
    validate_bundle(bundle)
    require(bundle['identity'] == identity, 'explicit bundle identity differs')
    require(os.environ.get('SLURM_JOB_ID'), 'Slurm allocation required')
    index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    require(index in [i for g in bundle['groups'] for i in g['indices']], 'array index differs')
    source = local_path(bundle['source']['path'], bundle['root'])
    require(Path(__file__).resolve().parents[1] == source.resolve(), 'loaded submission source differs')
    require(not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'],
                                        text=True).strip(), 'source checkout has tracked changes')
    require(subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
            == bundle['source']['commit'], 'source checkout differs')
    model_config = local_path(bundle['model']['path']+'/config.json', bundle['root'])
    require(file_digest(model_config) == bundle['model']['config_sha256'], 'model config digest differs')
    validate_prepared(bundle)
    import torch
    require(torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(), 'one assigned BF16 GPU required')
    uuid = 'GPU-'+str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
    os.environ['CUDA_VISIBLE_DEVICES'] = uuid
    item = bundle['resolved']['trajectories'][index]
    output = Path(bundle['output'])/f'B{item["width"]}-R{item["ratio"]}'
    require(not output.exists(), 'fresh training output required')
    # This durable launch marker blocks duplicate execution before learner setup.
    with Path(str(output)+'.launch.json').open('x') as stream:
        json.dump({'job_id': os.environ['SLURM_JOB_ID'], 'trajectory': index,
                   'identity': identity, 'output': str(output), 'device_uuid': uuid}, stream)
    command = [bundle['python'], str(source/'scripts/original13_training.py'), 'train',
               '--prepared', bundle['prepared']['path'], '--prepared-sha256', bundle['prepared']['sha256'],
               '--width', str(item['width']), '--ratio', str(item['ratio']),
               '--approved-source', bundle['source']['commit'], '--device-uuid', uuid,
               '--timeout-seconds', str(bundle['resources']['walltime_minutes']*60-300),
               '--resolved', bundle['output']+'/bundle.json', '--resolved-identity', identity,
               '--trajectory', str(index), '--output', str(output)]
    os.execv(bundle['python'], command)
