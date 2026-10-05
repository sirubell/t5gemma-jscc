"""YAML job preparation and durable, non-retrying Slurm submission.

Preparation reads small manifests/configs and selects an existing exact asset.
It never transfers data, loads model weights, or invokes a scheduler.
"""
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tempfile
import time

import yaml

from .activation_replay import canonical_digest, file_digest
from .source_lifetime import SourceConsumers, capture_source_binding, check_source_binding

SCHEMA = 'shared-codec-submission-v1'
CASE_SCHEMA = 'recipe-case-submission-v1'


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
    """Prepare only the current recorded case graph; saved legacy bundles stay readable."""
    job = _read_yaml(job_path)
    require(isinstance(job, dict) and job.get('schema') == CASE_SCHEMA,
            'old submission execution retired; use recipe-case-submission-v1')
    return prepare_case_job(job_path, limits_path)


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
    require(isinstance(bundle, dict) and bundle.get('schema') in (SCHEMA, CASE_SCHEMA), 'unsupported bundle schema')
    raw = deepcopy(bundle)
    identity = raw.pop('identity', None)
    require(identity == canonical_digest(raw), 'bundle identity differs')
    return bundle


def _sync_directory(path):
    directory = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


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
        _sync_directory(path.parent)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def submission_preflight(bundle):
    """Check the bound effective config, prepared inputs and fixed source/runtime."""
    validate_bundle(bundle)
    binding = bundle['source_binding']
    require(normalize_path(binding['path'], bundle['root']) == bundle['source']['path']
            and binding['commit'] == bundle['source']['commit']
            and normalize_path(binding['runtime']['python'], bundle['root']) == bundle['python'],
            'source/runtime binding differs from bundle')
    check_source_binding(binding)
    if bundle['schema'] == CASE_SCHEMA:
        _validate_case_root(bundle)
    else:
        validate_prepared(bundle)
    return binding


def source_consumers(bundle):
    """Shared host registry, outside all source and scientific output checkouts."""
    return SourceConsumers(Path(bundle['root'])/'.source-consumers')


def preview_job(bundle, bundle_path):
    """Preview the current recorded case graph without a scheduler call."""
    require(bundle.get('schema') == CASE_SCHEMA, 'old submission execution retired')
    return _preview_case_job(bundle, bundle_path)


def render_payload(bundle, bundle_path):
    require(bundle.get('schema') == CASE_SCHEMA, 'old submission execution retired')
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
                            '--identity', bundle['identity']]) +
        ' --group "${1:?group required}"', ''])


def scheduler_command(bundle, payload_path, group_index, dependency=None):
    require(bundle.get('schema') == CASE_SCHEMA, 'old submission execution retired')
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
    dependency_key = 'aftercorr'
    dependency = dependency or group.get('accepted_dependency_job_id')
    if group[dependency_key] is not None or group.get('accepted_dependency_job_id'):
        require(isinstance(dependency, str) and re.fullmatch('[1-9][0-9]*', dependency), 'job ID required')
        command += ['--dependency='+dependency_key+':'+str(dependency), '--kill-on-invalid-dep=yes']
    return command+[normalize_path(str(payload_path), bundle['root']), str(group_index)]


def _slurm_submit(args):
    return subprocess.run(args, text=True, capture_output=True, check=False)


def _submission_intent(bundle, receipt, group, command) -> dict[str, Any]:
    intent = {'bundle_identity': bundle['identity'], 'group': group,
              'arguments': command, 'payload_sha256': receipt['payload_sha256'],
              'attempted_at': datetime.now(timezone.utc).isoformat()}
    return dict(intent, attempt_identity=canonical_digest(intent), status='attempting')


def reconcile_job(bundle, receipt_dir, *, evidence):
    """Record scheduler evidence without submitting, retrying or releasing."""
    from .submission_reconciliation import reconcile_job as reconcile
    return reconcile(bundle, receipt_dir, evidence=evidence)


def submit_job(bundle, receipt_dir, *, scheduler=None, authorization=None):
    """Submit the current case graph with durable intent and explicit authority."""
    require(bundle.get('schema') == CASE_SCHEMA, 'old submission execution retired')
    from .submission_reconciliation import require_retry_authorization
    require_retry_authorization(bundle)
    return _submit_case_job(bundle, receipt_dir, scheduler=scheduler, authorization=authorization)


def run_job(bundle, identity, group_index=None, *, runner=None):
    """Launch one current bound case row without a scheduler."""
    require(bundle.get('schema') == CASE_SCHEMA, 'old submission execution retired')
    return _run_case_job(bundle, identity, group_index, runner=runner)


def _case_inputs(job_path, job):
    recipe = _relative(job_path, job['recipe']).resolve()
    registrations = _relative(job_path, job['registrations']).resolve()
    require(registrations.is_dir(), 'registrations directory required')
    return recipe, registrations, {'recipe_sha256': file_digest(recipe),
        'registrations': {str(p.relative_to(registrations)): file_digest(p)
                          for p in sorted(registrations.glob('*.json'))}}


def prepare_case_job(job_path, limits_path):
    """Prepare a bound public recipe root and freeze its selected case graph."""
    from .recipe_runs import prepare_recipe_run, recipe_run_record
    job = _read_yaml(job_path)
    _keys(job, ('schema', 'recipe', 'registrations', 'run_output', 'root', 'source',
                'python', 'cache', 'output', 'resources', 'cases', 'evaluation_cases',
                'concurrency', 'combined', 'attempt'), 'case job')
    require(job['schema'] == CASE_SCHEMA, 'unsupported case schema')
    root = normalize_path(job['root'])
    limits = _read_yaml(limits_path)
    _keys(limits, ('root', 'resources', 'max_concurrency', 'max_array_tasks'), 'limits')
    require(normalize_path(limits['root']) == root, 'approved root differs')
    _keys(job['source'], ('commit', 'path'), 'source')
    source = {'commit': _digest(job['source']['commit'], 40),
              'path': normalize_path(job['source']['path'], root)}
    python = normalize_path(job['python'], root)
    binding = capture_source_binding(local_path(source['path'], root), source['commit'], python)
    check_source_binding(binding)
    require(Path(__file__).resolve().parents[1] == Path(source['path']).resolve(),
            'loaded preparation source differs')
    resources = deepcopy(job['resources'])
    fields = ('partition', 'account', 'qos', 'node', 'gpus', 'cpus', 'memory_gb', 'walltime_minutes')
    _keys(resources, fields, 'resources')
    _keys(limits['resources'], fields, 'resource limits')
    for field in fields[:4]:
        require(isinstance(resources[field], str) and re.fullmatch(r'[A-Za-z0-9_.-]+', resources[field])
                and resources[field] == limits['resources'][field], 'unapproved '+field)
    for field in fields[4:]:
        _integer(limits['resources'][field], 1, 100000, 'limit '+field)
        _integer(resources[field], 10 if field == 'walltime_minutes' else 1,
                 limits['resources'][field], field)
    require(resources['gpus'] == 1, 'runtime requires one assigned GPU')
    concurrency = _integer(job['concurrency'], 1,
                           _integer(limits['max_concurrency'], 1, 10000, 'concurrency limit'), 'concurrency')
    array_limit = _integer(limits['max_array_tasks'], 1, 10000, 'array limit')
    require(type(job['combined']) is bool, 'combined must be boolean')
    require(isinstance(job['attempt'], str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', job['attempt']),
            'plain attempt name required')
    recipe, registrations, inputs = _case_inputs(job_path, job)
    run_output = normalize_path(job['run_output'], root)
    output = normalize_path(job['output'], root)
    require(run_output != output and PurePosixPath(run_output) not in PurePosixPath(output).parents
            and PurePosixPath(output) not in PurePosixPath(run_output).parents, 'run and submission outputs overlap')
    input_binding = {**inputs, 'source': source, 'source_binding': binding}
    run = Path(run_output)
    if run.exists():
        require((run/'submission-input.json').is_file(), 'existing root lacks submission input binding')
        require(json.loads((run/'submission-input.json').read_text()) == input_binding,
                'bound recipe inputs or source differ')
    else:
        run = prepare_recipe_run(recipe, run_output, registrations=registrations)
        write_json(run/'submission-input.json', input_binding)
    record = recipe_run_record(run)
    manifest = json.loads(Path(record['prepared']['path']).read_text())
    require(manifest['source_commit'] == source['commit'], 'prepared source commit differs')
    ids = [row['case_id'] for row in record['cases']]
    selected = ids if job['cases'] is None else job['cases']
    require(isinstance(selected, list) and selected and len(set(selected)) == len(selected)
            and all(i in ids for i in selected), 'explicit declared training cases required')
    evaluation = job['evaluation_cases']
    require(isinstance(evaluation, list) and len(set(evaluation)) == len(evaluation)
            and all(i in selected for i in evaluation), 'evaluations require selected declared cases')
    indices = [i for i, name in enumerate(ids) if name in selected]
    require(len(indices) <= array_limit, 'invalid array size')
    eval_indices = [i for i in indices if ids[i] in evaluation]
    groups = [{'stage': 'combined' if job['combined'] else 'training', 'indices': indices,
               'concurrency': concurrency, 'aftercorr': None}]
    if eval_indices and not job['combined']:
        groups.append({'stage': 'evaluation', 'indices': eval_indices,
                       'concurrency': concurrency, 'aftercorr': 0})
    cases = []
    for index, row in enumerate(record['cases']):
        case = {**row, 'index': index, 'commands': {}}
        if index in indices:
            train_attempt = job['attempt']+'-g0-i'+str(index)
            case['commands']['training'] = [python, source['path']+'/train.py', '--config',
                str(Path(row['path'])/'config.yaml'), '--output', run_output, '--case', row['case_id'],
                '--attempt', train_attempt, '--timeout-seconds', str(resources['walltime_minutes']*60-300)]
        if index in eval_indices:
            eval_group = 0 if job['combined'] else 1
            case['commands']['evaluation'] = [python, source['path']+'/evaluate.py', '--run', run_output,
                '--case', row['case_id'], '--attempt', job['attempt']+'-g'+str(eval_group)+'-i'+str(index),
                '--timeout-seconds', str(resources['walltime_minutes']*60-300)]
        description = json.loads((Path(row['path'])/'recipe.json').read_text())
        case['effective_config_sha256'] = canonical_digest(description['config'])
        if index in indices:
            case.update(effective_config=description['config'], locations=description['locations'],
                        selected_rows_request=description['config']['data'],
                        updates_per_location=description['updates_per_location'])
        cases.append(case)
    bundle = {'schema': CASE_SCHEMA, 'root': root, 'source': source, 'source_binding': binding,
              'python': python, 'cache': normalize_path(job['cache'], root), 'output': output,
              'run_output': run_output, 'run_identity': record['identity'], 'input_binding': input_binding,
              'resources': resources, 'groups': groups, 'cases': cases, 'attempt': job['attempt'],
              'evaluation_cases': evaluation, 'combined': job['combined'], 'launch_authorized': False}
    bundle['identity'] = canonical_digest(bundle)
    return bundle


def _validate_case_root(bundle):
    from .recipe_runs import recipe_run_record
    record = recipe_run_record(bundle['run_output'])
    require(record['identity'] == bundle['run_identity'], 'bound run identity differs')
    require(json.loads((Path(bundle['run_output'])/'submission-input.json').read_text()) == bundle['input_binding'],
            'bound input identity differs')
    manifest = json.loads(Path(record['prepared']['path']).read_text())
    require(manifest['source_commit'] == bundle['source']['commit'], 'prepared source differs')
    require(len(record['cases']) == len(bundle['cases']), 'bound case count differs')
    for row, frozen in zip(record['cases'], bundle['cases']):
        require(all(frozen[k] == v for k, v in row.items()), 'bound case differs')
        description = json.loads((Path(row['path'])/'recipe.json').read_text())
        require(frozen['effective_config_sha256'] == canonical_digest(description['config']),
                'bound effective config digest differs')
        if frozen['commands']:
            expected = {'effective_config': description['config'], 'locations': description['locations'],
                        'selected_rows_request': description['config']['data'],
                        'updates_per_location': description['updates_per_location']}
            require(all(frozen[key] == value for key, value in expected.items()),
                    'frozen scientific case description differs')


def _preview_case_job(bundle, bundle_path):
    require(bundle.get('schema') == CASE_SCHEMA, 'old submission execution retired')
    binding = submission_preflight(bundle)
    return {'bundle_identity': bundle['identity'], 'source_binding': binding,
            'run_output': bundle['run_output'], 'run_identity': bundle['run_identity'],
            'cases': [row for row in bundle['cases'] if row['commands']],
            'nonselected_cases': [row for row in bundle['cases'] if not row['commands']],
            'reservation_paths': [str(Path(row['path'])/'scheduler-reservations'/('training.json'))
                for row in bundle['cases'] if 'training' in row['commands']],
            'dependency_template_note': 'ID 1 stands for this graph training group job ID; aftercorr matches own task index',
            'resource_policy_note': 'Historical account allocation was four GPUs; per-array caps and per-task resources do not establish current account policy',
            'groups': [dict(g, group=i, task_count=len(g['indices']),
                command_template=scheduler_command(bundle, bundle['output']+'/job.sbatch', i,
                    None if g['aftercorr'] is None else '1')) for i, g in enumerate(bundle['groups'])],
            'resources_per_task': bundle['resources'],
            'task_count': sum(len(g['indices']) for g in bundle['groups']),
            'scientific_totals': {key: sum(row['counts'].get(key, 0) for row in bundle['cases']
                if 'training' in row['commands']) for key in
                ('selected_rows', 'optimizer_updates', 'base_stream_presentations', 'processed_presentations')},
            'payload': render_payload(bundle, bundle['output']+'/bundle.json'),
            'source_consumer_registry': str(source_consumers(bundle).directory),
            'launch_authorized': False, 'scheduler_invoked': False}


def _submit_case_job(bundle, receipt_dir, *, scheduler, authorization):
    require(bundle.get('schema') == CASE_SCHEMA, 'old submission execution retired')
    validate_bundle(bundle)
    directory = Path(receipt_dir)
    require(normalize_path(str(directory), bundle['root']) == bundle['output'], 'receipt/output differs')
    require(normalize_path(str(directory.parent.resolve(strict=True))) ==
            str(PurePosixPath(bundle['output']).parent), 'output parent symlink differs')
    mocked = scheduler is not None
    if not mocked:
        require(isinstance(authorization, dict) and authorization.get('approved') is True
            and authorization.get('operation') == 'submit-recipe'
            and authorization.get('bundle_identity') == bundle['identity']
            and authorization.get('source_commit') == bundle['source']['commit']
            and authorization.get('output') == bundle['output']
            and isinstance(authorization.get('user_authorization_reference'), str)
            and authorization['user_authorization_reference'].strip(), 'explicit exact live submission approval required')
        scheduler = _slurm_submit
    directory.mkdir(exist_ok=False)
    _sync_directory(directory.parent)
    receipt = {'schema': CASE_SCHEMA, 'bundle_identity': bundle['identity'],
        'source_commit': bundle['source']['commit'], 'status': 'reserved', 'jobs': [], 'mocked': mocked}
    receipt_path = directory/'SUBMISSION.json'
    try:
        (directory/'logs').mkdir()
        (directory/'tasks').mkdir()
        write_json(directory/'bundle.json', bundle)
        with (directory/'job.sbatch').open('x') as stream:
            stream.write(render_payload(bundle, directory/'bundle.json'))
            stream.flush()
            os.fsync(stream.fileno())
        _sync_directory(directory)
        receipt.update(payload_sha256=file_digest(directory/'job.sbatch'),
                       bundle_sha256=file_digest(directory/'bundle.json'))
        write_json(receipt_path, receipt)
        submission_preflight(bundle)
        source_consumers(bundle).reserve(bundle['source_binding'], bundle['identity'],
            owner=bundle['output'], reference=str(receipt_path))
        receipt['source_consumer_reserved'] = True
        training_indices = {i for g in bundle['groups'] if g['stage'] in ('training', 'combined') for i in g['indices']}
        for index, case in enumerate(bundle['cases']):
            if index in training_indices and 'training' in case['commands']:
                case_path = Path(case['path'])
                require(not list((case_path/'attempts').glob('training-*'))
                    and not (case_path/'training.json').exists() and not (case_path/'checkpoint.json').exists(),
                    'existing training science requires explicit reconciliation; do not retry')
        for i, group in enumerate(bundle['groups']):
            for index in group['indices']:
                case = bundle['cases'][index]
                stage_names = ['training', 'evaluation'] if group['stage'] == 'combined' else [group['stage']]
                case_reservations = Path(case['path'])/'scheduler-reservations'
                case_reservations.mkdir(exist_ok=True)
                _sync_directory(case_reservations.parent)
                for stage in stage_names:
                    if stage not in case['commands']:
                        continue
                    arguments = case['commands'][stage]
                    attempt = arguments[arguments.index('--attempt')+1]
                    stage_reservation = case_reservations/(stage+'.json')
                    with stage_reservation.open('x') as stream:
                        stream.flush()
                        os.fsync(stream.fileno())
                    _sync_directory(case_reservations)
                    metadata = stage_reservation.stat()
                    receipt.setdefault('created_stage_fences', []).append({
                        'path': str(stage_reservation), 'identity': bundle['identity'],
                        'group': i, 'index': index, 'stage': stage,
                        'device': metadata.st_dev, 'inode': metadata.st_ino})
                    write_json(receipt_path, receipt)
                    write_json(stage_reservation, {'identity': bundle['identity'], 'group': i, 'index': index,
                               'stage': stage, 'attempt': attempt, 'submission': bundle['output']})
                reservation = directory/'tasks'/f'g{i}-i{index}.reservation.json'
                with reservation.open('x'):
                    pass
                write_json(reservation, {'identity': bundle['identity'], 'group': i, 'index': index,
                           'case_id': bundle['cases'][index]['case_id'], 'status': 'reserved'})
        write_json(receipt_path, receipt)
        for i, group in enumerate(bundle['groups']):
            dependency = group.get('accepted_dependency_job_id') or (None if group['aftercorr'] is None else receipt['jobs'][group['aftercorr']]['job_id'])
            command = scheduler_command(bundle, directory/'job.sbatch', i, dependency)
            job = _submission_intent(bundle, receipt, i, command)
            receipt['jobs'].append(job)
            write_json(receipt_path, receipt)
            try:
                result = scheduler(command)
            except BaseException as error:
                job.update(status='unknown', failure={'type': type(error).__name__, 'message': str(error)})
                write_json(receipt_path, receipt)
                raise
            job.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
            match = re.fullmatch(r'([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?\s*', result.stdout)
            if result.returncode == 0 and match:
                job.update(status='accepted', job_id=match[1])
            else:
                job['status'] = 'unknown' if result.returncode == 0 or re.search(r'\d+', result.stdout+' '+result.stderr) else 'rejected'
            write_json(receipt_path, receipt)
            require(job['status'] == 'accepted', 'scheduler '+job['status']+'; inspect receipt; do not retry')
        receipt['status'] = 'submitted'
        write_json(receipt_path, receipt)
        source_consumers(bundle).record_state(bundle['source']['path'], bundle['identity'], 'queued',
            owner=bundle['output'], evidence=str(receipt_path))
    except BaseException as error:
        receipt.update(status='stopped-no-retry', failure={'type': type(error).__name__, 'message': str(error),
            'stage': 'submission' if receipt['jobs'] else 'preflight'})
        write_json(receipt_path, receipt)
        if receipt.get('source_consumer_reserved'):
            source_consumers(bundle).record_state(bundle['source']['path'], bundle['identity'], 'unresolved',
                owner=bundle['output'], evidence=str(receipt_path))
        raise
    return receipt


def _run_case_job(bundle, identity, group_index, *, runner=None):
    require(bundle.get('schema') == CASE_SCHEMA, 'old submission execution retired')
    validate_bundle(bundle)
    require(identity == bundle['identity'], 'explicit bundle identity differs')
    require(os.environ.get('SLURM_JOB_ID'), 'Slurm allocation required')
    if group_index is None:
        group_index = int(os.environ['CODEC_SUBMISSION_GROUP'])
    _integer(group_index, 0, len(bundle['groups'])-1, 'group index')
    index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    group = bundle['groups'][group_index]
    require(index in group['indices'], 'array index differs')
    case = bundle['cases'][index]
    marker = Path(bundle['output'])/'tasks'/f'g{group_index}-i{index}.launch.json'
    outcome = {'identity': identity, 'group': group_index, 'index': index, 'case_id': case['case_id'],
               'job_id': os.environ['SLURM_JOB_ID'], 'status': 'checking', 'commands': []}
    with marker.open('x'):
        pass
    write_json(marker, outcome)
    try:
        submission_preflight(bundle)
        source = local_path(bundle['source']['path'], bundle['root'])
        require(Path(__file__).resolve().parents[1] == source.resolve(), 'loaded submission source differs')
        source_consumers(bundle).record_state(source, identity, 'running', owner=bundle['output'], evidence=str(marker))
        outcome['preflight'] = 'passed'
        write_json(marker, outcome)
        deadline = time.monotonic()+bundle['resources']['walltime_minutes']*60-300
        stages = ['training', 'evaluation'] if group['stage'] == 'combined' else [group['stage']]
        for stage in stages:
            if stage not in case['commands']:
                continue
            entry = {'stage': stage, 'arguments': case['commands'][stage], 'status': 'preflight'}
            outcome['commands'].append(entry)
            write_json(marker, outcome)
            if stage == 'evaluation':
                from .recipe_evaluation import recipe_evaluation_target
                eligibility, *_ = recipe_evaluation_target(bundle['run_output'], case['case_id'])
                train_args = case['commands']['training']
                train_attempt = train_args[train_args.index('--attempt')+1]
                expected_attempt = str(Path(case['path'])/'attempts'/('training-'+train_attempt))
                require(eligibility['checkpoint']['training_attempt'] == expected_attempt,
                        'evaluation checkpoint is not this graph training attempt')
                entry['eligibility'] = eligibility
                write_json(marker, outcome)
            entry['status'] = 'running'
            write_json(marker, outcome)
            remaining = deadline-time.monotonic()
            require(remaining > 0, 'allocation command budget exhausted')
            entry['timeout_seconds'] = remaining
            write_json(marker, outcome)
            result = runner(entry['arguments']) if runner else subprocess.run(
                entry['arguments'], text=True, capture_output=True, check=False, timeout=remaining)
            entry.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode,
                         status='completed' if result.returncode == 0 else 'failed')
            write_json(marker, outcome)
            require(result.returncode == 0, stage+' command failed')
        outcome['status'] = 'completed'
        write_json(marker, outcome)
    except BaseException as error:
        outcome.update(status='failed', failure={'type': type(error).__name__, 'message': str(error)})
        if outcome['commands'] and outcome['commands'][-1]['status'] in ('running', 'preflight'):
            failed_command = outcome['commands'][-1]
            failed_command.update(status='failed', failure=outcome['failure'])
            if isinstance(error, subprocess.TimeoutExpired):
                def output_text(value):
                    return value.decode(errors='replace') if isinstance(value, bytes) else value or ''
                failed_command.update(stdout=output_text(error.stdout), stderr=output_text(error.stderr),
                                      returncode=None, timed_out=True)
        if 'preflight' not in outcome:
            outcome['preflight'] = 'failed'
        write_json(marker, outcome)
        raise
    return outcome
