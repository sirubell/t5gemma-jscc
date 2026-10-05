"""Offline preparation, exact Slurm arguments, and uncertain outcome retention."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from jscc.activation_replay import file_digest
from jscc.experiment_recipe import resolve_recipe
from jscc.encoder_stages_plan import Original13Plan
from jscc.recipe_submission import (SCHEMA, normalize_path, prepare_job, render_payload,
                                    scheduler_command, submit_job, validate_bundle)
from test_original13_joined import cli, prepared as prepared
from test_original13_plan_state import formal_views

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def job_files(tmp_path):
    source = tmp_path/'release'
    source.mkdir()
    (source/'uv.lock').write_text('fixture lock v1')
    (source/'worker.py').write_text('print("fixture")')
    subprocess.run(['git', 'init', '-q', str(source)], check=True)
    subprocess.run(['git', '-C', str(source), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(source), '-c', 'user.name=Fixture', '-c',
                    'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture'], check=True)
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    (tmp_path/'python').symlink_to(sys.executable)
    model = tmp_path/'model'
    model.mkdir()
    metadata = model/'config.json'
    metadata.write_text(json.dumps({'encoder': {'num_hidden_layers': 26, 'hidden_size': 1152},
                                    'decoder': {'num_hidden_layers': 26, 'hidden_size': 1152}}))
    from jscc.config import load_config
    task = load_config(ROOT/'configs/tasks/sharing_native16_long_b512.yaml')
    task['model']['name'] = str(model)
    task.pop('model_config', None)
    task_file = tmp_path/'task.yaml'
    task_file.write_text(yaml.safe_dump(task))
    recipe = yaml.safe_load((ROOT/'configs/recipes/original13.yaml').read_text())
    recipe['task_config'] = str(task_file)
    recipe_file = tmp_path/'recipe.yaml'
    recipe_file.write_text(yaml.safe_dump(recipe))
    resolved = resolve_recipe(recipe_file, model_config_path=metadata)
    views = formal_views()
    plans = [Original13Plan(views, 'submission-fixture', i['ratio'], i['width'], 'a'*64, 'b'*64).resolved()
             for i in resolved['trajectories']]
    manifest = {'schema': 'original13-prepared-v1', 'source_commit': commit, 'configs': {str(i['width']): i['config'] for i in resolved['trajectories']},
                'plans': plans, 'observation_request': {'panel': {'expected_items': 256},
                    'sites': recipe['observation']['sites'], 'conditions': recipe['observation']['conditions']}}
    prepared = tmp_path/'prepared.json'
    prepared.write_text(json.dumps(manifest))
    wrong = tmp_path/'wrong.json'
    wrong.write_text('{}')
    resources = dict(partition='h200q', account='research', qos='h200q_1g', node='hgx-cn01',
                     gpus=1, cpus=16, memory_gb=170, walltime_minutes=120)
    job = dict(schema=SCHEMA, recipe=str(recipe_file), root=str(tmp_path),
               source={'commit': commit, 'path': str(source)},
               model={'path': str(model), 'config_sha256': file_digest(metadata)},
               prepared={'sha256': file_digest(Path(prepared)),
                         'candidates': [str(tmp_path/'missing.json'), str(wrong), str(prepared)]},
               python=str(tmp_path/'python'), cache=str(tmp_path/'cache'), output=str(tmp_path/'campaign'),
               resources=resources, groups=[{'indices': [0, 1, 2], 'concurrency': 1, 'afterok': None},
                                            {'indices': list(range(3, 9)), 'concurrency': 3, 'afterok': 0}])
    limits = dict(root=str(tmp_path), resources=resources, max_concurrency=3, max_array_tasks=9)
    job_path, limits_path = tmp_path/'job.yaml', tmp_path/'limits.yaml'
    job_path.write_text(yaml.safe_dump(job))
    limits_path.write_text(yaml.safe_dump(limits))
    return job_path, limits_path, job


@pytest.mark.parametrize('value', ['/work/u/../escape', '/work/u/./data', 'relative', '/work/u/\nx'])
def test_path_escapes(value):
    with pytest.raises(ValueError):
        normalize_path(value, '/work/u')


def test_aliases_boundaries():
    assert normalize_path('/work/user/source', '/raid/work/user') == '/raid/work/user/source'
    assert normalize_path('/raid/work/user/source', '/work/user') == '/raid/work/user/source'
    for value in ['/work/user-other/data', '/work/user', '/other/user/data']:
        with pytest.raises(ValueError):
            normalize_path(value, '/work/user')


def test_bundle_tampering(job_files):
    b = deepcopy(bundle(job_files))
    b['resources']['gpus'] = 5
    with pytest.raises(ValueError, match='bundle identity'):
        validate_bundle(b)


def test_real_retained_loader_checks_persisted_resolution(prepared, tmp_path):
    path, manifest = prepared
    task_path = tmp_path/'task.yaml'
    task_path.write_text(yaml.safe_dump(manifest['configs']['8']))
    recipe = yaml.safe_load((ROOT/'configs/recipes/original13.yaml').read_text())
    recipe.update(task_config=str(task_path), updates_per_location=400,
                  widths=[8], reconstruction_percentages=[90])
    recipe_path = tmp_path/'recipe.yaml'
    recipe_path.write_text(yaml.safe_dump(recipe))
    resolved = resolve_recipe(recipe_path, model_config_path=tmp_path/'model/config.json')
    from jscc.activation_replay import canonical_digest
    b = {'schema': SCHEMA, 'resolved': resolved,
         'prepared': {'sha256': file_digest(path)},
         'source': {'commit': manifest['source_commit']}, 'model': {'path': str(tmp_path/'model')}}
    b['identity'] = canonical_digest(b)
    bundle_path = tmp_path/'bundle.json'
    bundle_path.write_text(json.dumps(b))
    args = ['check', '--prepared', str(path), '--prepared-sha256', file_digest(path),
            '--width', '8', '--ratio', '90', '--resolved', str(bundle_path),
            '--resolved-identity', b['identity'], '--trajectory', '0']
    check = cli('original13_training.py')
    assert check(args)['schema'] == 'original13-prepared-v1'
    b['resolved']['trajectories'][0]['config']['training']['lr'] = .001
    b['identity'] = canonical_digest({k: v for k, v in b.items() if k != 'identity'})
    bundle_path.write_text(json.dumps(b))
    args[args.index('--resolved-identity')+1] = b['identity']
    with pytest.raises(ValueError, match='complete config differs'):
        check(args)


@pytest.mark.parametrize('content', ['invalid json', '{"schema": "wrong"}', '[]', 'null'])
@pytest.mark.parametrize('mode', ['run', 'submit'])
def test_cli_invalid_launch_bundle_records_failure(tmp_path, monkeypatch, content, mode):
    path = tmp_path/'bundle.json'
    path.write_text(content)
    monkeypatch.setenv('SLURM_ARRAY_TASK_ID', '4')
    command = cli('submit_recipe.py')
    with pytest.raises(ValueError):
        command([mode, '--bundle', str(path), '--identity', 'expected'])
    marker = tmp_path/('trajectory-4.bundle-failure.json' if mode == 'run' else 'submission.bundle-failure.json')
    failure = json.loads(marker.read_text())
    assert failure['stage'] == 'bundle-preflight'
    assert failure['identity'] == 'expected'
    original = marker.read_bytes()
    with pytest.raises(FileExistsError):
        command([mode, '--bundle', str(path), '--identity', 'expected'])
    assert marker.read_bytes() == original


def test_cli_changed_old_bundle_rejects_without_failure_marker(job_files, tmp_path, monkeypatch):
    from jscc.activation_replay import canonical_digest
    b = bundle(job_files)
    approved = b['identity']
    b['resolved']['trajectories'][0]['config']['training']['lr'] = .001
    b['identity'] = canonical_digest({key: value for key, value in b.items() if key != 'identity'})
    path = tmp_path/'bound.json'
    path.write_text(json.dumps(b))
    monkeypatch.setenv('SLURM_ARRAY_TASK_ID', '7')
    command = cli('submit_recipe.py')
    with pytest.raises(ValueError, match='execution retired'):
        command(['run', '--bundle', str(path), '--identity', approved])
    assert not (tmp_path/'trajectory-7.bundle-failure.json').exists()


def bundle(job_files):
    """Construct a saved historical bundle, independently of execution APIs."""
    from jscc.activation_replay import canonical_digest
    from jscc.source_lifetime import capture_source_binding
    _, _, job = job_files
    prepared = str(Path(job['root'])/'prepared.json')
    result = {key: deepcopy(job[key]) for key in
              ('schema', 'root', 'source', 'model', 'python', 'cache', 'output', 'resources', 'groups')}
    result.update(prepared={'path': prepared, 'sha256': file_digest(Path(prepared))},
                  resolved=resolve_recipe(job['recipe'], model_config_path=Path(job['model']['path'])/'config.json'),
                  launch_authorized=False,
                  assets_inspected=[{'path': prepared, 'status': 'selected exact manifest'}])
    result['source_binding'] = capture_source_binding(job['source']['path'], job['source']['commit'], job['python'])
    result['identity'] = canonical_digest(result)
    return result


def saved_submission(job_files, *, accepted_first=True, missing_intent_fields=False):
    """Serialize a historical stopped receipt; no production scheduler/producer runs."""
    from jscc import recipe_submission as submission
    b = bundle(job_files)
    directory = Path(b['output'])
    directory.mkdir()
    submission.write_json(directory/'bundle.json', b)
    (directory/'job.sbatch').write_text('#!/bin/bash\n# retained historical payload\n')
    receipt = {'schema': SCHEMA, 'bundle_identity': b['identity'],
               'source_commit': b['source']['commit'], 'status': 'stopped-no-retry',
               'mocked': True, 'bundle_sha256': file_digest(directory/'bundle.json'),
               'payload_sha256': file_digest(directory/'job.sbatch'), 'jobs': [],
               'source_consumer_reserved': True}
    for index in range(2 if accepted_first else 1):
        row = {'bundle_identity': b['identity'], 'group': index,
               'arguments': ['sbatch', '--array='+','.join(map(str,b['groups'][index]['indices'])),
                             str(directory/'job.sbatch')],
               'payload_sha256': receipt['payload_sha256'], 'status': 'attempting'}
        if not missing_intent_fields:
            row.update(attempt_identity=f'historical-intent-{index}', attempted_at='2026-10-01T00:00:00+00:00')
        if index == 0 and accepted_first:
            row.update(status='submitted', job_id='9501', stdout='9501\n', stderr='', returncode=0)
        receipt['jobs'].append(row)
    submission.write_json(directory/'SUBMISSION.json', receipt)
    registry = submission.source_consumers(b)
    registry.reserve(b['source_binding'], b['identity'], owner=b['output'], reference=str(directory/'SUBMISSION.json'))
    registry.record_state(b['source']['path'], b['identity'], 'unresolved', owner=b['output'], evidence=str(directory/'SUBMISSION.json'))
    return b


@pytest.mark.parametrize('operation', ['preview', 'render', 'scheduler', 'submit', 'run', 'private-preview', 'private-submit', 'private-run', 'prepare-retry', 'authorize-retry'])
def test_old_producers_reject_before_effects(job_files, operation, monkeypatch):
    from jscc import recipe_submission as submission
    from jscc.submission_reconciliation import prepare_retry_bundle, authorize_retry_job
    b = bundle(job_files)
    before = sorted(str(p) for p in Path(b['root']).rglob('*'))
    monkeypatch.setattr(submission, 'submission_preflight', lambda *_: pytest.fail('preflight ran'))
    monkeypatch.setattr(submission, 'source_consumers', lambda *_: pytest.fail('registry consulted'))
    operations = {
        'preview': lambda: submission.preview_job(b, b['output']+'/bundle.json'),
        'render': lambda: render_payload(b, b['output']+'/bundle.json'),
        'scheduler': lambda: scheduler_command(b, b['output']+'/job.sbatch', 0),
        'submit': lambda: submit_job(b, b['output'], scheduler=lambda _: pytest.fail('scheduler ran')),
        'run': lambda: submission.run_job(b, b['identity'], runner=lambda _: pytest.fail('runner ran')),
        'private-preview': lambda: submission._preview_case_job(b, b['output']+'/bundle.json'),
        'private-submit': lambda: submission._submit_case_job(b, b['output'],
            scheduler=lambda _: pytest.fail('scheduler ran'), authorization=None),
        'private-run': lambda: submission._run_case_job(b, b['identity'], 0, runner=lambda _: pytest.fail('runner ran')),
        'prepare-retry': lambda: prepare_retry_bundle(b, b['output'], output=b['output']+'-retry', attempt='02'),
        'authorize-retry': lambda: authorize_retry_job(b, b['output'], new_bundle=b, authorization={})}
    with pytest.raises(ValueError, match='execution retired'):
        operations[operation]()
    assert sorted(str(p) for p in Path(b['root']).rglob('*')) == before


@pytest.mark.parametrize('mode', ['prepare', 'action'])
def test_old_job_rejects_before_inputs_or_output(job_files, mode, monkeypatch, tmp_path):
    import jscc.recipe_submission as submission
    job, limits, _ = job_files
    monkeypatch.setattr(submission, 'local_path', lambda *_: pytest.fail('asset read'))
    limits.unlink()
    with pytest.raises(ValueError, match='execution retired'):
        prepare_job(job, limits)
    output = tmp_path/'new-bundle.json'
    with pytest.raises(ValueError, match='execution retired'):
        cli('submit_recipe.py')([mode, '--job', str(job), '--limits', str(limits), '--output', str(output)])
    assert not output.exists()


@pytest.mark.parametrize('mode', ['preview', 'render', 'submit', 'run', 'prepare-retry', 'authorize-retry'])
def test_valid_old_cli_rejects_without_artifacts(job_files, mode, tmp_path):
    b = bundle(job_files)
    path = tmp_path/'saved-bundle.json'
    path.write_text(json.dumps(b))
    before = sorted(str(p) for p in tmp_path.rglob('*'))
    with pytest.raises(ValueError, match='execution retired'):
        cli('submit_recipe.py')([mode, '--bundle', str(path), '--identity', b['identity'],
                               '--output', str(tmp_path/'new-output')])
    assert sorted(str(p) for p in tmp_path.rglob('*')) == before


def test_saved_preflight_preserves_model_prepared_and_source_integrity(job_files):
    from jscc.recipe_submission import submission_preflight
    from jscc.activation_replay import canonical_digest
    b = bundle(job_files)
    assert submission_preflight(b) == b['source_binding']
    assert cli('submit_recipe.py')(['check', '--bundle', _save_bundle(b)])['trajectories'] == 9
    metadata = Path(b['model']['path'])/'config.json'
    original = metadata.read_bytes()
    metadata.write_text('{}')
    with pytest.raises(ValueError, match='model config digest'):
        submission_preflight(b)
    metadata.write_bytes(original)
    asset = Path(b['prepared']['path'])
    manifest = json.loads(asset.read_text())
    manifest['configs']['8']['training']['lr'] = .001
    asset.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='prepared digest'):
        submission_preflight(b)
    b['prepared']['sha256'] = file_digest(asset)
    b['identity'] = canonical_digest({k:v for k,v in b.items() if k != 'identity'})
    with pytest.raises(ValueError, match='complete config differs'):
        submission_preflight(b)


def _save_bundle(b):
    path = Path(b['root'])/'saved-check.json'
    path.write_text(json.dumps(b))
    return str(path)


@pytest.mark.parametrize('mutation,match', [('dirty', 'tracked changes'), ('runtime', 'runtime'),
                                            ('prepared-source', 'prepared source commit'),
                                            ('prepared-plan', 'exact trajectory')])
def test_saved_preflight_rejects_source_runtime_and_science_changes(job_files, mutation, match):
    from jscc.recipe_submission import submission_preflight
    from jscc.activation_replay import canonical_digest
    b = bundle(job_files)
    if mutation == 'dirty':
        (Path(b['source']['path'])/'worker.py').write_text('changed source')
    elif mutation == 'runtime':
        b['source_binding']['runtime']['python'] += '-different'
    else:
        path = Path(b['prepared']['path'])
        manifest = json.loads(path.read_text())
        if mutation == 'prepared-source':
            manifest['source_commit'] = '0'*40
        else:
            manifest['plans'].pop()
        path.write_text(json.dumps(manifest))
        b['prepared']['sha256'] = file_digest(path)
    b['identity'] = canonical_digest({k:v for k,v in b.items() if k != 'identity'})
    with pytest.raises(ValueError, match=match):
        submission_preflight(b)
    assert not Path(b['output']).exists()
