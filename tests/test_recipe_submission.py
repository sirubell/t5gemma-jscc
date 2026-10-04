"""Offline preparation, exact Slurm arguments, and uncertain outcome retention."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest
import yaml

from jscc.activation_replay import file_digest
from jscc.experiment_recipe import resolve_recipe
from jscc.encoder_stages_plan import Original13Plan
from jscc.recipe_submission import (SCHEMA, normalize_path, prepare_job, render_payload,
                                    scheduler_command, submit_job, validate_bundle)
from test_original13_joined import cli
from test_original13_plan_state import formal_views

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def job_files(tmp_path):
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
    manifest = {'configs': {str(i['width']): i['config'] for i in resolved['trajectories']},
                'plans': plans, 'observation_request': {'panel': {'expected_items': 256},
                    'sites': recipe['observation']['sites'], 'conditions': recipe['observation']['conditions']}}
    prepared = tmp_path/'prepared.json'
    prepared.write_text(json.dumps(manifest))
    wrong = tmp_path/'wrong.json'
    wrong.write_text('{}')
    resources = dict(partition='h200q', account='research', qos='h200q_1g', node='hgx-cn01',
                     gpus=1, cpus=16, memory_gb=170, walltime_minutes=120)
    job = dict(schema=SCHEMA, recipe=str(recipe_file), root=str(tmp_path),
               source={'commit': '1'*40, 'path': str(tmp_path/'release')},
               model={'path': str(model), 'config_sha256': file_digest(metadata)},
               prepared={'sha256': file_digest(prepared),
                         'candidates': [str(tmp_path/'missing.json'), str(wrong), str(prepared)]},
               python=str(tmp_path/'python'), cache=str(tmp_path/'cache'), output=str(tmp_path/'campaign'),
               resources=resources, groups=[{'indices': [0, 1, 2], 'concurrency': 1, 'afterok': None},
                                            {'indices': list(range(3, 9)), 'concurrency': 3, 'afterok': 0}])
    limits = dict(root=str(tmp_path), resources=resources, max_concurrency=3, max_array_tasks=9)
    job_path, limits_path = tmp_path/'job.yaml', tmp_path/'limits.yaml'
    job_path.write_text(yaml.safe_dump(job))
    limits_path.write_text(yaml.safe_dump(limits))
    return job_path, limits_path, job


def bundle(job_files):
    job, limits, _ = job_files
    return prepare_job(job, limits)


def test_resolve_assets_render_exact_commands(job_files):
    b = bundle(job_files)
    assert [r['status'] for r in b['assets_inspected']] == ['missing', 'digest differs', 'selected exact manifest']
    assert b['launch_authorized'] is False
    payload = render_payload(b, b['output']+'/bundle.json')
    assert payload.startswith('#!/bin/bash -l\nset -euo pipefail')
    for env in ('HF_HUB_OFFLINE=1', 'HF_DATASETS_OFFLINE=1', 'CUBLAS_WORKSPACE_CONFIG=:4096:8'):
        assert env in payload
    assert b['source']['path']+'/scripts/submit_recipe.py run' in payload
    assert scheduler_command(b, b['output']+'/job.sbatch', 1, '12345') == [
        'sbatch', '--parsable', '--export=NONE', '--no-requeue', '--job-name=codec-recipe',
        '--partition=h200q', '--account=research', '--qos=h200q_1g', '--nodelist=hgx-cn01', '--gres=gpu:1',
        '--cpus-per-task=16', '--mem=170G', '--time=120', '--array=3,4,5,6,7,8%3',
        '--output='+b['output']+'/logs/%A_%a.stdout', '--error='+b['output']+'/logs/%A_%a.stderr',
        '--dependency=afterok:12345', '--kill-on-invalid-dep=yes', b['output']+'/job.sbatch']


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


@pytest.mark.parametrize('mutation, match', [
    (lambda j: j['resources'].update(cpus=17), 'invalid cpus'),
    (lambda j: j['resources'].update(walltime_minutes=9), 'invalid walltime'),
    (lambda j: j['resources'].update(account='unapproved'), 'unapproved account'),
    (lambda j: j['resources'].update(node='bad\nnode'), 'invalid Slurm'),
    (lambda j: j['groups'][0].update(concurrency=True), 'invalid concurrency'),
    (lambda j: j['groups'][0].update(afterok=0), 'earlier group'),
    (lambda j: j['groups'][1].update(indices=[0, 1]), 'exactly once'),
    (lambda j: j['source'].update(commit='short'), 'full source'),
    (lambda j: j.update(output=j['root']+'/../escape'), 'path escape'),
    (lambda j: j['model'].update(config_sha256='f'*64), 'model config digest'),
    (lambda j: j['prepared'].update(sha256='f'*64), 'no compatible existing'),
])
def test_invalid_job(job_files, mutation, match):
    path, limits, job = job_files
    mutation(job)
    path.write_text(yaml.safe_dump(job))
    with pytest.raises(ValueError, match=match):
        prepare_job(path, limits)


def test_changed_recipe_needs_integration(job_files):
    path, limits, job = job_files
    recipe_path = Path(job['recipe'])
    recipe = yaml.safe_load(recipe_path.read_text())
    recipe['overrides']['training']['lr'] = .001
    recipe_path.write_text(yaml.safe_dump(recipe))
    with pytest.raises(ValueError, match='complete config differs'):
        prepare_job(path, limits)


def test_symlink_asset(job_files, tmp_path):
    path, limits, job = job_files
    link = tmp_path/'link.json'
    link.symlink_to(tmp_path/'prepared.json')
    job['prepared']['candidates'] = [str(link)]
    path.write_text(yaml.safe_dump(job))
    with pytest.raises(ValueError, match='symlink changes'):
        prepare_job(path, limits)


def test_receipts_and_duplicate_prevention(job_files):
    b = bundle(job_files)
    commands = []

    def scheduler(command):
        receipt = json.loads((Path(b['output'])/'SUBMISSION.json').read_text())
        assert receipt['jobs'][-1]['status'] == 'attempting'
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, str(100+len(commands))+';cluster\n', '')

    receipt = submit_job(b, b['output'], scheduler=scheduler)
    assert receipt['status'] == 'submitted' and receipt['mocked'] is True
    assert [j['job_id'] for j in receipt['jobs']] == ['101', '102']
    assert '--dependency=afterok:101' in commands[1]
    root = Path(b['output'])
    assert receipt['payload_sha256'] == file_digest(root/'job.sbatch')
    assert receipt['bundle_sha256'] == file_digest(root/'bundle.json')
    assert (root/'logs').is_dir()
    with pytest.raises(FileExistsError):
        submit_job(b, b['output'], scheduler=scheduler)
    assert len(commands) == 2


@pytest.mark.parametrize('failure', ['nonzero', 'ambiguous', 'exception', 'keyboard'])
def test_partial_ambiguous_outcomes(job_files, failure):
    b = bundle(job_files)
    calls = []

    def scheduler(command):
        calls.append(command)
        if len(calls) == 1:
            return subprocess.CompletedProcess(command, 0, '123\n', '')
        if failure == 'exception':
            raise OSError('connection lost after possible acceptance')
        if failure == 'keyboard':
            raise KeyboardInterrupt()
        return subprocess.CompletedProcess(command, 1 if failure == 'nonzero' else 0,
                                            'accepted but no ID', 'diagnostic')

    with pytest.raises((ValueError, OSError, KeyboardInterrupt)):
        submit_job(b, b['output'], scheduler=scheduler)
    receipt = json.loads((Path(b['output'])/'SUBMISSION.json').read_text())
    assert receipt['status'] == 'stopped-no-retry'
    assert receipt['jobs'][0]['job_id'] == '123'
    assert receipt['jobs'][1]['arguments'] == calls[1]
    if failure in ('nonzero', 'ambiguous'):
        assert receipt['jobs'][1]['stdout'] == 'accepted but no ID'
    with pytest.raises(FileExistsError):
        submit_job(b, b['output'], scheduler=scheduler)
    assert len(calls) == 2


def test_no_scheduler_without_approval(job_files, monkeypatch):
    b = bundle(job_files)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('unexpected scheduler'))
    with pytest.raises(ValueError, match='explicit exact live'):
        submit_job(b, b['output'])
    assert not Path(b['output']).exists()


def test_cli_offline_operations(job_files, tmp_path, monkeypatch):
    path, limits, _ = job_files
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('unexpected subprocess'))
    command = cli('submit_recipe.py')
    output = tmp_path/'bundle.json'
    b = command(['prepare', '--job', str(path), '--limits', str(limits), '--output', str(output)])
    assert command(['check', '--bundle', str(output)])['launch_authorized'] is False
    command(['render', '--bundle', str(output), '--output', str(tmp_path/'render.sbatch')])
    responses = tmp_path/'mock.json'
    responses.write_text(json.dumps([{'returncode': 0, 'stdout': '123\n', 'stderr': ''},
                                    {'returncode': 0, 'stdout': '124\n', 'stderr': ''}]))
    assert command(['submit', '--bundle', str(output), '--mock-results', str(responses)])['status'] == 'submitted'
    with pytest.raises(FileExistsError):
        command(['prepare', '--job', str(path), '--limits', str(limits), '--output', str(output)])
    with pytest.raises(SystemExit):
        command(['check', '--bundle', str(output), '--live'])
    assert b['identity'] == validate_bundle(json.loads(output.read_text()))['identity']


def test_bundle_tampering(job_files):
    b = deepcopy(bundle(job_files))
    b['resources']['gpus'] = 5
    with pytest.raises(ValueError, match='bundle identity'):
        validate_bundle(b)


def test_recheck_model_and_prepared_at_submission(job_files):
    b = bundle(job_files)
    Path(b['model']['path'], 'config.json').write_text('{}')
    calls = []
    with pytest.raises(ValueError, match='model config digest differs'):
        submit_job(b, b['output'], scheduler=calls.append)
    assert not calls and not Path(b['output']).exists()


def test_output_parent_symlink_escape(job_files, tmp_path):
    path, limits, job = job_files
    link = tmp_path/'link'
    link.symlink_to(tmp_path, target_is_directory=True)
    job['output'] = str(link/'campaign')
    path.write_text(yaml.safe_dump(job))
    b = bundle(job_files)
    with pytest.raises(ValueError, match='parent symlink'):
        submit_job(b, b['output'], scheduler=lambda command: None)
    assert not (tmp_path/'campaign').exists()


def test_run_payload_uses_bound_training_cli(job_files, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from jscc import recipe_submission as submission
    b = bundle(job_files)
    Path(b['source']['path']).mkdir()
    Path(b['output']).mkdir()
    monkeypatch.setattr(submission, '__file__', b['source']['path']+'/jscc/recipe_submission.py')
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    monkeypatch.setenv('SLURM_ARRAY_TASK_ID', '4')
    monkeypatch.setattr(subprocess, 'check_output',
                        lambda args, **kw: '' if 'status' in args else b['source']['commit']+'\n')
    monkeypatch.setattr('torch.cuda.device_count', lambda: 1)
    monkeypatch.setattr('torch.cuda.is_bf16_supported', lambda: True)
    monkeypatch.setattr('torch.cuda.get_device_properties', lambda index: SimpleNamespace(uuid='fixture-gpu'))
    calls = []
    monkeypatch.setattr('os.execv', lambda path, argv: calls.append((path, argv)))
    submission.run_job(b, b['identity'])
    executable, args = calls[0]
    assert executable == b['python']
    assert args[:3] == [b['python'], b['source']['path']+'/scripts/original13_training.py', 'train']
    assert args[args.index('--width')+1] == '1152'
    assert args[args.index('--ratio')+1] == '75'
    assert args[args.index('--trajectory')+1] == '4'
    assert args[args.index('--approved-source')+1] == b['source']['commit']
    assert args[args.index('--device-uuid')+1] == 'GPU-fixture-gpu'
    assert args[args.index('--timeout-seconds')+1] == '6900'
    with pytest.raises(FileExistsError):
        submission.run_job(b, b['identity'])
    assert len(calls) == 1
