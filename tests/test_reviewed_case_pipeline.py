"""One real tiny public pipeline with fake scheduling and explicit terminal evidence."""
import json
from pathlib import Path
import os
import subprocess
import sys
from uuid import uuid4
from time import perf_counter

import yaml

from jscc import recipe_submission as submission
from jscc.activation_replay import file_digest
from test_recipe_assets import registered
from test_source_lifetime import git

pytest_plugins = ['test_recipe_workflow']


def test_real_public_combined_case_exact_terminal(joined_assets, tmp_path, monkeypatch):
    source = Path(submission.__file__).resolve().parents[1]
    recipe, registrations, _ = registered(joined_assets)
    output = source/'runs'/('case-pipeline-'+uuid4().hex)
    resources = dict(partition='h200q', account='research', qos='h200q_1g', node='hgx-cn01',
                     gpus=1, cpus=16, memory_gb=170, walltime_minutes=120)
    job = dict(schema=submission.CASE_SCHEMA, recipe=str(recipe), registrations=str(registrations),
        run_output=str(output), root=str(source.parent), source={'path': str(source), 'commit': git(source, 'rev-parse', 'HEAD')},
        python=sys.executable, cache=str(source/'runs/cache'), output=str(output)+'-submission',
        resources=resources, cases=['width-8-ratio-50'], evaluation_cases=['width-8-ratio-50'],
        concurrency=1, combined=True, attempt='pipeline')
    path, limits = tmp_path/'job.yaml', tmp_path/'limits.yaml'
    path.write_text(yaml.safe_dump(job))
    limits.write_text(yaml.safe_dump(dict(root=str(source.parent), resources=resources, max_concurrency=3, max_array_tasks=9)))
    started = perf_counter()
    bundle = submission.prepare_case_job(path, limits)
    receipt = submission.submit_job(bundle, bundle['output'], scheduler=lambda args:
        subprocess.CompletedProcess(args, 0, '56789;fake\n', ''))
    assert receipt['mocked'] and len(receipt['jobs']) == 1
    monkeypatch.setenv('SLURM_JOB_ID', '56789')
    monkeypatch.setenv('SLURM_ARRAY_TASK_ID', '0')
    def public_runner(args):
        if Path(args[1]).name == 'train.py':
            from train import main
        else:
            from evaluate import main
        main(args[2:])
        return subprocess.CompletedProcess(args, 0, 'public command completed', '')
    outcome = submission.run_job(bundle, bundle['identity'], 0, runner=public_runner)
    assert outcome['status'] == 'completed'
    assert [row['status'] for row in outcome['commands']] == ['completed', 'completed']
    checkpoint = outcome['commands'][1]['eligibility']['checkpoint']
    assert checkpoint['role'] == 'terminal' and checkpoint['step'] == 24
    assert checkpoint['sha256'] == file_digest(checkpoint['path'])
    assert checkpoint['training_attempt'].endswith('/training-pipeline-g0-i0')
    gate = output/'evaluations/evaluation-pipeline-g0-i0/gate.json'
    assert json.loads(gate.read_text())['final_eligible'] is True
    all_terminal = Path(bundle['output'])/'MOCK-ALL-TASKS-TERMINAL.json'
    submission.write_json(all_terminal, {'mocked': True, 'tasks': [{'group': 0, 'index': 0, 'status': 'completed'}],
        'launch_receipt': str(Path(bundle['output'])/'tasks/g0-i0.launch.json'), 'real_scheduler': False})
    registry = submission.source_consumers(bundle)
    registry.record_state(source, bundle['identity'], 'completed', owner=bundle['output'], evidence=str(all_terminal))
    registry.release(source, bundle['identity'], owner=bundle['output'])
    if os.environ.get('TICKET56_EVIDENCE_DIR'):
        evidence = Path(os.environ['TICKET56_EVIDENCE_DIR'])
        submission.write_json(evidence/'REAL-PIPELINE.json', {'source': bundle['source'], 'seconds': perf_counter()-started,
            'optimizer_updates': 24, 'task_count': 1, 'mocked': True, 'output': str(output),
            'submission': receipt, 'outcome': outcome, 'terminal': checkpoint,
            'gate': json.loads(gate.read_text()), 'all_terminal': str(all_terminal)})
