"""Joined released interfaces; real tiny CPU science and injected scheduler facts."""
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from uuid import uuid4

import pytest
import yaml

from jscc import submission as submission
from jscc.result_collection import collect_snapshot, prepare_snapshot
from jscc.submission_reconciliation import finalize_source_consumer
from shared.test_recipe_assets import registered
from contracts.test_source_lifetime import git
from contracts.test_submission_reconciliation import evidence

pytest_plugins = ['shared.test_recipe_workflow']


def test_public_separate_arrays_terminal_release_and_compact_collection(joined_assets, tmp_path, monkeypatch):
    source = Path(submission.__file__).resolve().parents[1]
    timings, calls, loads, initializations = {}, Counter(), [], []
    active_loads = {}
    stage = 'registration_preparation'

    def profile(frame, event, _arg):
        name, filename = frame.f_code.co_name, frame.f_code.co_filename
        if '/jscc/' not in filename:
            return
        if event == 'call' and name in ('file_digest', '_digest', 'state_dict_identity'):
            path = frame.f_locals.get('path')
            suffix = (Path(path).suffix if isinstance(path, (str, Path))
                      else 'tensor-state' if name == 'state_dict_identity' else 'structured-record')
            kind = ('initializer' if isinstance(path, (str, Path)) and Path(path).name == 'initial.pt'
                    else 'backbone' if suffix in ('.safetensors', '.bin')
                    else 'cached_data' if suffix == '.arrow'
                    else 'checkpoint_or_tensor' if suffix == '.pt'
                    else 'tensor_identity' if name == 'state_dict_identity'
                    else 'structured_record' if suffix == 'structured-record' else 'small_record')
            calls[(stage, name, suffix, kind)] += 1
        if name in ('load_recipe_workflow', 'initialize_recipe'):
            if event == 'call':
                active_loads[id(frame)] = perf_counter()
            elif event == 'return' and id(frame) in active_loads:
                target = loads if name == 'load_recipe_workflow' else initializations
                target.append({'stage': stage, 'seconds': perf_counter() - active_loads.pop(id(frame))})

    acquisition = json.loads((joined_assets['root'] / 'model-acquisition-cost.json').read_text())
    assert acquisition['weight_files_hashed'] == ['model.safetensors']
    joined_assets['weight_hashes'].clear()
    previous_profile = sys.getprofile()
    sys.setprofile(profile)
    try:
        start = perf_counter()
        recipe, registrations, acquired = registered(joined_assets)
        timings[stage] = perf_counter() - start
        output = source / 'runs' / ('operations-joined-' + uuid4().hex)
        resources = dict(partition='h200q', account='research', qos='h200q_1g', node='hgx-cn01',
                         gpus=1, cpus=16, memory_gb=170, walltime_minutes=120)
        case_id = 'width-8-ratio-50'
        job = dict(schema=submission.CASE_SCHEMA, recipe=str(recipe), registrations=str(registrations),
                   run_output=str(output), root=str(source.parent),
                   source={'path': str(source), 'commit': git(source, 'rev-parse', 'HEAD')},
                   python=sys.executable, cache=str(source / 'runs/cache'), output=str(output) + '-submission',
                   resources=resources, cases=[case_id], evaluation_cases=[case_id], concurrency=1,
                   combined=False, attempt='joined')
        path, limits = tmp_path / 'job.yaml', tmp_path / 'limits.yaml'
        path.write_text(yaml.safe_dump(job))
        limits.write_text(yaml.safe_dump(dict(root=str(source.parent), resources=resources,
                                              max_concurrency=3, max_array_tasks=9)))
        stage = 'submission_preparation'
        start = perf_counter()
        bundle = submission.prepare_case_job(path, limits)
        timings[stage] = perf_counter() - start
        scheduler_calls = []

        def scheduler(args):
            scheduler_calls.append(args)
            return subprocess.CompletedProcess(args, 0, str(56789 + len(scheduler_calls) - 1) + ';fake\n', '')

        receipt = submission.submit_job(bundle, bundle['output'], scheduler=scheduler)
        assert receipt['mocked'] and len(receipt['jobs']) == 2
        assert bundle['groups'][1]['aftercorr'] == 0
        assert any('aftercorr:56789' in arg for arg in scheduler_calls[1])
        monkeypatch.setenv('SLURM_ARRAY_TASK_ID', '0')
        outcomes = []

        def public_runner(args):
            from experiment import main
            main(args[2:])
            return subprocess.CompletedProcess(args, 0, 'real tiny public command completed', '')

        for group, operation in enumerate(('public_training', 'public_evaluation')):
            stage = operation
            monkeypatch.setenv('SLURM_JOB_ID', str(56789 + group))
            start = perf_counter()
            outcomes.append(submission.run_job(bundle, bundle['identity'], group, runner=public_runner))
            timings[stage] = perf_counter() - start
            assert outcomes[-1]['status'] == 'completed'
        checkpoint = outcomes[1]['commands'][0]['eligibility']['checkpoint']
        assert checkpoint['role'] == 'terminal' and checkpoint['step'] == 24
        assert checkpoint['training_attempt'].endswith('/training-joined-g0-i0')
        gate = json.loads((output / 'evaluations/evaluation-joined-g1-i0/gate.json').read_text())
        assert gate['final_eligible'] is True

        stage = 'reconciliation'
        original_receipt = (Path(bundle['output']) / 'SUBMISSION.json').read_bytes()
        proof = evidence(bundle)
        accepted = submission.reconcile_job(bundle, bundle['output'], evidence=proof)
        for group in accepted['groups']:
            proof['tasks'].append(dict(group=group['group'], index=0, job_id=group['job_id'],
                                       scheduler_task_id=group['job_id'] + '_0', query_status='complete',
                                       authoritative=True, status='completed', reference='injected-fake-terminal'))
        incomplete = deepcopy(proof)
        incomplete['tasks'].pop()
        assert not submission.reconcile_job(bundle, bundle['output'], evidence=incomplete)['all_tasks_terminal']
        with pytest.raises(ValueError, match='terminal evidence'):
            finalize_source_consumer(bundle, bundle['output'], release=True)
        terminal = submission.reconcile_job(bundle, bundle['output'], evidence=proof)
        assert terminal['all_tasks_terminal']
        assert finalize_source_consumer(bundle, bundle['output'])['released'] is False
        released = finalize_source_consumer(bundle, bundle['output'], release=True)
        assert released['released'] is True
        submission.source_consumers(bundle).require_released(source)
        assert (Path(bundle['output']) / 'SUBMISSION.json').read_bytes() == original_receipt
        assert len(scheduler_calls) == 2

        snapshot, destination = tmp_path / 'snapshot', tmp_path / 'collected'
        stage = 'snapshot_export'
        start = perf_counter()
        manifest = prepare_snapshot(output, snapshot, source_uri='fixture://joined/run')
        timings[stage] = perf_counter() - start
        stage = 'collection_copy'
        start = perf_counter()
        copied = collect_snapshot(snapshot, destination)
        timings[stage] = perf_counter() - start
        stage = 'collection_retry'
        start = perf_counter()
        retry = collect_snapshot(snapshot, destination)
        timings[stage] = perf_counter() - start
        assert copied['status'] == retry['status'] == 'complete'
        assert retry['counts']['copied']['count'] == 0
        assert retry['counts']['reused'] == copied['counts']['selected']
        scalar_rows = [json.loads(row) for row in (snapshot / 'exports/scalars.jsonl').read_text().splitlines()]
        result_rows = [json.loads(row) for row in (snapshot / 'exports/results.jsonl').read_text().splitlines()]
        engine = Path(checkpoint['training_attempt']) / 'engine'
        raw_rows = [json.loads(row) for row in (engine / 'metrics.jsonl').read_text().splitlines()]
        assert len(scalar_rows) == len(raw_rows) > 0
        assert result_rows and all('row' in row for row in result_rows)
        assert not any(row['path'].endswith(('.pt', 'metrics.jsonl', 'items.jsonl')) for row in manifest['artifacts'])
        assert Path(checkpoint['path']).is_file()
        assert any(row['path'].endswith('.pt') for row in manifest['remote'])
        assert joined_assets['weight_hashes'] == []
        if os.environ.get('TICKET59_EVIDENCE_DIR'):
            submission.write_json(Path(os.environ['TICKET59_EVIDENCE_DIR']) / 'JOINED-PUBLIC-MEASUREMENTS.json', {
                'source': bundle['source'], 'runtime': {'python': sys.version, 'platform': platform.platform()},
                'case_id': case_id, 'output': str(output), 'seconds': timings,
                'acquisition_before_profile': acquisition, 'workflow_load_seconds': loads,
                'model_initialization_seconds': initializations,
                'hash_calls': [dict(stage=s, function=f, suffix=x, kind=k, count=n)
                               for (s, f, x, k), n in sorted(calls.items())],
                'denominators': {'cases': 1, 'scheduler_tasks': 2, 'scheduler_adapter_calls': 2,
                                 'optimizer_updates': 24, 'scalar_rows': len(scalar_rows),
                                 'raw_metric_rows': len(raw_rows), 'result_rows': len(result_rows)},
                'acquired': acquired, 'submission': receipt, 'outcomes': outcomes, 'gate': gate,
                'checkpoint': checkpoint, 'terminal': terminal, 'release': released,
                'snapshot': manifest, 'collection': copied, 'retry': retry,
                'limits': 'Real tiny cached CPU engine/scoring/copy; injected scheduler acceptance and terminal facts. No live/GPU/scientific acceptance.'})
    finally:
        sys.setprofile(previous_profile)


def test_public_coco_preselection_save_continues_without_last_alias(tmp_path, monkeypatch):
    """Scale the named100/500 save policy to3/15; restore a valid no-best state."""
    import torch
    from request_helpers import invoke
    from jscc import ordinary_continuation, ordinary_runs, training
    from ordinary.test_ordinary_continuation import task_fixture, assert_state_equal, read

    config, _, _ = task_fixture(tmp_path, monkeypatch, 'coco', max_steps=15, checkpoint=3, accumulation=1)
    config['training'].update(eval_every=15, selection_steps=[15], save_steps=[3, 15])
    authored = tmp_path/'coco.yaml'
    authored.write_text(yaml.safe_dump(config))
    baseline = ordinary_runs.train_task(config, tmp_path/'baseline')
    original = training.append_metrics

    def interrupt(path, row):
        original(path, row)
        if row.get('phase') == 'train' and row['step'] == 5:
            raise KeyboardInterrupt('unsaved work after pre-selection checkpoint')

    monkeypatch.setattr(training, 'append_metrics', interrupt)
    root = tmp_path/'interrupted'
    with pytest.raises(KeyboardInterrupt):
        invoke(tmp_path, 'ordinary', 'train', {'root': str(root), 'attempt': '0001'}, config)
    parent = root/'attempts/training-0001'
    checkpoint = parent/'step_000003.pt'
    state, _ = ordinary_continuation.load_checkpoint(checkpoint)
    assert state['best'] == float('inf') and state['continuation']['best_step'] is None
    assert not (parent/'last.pt').exists()
    unchanged = checkpoint.read_bytes()
    monkeypatch.setattr(training, 'append_metrics', original)
    receipt = read(checkpoint.with_suffix('.pt.json'))
    invoke(tmp_path, 'ordinary', 'resume', {'root': str(root), 'attempt': 'resumed', 'continuation': {
           'path': str(checkpoint), 'parent_attempt': str(parent), 'role': receipt['role'],
           'sha256': receipt['sha256'], 'size_bytes': receipt['size_bytes'], 'step': receipt['step']}})
    child = Path(read(root/'training.json')['path'])
    assert read(child/'attempt.json')['request']['next_optimizer_step'] == 4
    assert read(child/'completion.json')['attempt_optimizer_updates'] == 12
    assert checkpoint.read_bytes() == unchanged
    actual = torch.load(child/'last.pt', weights_only=True)
    expected = torch.load(baseline/'attempts/training-0001/last.pt', weights_only=True)
    for key in ('codec', 'memory_codec', 'channel', 'optimizer', 'scheduler', 'scaler', 'step'):
        assert_state_equal(actual[key], expected[key])
    for key in ('cursor', 'counters', 'rng', 'best_step'):
        assert_state_equal(actual['continuation'][key], expected['continuation'][key])
