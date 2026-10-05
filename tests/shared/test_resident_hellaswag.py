import json
from pathlib import Path

import pytest

from jscc import evaluation, panel_workers as core
from shared.tiny_parallel_helpers import tiny_runtime
from shared.resident_helpers import ResidentFactory, score_resident


def make_plan(tmp_path):
    checkpoint = tmp_path/'fixture-checkpoint'
    checkpoint.write_bytes(b'fixture checkpoint admission')
    fixture = tiny_runtime('hellaswag', checkpoint)
    request = fixture['request']
    request['conditions'] = [0.0, 6.0]
    fixture['patches'].undo()
    reference = core.file_reference(checkpoint)
    plan = core.make_plan(case_id='tiny-case', checkpoint={**reference, 'size': reference['bytes'],
        'role': 'terminal', 'step': 1}, scope='initial', sites=request['sites'], conditions=[0.0, 6.0],
        item_ids=request['input_ids'], identities={'source': 'fixture', 'runtime': 'cpu',
        'input': request, 'scorer': 'actual-hellaswag-body-fixture-admission'}, noise=request['noise'])
    return checkpoint, plan


def test_one_dataset_load_per_worker_for_multiple_real_panels(tmp_path, monkeypatch):
    checkpoint, plan = make_plan(tmp_path)
    runtime = ResidentFactory(str(checkpoint))(plan, 'cpu')
    serial = []
    try:
        for panel in plan['panels']:
            output = tmp_path/'serial'/panel['id']
            output.mkdir(parents=True)
            serial.append(score_resident(runtime, plan, panel, output))
        assert len(runtime['dataset_loads']) == 1
        from lm_eval import evaluator
        real_evaluate = evaluator.simple_evaluate
        def fail_after_real_scoring(*args, **kwargs):
            real_evaluate(*args, **kwargs)
            raise evaluation.TaskScoringError('fixture failure after real HellaSwag scoring')
        monkeypatch.setattr(evaluator, 'simple_evaluate', fail_after_real_scoring)
        failed = tmp_path/'failure'
        failed.mkdir()
        with pytest.raises(core.PanelScoringError, match='real HellaSwag scoring'):
            score_resident(runtime, plan, plan['panels'][0], failed)
        monkeypatch.setattr(evaluator, 'simple_evaluate', real_evaluate)
        recovered = tmp_path/'after-failure'
        recovered.mkdir()
        next_panel = score_resident(runtime, plan, plan['panels'][0], recovered)
        assert next_panel['scores'] == serial[0]['scores']
        assert next_panel['actual_epsilons'] == serial[0]['actual_epsilons']
        assert next_panel['task_evidence'] == serial[0]['task_evidence']
        assert next_panel['payload_counts'] == serial[0]['payload_counts']
        assert len(runtime['dataset_loads']) == 1
        with pytest.raises(ValueError, match='original recipe/data settings differ'):
            evaluation.evaluate_hellaswag(runtime['model'], runtime['processor'],
                {**runtime['request']['settings'], 'evidence_mode': 'compact-v1'},
                tmp_path/'mismatch', 0.0, {'name': 'other-fixture', 'revision': 'fixed'},
                prepared_task=runtime['data'])
        assert len(runtime['dataset_loads']) == 1
    finally:
        runtime['patches'].undo()
    attempt = core.reserve_attempt(tmp_path/'attempts', 'fresh', plan)
    states = core.run_workers(plan, attempt, core.assignments_for(plan, devices=['cpu', 'cpu'], workers=2),
        runtime_factory=ResidentFactory(str(checkpoint)), score_panel=score_resident, timeout_seconds=90)
    aggregate = core.collect_results(attempt, plan, worker_states=states, final=True)
    assert aggregate['counts']['complete'] == 4
    assert aggregate['observed_items'] == 8
    parallel = [json.loads(Path(row['receipt']['result']['path']).read_text()) for row in aggregate['rows']]
    assert len({r['pid'] for r in parallel}) == 2
    assert all(sum(r['pid'] == row['pid'] for r in parallel) == 2 for row in parallel)
    for expected, actual in zip(serial, parallel, strict=True):
        assert actual['dataset_loads'] == actual['initialize_calls'] == 1
        assert actual['item_ids'] == [0, 1]
        assert actual['scores'] == pytest.approx(expected['scores'], rel=1e-6, abs=1e-7)
        assert actual['actual_epsilons'] == expected['actual_epsilons']
        assert actual['actual_epsilons']
        assert actual['task_evidence'] == expected['task_evidence']
        assert actual['payload_counts'] == expected['payload_counts']
        receipt = json.loads(Path(actual['scientific_receipt']['path']).read_text())
        assert receipt['metrics']['candidate_forward_requests'] == 8


def test_ordinary_default_real_evaluation_remains_available(tmp_path):
    checkpoint = tmp_path/'checkpoint'
    checkpoint.write_bytes(b'fixture')
    runtime = tiny_runtime('hellaswag', checkpoint)
    try:
        settings = {**runtime['request']['settings'], 'evidence_mode': 'compact-v1', 'noise_seed': 13,
                    'noise_namespace': 'tiny-paired-hellaswag'}
        with runtime['model'].transmission(None):
            metrics = evaluation.evaluate_hellaswag(runtime['model'], runtime['processor'], settings,
                tmp_path/'ordinary', 'no_noise', runtime['request']['data_settings'])
        assert metrics['finite_scores']
        assert metrics['candidate_forward_requests'] == 8
        assert (tmp_path/'ordinary'/'compact_no_noise.jsonl').is_file()
    finally:
        runtime['patches'].undo()
