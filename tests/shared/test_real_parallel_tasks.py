"""Real tiny COCO/CIDEr and HellaSwag bodies through spawned recipe panels."""
import json
from pathlib import Path

import pytest

from jscc import evaluation, panel_workers as core
from shared.tiny_parallel_helpers import TinyFactory, score_tiny, tiny_runtime


@pytest.mark.parametrize('task', ['coco', 'hellaswag'])
def test_real_tiny_serial_parallel_scores_inventory_epsilon(tmp_path, task):
    checkpoint = tmp_path/'fixture-checkpoint'
    checkpoint.write_bytes(b'fixture checkpoint identity; state I/O explicitly injected')
    fixture = tiny_runtime(task, checkpoint)
    request = fixture['request']
    fixture['patches'].undo()
    reference = core.file_reference(checkpoint)
    target = {**reference, 'size': reference['bytes'], 'role': 'terminal', 'step': 1}
    plan = core.make_plan(case_id='tiny-case', checkpoint=target, scope='initial',
        sites=request['sites'], conditions=request['conditions'], item_ids=request['input_ids'],
        identities={'source': 'tiny-fixture', 'runtime': 'cpu', 'input': request,
                    'scorer': 'real-task-body-fixture-admission', 'training_observation_eligible': False},
        noise=request['noise'])
    serial = TinyFactory(task, str(checkpoint))(plan, 'cpu')
    serial_rows = []
    try:
        for panel in plan['panels']:
            output = tmp_path/'serial'/panel['id']
            output.mkdir(parents=True)
            serial_rows.append(score_tiny(serial, plan, panel, output))
        assert serial['initialize_calls'] == 1
    finally:
        serial['patches'].undo()
    attempt = core.reserve_attempt(tmp_path/'parallel', 'fresh', plan)
    states = core.run_workers(plan, attempt, core.assignments_for(plan, workers=2, devices=['cpu', 'cpu']),
        runtime_factory=TinyFactory(task, str(checkpoint)), score_panel=score_tiny, timeout_seconds=60)
    aggregate = core.collect_results(attempt, plan, worker_states=states, final=True)
    assert aggregate['counts']['complete'] == 2
    assert aggregate['observed_items'] == 4
    parallel_rows = [json.loads(Path(row['receipt']['result']['path']).read_text()) for row in aggregate['rows']]
    assert len({row['pid'] for row in parallel_rows}) == 2
    for expected, actual in zip(serial_rows, parallel_rows, strict=True):
        assert actual['initialize_calls'] == 1
        assert actual['item_ids'] == expected['item_ids'] == request['input_ids']
        assert actual['scores'] == pytest.approx(expected['scores'], rel=1e-6, abs=1e-7)
        assert actual['actual_epsilons'] == expected['actual_epsilons']
        assert actual['actual_epsilons']  # Actual Gaussian returns, not seed/policy hashes.
        scientific = json.loads(Path(actual['scientific_receipt']['path']).read_text())
        assert scientific['denominator'] == scientific['completed_items'] == 2
        if task == 'coco':
            assert scientific['metrics']['num_samples'] == 2
            assert 'cider' in scientific['metrics']
        else:
            assert scientific['metrics']['candidate_forward_requests'] == 8


def test_real_tiny_coco_restoration_on_scoring_failure(tmp_path, monkeypatch):
    checkpoint = tmp_path/'fixture-checkpoint'
    checkpoint.write_bytes(b'fixture')
    runtime = tiny_runtime('coco', checkpoint)
    runtime['initialize_calls'] = 1
    real_score = evaluation.score_coco_captions
    def fail_after_real_cider(*args, **kwargs):
        real_score(*args, **kwargs)
        raise evaluation.TaskScoringError('fixture failure after actual CIDEr')
    monkeypatch.setattr(evaluation, 'score_coco_captions', fail_after_real_cider)
    try:
        with pytest.raises(core.PanelScoringError, match='actual CIDEr'):
            score_tiny(runtime, {'expected_items': 2, 'noise': runtime['request']['noise']},
                {'site': 'dec_l0', 'condition': 0.0}, tmp_path)
        assert json.loads((tmp_path/'restoration.json').read_text())['restored'] is True
    finally:
        runtime['patches'].undo()


@pytest.mark.parametrize('mutation', ['bytes', 'payload'])
def test_resident_checkpoint_rejects_mutation_and_restores(tmp_path, mutation):
    from copy import deepcopy
    checkpoint = tmp_path/'checkpoint'
    checkpoint.write_bytes(b'validated fixture bytes')
    runtime = tiny_runtime('coco', checkpoint)
    runtime['checkpoint_payload'] = deepcopy(runtime['state'].payload)
    runtime['initialize_calls'] = 1
    if mutation == 'bytes':
        checkpoint.write_bytes(b'changed fixture bytes')
    else:
        runtime['state'].payload['metadata']['plan_identity'] = 'changed'
    try:
        with pytest.raises(ValueError, match='worker checkpoint'):
            score_tiny(runtime, {'expected_items': 2, 'noise': runtime['request']['noise']},
                {'site': 'dec_l0', 'condition': 0.0}, tmp_path)
    finally:
        runtime['patches'].undo()
