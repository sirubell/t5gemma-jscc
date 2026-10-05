import hashlib
import json
from pathlib import Path

import pytest

from jscc.result_collection import prepare_snapshot, collect_snapshot


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    write(root/'experiment.json', {'schema': 'experiment-run-v1'})
    (root/'config.yaml').write_text('task: coco\n')
    write(root/'case.json', {'case_id': 'one', 'config_sha256': hashlib.sha256((root/'config.yaml').read_bytes()).hexdigest()})
    attempt = root/'attempts/evaluation-0002'
    write(attempt/'attempt.json', {'stage': 'evaluation', 'attempt_id': '0002', 'status': 'failed', 'request': {'checkpoint': 'exact', 'source': 'exact'}})
    core = attempt/'panels/worker-attempts/0002'
    write(core/'plan.json', {'digest': 'immutable-plan', 'identities': {'checkpoint': 'exact', 'source': 'exact'}})
    write(core/'attempt.json', {'plan_digest': 'immutable-plan', 'parent': {'path': str(root/'attempts/evaluation-0001/panels/worker-attempts/0001/final.json'), 'sha256': '0'*64}, 'retry_count': 1})
    write(core/'workers/worker-0/worker-terminal.json', {'status': 'complete', 'plan_digest': 'immutable-plan'})
    write(core/'aggregate.json', {'schema': 'panel-worker-aggregate-v1', 'plan_digest': 'immutable-plan', 'rows': [{'status': 'complete', 'task_metric': 7}, {'status': 'failed'}, {'status': 'pending'}], 'counts': {'complete': 1, 'failed': 1, 'pending': 1}, 'incomplete': True})
    snap = core/'snapshots/digest.json'
    write(snap, {'counts': {'complete': 1, 'failed': 1, 'pending': 1}, 'plan_digest': 'immutable-plan'})
    write(core/'final.json', {'path': str(snap), 'sha256': hashlib.sha256(snap.read_bytes()).hexdigest(), 'bytes': snap.stat().st_size})
    return core, snap


def test_partial_worker_plan_retry_and_terminal_evidence_remain_exact(tmp_path):
    root = tmp_path/'run'
    core, terminal = fixture(root)
    snapshot = tmp_path/'snapshot'
    manifest = prepare_snapshot(root, snapshot)
    sources = {a['source'] for a in manifest['artifacts']}
    assert all(str(core/name) in sources for name in ['plan.json', 'aggregate.json', 'final.json', 'workers/worker-0/worker-terminal.json'])
    assert str(terminal) in sources
    rows = [json.loads(line) for line in (snapshot/'exports/results.jsonl').read_text().splitlines()]
    assert [r['row']['status'] for r in rows] == ['complete', 'failed', 'pending']
    assert rows[0]['row']['task_metric'] == 7
    assert rows[0]['result_record']['incomplete'] is True
    receipt = collect_snapshot(snapshot, tmp_path/'collected')
    assert receipt['status'] == 'incomplete'  # linked prior attempt is absent, never scientific success
    assert receipt['counts']['missing']['count'] == 1
    prior = receipt['receipt']
    retry = collect_snapshot(snapshot, tmp_path/'collected')
    assert Path(prior).exists() and retry['pass_id'] != receipt['pass_id']
    assert retry['counts']['reused']['count'] > 0
    with pytest.raises(FileExistsError):
        prepare_snapshot(root, snapshot)


def test_terminal_snapshot_identity_mismatch_rejects(tmp_path):
    root = tmp_path/'run'
    _, terminal = fixture(root)
    terminal.write_text('{}')
    with pytest.raises(ValueError, match='panel evidence identity differs'):
        prepare_snapshot(root, tmp_path/'snapshot')
