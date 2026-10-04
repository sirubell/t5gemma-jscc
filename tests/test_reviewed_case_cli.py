"""One action CLI and durable rejected bundle boundary, without a scheduler."""
import json
from pathlib import Path

import pytest

from scripts import submit_recipe as cli


def test_one_action_prepares_reviews_and_submits_same_bundle(tmp_path, monkeypatch, capsys):
    bundle = {'schema': 'recipe-case-submission-v1', 'output': str(tmp_path/'submission'),
              'identity': 'exact-bound-graph'}
    payload = '# exact reviewed payload\n'
    calls = []
    monkeypatch.setattr(cli, 'prepare_job', lambda *args: bundle)
    def preview(value, path):
        assert value is bundle
        assert json.loads(Path(path).read_text()) == bundle
        return {'payload': payload, 'identity': value['identity']}
    monkeypatch.setattr(cli, 'preview_job', preview)
    def submit(value, output, *, scheduler):
        assert value is bundle and output == bundle['output']
        calls.append(scheduler(['fake-sbatch']))
        return {'status': 'submitted', 'payload': payload}
    monkeypatch.setattr(cli, 'submit_job', submit)
    responses = tmp_path/'responses.json'
    responses.write_text(json.dumps([{'returncode': 0, 'stdout': '42\n', 'stderr': ''}]))
    result = cli.main(['action', '--job', 'job.yaml', '--limits', 'limits.yaml',
                       '--output', str(tmp_path/'bundle.json'), '--mock-results', str(responses)])
    assert result['payload'] == payload
    assert len(calls) == 1
    assert (calls[0].args, calls[0].returncode, calls[0].stdout, calls[0].stderr) == (['fake-sbatch'], 0, '42\n', '')
    assert capsys.readouterr().out.count(payload.replace('\n', '\\n')) == 2


def test_one_action_default_preview_has_no_submission(tmp_path, monkeypatch):
    bundle = {'output': str(tmp_path/'submission')}
    monkeypatch.setattr(cli, 'prepare_job', lambda *args: bundle)
    monkeypatch.setattr(cli, 'preview_job', lambda *args: {'scheduler_invoked': False})
    monkeypatch.setattr(cli, 'submit_job', lambda *args, **kwargs: pytest.fail('preview submitted'))
    assert cli.main(['action', '--job', 'job.yaml', '--limits', 'limits.yaml',
                     '--output', str(tmp_path/'bundle.json')]) == {'scheduler_invoked': False}


def test_case_run_passes_exact_group(tmp_path, monkeypatch):
    bundle = {'schema': 'recipe-case-submission-v1', 'identity': 'fixed'}
    path = tmp_path/'bundle.json'
    path.write_text(json.dumps(bundle))
    monkeypatch.setattr(cli, 'validate_bundle', lambda value: value)
    calls = []
    monkeypatch.setattr(cli, 'run_job', lambda value, identity, group: calls.append((identity, group)))
    cli.main(['run', '--bundle', str(path), '--identity', 'fixed', '--group', '2'])
    assert calls == [('fixed', 2)]


def test_wrong_bundle_records_group_specific_failures_before_runner(tmp_path, monkeypatch):
    path = tmp_path/'bundle.json'
    path.write_text('{invalid')
    monkeypatch.setenv('SLURM_ARRAY_TASK_ID', '1')
    monkeypatch.setattr(cli, 'run_job', lambda *args: pytest.fail('invalid bundle reached runner'))
    for group in [0, 1]:
        with pytest.raises(ValueError):
            cli.main(['run', '--bundle', str(path), '--identity', 'fixed', '--group', str(group)])
        receipt = json.loads((tmp_path/f'group-{group}-task-1.bundle-failure.json').read_text())
        assert receipt['status'] == 'failed' and receipt['stage'] == 'bundle-preflight'
        assert receipt['bundle_path'] == str(path)
    assert len(list(tmp_path.glob('*.bundle-failure.json'))) == 2
