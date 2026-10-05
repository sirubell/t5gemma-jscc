"""Separate public YAML operations and durable transport rejection without a scheduler."""
import json
from pathlib import Path

import pytest

from scripts import submit_recipe as cli


def test_public_preview_and_explicit_submit_are_separate_operations(tmp_path, monkeypatch):
    from request_helpers import invoke
    from jscc import submission
    bundle = {'schema': 'recipe-case-submission-v1', 'output': str(tmp_path/'submission'),
              'identity': 'exact-bound-graph'}
    payload = '# exact reviewed payload\n'
    calls = []
    monkeypatch.setattr(submission, 'prepare_job', lambda *args: bundle)
    def preview(value, path):
        assert value is bundle
        assert json.loads(Path(path).read_text()) == bundle
        calls.append('preview')
        return {'payload': payload, 'identity': value['identity'], 'scheduler_invoked': False}
    monkeypatch.setattr(submission, 'preview_job', preview)
    authorization = {'identity': bundle['identity'], 'explicit': True}
    def submit(value, output, *, authorization):
        assert value is bundle and output == str(tmp_path/'submission')
        assert authorization['identity'] == bundle['identity']
        calls.append('submit')
        Path(output).mkdir()
        receipt = {'status': 'submitted', 'payload': payload, 'identity': value['identity']}
        (Path(output)/'submission.json').write_text(json.dumps(receipt))
        return receipt
    monkeypatch.setattr(submission, 'submit_job', submit)
    result = invoke(tmp_path, 'shared', 'preview', {'plan': {'kind': 'submission', 'path': 'job.yaml'},
        'submission': {'limits': 'limits.yaml', 'bundle_output': 'reviewed-bundle.json'}})
    assert result['payload'] == payload and result['identity'] == bundle['identity']
    assert calls == ['preview'] and not (tmp_path/'submission').exists()
    submitted = invoke(tmp_path, 'shared', 'submit', {'submission': {'plan': 'job.yaml',
        'limits': 'limits.yaml', 'bundle_output': 'submission/bundle.json',
        'receipt_dir': 'submission', 'authorization': authorization}})
    assert calls == ['preview', 'submit']
    assert submitted == json.loads((tmp_path/'submission/submission.json').read_text())
    assert submitted['payload'] == payload and submitted['identity'] == result['identity']


def test_public_preview_has_no_automatic_submission(tmp_path, monkeypatch):
    from request_helpers import invoke
    from jscc import submission
    bundle = {'output': str(tmp_path/'submission')}
    monkeypatch.setattr(submission, 'prepare_job', lambda *args: bundle)
    monkeypatch.setattr(submission, 'preview_job', lambda *args: {'scheduler_invoked': False})
    monkeypatch.setattr(submission, 'submit_job', lambda *args, **kwargs: pytest.fail('preview submitted'))
    assert invoke(tmp_path, 'shared', 'preview', {'plan': {'kind': 'submission', 'path': 'job.yaml'},
        'submission': {'limits': 'limits.yaml', 'bundle_output': 'reviewed-bundle.json'}}) == {'scheduler_invoked': False}
    assert not (tmp_path/'submission').exists()


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


@pytest.mark.parametrize('case_schema', [True, False])
def test_check_preserves_legacy_count_and_selected_subset(tmp_path, monkeypatch, case_schema):
    bundle = {'schema': 'recipe-case-submission-v1' if case_schema else 'shared-codec-submission-v1',
              'identity': 'fixed', 'cases': [{'commands': {}}, {'commands': {'training': []}}],
              'resolved': {'trajectories': [1, 2, 3]}}
    path = tmp_path/'bundle.json'
    path.write_text(json.dumps(bundle))
    monkeypatch.setattr(cli, 'validate_bundle', lambda value: value)
    monkeypatch.setattr(cli, 'submission_preflight', lambda value: None)
    result = cli.main(['check', '--bundle', str(path)])
    if case_schema:
        assert result['selected_cases'] == 1 and result['declared_cases'] == 2
    else:
        assert result['trajectories'] == 3
