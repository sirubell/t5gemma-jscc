"""CPU-only guardrails for the bounded production-loop instrumentation."""
from typing import Any

import pytest

from scripts.native_policy_train import Budget, matrix, order_for, candidate_config


def test_balanced_matrix_and_caps():
    plan = matrix()
    assert [row[1] for row in plan[:4]] == [64, 128, 128, 64]
    assert all(row[2] == 1024 + 4096 for row in plan[:4])
    assert sum(row[2] for row in plan) <= 32768
    assert sum(row[2] // row[1] for row in plan) <= 512


def test_attempts_charge_before_failure_and_caps():
    budget = Budget()
    budget.charge(examples=32768, updates=512)
    with pytest.raises(RuntimeError, match="cap"):
        budget.charge(examples=1)
    with pytest.raises(RuntimeError, match="cap"):
        budget.charge(updates=1)
    assert (budget.presentations, budget.updates) == (32768, 512)
    budget.deadline = 0
    with pytest.raises(TimeoutError):
        budget.charge()


def test_paired_actual_order_independent_of_batch_and_repeat():
    data = [{"input_ids": [1] * i, "label_ids": [2]} for i in range(600)]
    first = order_for(data, "A", 512)
    assert first == order_for(data, "A", 512)
    assert first != order_for(data, "B", 512)
    assert len(set(first)) == 512
    assert order_for(data, "stress", 128) == list(range(599, 471, -1))
    with pytest.raises(ValueError, match="insufficient"):
        order_for(data, "A", 601)


def test_candidate_keeps_schedule_and_logging_without_mutation(tmp_path):
    base = {"run": {}, "training": {"log_every": 50}}
    config = candidate_config(base, tmp_path, "A-b64", 64, 5120)
    assert config["training"]["max_steps"] == 80
    assert config["training"]["schedule_steps"] == 20000
    assert config["training"]["lr"] == 2e-4
    assert config["training"]["log_every"] == 50
    assert base == {"run": {}, "training": {"log_every": 50}}


def test_policy_rejects_film_and_unapproved_routes():
    from jscc.config import load_config
    from scripts.native_policy_train import validate_policy
    config = load_config('configs/tasks/hellaswag.yaml')
    config['protocol'] = 'corrected-baseline-v2-native-decoder-inputs'
    config['split'] = {'stack': 'dec', 'where': 'after_layer', 'index': 8}
    config['codec']['layernorm'] = 'none'
    config['codec']['memory'] = {'layernorm': 'both'}
    validate_policy(config)
    config['codec']['memory']['snr_film'] = True
    with pytest.raises(ValueError, match='FiLM'):
        validate_policy(config)
    config['codec']['memory']['snr_film'] = False
    config['split']['index'] = 9
    with pytest.raises(ValueError, match='only encoder'):
        validate_policy(config)


def test_profiler_setup_failure_is_scoped(monkeypatch):
    from scripts import native_policy_train as runner
    def fail(**kwargs):
        raise RuntimeError('trace tools unavailable')
    monkeypatch.setattr(runner.torch.profiler, 'profile', fail)
    report = {}
    assert runner.start_profile(report) is None
    assert report['profiler']['status'] == 'unavailable'
    assert 'status' not in report


def test_profiler_export_failure_retains_partial_status(tmp_path):
    from scripts.native_policy_train import finish_profile
    class Profiler:
        def stop(self):
            pass
        def export_chrome_trace(self, path):
            raise OSError('trace export unavailable')
    report: dict[str, Any] = {'status': 'PASS'}
    finish_profile(Profiler(), tmp_path, report)
    assert report['status'] == 'PASS'
    assert report['profiler']['status'] == 'partial_or_unavailable'
    assert not report['profiler']['partial_trace_exists']


def test_expired_candidate_writes_partial_report_without_model_setup(tmp_path, monkeypatch):
    import json
    from scripts import native_policy_train as runner
    budget = Budget()
    budget.deadline = 0
    monkeypatch.setattr(runner.torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(runner.torch.cuda, 'empty_cache', lambda: None)
    def forbidden(*args, **kwargs):
        pytest.fail('expired candidate must not load model or data')
    monkeypatch.setattr(runner.training, 'train', forbidden)
    output = tmp_path / 'candidate'
    report = runner.run_candidate({}, output, ('A-b64', 64, 5120, 'A', False), budget, False)
    assert report['status'] == 'TIME_LIMIT'
    assert 'BudgetDeadline' in report['error']
    assert report['attempted_presentations'] == 0
    assert json.loads((output / 'report.json').read_text())['status'] == 'TIME_LIMIT'
