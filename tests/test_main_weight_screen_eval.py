"""Screen-specific cap and legal two-stream conditions."""
import json
import time

import pytest

from scripts.experiments.main_weight_screen_eval import GlobalBudget, score
from test_evening_eval import Budget, adapter, batch
from test_harness_payload import fixture_model


def test_screen_budget_does_not_accept_old_suite_authorization(tmp_path):
    auth = {'execution_authorized': True, 'plan_id': 'MAIN_WEIGHT_EARLY_SCREEN_V1',
            'gpu_work_stop_at_unix': time.time()+60, 'stop_new_gpu_at_unix': time.time()+60}
    (tmp_path/'AUTHORIZATION.json').write_text(json.dumps(auth))
    budget = GlobalBudget(tmp_path)
    budget.reserve(64, 'W005', 'control')
    budget.path.write_text(json.dumps({'count': 16384})+'\n')
    with pytest.raises(ValueError, match='budget exhausted'):
        budget.reserve(64, 'W5', 'core')
    auth['plan_id'] = 'EVENING_SMALL_SUITE_V1'
    (tmp_path/'AUTHORIZATION.json').write_text(json.dumps(auth))
    with pytest.raises(ValueError, match='authorization'):
        budget.authorized()


def test_screen_noise_both_streams_and_replay_keep_payload():
    tokenizer, model = fixture_model()
    harness, budget = adapter(model, tokenizer), Budget()
    policies = {'hidden': 'A', 'memory': 'A'}
    one, evidence, _ = score(harness, model, batch(), budget, 'awgn', 'test', policies, 20260921)
    two, again, _ = score(harness, model, batch(), budget, 'awgn', 'test', policies, 20260921)
    assert one == two
    assert evidence['valid'] == again['valid']
    assert all(e['noise_enabled'] and e['codec_calls'] == 1 and e['oracle_calls'] == 0
               for e in evidence['events'])


def test_identical_scores_cannot_hide_payload_mismatch():
    from scripts.experiments.main_weight_screen_eval import payload_check
    original = {"allocated": {"hidden": 64, "memory": 128}, "valid": {"hidden": 32, "memory": 64}}
    changed = {**original, "valid": {"hidden": 32, "memory": 63}}
    assert payload_check(original, original)["passed"]
    assert not payload_check(original, changed)["passed"]
