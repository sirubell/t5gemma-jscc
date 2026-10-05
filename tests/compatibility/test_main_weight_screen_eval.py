"""Screen-specific cap and legal two-stream conditions."""
import json
import time

import pytest

from scripts.evaluate_historical_weight_screen import GlobalBudget, score
from compatibility.test_evening_eval import Budget, adapter, batch
from contracts.test_harness_payload import fixture_model


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
    from scripts.evaluate_historical_weight_screen import payload_check
    original = {"allocated": {"hidden": 64, "memory": 128}, "valid": {"hidden": 32, "memory": 64}}
    changed = {**original, "valid": {"hidden": 32, "memory": 63}}
    assert payload_check(original, original)["passed"]
    assert not payload_check(original, changed)["passed"]


def evaluation_argv(root, route="dec_l20", step=500):
    return ["--root", str(root), "--route", route, "--step", str(step),
            "--checkpoint", str(root / "weight.pt"), "--checkpoint-sha256", "saved-hash",
            "--arm", "W005"]


@pytest.mark.parametrize("step", [500, 1000])
def test_parser_dispatches_supported_route_and_steps_only(step, tmp_path, monkeypatch):
    import scripts.evaluate_historical_weight_screen as runner
    calls = []
    monkeypatch.setattr(runner, "run", calls.append)
    runner.main(evaluation_argv(tmp_path, step=step))
    assert len(calls) == 1
    args = calls[0]
    assert (args.root, args.route, args.step, args.arm) == (tmp_path, "dec_l20", step, "W005")
    assert args.checkpoint == tmp_path / "weight.pt"
    assert args.checkpoint_sha256 == "saved-hash" and not args.vanilla
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("route,step", [("dec_l0", 500), ("enc_l9", 1000),
                                       ("dec_l20", 5000), ("dec_l20", 10000)])
def test_parser_rejects_unsupported_route_or_step_before_runtime(route, step, tmp_path, monkeypatch):
    import scripts.evaluate_historical_weight_screen as runner
    monkeypatch.setattr(runner, "run", lambda args: pytest.fail("must reject before runtime"))
    with pytest.raises(SystemExit) as failure:
        runner.main(evaluation_argv(tmp_path, route, step))
    assert failure.value.code == 2
    assert not list(tmp_path.iterdir())
