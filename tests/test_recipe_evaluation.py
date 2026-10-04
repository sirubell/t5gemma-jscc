"""Recorded public evaluation: real tiny CPU scorers and explicit boundary faults."""
import json
from pathlib import Path

import pytest
import torch
import yaml

from jscc.activation_replay import file_digest
from test_recipe_assets import registered
from test_recipe_runs import public, read

pytest_plugins = ['test_recipe_workflow']
CASE = 'width-8-ratio-50'


def prepared(assets, *, multiple=False, train=True):
    recipe, registrations, _ = registered(assets)
    if multiple:
        authored = yaml.safe_load(recipe.read_text())
        authored['reconstruction_percentages'] = [50, 75]
        authored['save'].update(reconstruction_fractions=[.5, 1], functional_fractions=[.5, 1])
        recipe.write_text(yaml.safe_dump(authored))
    root = assets['root']/'public-evaluation'
    public(recipe, registrations, root, *([] if train else ['--prepare-only']))
    return recipe, registrations, root


def evaluate(root, **kwargs):
    from jscc.recipe_evaluation import evaluate_recipe_run
    return evaluate_recipe_run(root, timeout_seconds=120, **kwargs)


def panels(root, aggregate):
    """Read the actual scientific engine receipt, separate from adapter summaries."""
    receipts = []
    for path in root.glob(f'cases/*/evaluations/{aggregate.name}/panels/result.json'):
        receipt = read(path)
        assert receipt['schema'] == 'recipe-evaluation-receipt-v1'
        receipts.append(receipt)
    return receipts


def test_real_terminal_supplemental_distinct_records_and_frozen_yaml(joined_assets):
    from evaluate import main
    recipe, _, root = prepared(joined_assets)
    terminal = read(root/'cases'/CASE/'checkpoint.json')
    recipe.write_text('author changed YAML after binding: [')
    output_link = joined_assets['root']/'evaluation-path.txt'
    initial = main(['--run', str(root), '--timeout-seconds', '120',
                    '--output-path-file', str(output_link)])
    assert output_link.read_text().strip() == str(initial)
    initial_bytes = (initial/'results.json').read_bytes()
    supplemental = evaluate(root, scope='supplemental', attempt_id='0002')
    assert initial != supplemental
    assert read(initial/'results.json')['status'] == 'complete'
    assert read(initial/'gate.json')['final_eligible'] is True
    assert read(supplemental/'gate.json')['final_eligible'] is False
    assert (initial/'results.json').read_bytes() == initial_bytes
    for aggregate, scope, condition in [(initial, 'initial', 'no_noise'), (supplemental, 'supplemental', -6)]:
        receipts = panels(root, aggregate)
        assert len(receipts) == 1
        receipt = receipts[0]
        assert receipt['status'] == 'complete'
        assert receipt['request']['scope'] == scope
        assert receipt['request']['training_run'] == terminal['engine_path']
        assert receipt['checkpoint']['sha256'] == terminal['sha256']
        assert receipt['checkpoint']['size'] == terminal['size']
        assert receipt['request']['step'] == terminal['step'] == 24
        assert len(receipt['panels']) == 3
        assert all(row['condition'] == condition and row['completed_items'] == 3 for row in receipt['panels'])
        assert all(row['metrics']['candidate_forward_requests'] == 12 for row in receipt['panels'])
        case_output = root/'cases'/CASE/'evaluations'/aggregate.name
        assert read(case_output/'request.json')['checkpoint_sha256'] == terminal['sha256']
        assert (case_output/'result.json').is_file()
    with pytest.raises(FileExistsError):
        evaluate(root)
    assert (initial/'results.json').read_bytes() == initial_bytes


def test_real_multiple_cases_complete_and_explicit_case_path(joined_assets):
    from jscc.recipe_evaluation import recipe_evaluation_target
    _, _, root = prepared(joined_assets, multiple=True)
    aggregate = evaluate(root)
    receipts = panels(root, aggregate)
    assert len(receipts) == 2
    assert sum(len(receipt['panels']) for receipt in receipts) == 6
    assert read(aggregate/'results.json')['status'] == 'complete'
    assert read(aggregate/'gate.json')['final_eligible'] is True
    selected = evaluate(root/'cases'/CASE, attempt_id='0002')
    assert len(panels(root, selected)) == 1
    assert read(selected/'results.json')['status'] == 'complete'
    inventory = read(root/'cases'/CASE/'attempts/training-0001/checkpoints.json')
    post_reset = next(row for row in inventory['checkpoints'] if row['role'] == 'post_reset')
    with pytest.raises(ValueError, match='observation checkpoint roles'):
        recipe_evaluation_target(root, CASE, scope='development',
                                 checkpoint=post_reset['path'], expected_step=post_reset['step'])
    with pytest.raises(ValueError, match='from this case'):
        recipe_evaluation_target(root, 'width-8-ratio-75', scope='development',
                                 checkpoint=post_reset['path'], expected_step=post_reset['step'])


def test_whole_root_missing_case_keeps_denominator_and_never_falls_back(joined_assets):
    recipe, registrations, root = prepared(joined_assets, multiple=True, train=False)
    public(recipe, registrations, root, '--case', CASE)
    aggregate = evaluate(root)
    report = read(aggregate/'results.json')
    assert report['status'] != 'complete'
    assert len(report['cases']) == 2
    assert report['counts']['cases'] == dict(requested=2, eligible=1, complete=1,
                                           missing=1, failed=0, pending=0, interrupted=0)
    assert report['counts']['panels'] == dict(requested=6, completed=3, failed=0,
                                            pending=3, requested_items=18, observed_items=9)
    assert read(aggregate/'gate.json')['final_eligible'] is False
    assert len(panels(root, aggregate)) == 1
    assert not list((root/'cases/width-8-ratio-75').glob('evaluations/*/panels/result.json'))


@pytest.mark.parametrize('mutation', ['hash', 'size', 'step', 'engine', 'ambiguous', 'attempt', 'source', 'runtime'])
def test_recorded_terminal_rejections_precede_model_initialization(joined_assets, monkeypatch, mutation):
    """Explicit boundary mutation; initialization is forbidden after invalid identity."""
    _, _, root = prepared(joined_assets)
    case = root/'cases'/CASE
    target = case/'checkpoint.json'
    value = read(target)
    if mutation == 'hash':
        value['sha256'] = '0'*64
    elif mutation == 'size':
        value['size'] += 1
    elif mutation == 'step':
        value['step'] -= 1
    elif mutation == 'engine':
        training = read(case/'training.json')
        training['engine_path'] = str(case/'different-engine')
        (case/'training.json').write_text(json.dumps(training))
    elif mutation == 'ambiguous':
        inventory = Path(value['training_attempt'])/'checkpoints.json'
        rows = read(inventory)
        rows['checkpoints'].append(dict(rows['checkpoints'][-1]))
        inventory.write_text(json.dumps(rows))
    elif mutation in ('attempt', 'source', 'runtime'):
        attempt = Path(value['training_attempt'])/'attempt.json'
        row = read(attempt)
        if mutation == 'attempt':
            row['status'] = 'interrupted'
        elif mutation == 'source':
            row['source']['revision'] = '0'*40
        else:
            row['runtime']['fixture_incompatible_runtime'] = True
        attempt.write_text(json.dumps(row))
    target.write_text(json.dumps(value))
    monkeypatch.setattr('transformers.AutoModelForSeq2SeqLM.from_pretrained',
                        lambda *a, **k: pytest.fail('invalid recorded target reached model construction'))
    aggregate = evaluate(root)
    assert read(aggregate/'results.json')['status'] != 'complete'
    assert read(aggregate/'gate.json')['final_eligible'] is False
    assert panels(root, aggregate) == []


def test_development_explicit_durable_pre_reset_real_scorer(joined_assets, monkeypatch):
    from jscc import original13_training
    recipe, registrations, root = prepared(joined_assets, train=False)
    original = original13_training.run_recipe_training
    def interrupt(*args, **kwargs):
        publish = kwargs['on_checkpoint']
        def save(saved, role):
            publish(saved, role)
            if role == 'pre_reset':
                raise KeyboardInterrupt('fixture stop after durable pre_reset')
        kwargs['on_checkpoint'] = save
        return original(*args, **kwargs)
    monkeypatch.setattr(original13_training, 'run_recipe_training', interrupt)
    with pytest.raises(KeyboardInterrupt):
        public(recipe, registrations, root, '--case', CASE)
    attempt = root/'cases'/CASE/'attempts/training-0001'
    checkpoint = read(attempt/'checkpoints.json')['checkpoints'][-1]
    assert checkpoint['role'] == 'pre_reset'
    before = file_digest(checkpoint['path'])
    rng = torch.get_rng_state().clone()
    aggregate = evaluate(root, selected=CASE, scope='development',
                         checkpoint=checkpoint['path'], expected_step=checkpoint['step'])
    assert read(aggregate/'results.json')['status'] == 'complete'
    assert read(aggregate/'gate.json')['final_eligible'] is False
    receipt = panels(root, aggregate)[0]
    assert receipt['request']['state_role'] == 'pre_reset'
    assert len(receipt['panels']) == 2
    assert all(row['completed_items'] == 2 for row in receipt['panels'])
    assert file_digest(checkpoint['path']) == before
    assert torch.equal(torch.get_rng_state(), rng)
    assert not (root/'cases'/CASE/'checkpoint.json').exists()


def test_panel_failure_retains_completed_pending_and_rerun_is_fresh(joined_assets, monkeypatch):
    from jscc import encoder_stages_evaluation as engine
    _, _, root = prepared(joined_assets)
    score = engine._score_original13_panel
    calls = []
    def fail_second(*args, **kwargs):
        calls.append(kwargs['site_id'])
        if len(calls) == 2:
            raise engine.ObservationScoringError('explicit fixture scorer failure')
        return score(*args, **kwargs)
    monkeypatch.setattr(engine, '_score_original13_panel', fail_second)
    failed = evaluate(root)
    before = {path.relative_to(failed): path.read_bytes() for path in failed.rglob('*') if path.is_file()}
    receipt = panels(root, failed)[0]
    assert [row['status'] for row in receipt['panels']] == ['complete', 'failed', 'pending']
    assert read(failed/'results.json')['status'] != 'complete'
    assert read(failed/'gate.json')['final_eligible'] is False
    monkeypatch.setattr(engine, '_score_original13_panel', score)
    rerun = evaluate(root, attempt_id='0002')
    assert rerun != failed
    assert read(rerun/'results.json')['status'] == 'complete'
    assert all(path.read_bytes() == before[path.relative_to(failed)] for path in failed.rglob('*') if path.is_file())


def test_development_requires_explicit_case_checkpoint_and_step(joined_assets):
    _, _, root = prepared(joined_assets, train=False)
    with pytest.raises(ValueError, match='explicit|selected|case'):
        evaluate(root, scope='development')
    with pytest.raises(ValueError, match='checkpoint|step'):
        evaluate(root, selected=CASE, scope='development')
    assert not list(root.glob('evaluations/*'))
    assert not list(root.glob('cases/*/evaluations/*'))


def test_duplicate_evaluation_override_and_changed_policy_reject_before_outputs(joined_assets):
    _, _, root = prepared(joined_assets, train=False)
    override = joined_assets['root']/'evaluation-override.yaml'
    override.write_text(yaml.safe_dump({'evaluation': {'conditions': [-6, -6.0]}}))
    with pytest.raises(ValueError, match='duplicate'):
        evaluate(root, overrides_path=override)
    override.write_text(yaml.safe_dump({'evaluation': {'conditions': [-6]}}))
    with pytest.raises(ValueError, match='frozen policy'):
        evaluate(root, overrides_path=override)
    assert not list(root.glob('evaluations/*'))
    assert not list(root.glob('cases/*/evaluations/*'))


def test_duplicate_numeric_conditions_reject_before_prepared_outputs(joined_assets):
    recipe, registrations, _ = registered(joined_assets)
    authored = yaml.safe_load(recipe.read_text())
    authored['evaluation']['conditions'] = [-6, -6.0]
    recipe.write_text(yaml.safe_dump(authored))
    root = joined_assets['root']/'duplicate-condition'
    with pytest.raises(ValueError, match='duplicate'):
        public(recipe, registrations, root, '--prepare-only')
    assert not root.exists()
