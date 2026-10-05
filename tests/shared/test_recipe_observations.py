"""Real tiny CPU scoring through changed YAML; no scientific acceptance."""
from copy import deepcopy
import json

import pytest
import torch
import yaml

from jscc import panel_evaluation as evaluation
from jscc.activation_replay import canonical_digest
from jscc.shared_config import resolve_recipe
from jscc.experiment_state import CheckpointRef, capture_rng
from jscc.shared_training import create_recipe_learner
from jscc.site_routing import export_codec
from jscc.shared_state import RecipeTrainingState, recipe_runtime_state
from jscc.shared_training import run_recipe_training
from jscc.shared_plan import RecipePlan
from jscc.observation_policy import recipe_observation_identity
from jscc.qualification import state_exact
from contracts.test_harness_payload import fixture_model
from shared.test_shared_case_evaluation import template
from shared.test_recipe_resume import setup


@pytest.fixture
def native_task(monkeypatch):
    from datasets import Dataset, DatasetDict
    def rows(count):
        return [{'ctx_a': 'x', 'ctx_b': 'x', 'activity_label': 'x',
                 'endings': ['x', 'x x', 'x x x', 'x x x x'], 'label': str(i % 4),
                 'source_id': f'family-{i}', 'ind': i} for i in range(count)]
    raw = DatasetDict({'train': Dataset.from_list(rows(8)), 'validation': Dataset.from_list(rows(8))})
    monkeypatch.setattr('datasets.load_dataset', lambda *a, **kw: raw)
    from lm_eval.tasks import TaskManager
    manager = TaskManager()
    monkeypatch.setattr('lm_eval.tasks.TaskManager', lambda: manager)
    return fixture_model()[0]


def changed_setup(path, channel=None):
    initial, batch, _ = setup(path, channel)
    recipe_path = path/'recipe.yaml'
    recipe = yaml.safe_load(recipe_path.read_text())
    recipe['save'].update(reconstruction_fractions=[.5, 1], functional_fractions=[.5, 1], functional_observation_fraction=.25)
    recipe['observation'] = {'items': 2, 'sites': ['enc_l2', 'dec_l3'], 'conditions': ['no_noise']}
    recipe['evaluation'] = {'items': 3, 'conditions': ['no_noise', -6],
                            'supplemental_conditions': [6], 'candidate_batch': 2,
                            'fewshot': 0, 'scoring_policy': 'fp32-v1'}
    recipe_path.write_text(yaml.safe_dump(recipe))
    description = resolve_recipe(recipe_path, model_config_path=path/'config.json')['trajectories'][0]
    plan = RecipePlan(description, initial.plan.views, initial.plan.study_pairing_id)
    model = initial.learner.model
    model.memory_codec = None
    model.universal_codec = False
    metadata = {k: v for k, v in initial.metadata.items()
                if k not in {'stream_identity', 'protocol_identity', 'noise_identity'}}
    metadata.update(config_identity=canonical_digest(description['config']),
                    observation_identity=recipe_observation_identity(description, scoring_template(model, 2)))
    learner = create_recipe_learner(model, plan=plan, metadata=metadata, synthetic=True)
    return RecipeTrainingState(learner, plan=plan, metadata=metadata), batch


def scoring_template(model, count):
    value = template(count)
    value['backend'] = model.base.config.decoder._attn_implementation
    value['settings'].update(num_fewshot=0, batch_size=2)
    value['prompt_policy'] = 'zero-shot'
    return value


def train(state, batch, tokenizer, path, observer, resume=None):
    state.learner.model.memory_codec = None
    state.learner.model.universal_codec = False
    return run_recipe_training(state.learner.model, tokenizer, plan=state.plan,
        batches=[batch]*state.plan.q, metadata=state.metadata, output=path,
        guard=lambda: None, data_ids={'rows': 'tiny'}, synthetic=True,
        observe=observer, resume=resume)


def scalar_updates(path):
    fields = ('completed_step', 'site_id', 'site_step', 'lr_used', 'lr_next',
              'recipe_progress', 'objective', 'snr', 'audit')
    rows = [json.loads(row) for row in (path/'metrics.jsonl').read_text().splitlines()]
    return [{k: row['payload'][k] for k in fields} for row in rows]


@pytest.mark.parametrize('channel', [None, {'type': 'identity', 'normalize_power': True}])
def test_real_observations_and_typed_failure_preserve_exact_training(tmp_path, native_task, monkeypatch, channel):
    baseline, batch = changed_setup(tmp_path/'baseline-setup', channel)
    def no_score(*args, **kwargs):
        return {'status': 'complete', 'request_identity': canonical_digest(kwargs['request'])}
    reference = train(baseline, batch, native_task, tmp_path/'baseline', no_score)
    expected = deepcopy(export_codec(baseline.learner.model))
    real, batch = changed_setup(tmp_path/'real-setup', channel)
    result = train(real, batch, native_task, tmp_path/'real',
                   evaluation.make_recipe_observer(scoring_template(real.learner.model, 2)))
    assert result['observations_status'] == 'complete'
    assert state_exact(export_codec(real.learner.model), expected)
    assert state_exact(scalar_updates(tmp_path/'real'), scalar_updates(tmp_path/'baseline'))
    assert [(r['step'], r['state']) for r in result['observations']] == [
        (8, 'pre_reset'), (10, 'functional_observation')]
    assert sum(c['role'] == 'pre_reset' for c in result['checkpoints']) == 1
    assert sum(c['role'] == 'post_reset' for c in result['checkpoints']) == 1
    roots = sorted((tmp_path/'real').glob('observation_*'))
    report = evaluation.recipe_report(tmp_path/'real', observations=roots)
    assert [r['observed_items'] for r in report['observation_receipts']] == [4, 4]
    assert len(report['training_scalars']['rows']) == reference['completed_updates']
    for root in roots:
        receipt = json.loads((root/'result.json').read_text())
        assert receipt['request']['settings']['num_fewshot'] == 0
        assert receipt['request']['settings']['batch_size'] == 2
        assert receipt['request']['sites'] == ['enc_l2', 'dec_l3']
        assert all(p['completed_items'] == 2 for p in receipt['panels'])
        assert all(p['metrics']['candidate_forward_requests'] == 8 for p in receipt['panels'])
        for panel in receipt['panels']:
            assert panel['payload']['valid']['hidden'] > 0
            assert (panel['payload']['valid']['memory'] > 0) == panel['site'].startswith('dec')
    failed, batch = changed_setup(tmp_path/'failure-setup', channel)
    from jscc.evaluation import TaskScoringError
    def fail(*args, **kwargs):
        torch.rand(19)
        raise TaskScoringError('injected task scoring failure')
    monkeypatch.setattr('jscc.evaluation.evaluate_hellaswag', fail)
    failure = train(failed, batch, native_task, tmp_path/'failed',
                    evaluation.make_recipe_observer(scoring_template(failed.learner.model, 2)))
    assert failure['observations_status'] == 'pending'
    assert all(r['status'] == 'failed' for r in failure['observations'])
    failed_report = evaluation.recipe_report(tmp_path/'failed',
        observations=sorted((tmp_path/'failed').glob('observation_*')))
    assert all(r['status'] == 'failed' and r['failed'] == 1 and r['pending'] == 1
               for r in failed_report['observation_receipts'])
    assert state_exact(export_codec(failed.learner.model), expected)
    assert state_exact(scalar_updates(tmp_path/'failed'), scalar_updates(tmp_path/'baseline'))


def test_same_terminal_requests_receipts_and_tamper(tmp_path, native_task):
    state, batch = changed_setup(tmp_path/'setup')
    train(state, batch, native_task, tmp_path/'train',
          evaluation.make_recipe_observer(scoring_template(state.learner.model, 2)))
    model = state.learner.model
    value = scoring_template(model, 3)
    initial = evaluation.terminal_recipe_request(tmp_path/'train', value)
    extra = evaluation.terminal_recipe_request(tmp_path/'train', value, scope='supplemental')
    assert initial['conditions'] == ['no_noise', -6] and extra['conditions'] == [6]
    assert initial['sites'] == ['dec_l3', 'enc_l2']
    assert initial['checkpoint'] == extra['checkpoint']
    requested = evaluation.recipe_report(tmp_path/'train', initial_request=initial)
    assert requested['terminal']['initial']['status'] == 'requested'
    assert requested['terminal']['initial']['requested'] == 4
    assert requested['terminal']['supplemental']['status'] == 'not_requested'
    rng, runtime, weights = deepcopy(capture_rng()), recipe_runtime_state(model), deepcopy(export_codec(model))
    for request, output in ((initial, tmp_path/'initial'), (extra, tmp_path/'extra')):
        result = evaluation.run_recipe_evaluation(model, native_task, request=request, output=output, guard=lambda: None)
        assert result['status'] == 'complete'
        assert all(p['completed_items'] == 3 for p in result['panels'])
        assert all(p['metrics']['candidate_forward_requests'] == 12 for p in result['panels'])
    assert state_exact(capture_rng(), rng)
    assert state_exact(recipe_runtime_state(model), runtime)
    assert state_exact(export_codec(model), weights)
    report = evaluation.recipe_report(tmp_path/'train', initial=tmp_path/'initial', supplemental=tmp_path/'extra')
    assert [report['terminal'][s]['observed_items'] for s in ('initial', 'supplemental')] == [12, 6]
    rows = report['training_scalars']['rows']
    assert rows[0]['objective']['resolved_weights']['hidden'] == .3
    for index in (0, 1, 7, 8, 9, 15):
        decoder = index % 2 == 0
        assert rows[index]['objective']['resolved_weights'] == {
            'hidden': .3 if decoder else .2, 'memory': .4 if decoder else 0.,
            'kl': .7 if index >= 8 else 0.}
    score = next((tmp_path/'extra').glob('*/identity_*.json'))
    score.write_text('tampered')
    with pytest.raises(ValueError, match='artifact bytes'):
        evaluation.recipe_report(tmp_path/'train', supplemental=tmp_path/'extra')


def test_resumed_report_carries_prior_obligations_without_rewriting(tmp_path, native_task):
    state, batch = changed_setup(tmp_path/'source-setup')
    original = train(state, batch, native_task, tmp_path/'source',
                     evaluation.make_recipe_observer(scoring_template(state.learner.model, 2)))
    pre = next(c for c in original['checkpoints'] if c['role'] == 'pre_reset')
    resumed, batch = changed_setup(tmp_path/'resume-setup')
    result = train(resumed, batch, native_task, tmp_path/'continued',
        evaluation.make_recipe_observer(scoring_template(resumed.learner.model, 2)),
        resume=CheckpointRef(**{k: pre[k] for k in ('path', 'sha256', 'size')}))
    assert result['observations'][0]['status'] == 'pending_prior_segment'
    roots = sorted((tmp_path/'source').glob('observation_*'))[:1]
    report = evaluation.recipe_report(tmp_path/'continued', observations=roots)
    assert report['observations_status'] == 'pending'
    assert report['observations'][0]['status'] == 'pending_prior_segment'
    assert report['observation_receipts'][0]['status'] == 'complete'
    assert report['training_scalars']['segment_ancestry'] == [{k: pre[k] for k in ('path', 'sha256', 'size')}]
    assert report['training_scalars']['segment_updates'] == 8


@pytest.mark.parametrize('changed', ['noise', 'family'])
def test_changed_fixed_panel_rejected_before_observation_output(tmp_path, native_task, changed):
    from shared.test_recipe_resume import advance
    state, batch = changed_setup(tmp_path/'setup')
    advance(state, batch, 8)
    saved = state.save(tmp_path/'pre.pt')
    policy = state.plan.description['observations'][0]
    request = {**policy, 'plan_identity': state.plan.identity, 'checkpoint_sha256': saved.reference.sha256}
    value = scoring_template(state.learner.model, 2)
    if changed == 'noise':
        value['noise']['seed'] += 1
    else:
        value['source_family_ids'][0] = 'other'
    before = deepcopy(capture_rng())
    with pytest.raises(ValueError, match='template identity'):
        evaluation.make_recipe_observer(value)(state.learner.model, native_task, state=saved,
            request=request, output=tmp_path/'bad', guard=lambda: None)
    assert not (tmp_path/'bad').exists()
    assert state_exact(capture_rng(), before)


def test_actual_nonfinite_scoring_is_fatal_and_restored(tmp_path, native_task, monkeypatch):
    from jscc import evaluation as task_evaluation
    from shared.test_recipe_resume import advance
    state, batch = changed_setup(tmp_path/'setup')
    advance(state, batch, 8)
    saved = state.save(tmp_path/'pre.pt')
    policy = state.plan.description['observations'][0]
    request = {**policy, 'plan_identity': state.plan.identity, 'checkpoint_sha256': saved.reference.sha256}
    original = task_evaluation.evaluate_hellaswag
    def nonfinite(*args, **kwargs):
        result = original(*args, **kwargs)
        result['finite_scores'] = False
        return result
    monkeypatch.setattr(task_evaluation, 'evaluate_hellaswag', nonfinite)
    model = state.learner.model
    before, runtime = deepcopy(capture_rng()), recipe_runtime_state(model)
    with pytest.raises(ValueError, match='finite score'):
        evaluation.make_recipe_observer(scoring_template(model, 2))(model, native_task,
            state=saved, request=request, output=tmp_path/'bad', guard=lambda: None)
    assert state_exact(capture_rng(), before)
    assert state_exact(recipe_runtime_state(model), runtime)
    assert json.loads((tmp_path/'bad/result.json').read_text())['status'] == 'incomplete'


def test_failed_terminal_receipt_rejects_impossible_counts_and_status(tmp_path, native_task, monkeypatch):
    state, batch = changed_setup(tmp_path/'setup')
    train(state, batch, native_task, tmp_path/'train',
          evaluation.make_recipe_observer(scoring_template(state.learner.model, 2)))
    request = evaluation.terminal_recipe_request(tmp_path/'train', scoring_template(state.learner.model, 3))
    def fail(*args, **kwargs):
        raise evaluation.ObservationScoringError('injected terminal scoring failure')
    monkeypatch.setattr(evaluation, '_score_panel', fail)
    with pytest.raises(evaluation.ObservationScoringError):
        evaluation.run_recipe_evaluation(state.learner.model, native_task, request=request,
            output=tmp_path/'failed', guard=lambda: None)
    report = evaluation.recipe_report(tmp_path/'train', initial=tmp_path/'failed')
    counts = report['terminal']['initial']
    assert counts['status'] == 'failed'
    assert counts['failed'] == 1 and counts['pending'] == 3 and counts['observed_items'] == 0
    path = tmp_path/'failed/result.json'
    original = json.loads(path.read_text())
    mutations = [(0, 999), (0, -1), (0, True), (1, 1)]
    for index, completed in mutations:
        changed = deepcopy(original)
        changed['panels'][index]['completed_items'] = completed
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            evaluation.recipe_report(tmp_path/'train', initial=tmp_path/'failed')
    changed = deepcopy(original)
    changed['status'] = 'invented'
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        evaluation.recipe_report(tmp_path/'train', initial=tmp_path/'failed')
    path.write_text(json.dumps(original))
