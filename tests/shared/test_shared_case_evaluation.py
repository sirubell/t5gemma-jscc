"""Current shared terminal lifecycle retaining original13 scientific panel counts."""

from copy import deepcopy
import json
import pytest
import torch
from jscc import panel_evaluation as evaluation, panel_evaluation as panels, site_routing
from jscc import panel_evaluation as _body_panel_evaluation
from jscc.qualification import state_exact
from jscc.shared_training import run_recipe_training
from jscc.observation_policy import recipe_observation_identity
from shared.test_shared_case_objectives import real_setup


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("current-original13-terminal")
    learner, machine, batches = real_setup(tmp_path/'setup')
    plan = learner.schedule
    metadata = dict(machine.metadata)
    metadata['observation_identity'] = recipe_observation_identity(plan.description, template(256))
    model = learner.model
    model.memory_codec = None
    model.universal_codec = False
    run = tmp_path/'training'
    def no_score(*args, **kwargs):
        from jscc.activation_replay import canonical_digest
        return {'status': 'complete', 'request_identity': canonical_digest(kwargs['request'])}
    run_recipe_training(model, None, plan=plan, batches=batches, metadata=metadata, output=run,
        guard=lambda: None, synthetic=True, observe=no_score)
    return model, plan, run

def template(count=10042):
    return {'task': 'hellaswag', 'expected_items': count, 'input_ids': list(range(count)),
            'source_family_ids': [f'family-{i}' for i in range(count)],
            'data_settings': {'name': 'fixture', 'revision': 'fixed'},
            'scorer': 'lm-eval-hellaswag-fp32-v1-acc-and-acc_norm', 'precision': 'fp32', 'backend': 'sdpa',
            'reference_corpus': {'split': 'validation'}, 'prompt_policy': 'five-shot', 'layout': 'native',
            'noise': {'seed': 0, 'namespace': 'fixed'}, 'settings': {'num_samples': count, 'num_fewshot': 5,
            'batch_size': 16, 'scoring_policy': 'fp32-v1', 'numerical_policy': 'native', 'vanilla': False}}

def score_stub(model, processor, *, state, request, site_id, condition, output):
    torch.rand(3)  # Restoration must protect subsequent training RNG.
    model.eval()
    (output / 'scores.json').write_text(json.dumps({'site': site_id, 'condition': condition}))
    return {'completed_items': request['expected_items'], 'denominator': request['expected_items'],
            'metrics': {'acc': .25, 'acc_norm': .25},
            'payload': {'valid': {'hidden': 16, 'memory': 16 if site_id.startswith('dec') else 0}}}

def test_initial_then_optional_share_one_terminal(frozen, tmp_path, monkeypatch):
    model, _, run = frozen
    monkeypatch.setattr(panels, '_score_panel', score_stub)
    from jscc.experiment_state import capture_rng
    before = deepcopy(capture_rng())
    initial = evaluation.terminal_recipe_request(run, template())
    first = evaluation.run_recipe_evaluation(model, None, request=initial, output=tmp_path/'initial', guard=lambda: None)
    assert first['requested_panels'] == first['completed_panels'] == 26
    assert state_exact(before, capture_rng())
    report = evaluation.recipe_report(run, initial=tmp_path/'initial')
    assert report['terminal']['supplemental']['status'] == 'not_requested'
    pending = evaluation.recipe_report(run, initial_request=initial)
    assert pending['terminal']['initial']['status'] == 'requested'
    assert pending['terminal']['initial']['requested'] == 26
    extra = evaluation.terminal_recipe_request(run, template(), scope='supplemental')
    second = evaluation.run_recipe_evaluation(model, None, request=extra, output=tmp_path/'extra', guard=lambda: None)
    assert second['requested_panels'] == second['completed_panels'] == 39
    assert initial['checkpoint_sha256'] == extra['checkpoint_sha256']
    assert first['checkpoint'] == second['checkpoint']
    report = evaluation.recipe_report(run, initial=tmp_path/'initial', supplemental=tmp_path/'extra')
    assert [report['terminal'][s]['completed'] for s in ('initial', 'supplemental')] == [26, 39]
    with pytest.raises(FileExistsError):
        evaluation.run_recipe_evaluation(model, None, request=initial, output=tmp_path/'initial', guard=lambda: None)
    path = next((tmp_path/'extra').glob('*/scores.json'))
    path.write_text('changed')
    with pytest.raises(ValueError, match='artifact bytes'):
        evaluation.recipe_report(run, supplemental=tmp_path/'extra')

@pytest.mark.parametrize('field,value', [('checkpoint_sha256','f'*64), ('step', 1), ('sites',['enc_fn']),
                                       ('kind','shared_encoder_v2'), ('stream_identity','b'*64),
                                       ('plan_identity','b'*64), ('source','wrong')])
def test_wrong_request_identity_rejected_before_output(frozen, tmp_path, field, value):
    model, _, run = frozen
    request = evaluation.terminal_recipe_request(run, template())
    request[field] = value
    with pytest.raises(ValueError):
        evaluation.run_recipe_evaluation(model, None, request=request, output=tmp_path/'bad', guard=lambda: None)
    assert not (tmp_path/'bad').exists()

def test_failed_requested_panel_independent_of_supplement(frozen, tmp_path, monkeypatch):
    model, _, run = frozen
    calls = []
    def fail(model, processor, **kwargs):
        calls.append(kwargs['site_id'])
        if len(calls) == 3:
            raise _body_panel_evaluation.ObservationScoringError('isolated scorer failure')
        return score_stub(model, processor, **kwargs)
    monkeypatch.setattr(panels, '_score_panel', fail)
    request = evaluation.terminal_recipe_request(run, template())
    with pytest.raises(_body_panel_evaluation.ObservationScoringError):
        evaluation.run_recipe_evaluation(model, None, request=request, output=tmp_path/'failed', guard=lambda: None)
    report = evaluation.recipe_report(run, initial=tmp_path/'failed')
    assert report['training'] == 'training_frozen'
    assert report['terminal']['initial'] == {'status': 'failed', 'requested': 26, 'completed': 2,
        'failed': 1, 'pending': 23, 'requested_items': 26*10042, 'observed_items': 2*10042, 'elapsed_seconds': report['terminal']['initial']['elapsed_seconds']}
    assert report['terminal']['supplemental']['status'] == 'not_requested'

def test_saved_request_check_and_report(frozen, tmp_path):
    from jscc.panel_evaluation import open_recipe_terminal, _validate_recipe_request, recipe_report, terminal_recipe_request
    from jscc.experiment_records import write_manifest
    _, _, run = frozen
    request = terminal_recipe_request(run, template())
    state, _ = open_recipe_terminal(run)
    _validate_recipe_request(state, request, observation=False)
    report = recipe_report(run)
    target = tmp_path/'report.json'
    with target.open('x'):
        pass
    write_manifest(target, report)
    assert json.loads(target.read_text()) == report
    with pytest.raises(FileExistsError):
        with target.open('x'):
            pass

def test_real_original13_decoder_and_harness(monkeypatch, tmp_path):
    """Actual canonical loader/routes/scorer. Joined source is required, no doubles."""
    from datasets import Dataset, DatasetDict
    from jscc.models import split_model
    if not hasattr(split_model, 'resolve_original13_site'):
        pytest.skip('real original13 routing consumer is not integrated in M1')
    from contracts.test_harness_payload import fixture_model
    torch.set_num_threads(1)
    tokenizer, _ = fixture_model()
    learner, machine, _ = real_setup(tmp_path/'real-setup')
    model = learner.model
    weights = deepcopy(dict(model.codec.state_dict()))
    site_routing.load_codec(model, weights)
    assert state_exact(site_routing.export_codec(model), weights)
    state = machine.save(tmp_path/'real-initial.pt')
    def rows(count):
        return [{'ctx_a': 'x', 'ctx_b': 'x', 'activity_label': 'x',
                 'endings': ['x', 'x x', 'x x x', 'x x x x'], 'label': str(i%4),
                 'source_id': f'family-{i}', 'ind': i} for i in range(count)]
    raw = DatasetDict({'train': Dataset.from_list(rows(8)), 'validation': Dataset.from_list(rows(16))})
    monkeypatch.setattr('datasets.load_dataset', lambda *a, **kw: raw)
    from lm_eval.tasks import TaskManager
    manager = TaskManager()
    monkeypatch.setattr('lm_eval.tasks.TaskManager', lambda: manager)
    request = template(16)
    request['backend'] = model.base.config.decoder._attn_implementation
    from jscc.experiment_state import capture_rng
    before = deepcopy(capture_rng())
    runtime = site_routing.observation_state(model)
    with panels._panel_runtime(model, state):
        for site in ('enc_fn', 'dec_l0', 'dec_l24'):
            for condition in ('no_noise', -6):
                path = tmp_path/f'{site}_{condition}'
                path.mkdir()
                result = panels._score_panel(model, tokenizer, state=state, request=request,
                    site_id=site, condition=condition, output=path)
                assert result['completed_items'] == result['denominator'] == 16
                assert result['metrics']['candidate_forward_requests'] == 64
                assert result['payload']['valid']['hidden'] > 0
                assert (result['payload']['valid']['memory'] > 0) == site.startswith('dec')
                assert json.loads((path/f'identity_{condition}.json').read_text())['sum_dtype'] == 'float32'
    assert state_exact(before, capture_rng())
    assert state_exact(runtime, site_routing.observation_state(model))
    assert model.memory_codec is model.codec and not torch.cuda.is_initialized()
