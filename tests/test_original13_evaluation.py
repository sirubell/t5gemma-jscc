"""Frozen request lifecycle; synthetic scores never claim scientific execution."""
from copy import deepcopy
import json

import pytest
import torch

from jscc.activation_replay import file_digest
from jscc import encoder_stages_evaluation as evaluation, original13_routing
from jscc.encoder_stages import run_original13_training
from jscc.sharing_qualification_guard import state_exact
from test_original13_plan_state import MechanicalRouting, mechanics, observe_complete


def template(count=10042):
    return {'task': 'hellaswag', 'expected_items': count, 'input_ids': list(range(count)),
            'source_family_ids': [f'family-{i}' for i in range(count)],
            'data_settings': {'name': 'fixture', 'revision': 'fixed'},
            'scorer': 'lm-eval-hellaswag-fp32-v1-acc-and-acc_norm', 'precision': 'fp32', 'backend': 'sdpa',
            'reference_corpus': {'split': 'validation'}, 'prompt_policy': 'five-shot', 'layout': 'native',
            'noise': {'seed': 0, 'namespace': 'fixed'}, 'settings': {'num_samples': count, 'num_fewshot': 5,
            'batch_size': 16, 'scoring_policy': 'fp32-v1', 'numerical_policy': 'native', 'vanilla': False}}


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    for name in ('canonical_codec', 'export_codec', 'load_codec', 'observation_state', 'restore_runtime_state'):
        monkeypatch.setattr(original13_routing, name, getattr(MechanicalRouting, name))
    model, plan, batches, metadata, _ = mechanics()
    run = tmp_path / 'training'
    run_original13_training(model, None, plan=plan, batches=batches, metadata=metadata,
        observation_request={'panel': __import__('test_original13_plan_state').panel()}, output=run,
        guard=lambda: None, routing=MechanicalRouting, observe=observe_complete)
    return model, plan, run


def score_stub(model, processor, *, state, request, site_id, condition, output):
    torch.rand(3)  # Restoration must protect subsequent training RNG.
    model.eval()
    (output / 'scores.json').write_text(json.dumps({'site': site_id, 'condition': condition}))
    return {'completed_items': request['expected_items'], 'denominator': request['expected_items'],
            'metrics': {'acc': .25, 'acc_norm': .25},
            'payload': {'valid': {'hidden': 16, 'memory': 16 if site_id.startswith('dec') else 0}}}


def test_initial_then_optional_share_one_terminal(frozen, tmp_path, monkeypatch):
    model, _, run = frozen
    monkeypatch.setattr(evaluation, '_score_original13_panel', score_stub)
    from jscc.experiment_state import capture_rng
    before = deepcopy(capture_rng())
    initial = evaluation.terminal_original13_request(run, template())
    first = evaluation.run_original13_evaluation(model, None, request=initial, output=tmp_path/'initial', guard=lambda: None)
    assert first['requested_panels'] == first['completed_panels'] == 26
    assert state_exact(before, capture_rng())
    report = evaluation.original13_report(run, initial=tmp_path/'initial')
    assert report['terminal']['supplemental']['status'] == 'not_requested'
    pending = evaluation.original13_report(run, initial_request=initial)
    assert pending['terminal']['initial']['status'] == 'pending'
    assert pending['terminal']['initial']['requested'] == 26
    extra = evaluation.terminal_original13_request(run, template(), scope='supplemental')
    second = evaluation.run_original13_evaluation(model, None, request=extra, output=tmp_path/'extra', guard=lambda: None)
    assert second['requested_panels'] == second['completed_panels'] == 39
    assert initial['checkpoint_sha256'] == extra['checkpoint_sha256']
    assert first['checkpoint'] == second['checkpoint']
    report = evaluation.original13_report(run, initial=tmp_path/'initial', supplemental=tmp_path/'extra')
    assert [report['terminal'][s]['completed'] for s in ('initial', 'supplemental')] == [26, 39]
    with pytest.raises(FileExistsError):
        evaluation.run_original13_evaluation(model, None, request=initial, output=tmp_path/'initial', guard=lambda: None)
    path = next((tmp_path/'extra').glob('*/scores.json'))
    path.write_text('changed')
    with pytest.raises(ValueError, match='artifact bytes'):
        evaluation.original13_report(run, supplemental=tmp_path/'extra')


@pytest.mark.parametrize('field,value', [('checkpoint_sha256','f'*64), ('step', 1), ('sites',['enc_fn']),
                                       ('kind','shared_encoder_v2'), ('input_identity','b'*64),
                                       ('plan_identity','b'*64), ('source','wrong')])
def test_wrong_request_identity_rejected_before_output(frozen, tmp_path, field, value):
    model, _, run = frozen
    request = evaluation.terminal_original13_request(run, template())
    request[field] = value
    with pytest.raises(ValueError):
        evaluation.run_original13_evaluation(model, None, request=request, output=tmp_path/'bad', guard=lambda: None)
    assert not (tmp_path/'bad').exists()


def test_failed_requested_panel_independent_of_supplement(frozen, tmp_path, monkeypatch):
    model, _, run = frozen
    calls = []
    def fail(model, processor, **kwargs):
        calls.append(kwargs['site_id'])
        if len(calls) == 3:
            raise evaluation.ObservationScoringError('isolated scorer failure')
        return score_stub(model, processor, **kwargs)
    monkeypatch.setattr(evaluation, '_score_original13_panel', fail)
    request = evaluation.terminal_original13_request(run, template())
    with pytest.raises(evaluation.ObservationScoringError):
        evaluation.run_original13_evaluation(model, None, request=request, output=tmp_path/'failed', guard=lambda: None)
    report = evaluation.original13_report(run, initial=tmp_path/'failed')
    assert report['training'] == 'training_frozen'
    assert report['terminal']['initial'] == {'status': 'failed', 'requested': 26, 'completed': 2,
        'failed': 1, 'pending': 23, 'requested_items': 26*10042, 'observed_items': 2*10042, 'elapsed_seconds': report['terminal']['initial']['elapsed_seconds']}
    assert report['terminal']['supplemental']['status'] == 'not_requested'


def test_cli_request_check_and_report(frozen, tmp_path):
    from scripts.original13_evaluation import main
    _, _, run = frozen
    source = tmp_path/'template.json'
    source.write_text(json.dumps(template()))
    request = tmp_path/'request.json'
    main(['request', '--training-run', str(run), '--template', str(source), '--template-sha256', file_digest(source),
          '--output', str(request)])
    main(['check', '--request', str(request), '--request-sha256', file_digest(request)])
    main(['report', '--training-run', str(run), '--output', str(tmp_path/'report.json')])
    with pytest.raises(FileExistsError):
        main(['request', '--training-run', str(run), '--template', str(source), '--template-sha256', file_digest(source),
              '--output', str(request)])


def test_real_original13_decoder_and_harness(monkeypatch, tmp_path):
    """Actual canonical loader/routes/scorer. Joined source is required, no doubles."""
    from datasets import Dataset, DatasetDict
    from jscc.models import split_model
    if not hasattr(split_model, 'resolve_original13_site'):
        pytest.skip('real original13 routing consumer is not integrated in M1')
    from jscc.models.channel import AWGNChannel
    from jscc.models.codec import Codec
    from model_helpers import tiny_backbone
    from test_harness_payload import fixture_model
    from jscc.encoder_stages import Original13TrainingState
    from jscc.sharing_preparation import state_dict_identity
    from dataclasses import replace
    torch.set_num_threads(1)
    tokenizer, _ = fixture_model()
    model = split_model.SplitModel(tiny_backbone(26), Codec(16, {
        'architecture': 'direct_affine', 'bottleneck_dim': 4, 'n_res_blocks': 0,
        'layernorm': 'none', 'snr_film': False, 'dropout': 0.0}), AWGNChannel(),
        {'stack': 'enc', 'where': 'after_final_norm'},
        {'normalize_power': True, 'clean_film_snr': 18.0})
    weights = deepcopy(dict(model.codec.state_dict()))
    _, plan, _, metadata, _ = mechanics()
    plan = replace(plan, width=4)
    metadata['initialization_identity'] = state_dict_identity(weights)
    metadata['model_revision'] = 'tiny-cpu'
    learner = original13_routing.create_original13_learner(model, plan=plan, metadata=metadata, event_sink=None)
    original13_routing.load_codec(model, weights)
    assert state_exact(original13_routing.export_codec(model), weights)
    machine = Original13TrainingState(learner, plan=plan, metadata=metadata, routing=original13_routing)
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
    runtime = original13_routing.observation_state(model)
    with evaluation._original13_runtime(model, state):
        for site in ('enc_fn', 'dec_l0', 'dec_l24'):
            for condition in ('no_noise', -6):
                path = tmp_path/f'{site}_{condition}'
                path.mkdir()
                result = evaluation._score_original13_panel(model, tokenizer, state=state, request=request,
                    site_id=site, condition=condition, output=path)
                assert result['completed_items'] == result['denominator'] == 16
                assert result['metrics']['candidate_forward_requests'] == 64
                assert result['payload']['valid']['hidden'] > 0
                assert (result['payload']['valid']['memory'] > 0) == site.startswith('dec')
                assert json.loads((path/f'identity_{condition}.json').read_text())['sum_dtype'] == 'float32'
    assert state_exact(before, capture_rng())
    assert state_exact(runtime, original13_routing.observation_state(model))
    assert model.memory_codec is model.codec and not torch.cuda.is_initialized()
