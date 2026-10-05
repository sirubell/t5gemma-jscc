"""Real tiny COCO generation/CIDEr; processor/PTB and checkpoint I/O are fixtures."""
from copy import deepcopy
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from jscc import evaluation, panel_evaluation as panels, site_routing
from jscc.data import TaskData
from jscc.experiment_state import capture_rng
from jscc.models.channel import AWGNChannel
from jscc.models.codec import Codec
from jscc.models.split_model import SplitModel
from jscc.observation_policy import scoring_template
from jscc.qualification import state_exact
from jscc.shared_state import recipe_runtime_state
from model_helpers import tiny_backbone


class Image:
    def __init__(self, value):
        self.value = value

    def convert(self, mode):
        assert mode == 'RGB'
        return self


class Tokenizer:
    eos_token_id = 1
    name_or_path = 'tiny-fixture'

    def decode(self, tokens, skip_special_tokens):
        assert skip_special_tokens and len(tokens) > 0
        return 'a cat on a mat. discarded sentence'

    def __call__(self, text, add_special_tokens):
        return {'input_ids': list(range(len(text.split()) + 1))}


class Processor:
    tokenizer = Tokenizer()

    def __call__(self, *, images, text, return_tensors, padding, truncation):
        assert return_tensors == 'pt' and padding and not truncation
        assert len(images) == len(text)
        assert all(len(group) == 5 for group in images)
        assert all('demo cat' in prompt for prompt in text)
        return {'input_ids': torch.tensor([[2, 31, 31, 31, 31, 4]] * len(images)),
                'attention_mask': torch.ones(len(images), 6, dtype=torch.long),
                'pixel_values': torch.stack([torch.full((3, 8, 8), float(group[-1].value))
                                             for group in images])}


@pytest.fixture
def scoring_inputs(monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = SplitModel(tiny_backbone(), Codec(16, {
        'architecture': 'direct_affine', 'bottleneck_dim': 8, 'n_res_blocks': 0,
        'layernorm': 'none', 'snr_film': False, 'dropout': 0.0}), AWGNChannel(),
        {'stack': 'enc', 'where': 'after_final_norm'},
        {'normalize_power': True, 'clean_film_snr': 18.0}).enable_universal_codec()
    processor = Processor()
    references = [['a cat on a mat', 'the cat rests'], ['a dog running', 'the dog walks']]
    data = TaskData(ids={'demo_ids': [31, 32, 33, 34], 'report_ids': [9, 4]},
                    demo_images=[Image(.1)] * 4, demo_captions=['demo cat'] * 4,
                    report=[{'image': Image(float(index + 1)),
                             'file_name': f'COCO_val2014_{image_id:012d}.jpg', 'answer': refs}
                            for index, (image_id, refs) in enumerate(zip([9, 4], references, strict=True))])
    # Exercise the actual Cider class while avoiding Java for this tiny test.
    # Reverse PTB map order to prove score_coco_captions restores image order.
    class PTB:
        def tokenize(self, values):
            return {key: [row['caption'].lower().strip() for row in values[key]]
                    for key in reversed(values)}
    monkeypatch.setattr('pycocoevalcap.tokenizer.ptbtokenizer.PTBTokenizer', PTB)
    template = {'task': 'coco', 'input_ids': [9, 4], 'source_family_ids': [9, 4],
                'cache': {'path': '/fixture/cache.json', 'sha256': 'a' * 64},
                'data_settings': {'caption_policy': 'indexed-reference-v1', 'caption_seed': 7},
                'reference_corpus': {'image_ids': [9, 4], 'references': references,
                                     'demo_ids': [31, 32, 33, 34], 'demo_captions': ['demo cat'] * 4,
                                     'cache_identity': 'b' * 64, 'processor_identity': 'c' * 64},
                'scorer': 'coco-ptb-cider-v1', 'precision': 'fp32',
                'backend': model.base.config.decoder._attn_implementation,
                'prompt_policy': 'native-demo-caption-v1', 'layout': 'native',
                'noise': {'seed': 13, 'namespace': 'tiny-paired-coco'},
                'settings': {'numerical_policy': 'native', 'vanilla': False}}
    policy = {'items': 2, 'batch_size': 2, 'max_new_tokens': 3, 'scoring_policy': 'coco-cider-v1'}
    request = {**scoring_template(template, policy, policy),
               'sites': ['enc_emb', 'dec_l0'], 'conditions': ['no_noise']}
    return model, processor, data, template, policy, request


def read_rows(output, condition='no_noise'):
    return [json.loads(line) for line in (output / f'captions_{condition}.jsonl').read_text().splitlines()]


def test_ordinary_real_generation_and_ordered_cider(scoring_inputs, tmp_path, monkeypatch):
    from pycocoevalcap.cider.cider import Cider
    model, processor, data, _, _, request = scoring_inputs
    calls = []
    generate = model.generate
    def record_generation(**kwargs):
        calls.append(kwargs)
        return generate(**kwargs)
    monkeypatch.setattr(model, 'generate', record_generation)
    model.eval()
    with model.transmission(None):
        metrics = evaluation.evaluate_coco(model, processor, data, request['settings'], tmp_path, 'no_noise')
    rows = read_rows(tmp_path)
    assert len(calls) == metrics['generation_requests'] == 1
    assert calls[0]['do_sample'] is False and calls[0]['num_beams'] == 1 and calls[0]['use_cache'] is True
    assert [row['image_id'] for row in rows] == [9, 4]
    assert metrics['num_samples'] == 2 and metrics['finite_scores']
    refs = {row['image_id']: row['ptb_references'] for row in rows}
    captions = {row['image_id']: row['ptb_caption'] for row in rows}
    score, per_image = Cider().compute_score(refs, captions)
    assert metrics['cider'] == pytest.approx(score)
    assert [row['cider'] for row in rows] == pytest.approx(per_image.tolist())
    assert all(row['token_ids'] and row['generated_tokens_through_eos'] <= 3 for row in rows)
    assert [row['references'] for row in rows] == request['reference_corpus']['references']
    assert all(row['raw_caption'] == 'a cat on a mat. discarded sentence' for row in rows)
    assert all(row['caption'] == 'a cat on a mat.' for row in rows)
    for key in ('allocated', 'mask_valid', 'generation_valid'):
        assert metrics[f'channel_uses_{key}'] == {
            stream: sum(row['payload'][key][stream] for row in rows) for stream in ('hidden', 'memory')}
    evidence = json.loads((tmp_path / 'coco_evidence_no_noise.json').read_text())
    assert evidence['batches'][0]['image_ids'] == [9, 4]
    assert evidence['generation']['max_new_tokens'] == 3


@pytest.mark.parametrize('fault', ['short_generation', 'short_scores', 'long_scores', 'nan_score', 'inf_image', 'missing_ptb'])
def test_ordinary_rejects_bad_generation_or_scores(scoring_inputs, tmp_path, monkeypatch, fault):
    model, processor, data, _, _, request = scoring_inputs
    if fault == 'short_generation':
        monkeypatch.setattr(model, 'generate', lambda **kwargs: torch.tensor([[0, 1]]))
    else:
        def bad_score(refs, captions):
            tokens = {key: ['caption'] for key in refs}
            values = [0., 1.]
            if fault == 'short_scores':
                values = [0.]
            elif fault == 'long_scores':
                values = [0., 1., 2.]
            elif fault == 'inf_image':
                values[1] = float('inf')
            elif fault == 'missing_ptb':
                tokens.pop(4)
            return float('nan') if fault == 'nan_score' else .5, values, tokens, tokens
        monkeypatch.setattr(evaluation, 'score_coco_captions', bad_score)
    with pytest.raises(ValueError, match='cardinality|finite score|membership'):
        evaluation.evaluate_coco(model.eval(), processor, data, request['settings'], tmp_path, 'no_noise')
    assert not (tmp_path / 'captions_no_noise.jsonl').exists()


def test_template_prefix_uses_image_ids_and_generation_settings(scoring_inputs):
    _, _, _, template, policy, _ = scoring_inputs
    result = scoring_template(template, {'items': 1}, policy)
    assert result['input_ids'] == result['source_family_ids'] == result['reference_corpus']['image_ids'] == [9]
    assert result['reference_corpus']['references'] == [['a cat on a mat', 'the cat rests']]
    assert result['settings']['max_new_tokens'] == 3 and result['settings']['batch_size'] == 2
    assert result['cache'] == template['cache']
    assert len(template['input_ids']) == 2  # Binding does not mutate acquired corpus.
    for field, value in [('source_family_ids', ['9', '4']), ('input_ids', [9, 9]), ('scorer', 'hellaswag')]:
        bad = deepcopy(template)
        bad[field] = value
        with pytest.raises(ValueError):
            scoring_template(bad, policy, policy)


@pytest.mark.parametrize('fail', [False, True])
def test_panels_load_once_and_restore_training_runtime_on_score_outcome(scoring_inputs, tmp_path, monkeypatch, fail):
    from jscc import shared_inputs
    model, processor, data, _, _, request = scoring_inputs
    model.train()
    model.channel_uses = {'hidden': 37, 'memory': 0}
    model.channel_uses_valid = {'hidden': 19, 'memory': 0}
    for parameter in model.codec.parameters():
        parameter.grad = torch.full_like(parameter, .125)
    before_weights = deepcopy(model.state_dict())
    before_grads = [parameter.grad.clone() for parameter in model.codec.parameters()]
    before_route, before_rng = recipe_runtime_state(model), capture_rng()
    checkpoint_weights = {key: value * .9 for key, value in site_routing.export_codec(model).items()}
    state = SimpleNamespace(payload={'kind': 'recipe_v1', 'model': checkpoint_weights, 'metadata': {
        'model_revision': 'tiny-scoring', 'recipe_plan': {'description': {'config': {'task': 'coco'},
                                                                      'locations': request['sites']}}}},
                            reference=SimpleNamespace(path='/fixture/checkpoint.pt', sha256='d' * 64, size=1))
    # Only checkpoint/plan I/O is mocked: routes, weight/gradient snapshots,
    # generation, counters, RNG restoration and actual CIDEr remain real.
    from jscc.experiment_state import CheckpointRef
    state.reference = CheckpointRef('/fixture/checkpoint.pt', 'd' * 64, 1)
    monkeypatch.setattr(panels, '_validate_recipe_request', lambda *args, **kwargs: (state, SimpleNamespace(identity='plan')))
    monkeypatch.setattr('jscc.experiment_state.open_state', lambda *args, **kwargs: state)
    loads = []
    def load(request, processor):
        loads.append(request['input_ids'])
        return data
    monkeypatch.setattr(shared_inputs, 'load_coco_panel_data', load)
    score = evaluation.score_coco_captions
    def mutate_rng_then_score(refs, captions):
        random.random()
        np.random.random()
        torch.rand(3)
        if fail:
            raise evaluation.TaskScoringError('fixture recoverable PTB failure')
        return score(refs, captions)
    monkeypatch.setattr(evaluation, 'score_coco_captions', mutate_rng_then_score)
    if fail:
        with pytest.raises(panels.ObservationScoringError, match='PTB failure'):
            panels._run_panels(model, processor, state=state, request=request, output=tmp_path / 'panels',
                               guard=lambda: None, observation=True)
    else:
        receipt = panels._run_panels(model, processor, state=state, request=request, output=tmp_path / 'panels',
                                     guard=lambda: None, observation=True)
        assert receipt['completed_panels'] == 2
        for row in receipt['panels']:
            assert row['completed_items'] == row['denominator'] == row['metrics']['num_samples'] == 2
            assert [item['image_id'] for item in read_rows(tmp_path / 'panels' / f"{row['site']}_no_noise")] == [9, 4]
            assert 'generation_valid' in row['payload']
    assert loads == [[9, 4]]
    assert state_exact(before_weights, model.state_dict())
    assert state_exact(before_grads, [parameter.grad for parameter in model.codec.parameters()])
    assert state_exact(before_route, recipe_runtime_state(model))
    assert state_exact(before_rng, capture_rng())
    result = json.loads((tmp_path / 'panels' / 'result.json').read_text())
    assert result['status'] == ('failed' if fail else 'complete')
    if fail:
        assert [row['status'] for row in result['panels']] == ['failed', 'pending']


@pytest.mark.parametrize('channel', ['awgn', 'identity'])
@pytest.mark.parametrize('fail', [False, True])
def test_actual_recipe_snapshot_and_checkpoint_restore_around_coco_scoring(
        scoring_inputs, tmp_path, monkeypatch, channel, fail):
    from jscc.shared_state import recipe_observation_snapshot, restore_recipe_observation
    from shared.test_shared_coco_capture_resume import coco_recipe_state, advance_indexed
    _, processor, data, _, _, request = scoring_inputs
    training, indexed, _ = coco_recipe_state(tmp_path / 'training', channel)
    advance_indexed(training, indexed, 3)
    model = training.learner.model
    saved = training.save(tmp_path / 'scoring-state.pt')
    checkpoint_bytes = (tmp_path / 'scoring-state.pt').read_bytes()
    request = deepcopy(request)
    request.update(sites=['enc_l2', 'dec_l1'])
    request['backend'] = model.base.config.decoder._attn_implementation
    before = recipe_observation_snapshot(training)
    score = evaluation.score_coco_captions
    def touch_training_state(refs, captions):
        # Labeled fault injection exercises the training-owned outer snapshot;
        # the scorer itself has no access to the learner in production.
        training.learner.attempted += 17
        training.learner.offset += 16
        training.learner.valid_tokens += 123
        training.learner.exposure.per_site['dec_l1']['sequences'] += 1
        training.learner.optimizer.param_groups[0]['lr'] = .25
        training.learner.scheduler.last_epoch += 2
        training.learner.scaler.update(new_scale=3.)
        random.random()
        np.random.random()
        torch.rand(11)
        if fail:
            raise evaluation.TaskScoringError('fixture recoverable CIDEr failure')
        return score(refs, captions)
    monkeypatch.setattr(evaluation, 'score_coco_captions', touch_training_state)
    try:
        with panels._panel_runtime(model, saved):
            if fail:
                with pytest.raises(panels.ObservationScoringError, match='CIDEr failure'):
                    panels._score_panel(model, processor, state=saved, request=request,
                                        site_id='dec_l1', condition='no_noise', output=tmp_path / 'caption-panel', data=data)
            else:
                receipt = panels._score_panel(model, processor, state=saved, request=request,
                                             site_id='dec_l1', condition='no_noise', output=tmp_path / 'caption-panel', data=data)
                assert receipt['denominator'] == receipt['completed_items'] == 2
                assert receipt['metrics']['finite_scores']
                assert len(read_rows(tmp_path / 'caption-panel')) == 2
    finally:
        restore_recipe_observation(training, before)
    assert state_exact(before, recipe_observation_snapshot(training))
    assert (tmp_path / 'scoring-state.pt').read_bytes() == checkpoint_bytes


def test_generation_valid_unavailable_keeps_observed_counts(scoring_inputs, tmp_path, monkeypatch):
    model, processor, data, _, _, request = scoring_inputs
    model.split = {'stack': 'dec', 'where': 'after_layer', 'index': 0}
    def generate(**kwargs):
        # Labeled non-single-token decoder execution preserves unavailable
        # generation-valid counts while still exporting exact observed totals.
        model._payload_observer.append({'stream': 'hidden', 'shape': [2, 2, 8],
                                        'allocated': [16, 16], 'mask_valid': [16, 8]})
        return torch.tensor([[0, 5, 1], [0, 6, 1]])
    monkeypatch.setattr(model, 'generate', generate)
    metrics = evaluation.evaluate_coco(model, processor, data, request['settings'], tmp_path, 'no_noise')
    assert metrics['channel_uses_generation_valid'] is None
    assert metrics['channel_uses_allocated']['hidden'] == 32
    assert metrics['channel_uses_mask_valid']['hidden'] == 24
    assert all(row['payload']['generation_valid'] is None for row in read_rows(tmp_path))


def test_coco_scorer_rejects_hellaswag_checkpoint_task_before_output(scoring_inputs, tmp_path):
    model, processor, data, _, _, request = scoring_inputs
    state = SimpleNamespace(payload={'kind': 'recipe_v1', 'metadata': {
        'recipe_plan': {'description': {'config': {'task': 'hellaswag'}}}}})
    with pytest.raises(ValueError, match='checkpoint task'):
        panels._score_panel(model, processor, state=state, request=request,
                            site_id='enc_emb', condition='no_noise', output=tmp_path / 'forbidden', data=data)
    assert not (tmp_path / 'forbidden').exists()
