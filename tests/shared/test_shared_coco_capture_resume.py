"""Real tiny multimodal shared routes and durable CPU continuation.

These tests establish fixture correctness only; no downloaded/full-weight
model, GPU, caption scorer or scientific qualification is exercised.
"""

import jscc
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import yaml
from torch.amp.grad_scaler import GradScaler

from jscc.activation_replay import canonical_digest
from jscc.asset_identity import state_dict_identity
from jscc.config import load_config
from jscc.experiment_state import capture_rng
from jscc.losses import reconstruction_loss_stats
from jscc.models.channel import AWGNChannel, IdentityChannel
from jscc.models.codec import Codec
from jscc.models.split_model import ResolvedSite, SplitModel, resolve_training_site, stack_module
from jscc.qualification import state_exact
from jscc.reconstruction import capture_clean_targets, reconstruction_batch_values
from jscc.runtime import model_inputs, seed_everything
from jscc.shared_config import resolve_recipe
from jscc.shared_inputs import batch_view, bind_batch_identity
from jscc.shared_plan import RecipePlan
from jscc.shared_state import RecipeTrainingState, recipe_observation_snapshot, recipe_runtime_state
from jscc.shared_training import create_recipe_learner
from jscc.site_routing import export_codec
from jscc.training import batch_losses
from model_helpers import tiny_backbone


ROOT = Path(jscc.__file__).resolve().parents[1]
SITES = ['enc_emb', 'enc_l2', 'enc_fn', 'dec_l1']
CODEC = dict(architecture='direct_affine', hidden_dim=16, bottleneck_dim=8,
             n_res_blocks=0, activation='gelu', dropout=0., layernorm='none', snr_film=False)


def native_image_batch(sequences=2, view=0):
    """Real tiny vision inputs with distinct source and native-target masks."""
    generator = torch.Generator().manual_seed(900 + view)
    repeats = sequences // 2
    return {
        'input_ids': torch.tensor([[2, 31, 31, 31, 31, 4, 6],
                                   [2, 31, 31, 31, 31, 5, 0]]).repeat(repeats, 1),
        'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1, 1, 0]]).repeat(repeats, 1),
        'pixel_values': torch.randn(sequences, 1, 3, 8, 8, generator=generator),
        'labels': torch.tensor([[7 + view % 3, 8, 9, 1],
                               [9, 1, -100, -100]]).repeat(repeats, 1),
        'task': 'coco',
        'query_ids': [f'coco/image/{view * sequences + index}' for index in range(sequences)],
        'source_family_ids': list(range(view * sequences, (view + 1) * sequences)),
        'demo_ids': [],
        'presentation_offset': view * sequences,
        'target_choices': [{'reference_index': view % 2, 'text': f'tiny caption {view}'}
                           for _ in range(sequences)],
    }


def tiny_shared_model(channel='awgn'):
    torch.set_num_threads(1)
    seed_everything(41)
    model = SplitModel(tiny_backbone(4), Codec(16, deepcopy(CODEC)),
        AWGNChannel() if channel == 'awgn' else IdentityChannel(),
        {'stack': 'enc', 'where': 'after_final_norm'},
        {'type': channel, 'normalize_power': True, 'clean_film_snr': 18.})
    return model.train().enable_universal_codec()


def route(model, site):
    return model.at_site(site, purpose='train', authorization={'sites': {site.site_id: 'trained'}})


def hook_inventory(model):
    return [(name, len(module._forward_pre_hooks), len(module._forward_hooks))
            for name, module in model.named_modules()]


def assert_capture_idle(model):
    assert model._forward_active == model._transmission_active == 0
    assert model.activation is model.reconstruction is None
    assert model.memory_activation is model.memory_reconstruction is None


def objective(site, functional=False):
    decoder = site.startswith('dec_')
    return {'temperature': .8, 'loss_weights': {'kl': .7 if functional else 0.,
            'hidden': .3 if decoder else .2, 'memory': .4 if decoder else 0.},
            'requires_memory': decoder}


@pytest.mark.parametrize('site_id', SITES)
def test_native_clean_image_targets_match_bypass_and_unwind_suffix(site_id, monkeypatch):
    model, batch = tiny_shared_model(), native_image_batch()
    site = resolve_training_site(model.base, site_id, 'tiny-coco')
    original_hooks = hook_inventory(model)
    with route(model, site):
        kwargs, labels = model_inputs(batch, model)
        memories = []
        memory_handle = model.base.get_encoder().register_forward_hook(
            lambda _module, _args, output: memories.append(output.last_hidden_state.detach().clone()))
        try:
            with torch.no_grad(), model.transmission(bypass=True,
                    encoder_mask=batch['attention_mask'], decoder_mask=labels != -100):
                model(**kwargs)
            assert torch.is_tensor(model.activation)
            expected_hidden = model.activation.detach().clone()
            expected_memory = memories[-1]
        finally:
            memory_handle.remove()

        observed_native = []
        native_forward = model.base.forward
        def record_native(**inputs):
            observed_native.append((torch.is_grad_enabled(), model.base.training, model.bypass,
                                    inputs['pixel_values'].detach().clone(),
                                    inputs['decoder_input_ids'].detach().clone()))
            return native_forward(**inputs)
        monkeypatch.setattr(model.base, 'forward', record_native)
        def forbidden(*_args, **_kwargs):
            pytest.fail('clean capture executed codec, receiver suffix or logits')
        monkeypatch.setattr(model, '_transmit', forbidden)
        monkeypatch.setattr(model.base.lm_head, 'forward', forbidden)
        stack = stack_module(model.base, site.split['stack'])
        if site_id.startswith('enc_'):
            assert isinstance(site, ResolvedSite)
            monkeypatch.setattr(stack_module(model.base, 'dec'), 'forward', forbidden)
            if site.where == 'after_embed':
                monkeypatch.setattr(stack.layers[0], 'forward', forbidden)
            elif site.where == 'after_layer':
                assert site.index is not None
                monkeypatch.setattr(stack.layers[site.index + 1], 'forward', forbidden)
        else:
            assert site.index is not None
            monkeypatch.setattr(stack.layers[site.index + 1], 'forward', forbidden)
        routed_hooks = hook_inventory(model)
        hidden, memory = capture_clean_targets(model, batch, site)
        torch.testing.assert_close(hidden, expected_hidden, rtol=0, atol=0)
        assert not hidden.requires_grad and hidden.dtype == expected_hidden.dtype
        if site_id.startswith('dec_'):
            assert torch.is_tensor(memory)
            torch.testing.assert_close(memory, expected_memory, rtol=0, atol=0)
            assert not memory.requires_grad
        else:
            assert memory is None
        assert observed_native[0][:3] == (False, False, True)
        torch.testing.assert_close(observed_native[0][3], batch['pixel_values'].flatten(0, 1))
        torch.testing.assert_close(observed_native[0][4], kwargs['decoder_input_ids'])
        changed = deepcopy(batch)
        changed['pixel_values'] = batch['pixel_values'].flip(-1)
        changed_hidden, changed_memory = capture_clean_targets(model, changed, site)
        assert not torch.equal(hidden, changed_hidden), 'pixels must influence every clean boundary'
        if memory is not None:
            assert changed_memory is not None
            assert not torch.equal(memory, changed_memory)
        assert hook_inventory(model) == routed_hooks
        assert_capture_idle(model)
    assert hook_inventory(model) == original_hooks
    assert not model._routing_active
    assert_capture_idle(model)


@pytest.mark.parametrize('site_id', SITES)
def test_capture_failure_removes_temporary_hooks_and_unwinds_route(site_id, monkeypatch):
    model, batch = tiny_shared_model(), native_image_batch()
    site = resolve_training_site(model.base, site_id, 'tiny-coco')
    original_hooks = hook_inventory(model)
    with route(model, site):
        routed_hooks = hook_inventory(model)
        def fail(**_kwargs):
            raise RuntimeError('injected native encoder failure')
        monkeypatch.setattr(model.base.get_encoder(), 'forward', fail)
        with pytest.raises(RuntimeError, match='injected native encoder'):
            capture_clean_targets(model, batch, site)
        assert hook_inventory(model) == routed_hooks
        assert_capture_idle(model)
    assert hook_inventory(model) == original_hooks
    assert not model._routing_active


@pytest.mark.parametrize('site_id', SITES)
@pytest.mark.parametrize('functional', [False, True])
def test_multimodal_mask_reductions_and_one_fp32_canonical_gradient(site_id, functional, monkeypatch):
    model, batch = tiny_shared_model(), native_image_batch()
    site = resolve_training_site(model.base, site_id, 'tiny-coco')
    with route(model, site):
        settings = objective(site_id, functional)
        if functional:
            values = cast(dict[str, Any], batch_losses(model, batch, settings, None,
                          return_stats=True, valid_only_kl=True))
            assert values['kl_denominator'] == 6
        else:
            values = cast(dict[str, Any], reconstruction_batch_values(model, batch, None, settings, site=site))
            assert_capture_idle(model)
        assert values['hidden_denominator'] == 2
        decoder = site_id.startswith('dec_')
        if decoder:
            assert values['memory_denominator'] == 2
        else:
            assert values['memory_denominator'] is None
        assert model.channel_uses_valid == {'hidden': (6 if decoder else 13) * 8,
                                            'memory': 13 * 8 if decoder else 0}
        assert model.channel_uses == {'hidden': (8 if decoder else 14) * 8,
                                      'memory': 14 * 8 if decoder else 0}
        values['loss'].backward()
        assert model.memory_codec is model.codec
        assert {id(p) for p in model.parameters() if p.requires_grad} == {id(p) for p in model.codec.parameters()}
        assert all(p.dtype == torch.float32 and p.grad is not None
                   and bool(torch.isfinite(p.grad).all()) for p in model.codec.parameters())
        assert all(float(p.grad.abs().sum()) > 0 for p in model.codec.parameters() if p.grad is not None)
        assert all(p.grad is None for p in model.base.parameters())

        # Padding in each transmitted role must not enter its nMSE reduction.
        transmit = model._transmit
        def poison_padding(hidden, codec, role, mask, **kwargs):
            result = transmit(hidden, codec, role, mask, **kwargs)
            return result + (~mask.bool()).unsqueeze(-1) * 10000.
        monkeypatch.setattr(model, '_transmit', poison_padding)
        if not functional:
            poisoned = cast(dict[str, Any], reconstruction_batch_values(model, batch, None, settings, site=site))
            torch.testing.assert_close(poisoned['hidden_numerator'], values['hidden_numerator'])
            if decoder:
                torch.testing.assert_close(poisoned['memory_numerator'], values['memory_numerator'])


def test_decoder_sender_uses_clean_memory_and_all_receivers_reconstructed_memory():
    model, batch = tiny_shared_model(), native_image_batch()
    site = resolve_training_site(model.base, 'dec_l1', 'tiny-coco')
    with route(model, site):
        memories, seen, handles = [], [[], [], [], []], []
        handles.append(model.base.get_encoder().register_forward_hook(
            lambda _module, _args, output: memories.append(output.last_hidden_state.detach().clone())))
        for index, layer in enumerate(stack_module(model.base, 'dec').layers):
            def observe(_module, _args, kwargs, index=index):
                if not model.bypass:
                    seen[index].append(kwargs['encoder_hidden_states'])
            handles.append(layer.self_attn.register_forward_pre_hook(observe, with_kwargs=True))
        try:
            kwargs, labels = model_inputs(batch, model)
            with model.transmission(0., encoder_mask=batch['attention_mask'], decoder_mask=labels != -100):
                model(**kwargs)
            for index in [0, 1]:
                torch.testing.assert_close(seen[index][0], memories[0], rtol=0, atol=0)
            for index in [2, 3]:
                assert seen[index][0] is model.memory_reconstruction
                assert not torch.equal(seen[index][0], memories[0])
            hidden = model.reconstruction
            memory = model.memory_reconstruction
            assert torch.is_tensor(hidden) and torch.is_tensor(memory)
            hidden.retain_grad()
            memory.retain_grad()
            assert model.activation is not None and model.memory_activation is not None
            h, _ = reconstruction_loss_stats(hidden, model.activation, labels != -100)
            m, _ = reconstruction_loss_stats(memory, model.memory_activation, batch['attention_mask'])
            (h + m).backward()
            assert hidden.grad is not None and memory.grad is not None
            assert float(hidden.grad.abs().sum()) > 0 and float(memory.grad.abs().sum()) > 0
        finally:
            for handle in handles:
                handle.remove()


class IndexedImages:
    """Independent immutable content per base view, with observable indexed seeks."""
    def __init__(self, q):
        self.batches = [native_image_batch(16, index) for index in range(q)]
        self.views = tuple(batch_view(batch) for batch in self.batches)
        self.reads = []

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        self.reads.append(index)
        return deepcopy(self.batches[index])


def coco_recipe_state(tmp_path, channel='awgn'):
    tmp_path.mkdir(parents=True, exist_ok=True)
    # Resolve a real COCO task/recipe with the accepted shared numerical envelope.
    task_config = load_config(ROOT / 'configs/experiments/shared/coco_base.yaml', validate=False)
    task_config['model'].update(device='cpu', dtype='float32', revision='tiny-coco',
                                sdpa_backend_policy='auto', numerical_policy='native')
    task_config['codec'] = deepcopy(CODEC)
    task_config['split'] = {'stack': 'enc', 'where': 'after_final_norm'}
    task_config['data'].update(num_train=128, num_demos=0)
    task_path = tmp_path / 'task.yaml'
    task_path.write_text(yaml.safe_dump(task_config))
    recipe = yaml.safe_load((ROOT / 'configs/experiments/shared/shared_hellaswag_reference.yaml').read_text())
    experiment = recipe['experiment']
    experiment.update(task_config=str(task_path), locations=['dec_l1', 'enc_l2'],
                      updates_per_location=8, widths=[8], reconstruction_percentages=[50])
    experiment['observation'] = {'items': 16, 'sites': ['dec_l1', 'enc_l2'], 'conditions': ['no_noise']}
    experiment['evaluation'] = {'items': 16, 'conditions': ['no_noise'], 'supplemental_conditions': [],
                                'batch_size': 16, 'max_new_tokens': 4, 'scoring_policy': 'coco-cider-v1'}
    experiment['overrides']['data'] = {'caption_policy': 'indexed-reference-v1', 'caption_seed': 23}
    experiment['overrides']['training'].update(lr=.001, weight_decay=.02, warmup_ratio=.25,
            min_lr_ratio=.2, betas=[.8, .95], eps=1e-7, temperature=.8)
    experiment['overrides']['channel'] = {'type': channel, 'train_noise': channel == 'awgn',
                                         'train_snr_range': [3., 3.], 'normalize_power': True}
    experiment['save']['functional_observation_fraction'] = .5
    recipe_path = tmp_path / 'recipe.yaml'
    recipe_path.write_text(yaml.safe_dump(recipe))
    model_path = tmp_path / 'config.json'
    model_path.write_text(json.dumps({'encoder': {'text_config': {'num_hidden_layers': 4, 'hidden_size': 16}},
                                     'decoder': {'num_hidden_layers': 4, 'hidden_size': 16}}))
    description = resolve_recipe(recipe_path, model_config_path=model_path)['trajectories'][0]
    batches = IndexedImages(8)
    assert len({view.view_sha256 for view in batches.views}) == 8
    plan = RecipePlan(description, batches.views, 'tiny-coco-indexed')
    model = tiny_shared_model(channel)
    # The released learner establishes the universal owner from a fresh enc_fn model.
    model.memory_codec = None
    model.universal_codec = False
    model.channel_config = deepcopy(description['config']['channel'])
    metadata = {'source_identity': 'tiny-coco-source',
        'config_identity': canonical_digest(description['config']),
        'data_identity': canonical_digest({'images': [view.view_sha256 for view in batches.views]}),
        'prepared_identity': 'tiny-coco-native16-indexed-v1',
        'observation_identity': canonical_digest(description['observations']),
        'initialization_identity': state_dict_identity(model.codec.state_dict()),
        'model_revision': 'tiny-coco'}
    events = []
    learner = create_recipe_learner(model, plan=plan, metadata=metadata, event_sink=events.append, synthetic=True)
    learner.scaler = GradScaler('cpu', enabled=True, init_scale=8., growth_interval=2)
    state = RecipeTrainingState(learner, plan=plan, metadata=metadata)
    assert learner.task == state.metadata['task'] == 'coco'
    assert state.metadata['protocol_identity'] == 'recipe-coco-training-state-v1'
    return state, batches, events


def advance_indexed(state, batches, target):
    while state.learner.completed < target:
        update = state.plan.update_at(state.learner.completed)
        index = update.site_local_index
        batch = batches[index]
        binding = bind_batch_identity(batch, expected_identity=update.view.view_sha256)
        before = [group['lr'] for group in state.learner.optimizer.param_groups]
        event = state.learner.update([batch], batch_identities=[binding])
        assert before == event['payload']['lr_used']
        assert event['task'] == 'coco'
        assert event['payload']['site_id'] == update.site
        assert event['payload']['audit']['complete_batch_identities'] == [binding.complete_sha256]


def continuation_snapshot(state):
    value = recipe_observation_snapshot(state)
    # Receipts contain timestamps and parent paths; compare their counts separately.
    value.pop('ancestry')
    value.pop('reset_parent')
    value['exposure'] = value['exposure'][:3]
    return value


@pytest.mark.parametrize('channel', ['awgn', 'identity'])
@pytest.mark.parametrize('position', ['reconstruction', 'pre_reset', 'post_reset', 'functional'])
def test_coco_indexed_native16_exact_continuation_at_each_phase_state(tmp_path, channel, position):
    state, batches, events = coco_recipe_state(tmp_path / 'first', channel)
    target = 3 if position == 'reconstruction' else state.plan.stages[1].start
    advance_indexed(state, batches, target)
    if position in {'post_reset', 'functional'}:
        parent = state.save(tmp_path / 'parent.pt')
        before = deepcopy(export_codec(state.learner.model))
        exposure = deepcopy(state.learner.exposure.per_site)
        state.reset(parent)
        assert state_exact(export_codec(state.learner.model), before)
        assert state.learner.exposure.per_site == exposure
        assert not state.learner.optimizer.state
        assert state.learner.scheduler.state_dict()['last_epoch'] == 0
        if position == 'functional':
            advance_indexed(state, batches, target + 3)
    saved = state.save(tmp_path / 'saved.pt')
    initial = state.learner.completed
    expected_progress = state.plan.progress(initial, state.learner.phase_start_step)
    assert saved.payload['stream']['next_view'] == initial // 2
    assert saved.payload['stream']['next_site'] == state.plan.trained_sites[initial % 2]
    assert saved.payload['stream']['phase_local_completed'] == expected_progress['phase_local_completed']
    assert saved.payload['metadata']['task'] == 'coco'
    assert saved.payload['metadata']['protocol_identity'] == 'recipe-coco-training-state-v1'
    assert saved.payload['metadata']['recipe_plan']['views'] == [asdict(view) for view in batches.views]
    assert saved.payload['scaler']['scale'] >= 8.
    if position == 'pre_reset':
        state.reset(saved)
    advance_indexed(state, batches, initial + 2)
    expected = continuation_snapshot(state)
    next_events = deepcopy(events[-2:])

    resumed, resumed_batches, resumed_events = coco_recipe_state(tmp_path / 'second', channel)
    resumed.restore(saved.reference)
    assert state_exact(capture_rng(), saved.payload['rng'])
    assert resumed.learner.completed == initial
    assert resumed.learner.offset == saved.payload['stream']['offset']
    assert resumed.learner.exposure.receipts == saved.payload['metadata']['exposure_receipts']
    assert resumed.ancestry == [asdict(saved.reference)]
    assert state_exact(recipe_runtime_state(resumed.learner.model), saved.payload['metadata']['route_state'])
    assert resumed.learner.scaler.state_dict() == saved.payload['scaler']
    if position == 'pre_reset':
        resumed.reset(saved)
    advance_indexed(resumed, resumed_batches, initial + 2)
    assert resumed_batches.reads == [initial // 2, (initial + 1) // 2]
    assert state_exact(continuation_snapshot(resumed), expected)
    assert len(resumed.learner.exposure.receipts) == initial + 2
    for actual, uninterrupted in zip(resumed_events, next_events):
        assert actual['task'] == uninterrupted['task'] == 'coco'
        for key in ('completed_step', 'site_id', 'site_step', 'lr_used', 'lr_next',
                    'recipe_progress', 'objective', 'snr', 'audit', 'exposure'):
            assert state_exact(actual['payload'][key], uninterrupted['payload'][key])
    continued = resumed.save(tmp_path / 'continued.pt')
    assert continued.payload['metadata']['segment_ancestry'] == [asdict(saved.reference)]
    assert continued.payload['metadata']['recipe_plan']['views'] == saved.payload['metadata']['recipe_plan']['views']


def test_coco_state_rejects_relabeled_hellaswag_protocol_before_mutation(tmp_path):
    import hashlib
    import io
    from jscc.experiment_state import CheckpointRef
    state, batches, _events = coco_recipe_state(tmp_path / 'setup')
    advance_indexed(state, batches, 3)
    saved = state.save(tmp_path / 'valid.pt')
    payload = deepcopy(saved.payload)
    payload['metadata']['protocol_identity'] = 'recipe-training-state-v1'
    payload['metadata'].pop('task')
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    data = buffer.getvalue()
    path = tmp_path / 'relabeled.pt'
    path.write_bytes(data)
    reference = CheckpointRef(str(path), hashlib.sha256(data).hexdigest(), len(data))
    before = recipe_observation_snapshot(state)
    optimizer = state.learner.optimizer
    with pytest.raises(ValueError):
        state.restore(reference)
    assert state.learner.optimizer is optimizer
    assert state_exact(recipe_observation_snapshot(state), before)
