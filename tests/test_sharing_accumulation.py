"""Explicit effective64 sharing: metadata binding and same-shape CPU guards."""
import copy
import json
from pathlib import Path

import pytest
import torch

from sharing_fixtures import sharing_model
from jscc.activation_replay import file_digest
from jscc.baseline_protocol import batch_identity, bind_batch_identity, read_prepared_batch
from jscc.sharing_accumulation import partition_native64
from jscc.sharing_protocol import SharingLearner, batch_view
from jscc.sharing_schedule import SharingPlan

REAL_INPUTS = Path('/Users/tim_c_wang/Documents/Codex/2026-09-30/task-11/sharing-production/inputs')


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
    assert not torch.cuda.is_initialized()


def parent64(index=0):
    """Unequal denominators across halves/quarters; padded metadata stays native."""
    source = torch.zeros(64, 5, dtype=torch.long)
    mask = torch.zeros_like(source)
    labels = torch.full((64, 4), -100, dtype=torch.long)
    for row in range(64):
        source_length = 2 + row // 16
        target_length = 1 + row // 16
        source[row, :source_length] = torch.tensor([2, 3 + index, 5, 6, 7][:source_length])
        mask[row, :source_length] = 1
        labels[row, :target_length] = torch.tensor([8, 9, 10, 1][:target_length])
    decoder = torch.zeros_like(labels)
    decoder[:, 0] = 2
    decoder[:, 1:] = labels[:, :-1].masked_fill(labels[:, :-1] == -100, 0)
    ids = list(range(64 * index, 64 * (index + 1)))
    roles = [['demo'] + ['query' if column < int(mask[row].sum()) else 'pad'
                        for column in range(1, 5)] for row in range(64)]
    return dict(input_ids=source, attention_mask=mask, labels=labels,
                decoder_input_ids=decoder, decoder_attention_mask=labels != -100,
                row_ids=torch.tensor(ids), source_family_ids=[f'train-{i}' for i in ids],
                batch_view_id=f'parent-{index}', view_ids=[f'view-{i}' for i in ids],
                demo_ids=[[10000, 10001] for _ in ids], position_roles=roles,
                token_roles=copy.deepcopy(roles), token_positions=torch.arange(5).repeat(64, 1),
                source_views=[{'row_id': i, 'native_prefix': 'fixture',
                               'demo_source_family_ids': ['demo-a', 'demo-b']} for i in ids])


def plan64():
    return SharingPlan(tuple(batch_view(parent64(i)) for i in range(4)),
                       'same-shape-explicit64', synthetic=True)


def learner64(micro, name='shared', reuse=False):
    return SharingLearner(sharing_model(), schedule=plan64().learner(name),
        run_id=name, identity={'source': 'source', 'config': 'config', 'data': 'data', 'parent': None},
        model_revision='tiny-cpu', microbatch_size=micro, enc_fn_reuse=reuse)


def assert_nested_equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def assert_partition(parent, micro):
    before = batch_identity(parent)
    parts = partition_native64(parent, microbatch_size=micro)
    assert len(parts) == 64 // micro
    for index, part in enumerate(parts):
        start, stop = index * micro, (index + 1) * micro
        assert part.keys() == parent.keys()
        for key, value in parent.items():
            if torch.is_tensor(value) and value.ndim and len(value) == 64:
                assert_nested_equal(part[key], value[start:stop])
                assert part[key].shape[1:] == value.shape[1:]
            elif isinstance(value, (list, tuple)) and len(value) == 64:
                assert_nested_equal(part[key], value[start:stop])
            else:
                assert_nested_equal(part[key], value)
        assert part['decoder_input_ids'].shape == part['labels'].shape
        assert part['decoder_attention_mask'].shape == part['labels'].shape
    assert torch.equal(torch.cat([p['row_ids'] for p in parts]), parent['row_ids'])
    assert sum(int(p['attention_mask'].sum()) for p in parts) == int(parent['attention_mask'].sum())
    assert sum(int((p['labels'] != -100).sum()) for p in parts) == int((parent['labels'] != -100).sum())
    assert batch_identity(parent) == before
    return parts


@pytest.mark.parametrize('micro', [32, 16])
def test_complete_padded_metadata_partition(micro):
    parts = assert_partition(parent64(), micro)
    assert len({int((p['labels'] != -100).sum()) for p in parts}) > 1


@pytest.mark.parametrize('micro', [32, 16])
def test_all202_real_inputs_hash_verified_readonly(micro):
    manifest_path = REAL_INPUTS / 'production-inputs.json'
    if not manifest_path.is_file():
        pytest.skip('real offline input package unavailable; not a real-input pass')
    assert file_digest(manifest_path) == 'feccf3232b67f9a99693c04f82cf9646556b2e31031fe9de09b62434ea623a55'
    manifest = json.loads(manifest_path.read_text())
    refs = manifest['updates'] + manifest['validation']
    assert len(refs) == 202
    for reference in refs:
        batch = read_prepared_batch(reference, REAL_INPUTS)
        assert_partition(batch, micro)


@pytest.mark.parametrize('micro', [0, 8, 64, True, 31])
def test_partition_rejects_unapproved_sizes(micro):
    with pytest.raises((ValueError, TypeError)):
        partition_native64(parent64(), microbatch_size=micro)


@pytest.mark.parametrize('micro', [32, 16])
def test_bound_parent_metadata_mutation_rejected(micro):
    item = learner64(micro)
    batch = parent64()
    binding = bind_batch_identity(batch)
    batch['source_views'][0]['native_prefix'] = 'changed'
    with pytest.raises(ValueError):
        item.update([batch], batch_identities=[binding])
    assert item.completed == 0 and item.exposure.completed == 0


@pytest.mark.parametrize('micro', [32, 16])
def test_shared_specialist_noisy_draw_pairing_and_terminal_skip(micro, monkeypatch):
    shared = learner64(micro)
    specialist = learner64(micro, 'specialist_enc_l19')
    shared.audit_policy = specialist.audit_policy = 'full'
    update64(shared, parent64())
    update64(shared, parent64())
    update64(specialist, parent64())
    assert shared.last_draws == specialist.last_draws
    assert len(shared.last_draws) == 64 // micro
    assert len({json.dumps(row['key'], sort_keys=True) for row in shared.last_draws}) == 64 // micro
    original = dict(specialist.model.split)
    monkeypatch.setattr(specialist.scaler, 'step', lambda optimizer: None)
    with pytest.raises(FloatingPointError, match='skipped'):
        update64(specialist, parent64(1))
    assert specialist.failed and specialist.completed == 1 and specialist.exposure.completed == 1
    assert specialist.model.split == original
    assert specialist.active_site is None and specialist.pending_update is None
    with pytest.raises(RuntimeError, match='terminal'):
        update64(specialist, parent64(1))


def update64(item, batch):
    return item.update([batch], batch_identities=[bind_batch_identity(batch)])


def manual_noisy_update(item, parent):
    """Independent explicit global reduction and optimizer action, same noise contract."""
    from dataclasses import asdict
    from jscc.baseline_protocol import objective_settings
    from jscc.experiment_schedule import noise_namespace
    from jscc.presentation import derived_seed
    from jscc.sharing_accumulation import micro_noise_key
    from jscc.training import batch_losses

    scheduled = item.schedule.update_at(item.completed)
    parts = partition_native64(parent, microbatch_size=item.microbatch_size)
    denominator = int((parent['labels'] != -100).sum())
    item.optimizer.zero_grad(set_to_none=True)
    losses, draws = [], []
    with item.model.at_site(item.sites[scheduled.site], purpose='train',
                            authorization={'sites': {scheduled.site: 'trained'}}):
        for index, batch in enumerate(parts):
            key = micro_noise_key(scheduled.noise_key, batch, micro=index,
                                  microbatch_size=item.microbatch_size)
            namespace = noise_namespace(key)
            generator = torch.Generator().manual_seed(derived_seed(0, namespace + ':snr'))
            snr = torch.empty((len(batch['labels']), 1, 1)).uniform_(-6, 18, generator=generator)
            with item.model.channel.replay(0, namespace, capture=True):
                values = batch_losses(item.model, batch, objective_settings('combined'), snr,
                                      return_stats=True, valid_only_kl=True)
            loss = values['kl_numerator'] / denominator + .1 * values['hidden_numerator'] / 64
            loss.backward()
            losses.append(float(loss.detach()))
            draws.append({'key': asdict(key), 'draws': list(item.model.channel.draw_summaries)})
    grads = [parameter.grad.clone() for parameter in item.parameters]
    torch.nn.utils.clip_grad_norm_(item.parameters, 1.0)
    item.optimizer.step()
    item.scheduler.step()
    item.completed += 1
    return grads, sum(losses), draws


@pytest.mark.parametrize('micro', [32, 16])
@pytest.mark.parametrize('site', ['enc_l9', 'enc_l19', 'enc_fn'])
def test_same_shape_noisy_global_gradient_single_actions(micro, site, monkeypatch):
    candidate = learner64(micro, 'specialist_' + site, reuse=True)
    reference = learner64(micro, 'specialist_' + site, reuse=False)
    candidate.audit_policy = 'full'
    initial = copy.deepcopy(candidate.model.codec.state_dict())
    assert_nested_equal(initial, reference.model.codec.state_dict())
    counts = {'clip': 0, 'optimizer': 0, 'scheduler': 0, 'encoder': 0}
    captured = []
    original_clip = torch.nn.utils.clip_grad_norm_
    original_step = type(candidate.scheduler).step
    original_encoder = candidate.model.base.get_encoder().forward
    def clip(parameters, *args, **kwargs):
        params = list(parameters)
        if params and params[0] is candidate.parameters[0]:
            counts['clip'] += 1
            captured.append([parameter.grad.clone() for parameter in params])
        return original_clip(params, *args, **kwargs)
    def scheduled_step(self, *args, **kwargs):
        if self is candidate.scheduler:
            counts['scheduler'] += 1
        return original_step(self, *args, **kwargs)
    def encode(*args, **kwargs):
        counts['encoder'] += 1
        return original_encoder(*args, **kwargs)
    def stepped(*_):
        counts['optimizer'] += 1
    handle = candidate.optimizer.register_step_post_hook(stepped)
    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', clip)
    monkeypatch.setattr(type(candidate.scheduler), 'step', scheduled_step)
    monkeypatch.setattr(candidate.model.base.get_encoder(), 'forward', encode)
    for index in range(2):
        event = update64(candidate, parent64(index))
        grads, loss, draws = manual_noisy_update(reference, parent64(index))
        assert candidate.last_draws == draws
        assert event['payload']['objective']['total']['value'] == pytest.approx(loss, abs=2e-6, rel=2e-6)
        components = event['payload']['objective']['components']
        assert components['K']['denominator'] == int((parent64(index)['labels'] != -100).sum())
        assert components['R']['denominator'] == 64
        for actual, expected in zip(captured[-1], grads):
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        assert any(bool(gradient.abs().sum()) for gradient in grads)
        for key, value in candidate.model.codec.state_dict().items():
            torch.testing.assert_close(value, reference.model.codec.state_dict()[key], atol=2e-7, rtol=2e-6)
        assert_nested_equal(candidate.scheduler.state_dict(), reference.scheduler.state_dict())
        for actual_state, expected_state in zip(candidate.optimizer.state.values(), reference.optimizer.state.values()):
            for key, value in actual_state.items():
                torch.testing.assert_close(value, expected_state[key], atol=2e-7, rtol=2e-5)
        assert counts['clip'] == counts['optimizer'] == counts['scheduler'] == index + 1
        assert all(parameter.grad is None for parameter in candidate.model.base.parameters())
        if index == 0:
            assert event['payload']['lr_used'] == [0.0]
            assert_nested_equal(candidate.model.codec.state_dict(), initial)
        else:
            assert event['payload']['lr_used'][0] > 0
            assert any(not torch.equal(value, initial[key]) for key, value in candidate.model.codec.state_dict().items())
    handle.remove()
    # l9/l19 always compute teacher and student; enc_fn reuse is scoped to EACH micro.
    assert counts['encoder'] == 2 * (64 // micro) * (1 if site == 'enc_fn' else 2)


@pytest.mark.parametrize('micro', [32, 16])
def test_full64_objective_partition_and_state_restoration(micro, monkeypatch):
    item = learner64(micro, 'specialist_enc_l19')
    update64(item, parent64())
    before = copy.deepcopy(item.model.codec.state_dict())
    optimizer = copy.deepcopy(item.optimizer.state_dict())
    scheduler = copy.deepcopy(item.scheduler.state_dict())
    seen = []
    original = item._values
    def values(batch, *args, **kwargs):
        seen.append((len(batch['labels']), list(batch['labels'].shape), args[3]))
        return original(batch, *args, **kwargs)
    monkeypatch.setattr(item, '_values', values)
    rows = item.validate_site([parent64()], 'enc_l19')
    assert [r['condition'] for r in rows] == ['no_noise', -6, 0, 6, 12, 18]
    assert len(seen) == 6 * (64 // micro)
    assert all(size == micro and shape == [micro, 4] and purpose == 'objective_validation'
               for size, shape, purpose in seen)
    assert all(r['completed'] == 64 for r in rows)
    assert_nested_equal(item.model.codec.state_dict(), before)
    assert_nested_equal(item.optimizer.state_dict(), optimizer)
    assert_nested_equal(item.scheduler.state_dict(), scheduler)
    assert item.active_site is None and item.completed == 1


@pytest.mark.parametrize('micro', [32, 16])
def test_full_saved_state_perturb_reload_replays_next_update(tmp_path, micro):
    from dataclasses import asdict
    from jscc.experiment_state import save_state, open_state, restore_state, capture_rng
    from jscc.sharing_state import build_sharing_state

    item = learner64(micro, 'specialist_enc_l19')
    item.audit_policy = 'full'
    update64(item, parent64())
    views = item.schedule.plan.views
    sharing, stream = build_sharing_state(synthetic=True, learner=item.schedule.name,
        ordered_view_identities=[view.view_sha256 for view in views], completed_updates=1,
        source_valid_per_view=[view.source_tokens for view in views],
        target_valid_per_view=[view.target_tokens for view in views],
        padded_per_view=[view.padded_tokens for view in views], batch_size=64)
    site = asdict(item.sites['enc_l19'])
    metadata = dict(source_identity='source', config_identity='config', parent_identity=None,
        initialization_identity='init', stream_identity='stream', protocol_identity='test-accum',
        completed_updates=1, phase='both', lineage=[], sharing=sharing, site=site,
        execution_partition=item.execution_partition,
        evaluation_sites={'enc_l19': {'site': site, 'role': 'trained'}})
    reference = save_state(tmp_path / 'state.pt', model=item.model.codec,
        optimizer=item.optimizer, scheduler=item.scheduler, scaler=item.scaler,
        metadata=metadata, stream_state=stream)
    saved = open_state(reference, expected=metadata)
    counters = {key: copy.deepcopy(getattr(item, key)) for key in
                ('completed', 'attempted', 'offset', 'valid_tokens', 'exposure')}
    update64(item, parent64(1))
    expected = (copy.deepcopy(item.model.codec.state_dict()), copy.deepcopy(item.optimizer.state_dict()),
                copy.deepcopy(item.scheduler.state_dict()), copy.deepcopy(item.last_draws))
    with torch.no_grad():
        for parameter in item.model.codec.parameters():
            parameter.add_(4)
        for state in item.optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    value.add_(3)
    item.scheduler.last_epoch += 5
    torch.manual_seed(123456)
    restored = restore_state(saved, model=item.model.codec, optimizer=item.optimizer,
                             scheduler=item.scheduler, scaler=item.scaler)
    assert restored == stream
    assert torch.equal(capture_rng()['torch'], saved.payload['rng']['torch'])
    assert_nested_equal(item.model.codec.state_dict(), saved.payload['model'])
    assert_nested_equal(item.optimizer.state_dict(), saved.payload['optimizer'])
    assert_nested_equal(item.scheduler.state_dict(), saved.payload['scheduler'])
    for key, value in counters.items():
        setattr(item, key, value)
    update64(item, parent64(1))
    for actual, wanted in zip((item.model.codec.state_dict(), item.optimizer.state_dict(),
                               item.scheduler.state_dict(), item.last_draws), expected):
        assert_nested_equal(actual, wanted)


@pytest.mark.parametrize('micro', [32, 16])
def test_micro_failure_has_no_optimizer_action_and_clears_reuse(micro, monkeypatch):
    from jscc import sharing_protocol

    item = learner64(micro, 'specialist_enc_fn', reuse=True)
    original_forward = item.model.base.get_encoder().forward
    original_losses = sharing_protocol.batch_losses
    calls = 0
    def fail_second(model, batch, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('injected second micro failure')
        return original_losses(model, batch, *args, **kwargs)
    monkeypatch.setattr(sharing_protocol, 'batch_losses', fail_second)
    with pytest.raises(RuntimeError, match='second micro'):
        update64(item, parent64())
    assert calls == 2 and item.failed
    assert item.completed == item.exposure.completed == item.scheduler.last_epoch == 0
    assert not item.optimizer.state
    assert item.model.base.get_encoder().forward == original_forward
    assert item.active_site is item.pending_update is None
    with pytest.raises(RuntimeError, match='terminal'):
        update64(item, parent64())
    assert calls == 2
