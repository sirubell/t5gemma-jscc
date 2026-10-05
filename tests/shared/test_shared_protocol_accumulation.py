"""Current primitive assertions extracted without resurrecting legacy execution."""

import copy
from typing import Any, cast
import json
from pathlib import Path
import pytest
import torch
from jscc.activation_replay import file_digest
from jscc.shared_inputs import batch_identity, read_prepared_batch
from jscc.batch_partition import partition_native64
from jscc.shared_inputs import bind_batch_identity
from shared.test_recipe_training import setup_recipe

REAL_INPUTS = Path("/Users/tim_c_wang/Documents/Codex/2026-09-30/task-11/sharing-production/inputs")


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
    assert not torch.cuda.is_initialized()


def parent64(index=0) -> dict[str, Any]:
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


def assert_partition(parent: dict[str, Any], micro):
    before = batch_identity(parent)
    parts: list[dict[str, Any]] = partition_native64(parent, microbatch_size=micro)
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


# Native16 current consumer replacements for numerical action/failure invariants.
# These do not claim successful effective64 qualification or micro equivalence.


def test_current_native16_single_gradient_optimizer_scheduler_actions(tmp_path, monkeypatch):
    learner, batch, events = setup_recipe(tmp_path, locations=['enc_fn'], q=8)
    counts = dict(clip=0, optimizer=0, scheduler=0)
    gradients = []
    original_clip = torch.nn.utils.clip_grad_norm_
    original_scheduler = learner.scheduler.step
    def clip(parameters, *args, **kwargs):
        parameters = list(parameters)
        counts['clip'] += 1
        gradients.append([p.grad.clone() for p in parameters if p.grad is not None])
        return original_clip(parameters, *args, **kwargs)
    def schedule(*args, **kwargs):
        counts['scheduler'] += 1
        return original_scheduler(*args, **kwargs)
    def stepped(*_):
        counts['optimizer'] += 1
    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', clip)
    monkeypatch.setattr(learner.scheduler, 'step', schedule)
    handle = learner.optimizer.register_step_post_hook(stepped)
    try:
        event = learner.update([batch], batch_identities=[bind_batch_identity(batch)])
    finally:
        handle.remove()
    assert counts == dict(clip=1, optimizer=1, scheduler=1)
    assert gradients and any(bool(g.abs().sum()) for g in gradients[0])
    assert all(p.grad is None for p in learner.model.base.parameters())
    assert event['payload']['objective']['components']['hidden']['denominator'] == 16
    assert event['payload']['exposure']['sequences'] == 16
    assert len(events) == learner.completed == learner.exposure.completed == 1
    assert learner.active_site is learner.pending_update is None


def test_current_native16_failure_has_no_optimizer_action_and_is_terminal(tmp_path, monkeypatch):
    learner, batch, events = setup_recipe(tmp_path, locations=['enc_fn'], q=8)
    route = dict(learner.model.split)
    weights = copy.deepcopy(learner.model.codec.state_dict())
    def fail(*args, **kwargs):
        raise RuntimeError('injected complete native view failure')
    monkeypatch.setattr(learner, '_values', fail)
    with pytest.raises(RuntimeError, match='native view failure'):
        learner.update([batch], batch_identities=[bind_batch_identity(batch)])
    assert learner.failed
    assert learner.completed == learner.exposure.completed == learner.scheduler.last_epoch == 0
    assert not learner.optimizer.state
    assert len(events) == 1 and events[0]["event_type"] == "failure"
    assert not any(event["event_type"] == "update" for event in events)
    assert_nested_equal(learner.model.codec.state_dict(), weights)
    assert learner.model.split == route
    assert learner.active_site is learner.pending_update is None
    with pytest.raises(RuntimeError, match='terminal'):
        learner.update([batch])


def test_current_native16_bound_metadata_rejects_before_any_training(tmp_path):
    learner, batch, events = setup_recipe(tmp_path, locations=['enc_fn'], q=8)
    binding = bind_batch_identity(batch)
    before = copy.deepcopy(learner.model.codec.state_dict())
    cast(dict[str, Any], batch)['source_views'] = [{'source': 'changed-after-binding'}]
    with pytest.raises(ValueError, match='changed after identity binding'):
        learner.update([batch], batch_identities=[binding])
    assert learner.completed == learner.attempted == learner.exposure.completed == 0
    assert not learner.optimizer.state and not events
    assert_nested_equal(before, learner.model.codec.state_dict())
    assert learner.active_site is learner.pending_update is None


def test_current_native16_paired_draws_and_gradient_updates(tmp_path):
    first_root, second_root = tmp_path/'first', tmp_path/'second'
    first_root.mkdir()
    second_root.mkdir()
    first, batch, _ = setup_recipe(first_root, locations=['enc_fn'], q=8)
    second, paired_batch, _ = setup_recipe(second_root, locations=['enc_fn'], q=8)
    first.audit_policy = second.audit_policy = 'full'
    for _ in range(2):
        first_event = first.update([batch])['payload']
        second_event = second.update([paired_batch])['payload']
        assert_nested_equal(first.last_draws, second.last_draws)
        assert len(first.last_draws) == 1
        assert_nested_equal(first.model.codec.state_dict(), second.model.codec.state_dict())
        assert_nested_equal(first.optimizer.state_dict(), second.optimizer.state_dict())
        assert_nested_equal(first.scheduler.state_dict(), second.scheduler.state_dict())
        assert first_event['objective'] == second_event['objective']
        assert all(parameter.grad is None for parameter in first.model.base.parameters())
