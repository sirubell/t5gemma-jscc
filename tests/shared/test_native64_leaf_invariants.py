"""Tiny CPU numerical leaves; no historical native64 execution policy."""

import json
from types import SimpleNamespace
import pytest
import torch
from torch.amp.grad_scaler import GradScaler
from transformers.modeling_outputs import BaseModelOutput
from jscc import qualification as leaves





@pytest.fixture(autouse=True)
def cpu_rng_boundary(monkeypatch):
    monkeypatch.setattr(leaves.torch.cuda, 'is_available', lambda: False)
    with torch.random.fork_rng(devices=[]):
        yield


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(2), requires_grad=False)
        self.calls = 0

    def get_encoder(self) -> "Encoder":
        return self

    def forward(self, inputs, attention_mask=None):
        self.calls += 1
        return BaseModelOutput(last_hidden_state=inputs @ self.weight)


class TinyModel:
    def __init__(self):
        self.encoder = Encoder().eval()
        self.base = self.encoder
        self.codec = torch.nn.Linear(2, 2)
        self.split = {'stack': 'enc', 'where': 'after_final_norm'}
        self.numerical_policy = 'native'
        self.bypass = True

    def _roundtrip(self, value):
        return self.codec(value)


def teacher(model, inputs, mask):
    with torch.no_grad():
        return model.encoder(inputs, attention_mask=mask)


def test_reuse_keeps_one_teacher_and_student_codec_gradient():
    model = TinyModel()
    original = model.encoder.forward
    inputs, mask = torch.ones(1, 2), torch.ones(1, 2)
    with leaves.reuse_encoder(model) as calls:
        output = teacher(model, inputs, mask)
        assert not output.last_hidden_state.requires_grad
        model.bypass = False
        student = model.encoder(inputs, attention_mask=mask)
        assert student.last_hidden_state.requires_grad
        student.last_hidden_state.sum().backward()
        assert calls == {'computed': 1, 'reused': 1}
    assert model.encoder.calls == 1
    assert model.encoder.forward == original
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad) > 0
               for p in model.codec.parameters())
    assert all(not p.requires_grad and p.grad is None for p in model.base.parameters())


@pytest.mark.parametrize('mutation', ['tensor_identity', 'tensor_version', 'shape', 'dtype', 'mask_identity', 'mask_version'])
def test_reuse_rejects_changed_input_binding_and_restores_forward(mutation):
    model = TinyModel()
    original = model.encoder.forward
    inputs, mask = torch.ones(1, 2), torch.ones(1, 2)
    with pytest.raises(ValueError, match='inputs differ'):
        with leaves.reuse_encoder(model):
            teacher(model, inputs, mask)
            model.bypass = False
            if mutation == 'tensor_identity':
                inputs = inputs.clone()
            elif mutation == 'tensor_version':
                inputs.add_(1)
            elif mutation == 'shape':
                inputs.data = inputs.data.reshape(2, 1)
            elif mutation == 'dtype':
                inputs = inputs.double()
            elif mutation == 'mask_identity':
                mask = mask.clone()
            else:
                mask.add_(1)
            model.encoder(inputs, attention_mask=mask)
    assert model.encoder.forward == original
    assert model.encoder.calls == 1
    assert all(p.grad is None for p in model.codec.parameters())


@pytest.mark.parametrize('failure', ['missing_student', 'missing_teacher', 'repeated_teacher', 'repeated_student', 'teacher_grad', 'teacher_training', 'injected'])
def test_reuse_lifecycle_failures_restore_original_forward(failure):
    model = TinyModel()
    original = model.encoder.forward
    inputs, mask = torch.ones(1, 2), torch.ones(1, 2)
    error = RuntimeError if failure == 'injected' else ValueError
    with pytest.raises(error):
        with leaves.reuse_encoder(model):
            if failure == 'missing_teacher':
                model.bypass = False
                model.encoder(inputs, attention_mask=mask)
            if failure == 'teacher_grad':
                model.encoder(inputs, attention_mask=mask)
            if failure == 'teacher_training':
                model.encoder.train()
            teacher(model, inputs, mask)
            if failure == 'missing_student':
                pass
            elif failure == 'repeated_teacher':
                teacher(model, inputs, mask)
            elif failure == 'injected':
                raise RuntimeError('consumer failed')
            else:
                model.bypass = False
                model.encoder(inputs, attention_mask=mask)
                if failure == 'repeated_student':
                    model.encoder(inputs, attention_mask=mask)
    assert model.encoder.forward == original


def tiny_learner():
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.codec.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    scaler = GradScaler('cpu')
    return SimpleNamespace(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                           completed=0, attempted=0, offset=0, valid_tokens=0, prepared_binding='fixture')


def tiny_update(learner):
    learner.optimizer.zero_grad()
    inputs = torch.randn(2, 2)
    loss = learner.model.codec(inputs).square().mean()
    learner.scaler.scale(loss).backward()
    learner.scaler.step(learner.optimizer)
    learner.scaler.update()
    learner.scheduler.step()
    learner.completed += 1
    learner.attempted += 1
    learner.offset += 2
    learner.valid_tokens += 4
    return loss.detach().clone()


def test_native_state_restore_replays_next_update_without_mutable_alias(tmp_path):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        learner = tiny_learner()
        tiny_update(learner)
        saved = leaves.native64_state(learner)
        torch.save(saved, tmp_path / 'state.pt')
        frozen = torch.load(tmp_path / 'state.pt', weights_only=True)
        expected_loss = tiny_update(learner)
        expected = leaves.native64_state(learner)
        assert leaves.exact(saved, frozen)
        assert saved['counters'] == [1, 1, 2, 4]
        assert saved['optimizer']['state'] and saved['scaler']
        leaves.native64_restore(learner, saved)
        assert leaves.exact(leaves.native64_state(learner), frozen)
        actual_loss = tiny_update(learner)
        assert torch.equal(actual_loss, expected_loss)
        assert leaves.exact(leaves.native64_state(learner), expected)
        assert leaves.exact(saved, frozen)
        assert all(not p.requires_grad and p.grad is None for p in learner.model.base.parameters())


def test_native_guard_resource_failure_prevents_batch_and_restores_full_state(tmp_path, monkeypatch):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        learner = tiny_learner()
        tiny_update(learner)
        initial = leaves.native64_state(learner)
        calls = []
        def resource_guard():
            calls.append('guard')
            tiny_update(learner)
            raise RuntimeError('resource denied before batch')
        monkeypatch.setattr(leaves, 'native64_load_cpu', lambda *a: pytest.fail('resource failure loaded historical batch'))
        output = tmp_path / 'failed'
        with pytest.raises(RuntimeError, match='resource denied'):
            leaves.native64_same_shape_guard(learner, [], tmp_path, output, resource_guard)
        assert calls == ['guard']
        assert leaves.exact(leaves.native64_state(learner), initial)
        assert learner.prepared_binding is None
        report = json.loads((output / 'acceptance.json').read_text())
        assert report['status'] == 'incomplete'
        assert report['optimizer_calls'] == report['formal_updates'] == 0
        assert report['error'] == {'type': 'RuntimeError', 'message': 'resource denied before batch'}
        assert not (output / 'step1.pt').exists()
        assert not (output / 'paired-audits.pt').exists()
