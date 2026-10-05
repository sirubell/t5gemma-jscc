"""Numerical and state qualification leaves; no launch authority."""





from __future__ import annotations
from contextlib import contextmanager
import copy
from dataclasses import asdict
from pathlib import Path
import hashlib
import random
from unittest.mock import patch
import numpy as np
import torch
from jscc.shared_inputs import read_prepared_batch, bind_batch_identity
from jscc.experiment_state import capture_rng, restore_rng, save_state, open_state, restore_state
from jscc.models.split_model import stack_module
from jscc.presentation import tensor_digest
from typing import Any
from jscc.experiment_records import write_json_atomic
import json
import os
from .activation_replay import canonical_digest, file_digest



# Transferred from sharing_qualification_guard; serialized meanings remain unchanged.
def require(value, message):
    if not value:
        raise ValueError(message)


def state_exact(left, right):
    if type(left) is not type(right):
        return False
    if torch.is_tensor(left):
        return left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(state_exact(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(state_exact(a, b) for a, b in zip(left, right))
    return left == right


def state_close(left, right):
    if type(left) is not type(right):
        return False
    if torch.is_tensor(left):
        if left.shape != right.shape or left.dtype != right.dtype:
            return False
        return bool(torch.allclose(left, right, rtol=.03, atol=.0005)) if left.is_floating_point() else torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(state_close(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(state_close(a, b) for a, b in zip(left, right))
    return state_exact(left, right)


def snapshot(learner):
    """CPU state copies; backbone is frozen and is never copied or replaced."""

    require(not learner.model._routing_active and not learner.model._forward_active,
            'cannot snapshot live site routing')
    return {'codec': cpu(learner.model.codec.state_dict()),
            'optimizer': cpu(learner.optimizer.state_dict()),
            'scheduler': copy.deepcopy(learner.scheduler.state_dict()),
            'scaler': copy.deepcopy(learner.scaler.state_dict()), 'rng': capture_rng(),
            'counters': [learner.completed, learner.attempted, learner.offset, learner.valid_tokens],
            'exposure': copy.deepcopy({k: v for k, v in vars(learner.exposure).items() if k != 'schedule'}),
            'failed': learner.failed, 'last_draws': copy.deepcopy(getattr(learner, 'last_draws', None)),
            'grads': [None if p.grad is None else p.grad.detach().cpu().clone() for p in learner.parameters],
            'training': learner.model.training,
            'channel_uses': copy.deepcopy(learner.model.channel_uses),
            'channel_uses_valid': copy.deepcopy(learner.model.channel_uses_valid),
            'channel_draws': copy.deepcopy(learner.model.channel.draw_summaries)}


def restore(learner, saved):
    require(not learner.model._routing_active and not learner.model._forward_active,
            'cannot restore live site routing')
    learner.model.codec.load_state_dict(saved['codec'])
    learner.optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
    learner.scheduler.load_state_dict(saved['scheduler'])
    learner.scaler.load_state_dict(saved['scaler'])
    learner.completed, learner.attempted, learner.offset, learner.valid_tokens = saved['counters']
    for key, value in saved['exposure'].items():
        setattr(learner.exposure, key, copy.deepcopy(value))
    learner.failed = saved['failed']
    if saved['last_draws'] is None:
        learner.__dict__.pop('last_draws', None)
    else:
        learner.last_draws = copy.deepcopy(saved['last_draws'])
    learner.__dict__.pop('last_failure', None)
    learner.active_site = learner.active_local = learner.pending_update = None
    for parameter, gradient in zip(learner.parameters, saved['grads']):
        parameter.grad = None if gradient is None else gradient.to(parameter.device).clone()
    learner.model.train(saved['training'])
    learner.model.activation = learner.model.reconstruction = None
    learner.model.memory_activation = learner.model.memory_reconstruction = None
    learner.model.channel_uses = copy.deepcopy(saved['channel_uses'])
    learner.model.channel_uses_valid = copy.deepcopy(saved['channel_uses_valid'])
    learner.model.channel.draw_summaries = copy.deepcopy(saved['channel_draws'])
    restore_rng(saved['rng'])


def comparable_state(saved):
    """Receipt timestamps vary physically; exposure totals and receipt counts do not."""
    result = copy.deepcopy(saved)
    result['exposure']['receipts'] = len(result['exposure']['receipts'])
    return result


def routing_state(model):
    return {'split': copy.deepcopy(model.split), 'bypass': model.bypass,
            'routing': model._routing_active, 'forward': model._forward_active,
            'hooks': [(name, len(module._forward_hooks), len(module._forward_pre_hooks))
                      for name, module in model.named_modules()],
            'encoder_forward': model.base.get_encoder().forward}


@contextmanager
def capture_accumulated(learner):
    """Independently capture all four teacher/features, draws and preclip gradient."""
    from jscc.checkpoint_readers import NATIVE16_PROTOCOL
    audit = {'microbatches': [], 'actions': {'clip': 0, 'optimizer': 0, 'scheduler': 0}}
    original_values, original_unscale = learner._values, learner.scaler.unscale_
    original_policy = learner.audit_policy
    clip, schedule = torch.nn.utils.clip_grad_norm_, learner.scheduler.step

    def values(batch, kind, step, micro, purpose, *args, **kwargs):
        require(purpose == 'train', 'guard captures training only')
        row = {'micro': micro, 'batch_sha256': bind_batch_identity(batch).complete_sha256}
        labels = batch['labels']
        site = learner.sites[learner.active_site]
        stack = stack_module(learner.model.base, 'enc')
        module = stack.norm if site.where == 'after_final_norm' else stack.layers[site.index]

        def feature(_module, _args, output):
            if learner.model.bypass:
                require('teacher_feature_sha256' not in row, 'duplicate teacher feature')
                hidden = output[0] if isinstance(output, tuple) else output
                row['teacher_feature_sha256'] = tensor_digest(hidden.detach())

        def head(_module, _args, output):
            if learner.model.bypass:
                require('teacher_valid_logits_sha256' not in row, 'duplicate teacher logits')
                row['teacher_valid_logits_sha256'] = tensor_digest(output.detach()[(labels != -100).to(output.device)])

        handles = [module.register_forward_hook(feature), learner.model.base.lm_head.register_forward_hook(head)]
        try:
            result = original_values(batch, kind, step, micro, purpose, *args, **kwargs)
            row['feature_sha256'] = tensor_digest(learner.model.activation)
            row['snr_sha256'] = tensor_digest(result[1])
            row['draws'] = copy.deepcopy(result[2])
            require('teacher_feature_sha256' in row and 'teacher_valid_logits_sha256' in row,
                    'missing teacher capture')
            audit['microbatches'].append(row)
            return result
        finally:
            for handle in handles:
                handle.remove()

    def unscale(optimizer):
        result = original_unscale(optimizer)
        require('gradient' not in audit, 'duplicate effective gradient capture')
        require(all(p.grad is not None for p in learner.parameters), 'missing codec gradient')
        audit['gradient'] = torch.cat([p.grad.detach().float().flatten().cpu() for p in learner.parameters])
        return result

    def clipping(*args, **kwargs):
        audit['actions']['clip'] += 1
        return clip(*args, **kwargs)

    def scheduling(*args, **kwargs):
        audit['actions']['scheduler'] += 1
        return schedule(*args, **kwargs)

    def stepped(*_):
        audit['actions']['optimizer'] += 1

    handle = learner.optimizer.register_step_post_hook(stepped)
    learner._values, learner.scaler.unscale_, learner.audit_policy = values, unscale, 'full'
    try:
        with patch('torch.nn.utils.clip_grad_norm_', clipping), patch.object(learner.scheduler, 'step', scheduling):
            yield audit
        expected_micros = 1 if learner.schedule.plan.protocol_id == NATIVE16_PROTOCOL else 4
        require([r['micro'] for r in audit['microbatches']] == list(range(expected_micros))
                and 'gradient' in audit, 'incomplete physical-batch audit')
        require(audit['actions'] == {'clip': 1, 'optimizer': 1, 'scheduler': 1},
                'effective update action counts differ')
    finally:
        handle.remove()
        learner._values, learner.scaler.unscale_, learner.audit_policy = original_values, original_unscale, original_policy


def audited_update(learner, parent):
    with capture_accumulated(learner) as audit:
        event = learner.update([parent], batch_identities=[bind_batch_identity(parent)])
    objective = event['payload']['objective']
    audit['losses'] = {k: objective['components'][k]['raw']['value'] for k in ('K', 'R')}
    audit['losses']['total'] = objective['total']['value']
    audit['denominators'] = {k: objective['components'][k]['denominator'] for k in ('K', 'R')}
    audit['lr_used'] = event['payload']['lr_used']
    return audit


def compare_accumulated(reference, actual, *, expected_micros=4):
    target, gradient = reference['gradient'], actual['gradient']
    shape = target.shape == gradient.shape
    finite = bool(torch.isfinite(target).all() and torch.isfinite(gradient).all())
    failures = int((abs(gradient - target) > .0005 + .03 * abs(target)).sum()) if shape else -1
    micros = (len(reference['microbatches']) == len(actual['microbatches']) == expected_micros
              and state_exact(reference['microbatches'], actual['microbatches']))
    objective = all(abs(actual['losses'][k] - reference['losses'][k])
                    <= .0005 + .005 * abs(reference['losses'][k]) for k in ('K', 'R', 'total'))
    denominators = reference['denominators'] == actual['denominators']
    actions = reference['actions'] == actual['actions'] == {'clip': 1, 'optimizer': 1, 'scheduler': 1}
    lr = reference['lr_used'] == actual['lr_used']
    return {'passed': shape and finite and failures == 0 and micros and objective and denominators and actions and lr,
            'gradient_failed_coordinates': failures, 'gradient_finite': finite,
            'gradient_relative_l2': float(torch.linalg.vector_norm(gradient - target)
                / torch.linalg.vector_norm(target).clamp_min(1e-30)) if shape else None,
            'all_four_micro_features_teachers_snr_draws_exact': micros and expected_micros == 4,
            'all_physical_features_teachers_snr_draws_exact': micros,
            'objective_tolerance_passed': objective, 'denominators_exact': denominators,
            'one_clip_optimizer_scheduler': actions, 'lr_exact': lr}


def checkpoint_roundtrip(learner, *, checkpoint_path, metadata):
    """Save full validated sharing state, perturb live values, reload durable bytes."""
    from jscc.checkpoint_readers import _controls
    from jscc.checkpoint_readers import build_sharing_state
    plan, name = learner.schedule.plan, learner.schedule.name
    sharing, stream = build_sharing_state(synthetic=plan.synthetic, learner=name,
        ordered_view_identities=[v.view_sha256 for v in plan.views], completed_updates=learner.completed,
        source_valid_per_view=[v.source_tokens for v in plan.views],
        target_valid_per_view=[v.target_tokens for v in plan.views],
        padded_per_view=[v.padded_tokens for v in plan.views], batch_size=plan.batch_size,
        protocol_id=plan.protocol_id)
    sites = {key: learner.sites[key] for key in ('enc_l9', 'enc_l19', 'enc_fn')} if name == 'shared' else {
        name.removeprefix('specialist_'): learner.sites[name.removeprefix('specialist_')]}
    details = {**metadata, 'parent_identity': None, 'lineage': [], 'phase': 'both',
        'sharing': sharing, 'completed_updates': learner.completed,
        'evaluation_sites': {key: {'site': asdict(site), 'role': 'trained'} for key, site in sites.items()},
        'snapshot_role': 'trained', 'execution_partition': learner.execution_partition,
        'model_state_contract': 'codec-only-stateless-channel-v1', 'numerical_policy': 'native'}
    if name != 'shared':
        details['site'] = asdict(next(iter(sites.values())))
    else:
        details['evaluation_sites']['enc_l14'] = {
            'site': asdict(learner.sites['enc_l14']), 'role': 'heldout_after_freeze'}
    details['comparison_controls'] = _controls(learner, details)
    before = snapshot(learner)
    reference = save_state(Path(checkpoint_path), model=learner.model.codec,
        optimizer=learner.optimizer, scheduler=learner.scheduler, scaler=learner.scaler,
        metadata=details, stream_state=stream)
    with torch.no_grad():
        for parameter in learner.model.codec.parameters():
            parameter.add_(1)
        for values in learner.optimizer.state.values():
            for value in values.values():
                if torch.is_tensor(value):
                    value.zero_()
    learner.scheduler.last_epoch = -1
    scaler = learner.scaler.state_dict()
    if scaler:
        scaler['scale'] *= 2
        learner.scaler.load_state_dict(scaler)
    learner.completed = learner.attempted = learner.offset = learner.valid_tokens = -1
    learner.exposure.completed = -1
    for counts in learner.exposure.per_site.values():
        for field in counts:
            counts[field] = -1
    random.random()
    np.random.rand()
    torch.rand(1)
    device = next(learner.model.base.parameters()).device
    if device.type == 'cuda':
        torch.rand(1, device=device)
    opened = open_state(reference, expected=details)
    restored = restore_state(opened, model=learner.model.codec, optimizer=learner.optimizer,
                             scheduler=learner.scheduler, scaler=learner.scaler)
    require(restored == stream, 'durable checkpoint stream differs')
    learner.completed = learner.attempted = restored['completed_updates']
    learner.offset = restored['offset']
    learner.valid_tokens = sum(opened.payload['metadata']['sharing']['source_valid_per_site'].values())
    learner.exposure.completed = restored['completed_updates']
    sharing = opened.payload['metadata']['sharing']
    learner.exposure.per_site = {
        site: {'updates': count, 'sequences': count * sharing['batch_size'],
               'source_tokens': sharing['source_valid_per_site'][site],
               'target_tokens': sharing['target_valid_per_site'][site],
               'padded_tokens': sharing['padded_per_site'][site]}
        for site, count in sharing['completed_per_site'].items()}
    require(state_exact(before, snapshot(learner)), 'checkpoint complete state/RNG reload differs')
    return {'exact': True, 'sha256': reference.sha256, 'path': reference.path, 'size': reference.size}


def _receipt(audit):
    return {**{k: v for k, v in audit.items() if k != 'gradient'},
            'preclip_gradient_sha256': tensor_digest(audit['gradient']),
            'preclip_gradient_l2': float(torch.linalg.vector_norm(audit['gradient']))}


def same_shape_guard(reference, candidate, parents, *, checkpoint_path, metadata, before_update=lambda: None):
    """Exactly ref2 + candidate2 + replay1; restore true zero even with shared model."""
    from jscc.checkpoint_readers import NATIVE16_PROTOCOL
    require(len(parents) == 2, 'guard requires exactly two ordered parent batches')
    site = candidate.schedule.name.removeprefix('specialist_')
    require(site in ('enc_l9', 'enc_l19', 'enc_fn') and reference.schedule == candidate.schedule,
            'guard requires matching trained-site specialist schedules')
    native16 = candidate.schedule.plan.protocol_id == NATIVE16_PROTOCOL
    require(reference.microbatch_size == candidate.microbatch_size == (None if native16 else 16)
            and reference.effective_batch == candidate.effective_batch == (16 if native16 else 64),
            'guard requires explicit native16 or16x4')
    require(not reference.enc_fn_reuse and candidate.enc_fn_reuse == (not native16 and site == 'enc_fn'),
            'reference must be generic; candidate cache is enc_fn only')
    zeros = snapshot(reference), snapshot(candidate)
    require(state_exact(comparable_state(zeros[0]), comparable_state(zeros[1]))
            and zeros[0]['counters'] == [0, 0, 0, 0] and not zeros[0]['failed'], 'guard requires identical true-zero state')
    routing = routing_state(reference.model), routing_state(candidate.model)
    succeeded = False

    def call(learner, index):
        before_update()
        return audited_update(learner, parents[index])

    try:
        restore(reference, zeros[0])
        ref_first = call(reference, 0)
        require(ref_first['lr_used'] == [0.0]
                and state_exact(zeros[0]['codec'], snapshot(reference)['codec'])
                and bool(reference.optimizer.state), 'first update must advance Adam at zero LR')
        ref_second = call(reference, 1)
        ref_end = snapshot(reference)
        restore(candidate, zeros[1])
        cand_first = call(candidate, 0)
        after_first = snapshot(candidate)
        cand_second = call(candidate, 1)
        terminal = snapshot(candidate)
        comparisons = [compare_accumulated(ref_first, cand_first, expected_micros=1 if native16 else 4),
                      compare_accumulated(ref_second, cand_second, expected_micros=1 if native16 else 4)]
        require(all(row['passed'] for row in comparisons), 'same-shape16x4 numerical guard failed')
        require(all(lr > 0 for lr in cand_second['lr_used'])
                and not state_exact(zeros[1]['codec'], terminal['codec']), 'second update must change weights at nonzero LR')
        require(state_close(comparable_state(ref_end), comparable_state(terminal)),
                'reference/candidate optimizer/scheduler/RNG/exposure state differs')
        checkpoint = checkpoint_roundtrip(candidate, checkpoint_path=checkpoint_path, metadata=metadata)
        restore(candidate, after_first)
        replay = call(candidate, 1)
        require(state_exact(comparable_state(terminal), comparable_state(snapshot(candidate)))
                and state_exact(cand_second, replay), 'restored second update differs')
        succeeded = True
        return {'status': 'passed', 'site': site, 'physical_updates': 5, 'comparisons': comparisons,
                'reference_candidate_terminal_state_exact': state_exact(comparable_state(ref_end), comparable_state(terminal)),
                'reference_candidate_terminal_state_tolerance_passed': True,
                'restored_second_update_exact': True, 'nonzero_lr_update_changed_weights': True,
                'checkpoint': checkpoint,
                'audits': {name: _receipt(value) for name, value in zip(
                    ('reference_first', 'reference_second', 'candidate_first', 'candidate_second', 'replay_second'),
                    (ref_first, ref_second, cand_first, cand_second, replay))}}
    finally:
        restore(reference, zeros[0])
        restore(candidate, zeros[1])
        if not succeeded:
            reference.failed = candidate.failed = True
        require(routing_state(reference.model) == routing[0] and routing_state(candidate.model) == routing[1],
                'guard leaked site hooks or encoder cache wrapper')




@contextmanager
def reuse_encoder(model):
    """Reuse exactly one teacher enc_fn output inside one teacher/student update."""
    if (
        model.split != {"stack": "enc", "where": "after_final_norm"}
        or model.numerical_policy != "native"
    ):
        raise ValueError("diagnostic reuse requires native enc_fn")
    encoder = model.base.get_encoder()
    original = encoder.forward
    cached = None
    calls = {"computed": 0, "reused": 0}
    signature = None

    def inputs(args, kwargs):
        def binding(value):
            if torch.is_tensor(value):
                return (
                    id(value),
                    value._version,
                    value.data_ptr(),
                    tuple(value.shape),
                    value.dtype,
                    value.device,
                )
            return value

        return tuple(binding(v) for v in args), tuple(
            (k, binding(v)) for k, v in sorted(kwargs.items())
        )

    def forward(*args, **kwargs):
        nonlocal cached, signature
        if model.bypass:
            if cached is not None or encoder.training or torch.is_grad_enabled():
                raise ValueError("expected one frozen no-grad teacher encoder")
            cached = original(*args, **kwargs)
            if (
                not hasattr(cached, "last_hidden_state")
                or cached.last_hidden_state.requires_grad
            ):
                raise ValueError("expected detached encoder ModelOutput")
            signature = inputs(args, kwargs)
            calls["computed"] += 1
            return cached
        if cached is None or calls["reused"] or signature != inputs(args, kwargs):
            raise ValueError("student encoder inputs differ or cache crossed update")
        # transmission() supplies the same public mask even when first-layer
        # capture is skipped. The normal roundtrip retains codec/channel graph.
        received = model._roundtrip(cached.last_hidden_state)
        calls["reused"] += 1
        return type(cached)(**{**dict(cached), "last_hidden_state": received})

    encoder.forward = forward
    try:
        yield calls
        if calls != {"computed": 1, "reused": 1}:
            raise ValueError(
                "encoder reuse did not execute one teacher and one student"
            )
    finally:
        encoder.forward = original
        cached = None


def cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [cpu(item) for item in value]
    return value


def exact(left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            exact(left[k], right[k]) for k in left
        )
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right))
    return left == right


@contextmanager
def capture(learner, batch):
    """Restore-only audit, excluded from timing; full hashes, actual preclip gradient."""
    result = {}
    labels = batch["labels"]

    def head(_module, _args, output):
        if learner.model.bypass:
            result["teacher_valid_logits_sha256"] = tensor_digest(
                output.detach()[labels != -100]
            )

    handle = learner.model.base.lm_head.register_forward_hook(head)
    original = learner.scaler.unscale_

    def unscale(optimizer):
        returned = original(optimizer)
        result["gradient"] = torch.cat(
            [p.grad.detach().float().flatten().cpu() for p in learner.parameters]
        )
        return returned

    learner.scaler.unscale_ = unscale
    previous = learner.audit_policy
    learner.audit_policy = "full"
    try:
        yield result
        result["feature_sha256"] = tensor_digest(learner.model.activation)
        result["draws"] = copy.deepcopy(learner.last_draws)
    finally:
        learner.audit_policy = previous
        learner.scaler.unscale_ = original
        handle.remove()


def native64_same_shape_guard(learner, references, root, output, resource_guard):
    from .shared_training import RecipeLearner
    """Two native baseline calls and one restored C call; then reset to true init."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    zero = cpu(native64_state(learner))
    report = {
        "status": "incomplete",
        "optimizer_calls": 0,
        "formal_updates": 0,
        "gate": "native64_same_shape_v1",
        "cross_shape_gate": "not_applicable_no_microbatch",
    }
    try:
        resource_guard()
        batch, binding = native64_load_cpu(references[0][0], root, True)
        batch, binding = native64_move(
            batch, binding, next(learner.model.base.parameters()).device, True
        )
        RecipeLearner._update_complete_view(learner, [batch], batch_identities=[binding])
        report["optimizer_calls"] += 1
        step1 = cpu(native64_state(learner))
        if not exact(zero["codec"], step1["codec"]) or not step1["optimizer"]["state"]:
            raise ValueError(
                "zero-LR first call must preserve weights and advance moments"
            )
        torch.save(step1, output / "step1.pt")
        records = []
        for optimized in (False, True):
            resource_guard()
            native64_restore(
                learner,
                torch.load(output / "step1.pt", map_location="cpu", weights_only=True),
            )
            batch, binding = native64_load_cpu(references[1][0], root, True)
            batch, binding = native64_move(
                batch, binding, next(learner.model.base.parameters()).device, True
            )
            with capture(learner, batch) as record:
                event = (
                    learner.update([batch], batch_identities=[binding])
                    if optimized
                    else RecipeLearner._update_complete_view(
                        learner, [batch], batch_identities=[binding]
                    )
                )
            report["optimizer_calls"] += 1
            objective = event["payload"]["objective"]
            record["losses"] = {
                "K": objective["components"]["K"]["raw"]["value"],
                "R": objective["components"]["R"]["raw"]["value"],
                "total": objective["total"]["value"],
            }
            record["denominators"] = {
                k: objective["components"][k]["denominator"] for k in ("K", "R")
            }
            record["state"] = cpu(native64_state(learner))
            records.append(record)
            if any(
                p.requires_grad or p.grad is not None
                for p in learner.model.base.parameters()
            ):
                raise ValueError("backbone must be frozen and gradient-free")
            if any(
                p.grad is None or not bool(torch.isfinite(p.grad).all())
                for p in learner.parameters
            ):
                raise ValueError("every codec parameter must have a finite gradient")
        comparison = native64_compare(records[0], records[1])
        report.update(
            comparison=comparison,
            exact_restore_state=exact(records[0]["state"], records[1]["state"]),
            exact_denominators=records[0]["denominators"] == records[1]["denominators"],
            nonzero_lr_weights_changed=not exact(
                step1["codec"], records[1]["state"]["codec"]
            ),
        )
        torch.save(records, output / "paired-audits.pt")
        if not (
            comparison["passed"]
            and report["exact_restore_state"]
            and report["exact_denominators"]
            and report["nonzero_lr_weights_changed"]
        ):
            raise ValueError("per-arm native64+C correctness guard failed")
        report["status"] = "passed"
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        native64_restore(learner, zero)
        learner.prepared_binding = None
        write_json_atomic(output / "acceptance.json", report)
    return report



def native64_state(learner):
    return {
        "codec": copy.deepcopy(learner.model.codec.state_dict()),
        "optimizer": copy.deepcopy(learner.optimizer.state_dict()),
        "scheduler": copy.deepcopy(learner.scheduler.state_dict()),
        "scaler": copy.deepcopy(learner.scaler.state_dict()),
        "counters": [
            learner.completed,
            learner.attempted,
            learner.offset,
            learner.valid_tokens,
        ],
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def native64_restore(learner, saved):
    learner.model.codec.load_state_dict(saved["codec"])
    learner.optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
    learner.scheduler.load_state_dict(saved["scheduler"])
    learner.scaler.load_state_dict(saved["scaler"])
    learner.completed, learner.attempted, learner.offset, learner.valid_tokens = saved[
        "counters"
    ]
    torch.set_rng_state(saved["torch_rng"])
    if saved["cuda_rng"]:
        torch.cuda.set_rng_state_all(saved["cuda_rng"])


def native64_load_cpu(reference, prepared, optimized=False):
    if optimized:
        batch = torch.load(
            verified(prepared, reference), map_location="cpu", weights_only=True
        )
        binding = bind_batch_identity(batch, expected_identity=reference["view_sha256"])
    else:
        batch = read_prepared_batch(reference, prepared)
        binding = None
    if len(batch["labels"]) != 64:
        raise ValueError("native64 required")
    return batch, binding


def native64_move(batch, cpu_binding, device, optimized):
    if cpu_binding is not None:
        cpu_binding.validate(batch)
    batch = {
        k: v.to(device, non_blocking=optimized) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }
    if optimized:
        if cpu_binding is None:
            raise ValueError("optimized transfer requires verified CPU binding")
        # Existing token API checks versions/layout. Build its device binding
        # from the once-verified CPU content; no duplicate device->CPU hash.
        from jscc.shared_inputs import BoundBatchIdentity, _tensor_binding
        from jscc.activation_replay import canonical_digest

        binding = BoundBatchIdentity(
            cpu_binding.view_sha256,
            cpu_binding.complete_sha256,
            tuple(
                (k, _tensor_binding(v))
                for k, v in sorted(batch.items())
                if torch.is_tensor(v)
            ),
            canonical_digest(
                {k: v for k, v in batch.items() if not torch.is_tensor(v)}
            ),
        )
    else:
        binding = bind_batch_identity(batch)
    return batch, binding


def native64_compare(reference, actual):
    gradient = actual["gradient"]
    target = reference["gradient"]
    finite = bool(torch.isfinite(gradient).all())
    failures = int((abs(gradient - target) > 0.0005 + 0.03 * abs(target)).sum())
    exact_inputs = all(
        reference[k] == actual[k]
        for k in ("feature_sha256", "teacher_valid_logits_sha256", "draws")
    )
    losses = all(
        abs(actual["losses"][k] - reference["losses"][k])
        <= 0.0005 + 0.005 * abs(reference["losses"][k])
        for k in ("K", "R", "total")
    )
    return {
        "passed": finite and failures == 0 and exact_inputs and losses,
        "gradient_failed_coordinates": failures,
        "gradient_relative_l2": float(
            torch.linalg.vector_norm(gradient - target)
            / torch.linalg.vector_norm(target).clamp_min(1e-30)
        ),
        "exact_feature_teacher_draws": exact_inputs,
        "loss_tolerance_passed": losses,
    }



def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verified(root, reference):
    path = (root / reference["path"]).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or sha(path) != reference["sha256"]:
        raise ValueError("input binding differs: " + reference["path"])
    return path




def readiness_view_indices(batches):
    """Test both extreme layouts, taking the larger companion length on ties."""
    lengths = [(batch['input_ids'].shape[1], batch['labels'].shape[1]) for batch in batches]
    return {max(range(len(batches)), key=lambda i: lengths[i]),
            max(range(len(batches)), key=lambda i: tuple(reversed(lengths[i])))}


def controller_binding_identity(manifest):
    """Qualification and reservation must match these exact scientific inputs."""
    return {key: manifest[key] for key in (
        'prepared_identity', 'cell', 'source_identity', 'config_identity',
        'model_revision', 'recipe_identity', 'architecture_decision', 'protocol_id'
    )} | {
        'bottleneck_dim': manifest['resolved_config']['codec']['bottleneck_dim'],
        'partition': manifest.get('execution_partition'),
        'input_identity': canonical_digest({key: manifest[key] for key in (
            'updates', 'validation', 'geometry', 'task_request', 'data_ids')}),
        'initialization_identity': manifest['initialization']['state_identity'],
        'model_identity': canonical_digest(manifest['resolved_config']['model']),
    }


def read_controller_reference(reference):
    path = Path(reference['path'])
    if not path.is_absolute() or file_digest(path) != reference['sha256']:
        raise ValueError('absolute immutable controller reference required')
    return json.loads(path.read_text())


def output_bytes(root):
    total = 0
    for path in Path(root).rglob('*'):
        if path.is_symlink():
            raise ValueError('output symlinks forbidden')
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            pass
    return total


def check_target_device(manifest):
    """Verify actual scheduler-visible CUDA UUID before loading model weights."""
    if manifest['synthetic_cpu']:
        return
    path = os.environ.get('SHARING_CONTROLLER_CONTRACT')
    if not path or file_digest(Path(path)) != os.environ.get('SHARING_CONTROLLER_CONTRACT_SHA256'):
        raise ValueError('immutable controller required before target device check')
    contract = json.loads(Path(path).read_text())
    expected = contract.get('device_uuid')
    if expected is None:
        return  # Retained single-device legacy contracts have separate target acceptance.
    model_config = manifest['resolved_config']['model']
    if model_config['device'] != 'cuda' or model_config['dtype'] != 'bfloat16':
        raise ValueError('CUDA BF16 model configuration required')
    import torch
    if torch.cuda.device_count() != 1:
        raise ValueError('one actual scheduler-visible CUDA device required')
    properties = torch.cuda.get_device_properties(0)
    actual = str(getattr(properties, 'uuid', ''))
    if actual and not actual.startswith('GPU-'):
        actual = 'GPU-' + actual
    if actual.lower() != expected.lower():
        raise ValueError('visible CUDA UUID differs from sealed assigned device')
    if not torch.cuda.is_bf16_supported():
        raise ValueError('actual CUDA BF16 target required')
    if contract.get('entry_route') == '5090-q400-target-startup-v1':
        if str(properties.name) != 'NVIDIA GeForce RTX 5090':
            raise ValueError('q400 startup requires actual RTX 5090')
    if contract.get('device_binding_receipt') is not None:
        receipt = read_controller_reference(contract['device_binding_receipt'])
        if receipt.get('device_uuid') != expected or not str(properties.name).startswith('NVIDIA H200'):
            raise ValueError('sealed H200 allocation differs from actual CUDA device')


def checked_gradient_names(model):
    names = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise RuntimeError(f"missing trainable gradient: {name}")
        if not torch.isfinite(parameter.grad).all().item():
            raise FloatingPointError(f"nonfinite trainable gradient: {name}")
        names.append(name)
    if not names:
        raise RuntimeError("no trainable gradients")
    return names


def check_reconstruction_recipe(config: dict) -> dict:
    settings = config["training"]
    if config["task"] != "hellaswag" or config["split"] != {"stack": "enc", "where": "after_layer", "index": 9}:
        raise ValueError("two-stage pilot requires HellaSwag enc_l9")
    stream = settings.get("presentation_stream")
    if not stream or stream.get("policy") != "epoch-permutations-v1":
        raise ValueError("pilot requires a fixed presentation stream")
    updates = settings["max_steps"]
    presentations = updates * settings["batch_size"] * settings["gradient_accumulation"]
    if stream["total_presentations"] != presentations:
        raise ValueError("declared presentation count differs from update budget")
    if config["channel"]["type"] != "awgn" or config["codec"].get("snr_film", False):
        raise ValueError("pilot requires AWGN and FiLM off")
    if settings.get("paired_randomness", {}).get("policy") != "step-microbatch-stream-v1":
        raise ValueError("pilot requires explicit paired randomness")
    if settings.get("patience", 0):
        raise ValueError("fixed pilot must complete the declared terminal step")
    return {"updates": updates, "presentations": presentations,
            "batch_size": settings["batch_size"],
            "accumulation": settings["gradient_accumulation"],
            "schedule_steps": settings.get("schedule_steps", updates),
            "selection_steps": settings.get("selection_steps"),
            "presentation_seed": stream["seed"],
            "presentation_start": stream.get("start_presentation", 0),
            "noise_seed": settings["paired_randomness"]["seed"]}

