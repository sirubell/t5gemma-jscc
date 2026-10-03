"""Approved native64 enc_fn optimization, bounded input preload and same-shape guard."""

from __future__ import annotations
import copy
import hashlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any
import torch
from jscc.baseline_protocol import (
    BaselineLearner,
    bind_batch_identity,
    read_prepared_batch,
)
from jscc.presentation import tensor_digest
from jscc.experiment_records import write_json_atomic


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verified(root, reference):
    path = (root / reference["path"]).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or sha(path) != reference["sha256"]:
        raise ValueError("input binding differs: " + reference["path"])
    return path


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


def state(learner):
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


def restore(learner, saved):
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


def load_cpu(reference, prepared, optimized=False):
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


def move(batch, cpu_binding, device, optimized):
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
        from jscc.baseline_protocol import BoundBatchIdentity, _tensor_binding
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


def compare(reference, actual):
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


class Native64Learner(BaselineLearner):
    """Only the approved combined, single native batch route uses encoder reuse."""

    prepared_binding = None

    def update(
        self, batches, *, kind="combined", data_cache_seconds=0.0, batch_identities=None
    ):
        if kind != "combined" or len(batches) != 1:
            raise ValueError("native64+C only supports one combined native batch")
        if batch_identities is None:
            if self.prepared_binding is None or self.prepared_binding[0] != id(
                batches[0]
            ):
                raise ValueError("missing once-verified prepared binding")
            batch_identities = [self.prepared_binding[1]]
        self.prepared_binding = None
        with reuse_encoder(self.model):
            return super().update(
                batches,
                kind=kind,
                data_cache_seconds=data_cache_seconds,
                batch_identities=batch_identities,
            )


class PreparedUpdates:
    """One worker, at most current/next CPU batch; no input cache across arms."""

    def __init__(self, learner, references, root):
        self.learner, self.references, self.root = learner, references, root
        self.index = 0
        self.pool = None
        self.future = None

    def __enter__(self):
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = self.pool.submit(load_cpu, self.references[0][0], self.root, True)
        return self

    def __call__(self, index, kind):
        if kind != "combined" or index != self.index or self.future is None:
            raise ValueError("prepared stream must be consumed once in declared order")
        assert self.pool is not None
        batch, binding = self.future.result()
        self.index += 1
        self.future = (
            self.pool.submit(load_cpu, self.references[self.index][0], self.root, True)
            if self.index < len(self.references)
            else None
        )
        batch, binding = move(
            batch, binding, next(self.learner.model.base.parameters()).device, True
        )
        self.learner.prepared_binding = (id(batch), binding)
        return [batch]

    def __exit__(self, *_):
        assert self.pool is not None
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.future = None
        self.learner.prepared_binding = None


def same_shape_guard(learner, references, root, output, resource_guard):
    """Two native baseline calls and one restored C call; then reset to true init."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    zero = cpu(state(learner))
    report = {
        "status": "incomplete",
        "optimizer_calls": 0,
        "formal_updates": 0,
        "gate": "native64_same_shape_v1",
        "cross_shape_gate": "not_applicable_no_microbatch",
    }
    try:
        resource_guard()
        batch, binding = load_cpu(references[0][0], root, True)
        batch, binding = move(
            batch, binding, next(learner.model.base.parameters()).device, True
        )
        BaselineLearner.update(learner, [batch], batch_identities=[binding])
        report["optimizer_calls"] += 1
        step1 = cpu(state(learner))
        if not exact(zero["codec"], step1["codec"]) or not step1["optimizer"]["state"]:
            raise ValueError(
                "zero-LR first call must preserve weights and advance moments"
            )
        torch.save(step1, output / "step1.pt")
        records = []
        for optimized in (False, True):
            resource_guard()
            restore(
                learner,
                torch.load(output / "step1.pt", map_location="cpu", weights_only=True),
            )
            batch, binding = load_cpu(references[1][0], root, True)
            batch, binding = move(
                batch, binding, next(learner.model.base.parameters()).device, True
            )
            with capture(learner, batch) as record:
                event = (
                    learner.update([batch], batch_identities=[binding])
                    if optimized
                    else BaselineLearner.update(
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
            record["state"] = cpu(state(learner))
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
        comparison = compare(records[0], records[1])
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
        restore(learner, zero)
        learner.prepared_binding = None
        write_json_atomic(output / "acceptance.json", report)
    return report


def _require(value, message):
    if not value:
        raise ValueError(message)


def evaluate_vanilla_panel(model, processor, request, output):
    """Evaluate and retain the complete shared vanilla panel on CPU or CUDA.

    The compact writer still verifies observed payload counters. The condition
    describes the bypass route consistently through scoring, readback and receipt.
    """
    import torch
    from jscc.evaluation import evaluate_hellaswag, _observation_items
    from jscc.experiment_records import write_json_atomic

    _require(request["expected_items"] == 256, "development panel must contain256")
    _require(request["settings"]["num_samples"] == 256, "vanilla sample limit differs")
    directory = Path(output)
    settings = {**request["settings"], "evidence_mode": "compact-v1"}
    model.eval()
    with torch.no_grad(), model.transmission(bypass=True):
        scores = evaluate_hellaswag(
            model, processor, settings, directory, "vanilla", request["data_settings"]
        )
    items, ids, families = _observation_items(directory, "hellaswag", "vanilla")
    _require(
        ids == request["input_ids"] and families == request["source_family_ids"],
        "vanilla membership differs",
    )
    _require(len(items) == 256, "vanilla panel incomplete")
    receipt = {
        "schema": "vanilla-panel-v1",
        "requested": 256,
        "completed": len(items),
        "failed": 0,
        "condition": "vanilla",
        "codec_bypassed": True,
        "request": request,
        "metrics": scores,
        "items": items,
    }
    write_json_atomic(directory / "receipt.json", receipt)
    model.train()
    return receipt
