"""Versioned baseline learner: online functional and complete-sequence local work.

This module emits plain records. Durable records integration and scientific
package acceptance are separate gates; it never chooses an architecture.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import time
from pathlib import Path

import torch
from torch.amp.grad_scaler import GradScaler

from .activation_replay import canonical_digest, file_digest
from .local_reconstruction import local_batch_values
from .models.channel import AWGNChannel
from .presentation import derived_seed, tensor_digest
from .runtime import prepare_trainable_parameters
from .experiment_records import (append_event, artifact_ref, completion_status,
                                 observation_identity, read_events, write_json_atomic)
from .experiment_state import isolated_rng
from .experiment_schedule import NoiseKey, noise_namespace, build_baseline_scheduler
from .training import batch_losses
from .training_objectives import (
    aggregate_batch_losses, detached_batch_values, effective_batch_denominators,
    scaled_batch_loss,
)

CONDITIONS = ("no_noise", -6, 0, 6, 12, 18)


def measure(value=None, reason=None):
    if value is None and not reason:
        raise ValueError("unavailable measurement requires a reason")
    return {"value": value, "reason": reason}


def objective_settings(kind):
    if kind not in {"combined", "local"}:
        raise ValueError("objective must be combined or local; cached functional suffix is disabled")
    return {"temperature": 1.0, "kl_weight": 1.0 if kind == "combined" else 0.0,
            "mse_weight": 0.1, "loss_weights": {
                "kl": 1.0 if kind == "combined" else 0.0, "hidden": 0.1, "memory": 0.0}}


def batch_identity(batch):
    """Bind complete padded inputs and view metadata, including image tensors."""
    return canonical_digest({key: tensor_digest(value) if torch.is_tensor(value) else value
                             for key, value in batch.items() if key not in {"activation", "site_id"}})


def _tensor_binding(value):
    return (id(value), value._version, tuple(value.shape), tuple(value.stride()),
            str(value.dtype), str(value.device), value.data_ptr())


@dataclass(frozen=True)
class BoundBatchIdentity:
    """A complete prepared-batch binding for reuse outside the update timer.

    Keep the bound tensors read-only: ordinary in-place mutation, replacement,
    layout changes, and metadata edits invalidate this token. Unsafe writes
    through ``.data`` or a foreign storage alias are outside this contract.
    The complete digest also binds replay activation/site provenance, while
    the view digest preserves the established paired-noise namespace.
    """
    view_sha256: str
    complete_sha256: str
    _tensor_bindings: tuple
    _metadata_sha256: str

    def validate(self, batch):
        tensors = tuple((key, _tensor_binding(value)) for key, value in sorted(batch.items())
                        if torch.is_tensor(value))
        metadata = canonical_digest({key: value for key, value in batch.items()
                                     if not torch.is_tensor(value)})
        if tensors != self._tensor_bindings or metadata != self._metadata_sha256:
            raise ValueError("prepared batch changed after identity binding")
        return self.view_sha256


def bind_batch_identity(batch, *, expected_identity=None):
    """Hash each complete tensor once, before timed updates; verify a known view."""
    contents = {key: tensor_digest(value) if torch.is_tensor(value) else value
                for key, value in batch.items()}
    view = canonical_digest({key: value for key, value in contents.items()
                             if key not in {"activation", "site_id"}})
    if expected_identity is not None and view != expected_identity:
        raise ValueError("prepared input/view identity mismatch")
    return BoundBatchIdentity(view, canonical_digest(contents),
        tuple((key, _tensor_binding(value)) for key, value in sorted(batch.items())
              if torch.is_tensor(value)),
        canonical_digest({key: value for key, value in batch.items() if not torch.is_tensor(value)}))


def _check_batches(batches, kind, expected_sequences=None):
    if not batches:
        raise ValueError("empty effective batch")
    count = 0
    for batch in batches:
        mask, labels = batch["attention_mask"], batch["labels"]
        if mask.ndim != 2 or labels.ndim != 2 or len(mask) != len(labels):
            raise ValueError("complete source/target sequences required")
        if not bool(((mask == 0) | (mask == 1)).all()) or not bool(mask.bool().any(1).all()):
            raise ValueError("source mask must be binary with no empty real sequence")
        if not bool((labels != -100).any(1).all()):
            raise ValueError("target sequence has no valid native-reference position")
        if batch["input_ids"].shape != mask.shape:
            raise ValueError("source layout differs from mask")
        if kind == "local":
            activation = batch.get("activation")
            if activation is None or activation.ndim != 3 or activation.shape[:2] != mask.shape:
                raise ValueError("local learning requires complete-sequence replay")
        count += len(mask)
    if expected_sequences is not None and count != expected_sequences:
        raise ValueError("effective-batch sequence count differs from frozen plan")
    return count


def _objective_record(totals, kind):
    def component(name, measured=True):
        if not measured:
            return {"raw": measure(reason="unmeasured"), "weighted": measure(reason="unmeasured"),
                    "numerator": measure(reason="unmeasured"), "denominator": 0}
        stat = "kl" if name == "K" else "hidden"
        return {"raw": measure(float(totals[stat])),
                "weighted": measure(float(totals["weighted_" + stat])),
                "numerator": measure(float(totals[stat + "_numerator"])),
                "denominator": int(totals[stat + "_denominator"])}
    return {"kind": kind, "total": measure(float(totals["loss"])),
            "components": {"K": component("K", kind == "combined"), "R": component("R")}}


class BaselineLearner:
    """A finite, fail-stop learner with paired draws and explicit stream position.

    The caller supplies verified ordered batches (online inputs or v2 replay).
    A failed update cannot be retried on this object. Saved state is handled by
    experiment_state; ordinary legacy train/resume remains unchanged.
    """
    def __init__(self, model, *, run_id, task, identity, pairing_id, final_step=400,
                 effective_batch=64, lr=2e-4, weight_decay=0.01, grad_clip=1.0,
                 event_sink=None, synthetic=False, audit_policy="full"):
        if audit_policy not in {"full", "sparse_first_final"}:
            raise ValueError("audit policy must be full or sparse_first_final")
        if task not in {"hellaswag", "coco"}:
            raise ValueError("undeclared mixed-task baseline is unsupported")
        if model.split["stack"] != "enc" or model.split["where"] != "after_final_norm":
            raise ValueError("baseline requires post-final-norm enc_fn")
        if any(True for _ in model.channel.parameters()) or model.channel.state_dict():
            raise ValueError("baseline requires a stateless channel")
        if model.memory_codec is not None or model.codec.film is not None:
            raise ValueError("baseline has one hidden stream and FiLM disabled")
        if not synthetic and (final_step != 400 or effective_batch != 64):
            raise ValueError("accepted baseline has 400 updates and effective batch 64")
        if not synthetic and (lr != 2e-4 or weight_decay != 0.01 or grad_clip != 1.0):
            raise ValueError("accepted baseline optimizer is AdamW lr2e-4/decay0.01/clip1")
        if final_step < 1 or effective_batch < 1 or not pairing_id:
            raise ValueError("invalid finite baseline plan")
        if set(identity) != {"source", "config", "data", "parent"}:
            raise ValueError("identity must bind source/config/data/parent")
        self.synthetic = synthetic
        self.audit_policy = audit_policy
        self.model = prepare_trainable_parameters(model)
        self.model.train()
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        if any(p.requires_grad for p in model.base.parameters()):
            raise ValueError("backbone must remain frozen")
        self.optimizer = torch.optim.AdamW(self.parameters, lr=lr, weight_decay=weight_decay)
        self.scheduler = build_baseline_scheduler(self.optimizer)
        device = next(model.base.parameters()).device
        dtype = next(model.base.parameters()).dtype
        self.scaler = GradScaler("cuda", enabled=device.type == "cuda" and dtype == torch.float16)
        self.run_id, self.task, self.identity, self.pairing_id = run_id, task, identity, pairing_id
        self.noise_identity = canonical_digest({"pairing_id": pairing_id, "schema": "runtime-awgn-replay-v2",
            "torch": str(torch.__version__), "device": str(device), "dtype": str(dtype),
            "snr": "per-sequence-uniform[-6,18]", "streams": ["hidden"]})
        self.final_step, self.effective_batch, self.grad_clip = final_step, effective_batch, grad_clip
        self.event_sink = event_sink
        self.completed = self.attempted = self.offset = self.valid_tokens = 0
        self.phase_start_step = 0
        self.failed = False
        self.started = time.monotonic()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    def _event(self, event_type, phase, payload):
        event = {"schema": "experiment-records-v1", "event_type": event_type,
                 "run_id": self.run_id, "phase_id": phase, "task": self.task,
                 "timestamp": datetime.now(timezone.utc).isoformat(),
                 "units": {"time": "seconds", "memory": "bytes", "channel": "real_coordinates"},
                 "identity": self.identity, "payload": payload}
        if event_type == "observation" and payload.get("reuse") is not None:
            event["identity"] = {key: payload["request"][key] for key in ("source", "config", "data", "parent")}
        if self.event_sink is not None:
            self.event_sink(event)
        return event

    def _values(self, batch, kind, step, micro, purpose, condition: str | int = "uniform", *,
                capture=True, bound_identity=None):
        view = None if bound_identity is None else bound_identity.view_sha256
        key = {"schema": "baseline-draw-v2", "pairing_id": self.pairing_id,
               "purpose": purpose, "site": "enc_fn", "site_local_batch": step if purpose == "train" else 0,
               "view": batch_identity(batch) if view is None else view, "microbatch": micro,
               "layout": list(batch["attention_mask"].shape), "condition": condition}
        namespace = noise_namespace(NoiseKey(self.pairing_id,
            "training" if purpose == "train" else "validation", "enc_fn",
            step if purpose == "train" else 0, batch_identity(batch) if view is None else view,
            canonical_digest({"shape": key["layout"], "micro": micro}), condition=str(condition),
            draw_schema="runtime-awgn-replay-v2"))
        device = next(self.model.base.parameters()).device
        if condition == "uniform":
            generator = torch.Generator(device=device).manual_seed(derived_seed(0, namespace + ":snr"))
            snr = torch.empty((len(batch["labels"]), 1, 1), device=device).uniform_(-6, 18, generator=generator)
        else:
            snr = None if condition == "no_noise" else float(condition)
        channel = self.model.channel
        context = channel.replay(0, namespace, capture=capture) if isinstance(channel, AWGNChannel) else nullcontext()
        with context:
            values = (local_batch_values(self.model, batch, snr, objective_settings(kind)) if kind == "local"
                      else batch_losses(self.model, batch, objective_settings(kind), snr,
                                        return_stats=True, valid_only_kl=True))
        draws = list(channel.draw_summaries) if isinstance(channel, AWGNChannel) else []
        return values, snr, {"key": key, "draws": draws}

    def update(self, batches, *, kind="combined", data_cache_seconds=0.0, batch_identities=None):
        if self.failed or self.completed >= self.final_step:
            raise RuntimeError("learner is terminal; no automatic retries or extra updates")
        settings = objective_settings(kind)
        _check_batches(batches, kind, self.effective_batch)
        if batch_identities is not None:
            if len(batch_identities) != len(batches):
                raise ValueError("prepared batch identity count differs from microbatches")
            for batch, binding in zip(batches, batch_identities):
                if not isinstance(binding, BoundBatchIdentity):
                    raise TypeError("prepared batch identity must be a bound identity token")
                binding.validate(batch)
        step = self.completed + 1
        capture = self.audit_policy == "full" or step in {self.phase_start_step + 1, self.final_step}
        self.attempted += 1
        started = time.monotonic()
        self.optimizer.zero_grad(set_to_none=True)
        sampled = step == self.phase_start_step + 1 or step % 10 == 0 or step == self.final_step
        before = [p.detach().clone() for p in self.parameters] if sampled else None
        lr_used = [group["lr"] for group in self.optimizer.param_groups]
        source = sum(int(b["attention_mask"].sum()) for b in batches)
        target = sum(int((b["labels"] != -100).sum()) for b in batches)
        allocated = sum(b["attention_mask"].numel() for b in batches)
        self.offset += self.effective_batch  # attempted exposure is charged even on failure
        self.valid_tokens += source
        results, snrs, draws = [], [], []
        optimizer_called = False
        try:
            denominator = effective_batch_denominators(self.model, batches, next(self.model.base.parameters()).device)
            for micro, batch in enumerate(batches):
                values, snr, draw = self._values(batch, kind, step, micro, "train",
                    capture=capture, bound_identity=None if batch_identities is None else batch_identities[micro])
                loss = scaled_batch_loss(values, settings, denominator)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite loss")
                self.scaler.scale(loss).backward()
                results.append(detached_batch_values(values))
                if not torch.is_tensor(snr):
                    raise RuntimeError("training SNR must be a per-sequence tensor")
                snrs.append(snr.detach().flatten())
                draws.append(draw)
            totals = aggregate_batch_losses(results, settings)
            self.scaler.unscale_(self.optimizer)
            pre_clip = torch.nn.utils.clip_grad_norm_(self.parameters, self.grad_clip)
            if not bool(torch.isfinite(pre_clip)):
                raise FloatingPointError("nonfinite gradient")
            post_clip = torch.linalg.vector_norm(torch.stack([
                torch.linalg.vector_norm(p.grad.detach().float()) for p in self.parameters if p.grad is not None]))
            scale = self.scaler.get_scale()
            # A post-hook distinguishes a genuinely invoked zero-LR optimizer step from a skip.
            def called(*_):
                nonlocal optimizer_called
                optimizer_called = True
            handle = self.optimizer.register_step_post_hook(called)
            try:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            finally:
                handle.remove()
            if not optimizer_called or self.scaler.get_scale() < scale:
                raise FloatingPointError("skipped optimizer update")
            if any(not bool(torch.isfinite(p).all()) for p in self.parameters):
                raise FloatingPointError("nonfinite parameter after optimizer update")
            if any(torch.is_tensor(value) and not bool(torch.isfinite(value).all())
                   for state in self.optimizer.state.values() for value in state.values()):
                raise FloatingPointError("nonfinite optimizer state")
            self.scheduler.step()
            self.completed = step
            all_snrs = torch.cat(snrs)
            width = self.model.codec.config["bottleneck_dim"]
            elapsed = time.monotonic() - started
            update_l2 = (float(torch.sqrt(torch.stack([(p.detach() - old).float().square().sum()
                                             for p, old in zip(self.parameters, before)]).sum())) if before else None)
            self.last_draws = draws
            device = next(self.model.base.parameters()).device
            memory = ({"allocated_bytes": measure(torch.cuda.max_memory_allocated(device)),
                       "reserved_bytes": measure(torch.cuda.max_memory_reserved(device)),
                       "reset_scope": "learner_initialization_high_water_no_extra_sync"} if device.type == "cuda" else
                      {"allocated_bytes": measure(reason="not_applicable_cpu"),
                       "reserved_bytes": measure(reason="not_applicable_cpu"), "reset_scope": "not_applicable_cpu"})
            payload = {"site_id": "enc_fn", "attempted_step": self.attempted, "completed_step": self.completed,
                       "final_step": self.final_step, "phase_start_step": self.phase_start_step,
                       "site_step": step, "sweep": step,
                       "exposure": {"source_tokens": source, "target_tokens": target, "sequences": self.effective_batch,
                                    "padded_tokens": allocated, "valid_tokens": source,
                                    "cumulative_valid_tokens": self.valid_tokens, "latent_width": width,
                                    "hidden_coordinates": source * width, "hidden_coordinates_allocated": allocated * width,
                                    "memory_coordinates": measure(reason="not_applicable"),
                                    "memory_coordinates_allocated": measure(reason="not_applicable")},
                       "objective": _objective_record(totals, kind), "lr_used": lr_used,
                       "lr_next": [group["lr"] for group in self.optimizer.param_groups],
                       "gradient": {"pre_clip": measure(float(pre_clip)), "post_clip": measure(float(post_clip)),
                                    "clip_threshold": measure(self.grad_clip), "clipped": float(pre_clip) > self.grad_clip},
                       "nonfinite": False, "skipped": False,
                       "update_l2": measure(update_l2, None if sampled else "sampled_first_every10_final"),
                       "snr": {"condition": "uniform[-6,18]", "min": measure(float(all_snrs.min())),
                               "mean": measure(float(all_snrs.mean())), "max": measure(float(all_snrs.max())), "draw_ref": canonical_digest(draws)},
                       "timing": {"step_seconds": measure(elapsed), "elapsed_seconds": measure(time.monotonic() - self.started),
                                  "data_cache_seconds": measure(data_cache_seconds),
                                  "throughput": measure(self.effective_batch / elapsed), "throughput_denominator": "sequences",
                                  "scope": "host_wall_including_scalar_reads; no_extra_cuda_synchronize"},
                       "memory": memory}
            if self.audit_policy != "full":
                payload["audit"] = {"policy": self.audit_policy, "noise_capture": capture,
                                    "complete_batch_identities": None if batch_identities is None else
                                        [b.complete_sha256 for b in batch_identities]}
            return self._event("update", kind, payload)
        except Exception as exc:
            self.failed = True
            self.last_failure = {"site_id": "enc_fn", "attempted_step": self.attempted,
                        "completed_step": self.completed, "attempted_sequences": self.offset,
                        "attempted_source_tokens": self.valid_tokens, "optimizer_called": optimizer_called,
                        "reason": type(exc).__name__, "message": str(exc), "status": "incomplete"}
            self._event("failure", kind, {"reason": type(exc).__name__,
                        "reference": f"failure-attempt-{self.attempted}.json"})
            raise

    @torch.no_grad()
    def validate(self, batches, *, kind="combined"):
        """Six fixed paired streams; preserve training mode/RNG and all denominators."""
        _check_batches(batches, kind)
        was_training = self.model.training
        self.model.eval()
        records = []
        try:
            with isolated_rng(derived_seed(0, self.pairing_id + ":objective_validation")):
                for condition in CONDITIONS:
                    evaluated = [self._values(b, kind, 0, micro, "objective_validation", condition)
                                 for micro, b in enumerate(batches)]
                    values = [value[0] for value in evaluated]
                    totals = aggregate_batch_losses(values, objective_settings(kind))
                    if not bool(torch.isfinite(totals["loss"])):
                        raise FloatingPointError("nonfinite objective validation")
                    records.append({"condition": condition, "objective": _objective_record(totals, kind),
                                    "requested": sum(len(b["labels"]) for b in batches),
                                    "completed": sum(len(b["labels"]) for b in batches), "failed": 0,
                                    "draws": [value[2] for value in evaluated]})
        finally:
            self.model.train(was_training)
        return records


def read_prepared_batch(reference, root):
    """Load only bytes bound by the preparation inventory; no layout mutation."""
    path = Path(root) / reference["path"]
    if file_digest(path) != reference["sha256"]:
        raise ValueError("prepared batch checksum mismatch")
    batch = torch.load(path, map_location="cpu", weights_only=True)
    if batch_identity(batch) != reference["view_sha256"]:
        raise ValueError("prepared input/view identity mismatch")
    return batch


def write_manifest(path, value):
    """Durably publish JSON; any write failure stops the caller."""
    write_json_atomic(path, value)


def run_baseline(learner, *, output, metadata, update_batches, validation_batches,
                 assess, segment=None, parent=None, reused_assessments=None, task_request=None,
                 resource_guard=None, allocation_required=False, on_output_created=None):
    """Execute and reconcile a finite segment against durable records and bytes.

    Assessment callbacks must emit full observation receipts and their referenced
    files, including for synthetic CPU models. A normal return alone never
    certifies a completed arm. A caller may inject an allocation deadline guard;
    the execution host must additionally enforce its external hard timer.
    """
    from dataclasses import asdict
    from .experiment_schedule import Segment
    from .experiment_state import save_state, open_state, restore_state, branch_metadata
    from .evaluation import (codec_observation_identity, observation_event_payload,
                             verify_observation_artifacts)
    segment = segment or Segment("single", "both", 0, learner.final_step,
                                 tuple(sorted({learner.final_step // 4, learner.final_step // 2,
                                               3 * learner.final_step // 4, learner.final_step} - {0})),
                                 (0, learner.final_step // 2, learner.final_step))
    if segment.stop > learner.final_step or segment.start not in {0, 200}:
        raise ValueError("segment exceeds finite learner plan")
    if segment.strategy not in {"both", "reconstruction-prefix", "reconstruction", "staged"}:
        raise ValueError("unknown segment objective")
    if segment.start and parent is None:
        raise ValueError("continuation requires validated immutable parent state")
    if not segment.start and parent is not None:
        raise ValueError("fresh segment cannot load parent weights")
    if task_request is None:
        raise ValueError("runner requires full task observation request, including CPU fixtures")
    output = Path(output)
    learner.phase_start_step = segment.start
    kind = "combined" if segment.strategy in {"both", "staged"} else "local"
    metadata = dict(metadata)
    metadata.setdefault("data_identity", learner.identity["data"])
    if metadata.get("noise_identity", learner.noise_identity) != learner.noise_identity:
        raise ValueError("learner pairing/noise identity differs from preparation")
    metadata["noise_identity"] = learner.noise_identity
    if parent is not None:
        parent = open_state(parent.reference, expected=parent.payload["metadata"])
        for key in ("source_identity", "config_identity", "data_identity", "initialization_identity",
                    "stream_identity", "protocol_identity", "noise_identity"):
            if parent.payload["metadata"].get(key) != metadata[key]:
                raise ValueError(f"phase continuation {key} differs from parent")
        prepared_inputs = {key: metadata[key] for key in ("retained_inputs", "prepared_manifest") if key in metadata}
        metadata = {**branch_metadata(parent, phase=segment.strategy), **prepared_inputs}
        learner.identity = {**learner.identity, "parent": parent.reference.sha256}
        stream = restore_state(parent, model=learner.model.codec, optimizer=learner.optimizer,
                               scheduler=learner.scheduler, scaler=learner.scaler)
        learner.completed = learner.attempted = stream["completed_updates"]
        learner.offset, learner.valid_tokens = stream["offset"], stream["source_valid_tokens"]
    elif learner.completed or learner.attempted:
        raise ValueError("fresh segment learner already consumed work")
    metadata["comparison_controls"] = comparison_refs(learner, metadata, kind)
    for field, key in (("source", "source_identity"), ("config", "config_identity"), ("data", "data_identity")):
        if learner.identity[field] != metadata[key] or task_request[field] != metadata[key]:
            raise ValueError(f"actual learner/task {field} binding mismatch")
    if task_request["comparison"] != metadata["comparison_controls"]:
        raise ValueError("task comparison controls differ from actual learner")
    output.mkdir(parents=True, exist_ok=False)
    if on_output_created is not None:
        on_output_created()
    learner.final_step = segment.stop
    checkpoints, objectives, assessments = [], [], []
    original_sink = learner.event_sink
    points = sorted({segment.start, *segment.checkpoints})
    expected = {f"objective:{step}": f"pending:objective:{step}" for step in points}
    expected.update({f"task:{step}": f"pending:task:{step}" for step in {segment.start, *segment.assessments}})
    required_artifacts = ["run.json", "config.yaml", "data_ids.json", "metrics.jsonl", "checkpoints.json", "baseline-result.json"]
    if allocation_required:
        required_artifacts.append("allocation.json")
    required_artifacts += [f"step_{step:06d}.pt" for step in points]
    required_artifacts += [f"objective_{step:06d}.json" for step in points]
    required_artifacts += [f"evaluations/step_{step:06d}/observation.json" for step in segment.assessments]
    manifest = {"schema": "experiment-records-v1", "run_id": learner.run_id,
                "phases": [{"phase_id": kind, "updates": segment.stop - segment.start,
                            "start_step": segment.start, "start_valid_tokens": learner.valid_tokens}],
                "expected_observations": list(expected.values()), "expected_artifacts": required_artifacts}
    def persist_plan():
        manifest["expected_observations"] = list(expected.values())
        write_manifest(output / "manifest.json", manifest)
    def guard():
        if resource_guard is not None:
            resource_guard()
    def sink(event):
        if event["event_type"] == "update":
            digest = event["payload"]["snr"]["draw_ref"]
            draw_path = output / "draws" / (digest + ".json")
            draw_path.parent.mkdir(exist_ok=True)
            if not draw_path.exists():
                write_manifest(draw_path, learner.last_draws)
        if event["event_type"] == "failure":
            write_manifest(output / event["payload"]["reference"], learner.last_failure)
        append_event(output / "metrics.jsonl", event)
        if original_sink is not None:
            original_sink(event)
    learner.event_sink = sink
    def cost(category, started, attribution="first_use"):
        learner._event("cost", kind, {"category": category, "seconds": measure(time.monotonic() - started),
            "attribution": attribution, "site_id": "enc_fn", "scope": "host_wall:" + category,
            "concurrency": "single_process", "device_count": int(next(learner.model.base.parameters()).is_cuda)})
    def save(step):
        guard()
        started = time.monotonic()
        state_metadata = {**metadata, "completed_updates": step, "phase": segment.strategy,
                          "snapshot_role": "initialization" if step == 0 else "trained"}
        if getattr(learner.model, "numerical_policy", "native") != "native":
            state_metadata["numerical_policy"] = learner.model.numerical_policy
        reference = save_state(output / f"step_{step:06d}.pt", model=learner.model.codec,
            optimizer=learner.optimizer, scheduler=learner.scheduler, scaler=learner.scaler,
            metadata=state_metadata, stream_state={"completed_updates": step, "offset": learner.offset,
                "source_valid_tokens": learner.valid_tokens})
        checkpoints.append(asdict(reference))
        write_manifest(output / "checkpoints.json", checkpoints)
        cost("checkpoint_io", started)
        return open_state(reference, expected=state_metadata)
    def task_at(step, state):
        return {**task_request, "checkpoint_sha256": state.reference.sha256,
                "step": step, "parent": state.payload["metadata"]["parent_identity"],
                "comparison": state.payload["metadata"]["comparison_controls"]}
    def measure_at(step, state, task_required):
        guard()
        started = time.monotonic()
        validation = validation_batches(kind)
        planned = objective_record_request(learner, state, validation, kind)
        expected[f"objective:{step}"] = observation_identity(planned)
        request = task_at(step, state)
        if task_required:
            expected[f"task:{step}"] = codec_observation_identity(request)
        persist_plan()  # Commit expectations before acquisition, never from results.
        records = learner.validate(validation, kind=kind)
        path = output / f"objective_{step:06d}.json"
        write_manifest(path, {"schema": "baseline-objective-v2", "checkpoint_sha256": state.reference.sha256,
                             "kind": kind, "conditions": records})
        objectives.append({"step": step, "path": path.name, "sha256": file_digest(path)})
        learner._event("observation", kind, objective_event_payload(learner, state, records, validation, path))
        cost("objective_validation", started)
        if task_required:
            guard()
            started = time.monotonic()
            directory = output / "evaluations" / f"step_{step:06d}"
            receipt = assess(state, directory)
            validate_task_receipt(receipt, request)
            verify_observation_artifacts(receipt, directory)
            learner._event("observation", kind, observation_event_payload(receipt))
            assessments.append({"step": step, "receipt": receipt})
            cost("task_scoring", started)
    def finalize(result):
        write_manifest(output / "baseline-result.json", result)
        persist_plan()
        paths = sorted(p for p in output.rglob("*") if p.is_file()
                       and p.name not in {"inventory.json", "completion.json"})
        inventory = {"schema": "experiment-records-v1", "mode": "tensor-complete", "omitted_tensors": [],
                     "artifacts": [artifact_ref(p, output, "tensor" if p.suffix == ".pt" else "metadata") for p in paths]}
        write_manifest(output / "inventory.json", inventory)
        status = completion_status(manifest, read_events(output / "metrics.jsonl"), inventory, output)
        write_manifest(output / "completion.json", status)
        return status
    started_run = time.monotonic()
    try:
        guard()
        write_manifest(output / "run.json", {"schema": "baseline-run-v2", "segment": asdict(segment), "metadata": metadata})
        from .config import save_config
        save_config(metadata.get("config", {"synthetic": True}), output / "config.yaml")
        write_manifest(output / "data_ids.json", metadata.get("data_ids", {"synthetic": True}))
        for name, reference in metadata.get("retained_inputs", {}).items():
            import shutil
            source = Path(reference["path"])
            if Path(name).name != name or file_digest(source) != reference["sha256"]:
                raise ValueError("retained input identity/path mismatch")
            destination = output / "inputs" / name
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, destination)
            if file_digest(destination) != reference["sha256"]:
                raise ValueError("retained input copy differs")
            manifest["expected_artifacts"].append(str(destination.relative_to(output)))
        persist_plan()
        for category in ("precommand_setup", "feature_acquisition", "gpu_active", "queue"):
            learner._event("cost", kind, {"category": category,
                "seconds": measure(reason="external_preparation_or_unmeasured; see bound allocation/input receipts"),
                "attribution": "first_use", "site_id": "enc_fn", "scope": category,
                "concurrency": "unavailable", "device_count": int(next(learner.model.base.parameters()).is_cuda)})
        initial = save(segment.start)
        measure_at(segment.start, initial, segment.start in segment.assessments)
        if segment.start not in segment.assessments:
            guard()
            started = time.monotonic()
            reuse = (reused_assessments or {}).get(segment.start)
            if reuse is None:
                raise ValueError("deduplicated task assessment requires exact prior state and receipt")
            reused_state, receipt = reuse["state"], reuse["receipt"]
            reused_state = open_state(reused_state.reference, expected=reused_state.payload["metadata"])
            request = task_at(segment.start, reused_state)
            validate_task_receipt(receipt, request)
            verify_observation_artifacts(receipt, reuse["root"])
            for key in ("source_identity", "config_identity", "data_identity", "initialization_identity",
                        "stream_identity", "protocol_identity", "noise_identity"):
                if reused_state.payload["metadata"][key] != metadata[key]:
                    raise ValueError("reused task assessment provenance mismatch")
            current = learner.model.codec.state_dict()
            if current.keys() != reused_state.payload["model"].keys() or any(
                not torch.equal(value.cpu(), reused_state.payload["model"][key].cpu()) for key, value in current.items()):
                raise ValueError("reused task assessment weights differ")
            # Retain the verified external evidence locally, preserving byte identities.
            import shutil
            destination = output / "reused" / f"step_{segment.start:06d}"
            destination.mkdir(parents=True)
            shutil.copyfile(Path(reuse["root"]) / "observation.json", destination / "observation.json")
            for row in receipt["conditions"]:
                condition_dir = destination / str(row["condition"])
                condition_dir.mkdir()
                for name in row["output_hashes"]:
                    shutil.copyfile(Path(reuse["root"]) / str(row["condition"]) / name, condition_dir / name)
            verify_observation_artifacts(receipt, destination)
            payload = observation_event_payload(receipt)
            payload["reuse"] = {"identity": payload["identity"], "status": "complete",
                                "reference": str(destination.relative_to(output) / "observation.json"),
                                "acquisition_seconds": measure(sum(r["elapsed_seconds"] for r in receipt["conditions"]))}
            expected[f"task:{segment.start}"] = payload["identity"]
            manifest["expected_artifacts"].append(str(destination.relative_to(output) / "observation.json"))
            persist_plan()
            learner._event("observation", kind, payload)
            assessments.append({"step": segment.start, "reuse": reused_state.reference.sha256, "receipt": receipt})
            cost("task_reuse_read", started, "reuse")
        for step in range(segment.start + 1, segment.stop + 1):
            guard()
            started = time.monotonic()
            batches = update_batches(step - 1, kind)
            loaded = time.monotonic()
            learner.update(batches, kind=kind, data_cache_seconds=loaded - started)
            cost("training_update", loaded)
            learner._event("cost", kind, {"category": "data_cache_io", "seconds": measure(loaded - started),
                "attribution": "reuse" if kind == "local" else "first_use", "site_id": "enc_fn",
                "scope": "host_wall:data_cache_io", "concurrency": "single_process",
                "device_count": int(next(learner.model.base.parameters()).is_cuda)})
            if step in segment.checkpoints:
                state = save(step)
                measure_at(step, state, step in segment.assessments)
        guard()
        learner._event("footprint", kind, {"scope": "per_site", "site_id": "enc_fn",
            "parameter_count": measure(sum(p.numel() for p in learner.model.codec.parameters())),
            "checkpoint_bytes": measure(sum(ref["size"] for ref in checkpoints)),
            "checkpoint_refs": [Path(ref["path"]).name for ref in checkpoints]})
        cost("command_wall", started_run)
        result = {"schema": "baseline-result-v2", "segment": asdict(segment),
                  "attempted_updates": learner.attempted, "completed_updates": learner.completed,
                  "checkpoints": checkpoints, "objective_observations": objectives,
                  "task_assessments": assessments, "scientific_acceptance": False}
        status = finalize(result)
        pending_allocation = (allocation_required and status["missing_artifacts"] == ["allocation.json"]
                              and status["reasons"] == ["missing mandatory artifacts"]
                              and not status["missing_observations"])
        if status["status"] != "complete" and not pending_allocation:
            raise RuntimeError("mandatory records or artifact inventory incomplete")
        return {**result, **status}
    except Exception as error:
        learner.failed = True
        failure = {"schema": "baseline-failure-v1", "attempted_updates": learner.attempted,
                   "completed_updates": learner.completed, "attempted_sequences": learner.offset,
                   "error": {"type": type(error).__name__, "message": str(error)},
                   "last_update": getattr(learner, "last_failure", None)}
        # Failure receipts are independent of the potentially broken metrics writer.
        try:
            write_manifest(output / "failure.json", failure)
            learner.last_failure = failure
            learner._event("failure", kind, {"reason": str(error) or type(error).__name__, "reference": "failure.json"})
        except Exception as recording_error:
            failure["recording_error"] = {"type": type(recording_error).__name__, "message": str(recording_error)}
        try:
            finalize({**failure, "checkpoints": checkpoints, "objective_observations": objectives,
                      "task_assessments": assessments})
        except Exception as inventory_error:
            write_manifest(output / "completion.json", {"schema": "baseline-failure-v1", "status": "incomplete",
                **failure, "inventory_error": str(inventory_error)})
        raise
    finally:
        learner.event_sink = original_sink


def objective_record_request(learner, state, batches, kind):
    """Declare an observation before executing it; actual draws stay in artifacts."""
    items = []
    for batch in batches:
        ids, families = batch.get("row_ids"), batch.get("source_family_ids")
        if (ids is None or families is None) and not learner.synthetic:
            raise ValueError("objective records require actual row/source-family IDs")
        for index in range(len(batch["labels"])):
            synthetic_id = "synthetic:" + batch_identity({k: v[index:index + 1] if torch.is_tensor(v) else v
                                                        for k, v in batch.items()})
            item_id = str(int(ids[index]) if torch.is_tensor(ids) else ids[index]) if ids is not None else synthetic_id
            source_id = str(families[index]) if families is not None else synthetic_id
            items.append({"item_id": item_id, "source_id": source_id})
    metadata = state.payload["metadata"]
    panel = canonical_digest([batch_identity(b) for b in batches])
    request = {**learner.identity, "checkpoint": state.reference.sha256, "panel": panel,
               "site": "enc_fn", "role": "trained", "task": learner.task,
               "protocol": metadata["protocol_identity"], "noise": canonical_digest({"noise": learner.noise_identity, "panel": panel, "purpose": "objective_validation"}),
               "scorer": "fp32-full-vocabulary-K-masked-sequence-R-v1", "layout": panel,
               "precision": str(next(learner.model.base.parameters()).dtype),
               "backend": str(learner.model.sdpa_backend_policy), "conditions": list(CONDITIONS),
               "expected_items": len(items), "learner_kind": "initialization" if learner.completed == 0 else "specialist",
               "step": learner.completed, "site_step": learner.completed, "purpose": "objective",
               "objective_kind": kind,
               "comparison": comparison_refs(learner, metadata, kind),
               "details": {"prompt": panel, "native_policy": panel,
                           "draws": canonical_digest({"noise": learner.noise_identity, "panel": panel, "purpose": "objective_validation"}),
                           "outputs": "baseline-objective-v2"}}
    if getattr(learner.model, "numerical_policy", "native") != "native":
        from .models.precision import compute_provenance
        request["precision"] += ":" + learner.model.numerical_policy
        request["details"]["outputs"] = "baseline-objective-v2:" + canonical_digest({
            "numerical_runtime": compute_provenance(learner.model)})
    return request


def objective_event_payload(learner, state, records, batches, artifact):
    request = objective_record_request(learner, state, batches, records[0]["objective"]["kind"])
    items = []
    for batch in batches:
        ids, families = batch.get("row_ids"), batch.get("source_family_ids")
        for index in range(len(batch["labels"])):
            fallback = f"synthetic:{batch_identity(batch)}:{len(items)}"
            item_id = str(int(ids[index]) if torch.is_tensor(ids) else ids[index]) if ids is not None else fallback
            source_id = str(families[index]) if families is not None else fallback
            items.append({"item_id": item_id, "source_id": source_id})
    return {"request": request, "identity": observation_identity(request), "status": "complete", "reuse": None,
            "conditions": [{"condition": row["condition"], "requested": len(items), "completed": len(items),
                            "failed": 0, "failure_reason": None, "denominator": len(items), "metrics": {},
                            "items": items, "objectives": row["objective"]["components"]} for row in records]}


def comparison_refs(learner, metadata, kind):
    """Derive immutable controls from actual configured codec and bound state."""
    result = {
        "architecture": canonical_digest(learner.model.codec.config),
        "initialization": metadata["initialization_identity"],
        "training_data": canonical_digest({"data": metadata.get("data_identity", learner.identity["data"]),
                                           "stream": metadata["stream_identity"]}),
        "objective": canonical_digest(objective_settings(kind)),
        "exposure": canonical_digest({"effective_batch": learner.effective_batch,
                                      "stream": metadata["stream_identity"], "horizon": 400}),
        "schedule": canonical_digest({"schema": "baseline-lr-v2", "horizon": 400, "warmup": 20,
                                      "pairing": learner.pairing_id}),
    }
    declared = metadata.get("comparison")
    if declared is not None and declared != result:
        raise ValueError("declared comparison controls differ from actual config/state")
    return result


def validate_task_receipt(receipt, expected_request):
    """A complete flag cannot replace exact immutable observation evidence."""
    from .evaluation import codec_observation_identity
    expected_identity = codec_observation_identity(expected_request)
    if receipt.get("request") != expected_request or receipt.get("identity") != expected_identity:
        raise ValueError("task observation identity differs from required panel/noise/scorer contract")
    rows = receipt.get("conditions", [])
    if receipt.get("status") != "complete" or [r.get("condition") for r in rows] != list(CONDITIONS):
        raise ValueError("task observation conditions incomplete")
    count = expected_request["expected_items"]
    for row in rows:
        if row.get("status") != "complete" or any(row.get(key) != count for key in ("requested", "completed", "denominator")) or row.get("failed") != 0:
            raise ValueError("task observation item accounting incomplete")
        items = row.get("items", [])
        ids = [item.get("sample_id", item.get("image_id")) for item in items]
        families = [item.get("source_id", item.get("image_id")) for item in items]
        if ids != expected_request["input_ids"] or families != expected_request["source_family_ids"]:
            raise ValueError("task observation item/source membership mismatch")
        if not row.get("metrics") or row.get("error"):
            raise ValueError("task observation scores unavailable or condition failed")


def open_baseline_replay(directory, requirement, *, source_identity, role, site, backbone, dtype):
    """Bind a producer bank to this consumer, not to self-asserted requirements."""
    from dataclasses import asdict
    from .activation_replay import open_replay
    capability = {"optimization": "local_reconstruction", "objective_validation": "objective_validation"}.get(role)
    if (capability is None or requirement.get("capability") != capability
            or requirement.get("data_role") != role
            or requirement.get("learner_source_sha256") != source_identity):
        raise ValueError("replay consumer source/role/capability differs from current learner")
    replay = open_replay(directory, requirement)
    producer = replay.manifest["producer"]
    expected_site = {**asdict(site), "site_id": site.site_id, "split": site.split}
    if (producer["sites"] != [expected_site] or producer["backbone"] != backbone
            or producer["environment"]["activation_dtype"] != str(dtype)):
        raise ValueError("replay backbone/site/precision differs from actual consumer binding")
    return replay


def read_replay_batch(replay, reference):
    batch = replay.read(reference["batch_view_id"], "enc_fn")
    if batch_identity(batch) != reference["view_sha256"]:
        raise ValueError("replayed input/view differs from prepared ordered stream")
    return batch


def write_baseline_comparisons(campaign, *, steps=(0, 200, 400)):
    """Bind the four declared topology contrasts to verified physical observations.

    The caller names an exact campaign directory; no newest-run discovery or
    scientific selection is performed. Missing/incomplete arms fail closed.
    """
    from .experiment_schedule import CELLS
    campaign = Path(campaign)
    arms = {}
    for path in sorted(campaign.iterdir()):
        if not path.is_dir() or not (path / "run.json").is_file():
            continue
        run = json.loads((path / "run.json").read_text())
        segment = run["segment"]
        if segment["strategy"] != "both":
            continue
        cell = segment["cell"]
        if cell not in CELLS or cell in arms:
            raise ValueError("campaign requires unique explicitly named baseline cells")
        events = read_events(path / "metrics.jsonl")
        inventory = json.loads((path / "inventory.json").read_text())
        manifest = json.loads((path / "manifest.json").read_text())
        result = completion_status(manifest, events, inventory, path)
        if result["status"] != "complete" or not result["tensor_bytes_verified"]:
            raise ValueError("comparison arm is not tensor-complete")
        observations = {}
        for event in events:
            if event["event_type"] == "observation" and event["payload"]["request"]["purpose"] == "task":
                payload = event["payload"]
                step = payload["request"]["step"]
                if step in observations or payload["status"] != "complete":
                    raise ValueError("ambiguous or incomplete task observation")
                observations[step] = payload["identity"]
        if not set(steps) <= observations.keys():
            raise ValueError("missing requested baseline comparison checkpoint")
        arms[cell] = observations
    if set(arms) != set(CELLS):
        raise ValueError("baseline comparison requires all four complete cells")
    pairs = (("D-none", "R-none"), ("D-LN", "R-LN"), ("D-LN", "D-none"), ("R-LN", "R-none"))
    result = {"schema": "experiment-comparisons-v1", "contrasts": [
        {"id": f"{left}-minus-{right}-step{step}", "left_observation": arms[left][step],
         "right_observation": arms[right][step], "metrics": ["normalized_correct", "raw_correct"],
         "allowed_control_differences": ["architecture", "initialization"]}
        for left, right in pairs for step in steps]}
    write_manifest(campaign / "comparisons.json", result)
    return result
