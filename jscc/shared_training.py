"""Current shared training bodies; saved record meanings preserved."""

from __future__ import annotations
import torch
from dataclasses import asdict
from pathlib import Path
from .activation_replay import canonical_digest, file_digest
from .experiment_records import write_manifest
from .experiment_state import open_state
from .shared_inputs import batch_view
from .checkpoint_readers import measure as measure
from datetime import datetime, timezone
import time
from dataclasses import replace
from copy import deepcopy
from typing import cast
from torch.amp.grad_scaler import GradScaler
from .runtime import prepare_trainable_parameters
from .site_routing import canonical_codec
from .checkpoint_readers import LearnerSchedule
from .shared_plan import SharingExposureLedger
from .shared_inputs import _check_batches, BoundBatchIdentity, bind_batch_identity
from .models.channel import AWGNChannel
from .presentation import derived_seed
from .experiment_schedule import noise_namespace
from .reconstruction import reconstruction_batch_values
from .training import batch_losses
from .training_objectives import aggregate_batch_losses, detached_batch_values, effective_batch_denominators, scaled_batch_loss


def build_recipe_scheduler(optimizer, plan, phase_start):
    """Build a phase-local scheduler; state owner installs it after its reset."""
    stage = next((s for s in plan.stages if s.start == phase_start), None)
    if stage is None:
        raise ValueError("scheduler requires a declared recipe phase start")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda local:
        plan.update_at(stage.start + local).lr_factor if local < stage.horizon else 0.0)


def create_recipe_learner(model, *, plan, metadata, event_sink=None, synthetic=False):
    return RecipeLearner(model, plan=plan, metadata=metadata, event_sink=event_sink, synthetic=synthetic)


def run_recipe_training(model, processor, *, plan, batches, metadata, output, guard,
                        observe=None, resume=None, data_ids=None, synthetic=False, on_checkpoint=None, finalize_terminal=False):
    """Run one YAML segment over indexed native16 inputs; no loading or submission.

    observe(model, processor, state=ValidatedState, request=dict, output=Path,
    guard=callable) is a released consumer seam. Required requests without a
    consumer fail at setup. on_checkpoint is an optional durable-save notification.
    """

    from .shared_state import RecipeTrainingState, recipe_observation_snapshot, restore_recipe_observation, _recipe_progress
    from .shared_plan import RecipePlan
    from .config import save_config
    from .shared_inputs import bind_batch_identity
    from .panel_evaluation import ObservationScoringError
    if not isinstance(plan, RecipePlan):
        raise TypeError('recipe runner requires RecipePlan')
    description = plan.description
    if (metadata['config_identity'] != canonical_digest(description['config'])
            or (data_ids is not None and canonical_digest(data_ids) != metadata['data_identity'])
            or (not synthetic and data_ids is None)):
        raise ValueError('recipe run config/data identities differ')
    if len(batches) != plan.q:
        raise ValueError('recipe runner requires Q indexed batches')
    if description['observations'] and observe is None:
        raise ValueError('recipe observations require an explicit released consumer')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = {'schema': 'recipe-training-v1', 'status': 'incomplete', 'plan_identity': plan.identity,
        'completed_updates': 0, 'checkpoints': [], 'observations': [
            {**r, 'status': 'pending', 'requires_explicit_completion': False} for r in description['observations']],
        'segment_parent': None if resume is None else asdict(resume)}
    write_manifest(output/'campaign.json', result)
    learner = None
    metrics = None
    try:
        import os
        import json
        metrics = (output/'metrics.jsonl').open('xb')
        def append_recipe_event(event):
            # The retained writer validates fixed original13 objectives. Recipe
            # events carry generalized weights and progress in their own schema.
            event['schema'] = 'recipe-training-event-v1'
            encoded = json.dumps(event, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n'
            if metrics.write(encoded) != len(encoded):
                raise OSError('partial recipe event append')
            metrics.flush()
            os.fsync(metrics.fileno())
        save_config(description['config'], output/'config.yaml')
        write_manifest(output/'recipe.json', description)
        if data_ids is not None:
            write_manifest(output/'data_ids.json', data_ids)
        learner = create_recipe_learner(model, plan=plan, metadata=metadata,
            event_sink=append_recipe_event, synthetic=synthetic)
        state_machine = RecipeTrainingState(learner, plan=plan, metadata=metadata)
        restored = None
        if resume is not None:
            candidate = state_machine._open(resume)
            terminal = candidate.payload['metadata']['completed_updates'] == plan.horizon
            if terminal and not finalize_terminal:
                raise ValueError('terminal recipe state requires evaluation or explicit finalization, not training')
            if finalize_terminal and not terminal:
                raise ValueError('recipe finalization requires a full-budget terminal checkpoint')
            restored = state_machine.restore(resume)
            for observation in result['observations']:
                if observation['step'] <= learner.completed:
                    observation.update(status='pending_prior_segment', requires_explicit_completion=True)
        if finalize_terminal and resume is None:
            raise ValueError('recipe finalization requires an exact terminal checkpoint')
        initial_completed = learner.completed

        def publish(role):
            guard()
            saved = state_machine.save(output/f'step_{learner.completed:06d}_{role}.pt')
            guard()
            result['checkpoints'].append({'role': role, **asdict(saved.reference)})
            result['completed_updates'] = learner.completed
            write_manifest(output/'campaign.json', result)
            if on_checkpoint is not None:
                on_checkpoint(saved, role)
                guard()
            return saved

        if finalize_terminal:
            # Reuse the full state through the same restoration/save boundary;
            # the loop emits no optimizer update. Earlier observations remain
            # pending_prior_segment rather than inventing completion.
            publish('terminal')
        if restored is not None and learner.completed == plan.stages[1].start and learner.phase_start_step == 0:
            state_machine.reset(restored)
            publish('post_reset')
        view_index, batch, binding = None, None, None
        while True:
            guard()
            step = learner.completed
            for save_step, role in description['checkpoint_states']:
                if save_step != step or role == 'post_reset' or (resume is not None and step == initial_completed):
                    continue
                saved = publish(role)
                for row in result['observations']:
                    if row['step'] != step or row['state'] != role:
                        continue
                    request = {**{k: v for k, v in row.items() if k not in {
                        'status', 'requires_explicit_completion'}}, 'plan_identity': plan.identity,
                        'checkpoint_sha256': saved.reference.sha256}
                    if observe is None:
                        raise ValueError('missing recipe observation consumer')
                    snapshot = recipe_observation_snapshot(state_machine)
                    error = None
                    try:
                        receipt = observe(model, processor, state=saved, request=request,
                            output=output/f'observation_{step:06d}_{role}', guard=guard)
                        if (receipt.get('status') != 'complete'
                                or receipt.get('request_identity') != canonical_digest(request)):
                            raise ValueError('recipe observation receipt identity/status differs')
                    except ObservationScoringError as exc:
                        error = {'type': type(exc).__name__, 'message': str(exc)}
                    finally:
                        restore_recipe_observation(state_machine, snapshot)
                    open_state(saved.reference, expected=saved.payload['metadata'])
                    row.update(status='failed' if error else 'complete', error=error,
                               requires_explicit_completion=bool(error), request_identity=canonical_digest(request))
                    write_manifest(output/'campaign.json', result)
                if role == 'pre_reset':
                    state_machine.reset(saved)
                    publish('post_reset')
            if step == plan.horizon:
                break
            expected = plan.update_at(step)
            if view_index != expected.site_local_index:
                view_index = expected.site_local_index
                batch = batches[view_index]
                if batch_view(batch) != expected.view:
                    raise ValueError('recipe indexed batch differs from bound view')
                binding = bind_batch_identity(batch, expected_identity=expected.view.view_sha256)
            used = [g['lr'] for g in learner.optimizer.param_groups]
            event = learner.update([batch], batch_identities=[binding])
            if (learner.completed != step + 1 or learner.exposure.completed != learner.completed
                    or any(lr != description['config']['training']['lr'] * expected.lr_factor for lr in used)
                    or event['payload']['lr_used'] != used):
                raise ValueError('recipe update progress or consumed LR differs')
        counts = learner.exposure.finalize()
        events = [e for e in (json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines())
                  if e['event_type'] == 'update']
        if [e['payload']['completed_step'] for e in events] != list(range(initial_completed+1, plan.horizon+1)):
            raise ValueError('recipe segment update log incomplete')
        terminal = next(c for c in result['checkpoints'] if c['role'] == 'terminal')
        freeze = {'schema': 'recipe-freeze-v1', 'status': 'training_frozen', 'plan_identity': plan.identity,
            'checkpoint': {k: terminal[k] for k in ('path', 'sha256', 'size')}, 'counts': counts,
            'progress': _recipe_progress(plan, learner.completed, learner.phase_start_step),
            'segment_ancestry': state_machine.ancestry,
            'metrics': {'path': str(output/'metrics.jsonl'), 'sha256': file_digest(output/'metrics.jsonl')}}
        guard()
        write_manifest(output/'freeze.json', freeze)
        guard()
        result.update(status='training_frozen', completed_updates=learner.completed, segment_updates=len(events),
            observations_status='complete' if all(r['status'] == 'complete' for r in result['observations']) else 'pending',
            freeze_reference={'path': str(output/'freeze.json'), 'sha256': file_digest(output/'freeze.json')})
        write_manifest(output/'campaign.json', result)
        return result
    except Exception as exc:
        result.update(status='incomplete', completed_updates=0 if learner is None else learner.completed,
                      error={'type': type(exc).__name__, 'message': str(exc)})
        write_manifest(output/'campaign.json', result)
        raise

    finally:
        if metrics is not None:
            metrics.close()


def _objective_record(totals, kind, settings=None):
    def component(name, measured=True):
        if not measured:
            return {"raw": measure(reason="unmeasured"), "weighted": measure(reason="unmeasured"),
                    "numerator": measure(reason="unmeasured"), "denominator": 0}
        stat = "kl" if name == "K" else "hidden"
        return {"raw": measure(float(totals[stat])),
                "weighted": measure(float(totals["weighted_" + stat])),
                "numerator": measure(float(totals[stat + "_numerator"])),
                "denominator": int(totals[stat + "_denominator"])}
    result = {"kind": kind, "total": measure(float(totals["loss"])),
              "components": {"K": component("K", kind == "combined"), "R": component("R")}}
    if settings and "requires_memory" in settings:
        result["resolved_weights"] = dict(settings["loss_weights"])
        result["components"] = {}
        for name in ("kl", "hidden", "memory"):
            measured = (name != "memory" or settings["requires_memory"]) and int(totals[name + "_denominator"]) > 0
            reason = "not_computed_reconstruction" if name == "kl" else "not_required"
            result["components"][name] = {
                "raw": measure(float(totals[name])) if measured else measure(reason=reason),
                "weighted": measure(float(totals["weighted_" + name])) if measured else measure(reason=reason),
                "numerator": measure(float(totals[name + "_numerator"])) if measured else measure(reason=reason),
                "denominator": int(totals[name + "_denominator"]) if measured else 0}
    return result



class RecipeLearner:
    """One canonical shared codec with complete-view numerical updates and phases."""
    def __init__(self, model, *, plan, metadata, event_sink=None, synthetic=False):
        from .shared_plan import RecipePlan
        from .models.channel import IdentityChannel
        from .models.split_model import resolve_training_site
        if not isinstance(plan, RecipePlan) or type(synthetic) is not bool:
            raise TypeError("recipe learner requires a resolved RecipePlan and explicit synthetic marker")
        description = plan.description
        config = model.codec.config
        if (config["architecture"] != "direct_affine" or config["layernorm"] != "none"
                or config["snr_film"] or config["n_res_blocks"] != 0 or config["dropout"] != 0
                or config["bottleneck_dim"] != plan.width or model.numerical_policy != "native"):
            raise ValueError("recipe requires a shared direct-affine native codec matching its width")
        if metadata['model_revision'] != description['config']['model']['revision']:
            raise ValueError('runtime model revision differs from resolved recipe')
        self.recipe_channel = description['config']['channel']
        self.recipe_seed = description['config']['seed']
        expected_channel = AWGNChannel if self.recipe_channel['type'] == 'awgn' else IdentityChannel
        if (type(model.channel) is not expected_channel
                or model.channel_config['normalize_power'] != self.recipe_channel['normalize_power']):
            raise ValueError("runtime channel differs from resolved recipe")
        sites = {name: resolve_training_site(model.base, name, metadata['model_revision'])
                 for name in plan.trained_sites}
        if any(site.hidden_dim != model.codec.encoder[0].weight.shape[1] for site in sites.values()):
            raise ValueError("recipe roles must match the canonical codec hidden dimension")
        training = description['config']['training']
        self.recipe_training = training
        self.recipe_counts = description['counts']
        if model.split != {'stack': 'enc', 'where': 'after_final_norm'}:
            raise ValueError('shared initialization requires a clean enc_fn route')
        if any(True for _ in model.channel.parameters()) or model.channel.state_dict():
            raise ValueError('shared codec requires a stateless channel')
        if model.memory_codec is not None or model.codec.film is not None:
            raise ValueError('shared initialization requires one hidden codec with FiLM disabled')
        task = description['config']['task']
        if task not in {'hellaswag', 'coco'}:
            raise ValueError('unsupported shared task')
        run_id = f'b{plan.width}_r{plan.ratio}'
        identity = {'source': metadata['source_identity'], 'config': metadata['config_identity'],
                    'data': metadata['data_identity'], 'parent': None}
        pairing_id = plan.study_pairing_id
        final_step, effective_batch = plan.horizon, 16
        lr, weight_decay, grad_clip = training['lr'], training['weight_decay'], training['grad_clip']
        audit_policy = 'sparse_first_final'
        self.audit_policy = audit_policy
        self.model = prepare_trainable_parameters(model)
        self.model.train()
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        if any(p.requires_grad for p in model.base.parameters()):
            raise ValueError("backbone must remain frozen")
        self.optimizer = torch.optim.AdamW(self.parameters, lr=lr, weight_decay=weight_decay)
        device = next(model.base.parameters()).device
        dtype = next(model.base.parameters()).dtype
        self.scaler = GradScaler("cuda", enabled=device.type == "cuda" and dtype == torch.float16)
        self.run_id, self.task, self.identity, self.pairing_id = run_id, task, identity, pairing_id
        self.final_step, self.effective_batch, self.grad_clip = final_step, effective_batch, grad_clip
        self.event_sink = event_sink
        self.completed = self.attempted = self.offset = self.valid_tokens = 0
        self.phase_start_step = 0
        self.failed = False
        self.started = time.monotonic()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # Configure the existing optimizer before any state/update exists.
        for group in [self.optimizer.defaults, *self.optimizer.param_groups]:
            group.update(betas=tuple(training['betas']), eps=training['eps'])
        model.enable_universal_codec()
        canonical_codec(model)
        self.synthetic = synthetic
        self.schedule, self.sites = plan, sites
        self.trained_sites = plan.trained_sites
        self.exposure = SharingExposureLedger(cast(LearnerSchedule, plan), sites=plan.trained_sites)
        self.microbatch_size = None
        self.active_site = self.active_local = self.pending_update = None
        self.noise_identity = canonical_digest({'pairing_id': plan.study_pairing_id,
            'schema': 'original13-role-awgn-v1', 'streams': ['hidden', 'memory'],
            'torch': str(torch.__version__), 'device': str(next(model.base.parameters()).device),
            'dtype': str(next(model.base.parameters()).dtype), 'channel': self.recipe_channel,
            'seed': self.recipe_seed})
        self.scheduler = build_recipe_scheduler(self.optimizer, plan, 0)


    def _objective_settings(self, kind):
        return {'temperature': self.recipe_training['temperature'],
                'loss_weights': self.schedule.objective_at(self.completed),
                'requires_memory': self.model.requires_memory_transmission}


    def _values(self, batch, kind, step, micro, purpose, condition='uniform', *,
                capture=True, bound_identity=None):
        from contextlib import nullcontext
        if purpose != 'train' or micro != 0 or self.pending_update is None or self.active_site is None:
            raise ValueError('recipe requires one scheduled complete native training view')
        key = self.pending_update.noise_key
        namespace = noise_namespace(key)
        device = next(self.model.base.parameters()).device
        channel = self.model.channel
        noisy = self.recipe_channel['train_noise'] and isinstance(channel, AWGNChannel)
        if noisy:
            low, high = self.recipe_channel['train_snr_range']
            generator = torch.Generator(device=device).manual_seed(derived_seed(self.recipe_seed, namespace + ':snr'))
            snr = torch.empty((len(batch['labels']), 1, 1), device=device).uniform_(low, high, generator=generator)
        else:
            snr = torch.full((len(batch['labels']), 1, 1), self.recipe_channel['clean_film_snr'], device=device)
        replay = channel.replay(self.recipe_seed, namespace, capture=capture) if isinstance(channel, AWGNChannel) else nullcontext()
        settings = self._objective_settings(kind)
        with replay:
            if self.schedule.stage_at(self.completed).objective_kind == 'reconstruction':
                values = reconstruction_batch_values(self.model, batch, snr if noisy else None,
                                                                settings, site=self.sites[self.active_site])
            else:
                values = batch_losses(self.model, batch, settings, snr if noisy else None,
                                      return_stats=True, valid_only_kl=True)
        draws = list(channel.draw_summaries) if isinstance(channel, AWGNChannel) else []
        return values, snr, {'key': asdict(key), 'draws': draws}


    def update(self, batches, *, kind='combined', data_cache_seconds=0.0, batch_identities=None):
        if self.failed or self.completed >= self.schedule.horizon:
            raise RuntimeError('learner is terminal; no retries')
        if self.phase_start_step != self.schedule.stage_at(self.completed).start:
            raise ValueError('declared phase reset must precede the next update')
        if len(batches) != 1 or kind != 'combined':
            raise ValueError('sharing supports the frozen native batch and combined recipe only')
        expected = self.schedule.update_at(self.completed)
        if batch_identities is None:
            batch_identities = [bind_batch_identity(batches[0])]
        if len(batch_identities) != 1:
            raise ValueError('sharing requires one complete batch identity')
        actual = replace(expected, view=batch_view(batches[0], binding=batch_identities[0]))
        if actual != expected:
            raise ValueError('actual complete sequence view/layout differs from ordered sharing plan')
        self.pending_update = actual
        self.active_site, self.active_local = actual.site, actual.site_local_index
        try:
            with self.model.at_site(self.sites[actual.site], purpose='train',
                                    authorization={'sites': {actual.site: 'trained'}}):
                event = self._update_complete_view(batches, kind=kind, data_cache_seconds=data_cache_seconds,
                                       batch_identities=batch_identities)
            self.exposure.record(actual, receipt=canonical_digest(event))
            return event
        except Exception:
            self.failed = True
            raise
        finally:
            self.active_site = self.active_local = self.pending_update = None


    def _update_complete_view(self, batches, *, kind="combined", data_cache_seconds=0.0, batch_identities=None):
        if self.failed or self.completed >= self.final_step:
            raise RuntimeError("learner is terminal; no automatic retries or extra updates")
        settings = self._objective_settings(kind)
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
        coordinates = {"hidden": 0, "memory": 0}
        coordinates_allocated = {"hidden": 0, "memory": 0}
        optimizer_called = False
        try:
            denominator = effective_batch_denominators(self.model, batches, next(self.model.base.parameters()).device)
            for micro, batch in enumerate(batches):
                values, snr, draw = self._values(batch, kind, step, micro, "train",
                    capture=capture, bound_identity=None if batch_identities is None else batch_identities[micro])
                for role in coordinates:
                    coordinates[role] += self.model.channel_uses_valid[role]
                    coordinates_allocated[role] += self.model.channel_uses[role]
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
                                    "hidden_coordinates": coordinates["hidden"], "hidden_coordinates_allocated": coordinates_allocated["hidden"],
                                    "memory_coordinates": coordinates["memory"] if self.model.requires_memory_transmission else measure(reason="not_applicable"),
                                    "memory_coordinates_allocated": coordinates_allocated["memory"] if self.model.requires_memory_transmission else measure(reason="not_applicable")},
                       "objective": _objective_record(totals, kind, settings), "lr_used": lr_used,
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


    def _event(self, event_type, phase, payload):
        if event_type == 'update':
            payload['recipe_identity'] = self.schedule.identity
            payload['configured_counts'] = deepcopy(self.recipe_counts)
            payload['recipe_progress'] = self.schedule.progress(self.completed, self.phase_start_step)
            payload['channel'] = deepcopy(self.recipe_channel)
            payload['snr']['condition'] = ('uniform' if self.recipe_channel['train_noise']
                and isinstance(self.model.channel, AWGNChannel) else 'no_noise')
        stage = next(s for s in self.schedule.stages if s.start == self.phase_start_step)
        if event_type == 'update':
            payload['objective']['kind'] = stage.objective_kind
            update = self.pending_update
            if update is None:
                raise RuntimeError('update outside declared shared schedule')
            payload = {**payload, 'site_id': update.site,
                       'site_step': update.site_local_index + 1, 'sweep': update.site_local_index + 1}
        phase = stage.phase_id
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



# Same numerical body used by transitional diagnostic learners.
complete_view_update = RecipeLearner._update_complete_view
