"""Finite online shared/specialist learner; reuse baseline objective/update code."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, replace

import torch

from .activation_replay import canonical_digest
from .baseline_protocol import BaselineLearner, BoundBatchIdentity, batch_identity, bind_batch_identity, objective_settings
from .experiment_schedule import NoiseKey, noise_namespace
from .models.channel import AWGNChannel
from .models.split_model import resolve_encoder_site
from .native64_baseline import reuse_encoder
from .presentation import derived_seed, tensor_digest
from .sharing_schedule import BatchView, SharingExposureLedger
from .sharing_accumulation import micro_noise_key, partition_native64, partition_policy
from .training import batch_losses


def batch_view(batch):
    """Bind the actual complete CPU/native batch; no geometry token repacking."""
    mask = batch['attention_mask']
    return BatchView(
        view_sha256=batch_identity(batch), mask_sha256=tensor_digest(mask),
        layout_sha256=canonical_digest({'shape': list(mask.shape), 'micro': 0}),
        sequences=len(mask), source_tokens=int(mask.sum()),
        target_tokens=int((batch['labels'] != -100).sum()), padded_tokens=mask.numel())


def resolved_sites(model, revision):
    result = {}
    for name in ('enc_l9', 'enc_l19', 'enc_fn', 'enc_l14'):
        spec = ({'stack': 'enc', 'where': 'after_final_norm'} if name == 'enc_fn' else
                {'stack': 'enc', 'where': 'after_layer', 'index': int(name[5:])})
        result[name] = resolve_encoder_site(model.base, spec, revision)
    return result


class SharingLearner(BaselineLearner):
    """One codec/optimizer, one authorized site per update, no detached suffix.

    The inherited updater owns AdamW, finite checks, clipping and K/R reduction.
    This adapter owns the finite sharing schedule, real site-local draw keys,
    actual-batch receipts and horizon. No baseline protocol guard is relaxed.
    """
    def __init__(self, model, *, schedule, run_id, identity, model_revision,
                 event_sink=None, enc_fn_reuse=False, microbatch_size=None):
        plan = schedule.plan
        self.microbatch_size = microbatch_size
        self.execution_partition = None if microbatch_size is None else partition_policy(microbatch_size)
        if microbatch_size is not None and plan.batch_size != 64:
            raise ValueError('explicit accumulation requires effective64 parent views')
        if model.numerical_policy != 'native':
            raise ValueError('sharing uses native numerical policy')
        if enc_fn_reuse and plan.batch_size != 64:
            raise ValueError('encoder reuse acceptance is native64 only')
        # The enclosing sharing plan validates the production q/batch/horizon.
        # Baseline's synthetic flag only bypasses its enc_fn400 protocol quota;
        # the update/objective implementation is otherwise identical.
        super().__init__(model, run_id=run_id, task='hellaswag', identity=identity,
                         pairing_id=plan.study_pairing_id, synthetic=True,
                         final_step=schedule.horizon, effective_batch=plan.batch_size,
                         event_sink=event_sink, audit_policy='sparse_first_final')
        self.synthetic = plan.synthetic
        self.schedule = schedule
        self.exposure = SharingExposureLedger(schedule)
        self.sites = resolved_sites(model, model_revision)
        self.enc_fn_reuse = enc_fn_reuse
        self.active_site = None
        self.active_local = None
        self.pending_update = None
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda index: schedule.update_at(min(index, schedule.horizon - 1)).lr_factor
            if index < schedule.horizon else 0.0)

    def _event(self, event_type, phase, payload):
        if event_type == 'update':
            update = self.pending_update
            if update is None:
                raise RuntimeError('update outside declared sharing schedule')
            payload = {**payload, 'site_id': update.site,
                       'site_step': update.site_local_index + 1,
                       'sweep': update.site_local_index + 1}
        return super()._event(event_type, phase, payload)

    def _values(self, batch, kind, step, micro, purpose, condition='uniform', *,
                capture=True, bound_identity=None):
        if kind != 'combined' or (purpose == 'train' and micro != 0 and self.microbatch_size is None) or self.active_site is None:
            raise ValueError('sharing requires one complete native batch and combined objective')
        if purpose == 'train' and (self.active_local is None or self.pending_update is None):
            raise ValueError('training draw lacks scheduled local update')
        local = self.active_local if self.active_local is not None else 0
        view = batch_identity(batch) if bound_identity is None else bound_identity.view_sha256
        key = NoiseKey(self.pairing_id, 'training' if purpose == 'train' else 'validation',
                       self.active_site, local if purpose == 'train' else 0,
                       view, canonical_digest({'shape': list(batch['attention_mask'].shape), 'micro': micro}),
                       condition=str(condition), draw_schema='runtime-awgn-replay-v2')
        if purpose == 'train' and self.microbatch_size is not None:
            if self.pending_update is None:
                raise ValueError('training partition lacks scheduled parent update')
            key = micro_noise_key(self.pending_update.noise_key, batch, micro=micro,
                                  microbatch_size=self.microbatch_size)
        if purpose == 'train' and self.microbatch_size is None and self.pending_update is not None and key != self.pending_update.noise_key:
            raise ValueError('actual site-local noise/view key differs from frozen schedule')
        namespace = noise_namespace(key)
        device = next(self.model.base.parameters()).device
        if condition == 'uniform':
            generator = torch.Generator(device=device).manual_seed(derived_seed(0, namespace + ':snr'))
            snr = torch.empty((len(batch['labels']), 1, 1), device=device).uniform_(-6, 18, generator=generator)
        else:
            snr = None if condition == 'no_noise' else float(condition)
        channel = self.model.channel
        replay = channel.replay(0, namespace, capture=capture) if isinstance(channel, AWGNChannel) else nullcontext()
        reuse = reuse_encoder(self.model) if self.enc_fn_reuse and self.active_site == 'enc_fn' else nullcontext()
        with replay, reuse:
            values = batch_losses(self.model, batch, objective_settings(kind), snr,
                                  return_stats=True, valid_only_kl=True)
        draws = list(channel.draw_summaries) if isinstance(channel, AWGNChannel) else []
        return values, snr, {'key': asdict(key), 'draws': draws}

    def update(self, batches, *, kind='combined', data_cache_seconds=0.0, batch_identities=None):
        if self.failed or self.completed >= self.schedule.horizon:
            raise RuntimeError('learner is terminal; no retries')
        if len(batches) != 1 or kind != 'combined':
            raise ValueError('sharing supports the frozen native batch and combined recipe only')
        expected = self.schedule.update_at(self.completed)
        actual = replace(expected, view=batch_view(batches[0]))
        if actual != expected:
            raise ValueError('actual complete sequence view/layout differs from ordered sharing plan')
        if self.microbatch_size is not None:
            if (batch_identities is None or len(batch_identities) != 1
                    or not isinstance(batch_identities[0], BoundBatchIdentity)):
                raise ValueError('accumulation requires one verified parent binding')
            batch_identities[0].validate(batches[0])
            batches = partition_native64(batches[0], microbatch_size=self.microbatch_size)
            batch_identities = [bind_batch_identity(batch) for batch in batches]
        self.pending_update = actual
        self.active_site, self.active_local = actual.site, actual.site_local_index
        try:
            with self.model.at_site(self.sites[actual.site], purpose='train',
                                    authorization={'sites': {actual.site: 'trained'}}):
                event = super().update(batches, kind=kind, data_cache_seconds=data_cache_seconds,
                                       batch_identities=batch_identities)
            self.exposure.record(actual, receipt=canonical_digest(event))
            return event
        except Exception:
            self.failed = True
            raise
        finally:
            self.active_site = self.active_local = self.pending_update = None
    
    def validate_site(self, batches, site_id, *, authorization=None):
        if self.active_site is not None:
            raise RuntimeError('cannot nest objective validation')
        authorization = authorization or {'sites': {site: 'trained' for site in ('enc_l9','enc_l19','enc_fn')}}
        if site_id == 'enc_l14' and not authorization.get('heldout_freeze_receipt'):
            raise ValueError('heldout objective requires verified freeze')
        if self.microbatch_size is not None:
            partitioned = []
            for batch in batches:
                if len(batch['labels']) == 64:
                    partitioned.extend(partition_native64(batch, microbatch_size=self.microbatch_size))
                elif len(batch['labels']) <= self.microbatch_size:
                    partitioned.append(batch)
                else:
                    raise ValueError('objective batch must be parent64 or at most one microbatch')
            batches = partitioned
        self.active_site, self.active_local = site_id, 0
        try:
            with self.model.at_site(self.sites[site_id], purpose='evaluate', authorization=authorization):
                return super().validate(batches, kind='combined')
        finally:
            self.active_site = self.active_local = None
