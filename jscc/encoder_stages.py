"""Nine independent, matched shared trajectories; reuse established lifecycle code."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
import copy
import json
from pathlib import Path
import time

import torch

from .activation_replay import canonical_digest, file_digest
from .baseline_protocol import objective_settings, read_prepared_batch, write_manifest
from .config import load_config
from .encoder_pilot_plan import SITE_SPECS, SITES
from .encoder_shared_pilot import readiness_view_indices
from .encoder_stages_plan import ENCODER_STAGES_PROTOCOL, EncoderStagesPlan, build_stages_state
from .encoder_stages_evaluation import open_stages_terminal, run_stages_evaluation  # noqa: F401
from .experiment_records import AppendOnlyEventWriter, read_events
from .experiment_schedule import noise_namespace
from .experiment_state import open_state, restore_state, save_state
from .models.channel import AWGNChannel
from .presentation import PresentationSampler, derived_seed
from .sharing_preparation import _reference, source_inventory, state_dict_identity
from .sharing_protocol import SharingLearner, batch_view
from .sharing_schedule import NATIVE16_WEIGHT_DECAY

CONDITIONS = ('no_noise', -6, 0, 6, 18)


def stages_source_inventory():
    root = Path(__file__).resolve().parents[1]
    return {**source_inventory(root), **{name: file_digest(root / name) for name in
            ('scripts/encoder_stages.py', 'scripts/render_encoder_stages.py',
             'scripts/original13_training.py', 'scripts/original13_evaluation.py')}}


class EncoderStagesLearner(SharingLearner):
    """Preserve the paired global stream through a weight-only phase transition."""

    def __init__(self, model, *, plan, metadata):
        super().__init__(model, schedule=plan, run_id=f'b{plan.width}_{plan.arm}',
                         identity={'source': metadata['source_identity'], 'config': metadata['config_identity'],
                                   'data': metadata['data_identity'], 'parent': None},
                         model_revision=metadata['model_revision'], site_specs=SITE_SPECS)
        self.reset_parent = None
        self._new_optimizer()

    def _new_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.parameters, lr=2e-4, weight_decay=NATIVE16_WEIGHT_DECAY)
        phase_start = self.phase_start_step
        stage = self.schedule.stage_at(phase_start)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
            lambda index: self.schedule.update_at(phase_start + index).lr_factor
            if index < stage.horizon else 0.0)

    def reset_phase(self, parent):
        boundary = self.schedule.horizon // 2
        details = parent.payload['metadata']
        if (self.failed or self.schedule.arm == 'A' or self.completed != boundary
                or self.phase_start_step != 0 or self.reset_parent is not None
                or details['completed_updates'] != boundary or details['phase'] != 'phase1'
                or details['sharing']['arm'] != self.schedule.arm
                or details['sharing']['width'] != self.schedule.width
                or state_dict_identity(parent.payload['model']) != state_dict_identity(self.model.codec.state_dict())):
            raise ValueError('reset requires this exact completed phase-one checkpoint and retained codec')
        self.optimizer.zero_grad(set_to_none=True)
        self.phase_start_step = boundary
        self._new_optimizer()
        self.reset_parent = asdict(parent.reference)
        self.identity = {**self.identity, 'parent': parent.reference.sha256}

    def _values(self, batch, kind, step, micro, purpose, condition: str | int = 'uniform', *,
                capture=True, bound_identity=None):
        if self.schedule.objective_kind_at(self.completed) != 'reconstruction':
            return super()._values(batch, kind, step, micro, purpose, condition,
                                   capture=capture, bound_identity=bound_identity)
        from .encoder_online_reconstruction import online_reconstruction_batch_values
        if purpose != 'train' or micro != 0 or self.pending_update is None:
            raise ValueError('online reconstruction requires one scheduled native training view')
        key = self.pending_update.noise_key
        namespace = noise_namespace(key)
        device = next(self.model.base.parameters()).device
        generator = torch.Generator(device=device).manual_seed(derived_seed(0, namespace + ':snr'))
        snr = torch.empty((len(batch['labels']), 1, 1), device=device).uniform_(-6, 18, generator=generator)
        channel = self.model.channel
        replay = channel.replay(0, namespace, capture=capture) if isinstance(channel, AWGNChannel) else nullcontext()
        with replay:
            values = online_reconstruction_batch_values(self.model, batch, snr, objective_settings('local'),
                                                        site=self.sites[self.active_site])
        return values, snr, {'key': asdict(key), 'draws': list(channel.draw_summaries)}

    def update(self, batches, **kwargs):
        stage = self.schedule.stage_at(self.completed)
        if self.phase_start_step != stage.start:
            raise ValueError('declared phase reset must precede the next update')
        return super().update(batches, **kwargs)

    def _event(self, event_type, phase, payload):
        stage = next(s for s in self.schedule.stages if s.start == self.phase_start_step)
        if event_type == 'update' and stage.objective_kind == 'reconstruction':
            payload = copy.deepcopy(payload)
            payload['objective']['kind'] = 'local'
            payload['objective']['components']['K'] = {
                'raw': {'value': None, 'reason': 'not_computed_online_reconstruction'},
                'weighted': {'value': None, 'reason': 'not_computed_online_reconstruction'},
                'numerator': {'value': None, 'reason': 'not_computed_online_reconstruction'}, 'denominator': 0}
        return super()._event(event_type, stage.phase_id, payload)


def load_stages(path, *, arm, width):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    raw = {k: v for k, v in manifest.items() if k != 'manifest_identity'}
    if (manifest.get('schema') != 'encoder-stages-prepared-v1'
            or manifest.get('protocol_id') != ENCODER_STAGES_PROTOCOL
            or manifest.get('manifest_identity') != canonical_digest(raw)
            or manifest.get('source_inventory') != stages_source_inventory()
            or manifest.get('source_identity') != canonical_digest(manifest['source_inventory'])):
        raise ValueError('stages source or prepared manifest identity differs')
    root = path.parent
    config = load_config(_reference(root, manifest['configs'][str(width)]))
    batches = [read_prepared_batch(ref, root) for ref in manifest['updates']]
    plan = EncoderStagesPlan(tuple(batch_view(batch) for batch in batches), manifest['study_pairing_id'],
                             arm, width=width, synthetic=manifest['synthetic_cpu'])
    template = json.loads(_reference(root, manifest['task_request']).read_text())
    train, codec, channel = config['training'], config['codec'], config['channel']
    if (config['task'] != 'hellaswag' or config['seed'] != 0 or config['split'] != SITE_SPECS['enc_fn']
            or codec['architecture'] != 'direct_affine' or codec['layernorm'] != 'none'
            or codec['n_res_blocks'] != 0 or codec['snr_film'] or codec['dropout'] != 0
            or codec['bottleneck_dim'] != width or train['batch_size'] != 16 or train['gradient_accumulation'] != 1
            or train['lr'] != 2e-4 or train['weight_decay'] != NATIVE16_WEIGHT_DECAY
            or train['schedule_steps'] != plan.horizon or train['max_steps'] != plan.horizon
            or train['kl_weight'] != 1 or train['mse_weight'] != .1 or train['temperature'] != 1
            or train['grad_clip'] != 1 or train.get('patience') is not None
            or train['warmup_ratio'] != .05 or train['min_lr_ratio'] != 0 or train['valid_only_kl'] is not True
            or train.get('loss_weights') is not None or channel['type'] != 'awgn'
            or not channel['normalize_power'] or not channel['train_noise']
            or channel['train_snr_range'] != [-6.0, 18.0]):
        raise ValueError('stages native batch, codec, objective or channel recipe differs')
    if not manifest['synthetic_cpu']:
        pool = manifest['data_ids']['train_rows']
        rows = [int(i) for batch in batches for i in batch['row_ids']]
        expected = PresentationSampler(pool, 160000, seed=20260920).actual_ids.tolist()
        if (len(pool) != 39058 or len(set(pool)) != 39058 or rows != expected
                or manifest['presentation_stream'] != {'policy': 'epoch-permutations-v1',
                    'pool_identity': canonical_digest(pool), 'seed': 20260920,
                    'start_presentation': 0, 'total_presentations': 160000}
                or config['model']['revision'] != 'dd0a2683227859151b1730ca3a63087df5b5f39b'
                or config['model']['dtype'] != 'bfloat16' or config['model']['device'] != 'cuda'
                or config['model']['attn_implementation'] != 'sdpa'
                or template['expected_items'] != 10042 or template['conditions'] != list(CONDITIONS)
                or template['input_ids'] != manifest['data_ids']['final_validation']['input_ids']
                or template['source_family_ids'] != manifest['data_ids']['final_validation']['source_family_ids']
                or template['settings']['batch_size'] != 16 or template['settings']['num_fewshot'] != 5
                or template['settings']['scoring_policy'] != 'fp32-v1'):
            raise ValueError('stages full pool, precision or terminal evaluation contract differs')
    reference = manifest['initializations'][str(width)]
    initial = torch.load(_reference(root, reference), map_location='cpu', weights_only=True)
    if state_dict_identity(initial) != reference['state_identity']:
        raise ValueError('stages initial codec state differs')
    metadata = {'source_identity': manifest['source_identity'], 'config_identity': canonical_digest(config),
                'data_identity': canonical_digest(manifest['data_ids']), 'parent_identity': None,
                'initialization_identity': reference['state_identity'], 'stream_identity': canonical_digest(manifest['updates']),
                'protocol_identity': ENCODER_STAGES_PROTOCOL, 'prepared_identity': manifest['manifest_identity'],
                'model_revision': config['model']['revision'], 'final_task_identity': manifest['task_request']['sha256'],
                'lineage': [], 'phase': 'phase1'}
    return manifest, config, plan, batches, template, initial, metadata


def save_stages(learner, metadata, path):
    sharing, stream = build_stages_state(learner.schedule, learner.completed,
                                        phase_start_step=learner.phase_start_step)
    details = {**metadata, 'sharing': sharing, 'completed_updates': learner.completed, 'phase': sharing['phase_id'],
               'evaluation_sites': {site: {'site': asdict(resolved), 'role': 'trained'}
                                    for site, resolved in learner.sites.items()},
               'snapshot_role': 'initialization' if learner.completed == 0 else 'trained',
               'model_state_contract': 'codec-only-stateless-channel-v1', 'numerical_policy': 'native',
               'comparison_controls': {'architecture': canonical_digest(learner.model.codec.config),
                    'initialization': metadata['initialization_identity'], 'training_data': metadata['stream_identity'],
                    'objective': canonical_digest({'combined': {'K': 1, 'R': .1}, 'reconstruction': {'R': .1}}),
                    'exposure': canonical_digest({'sites': SITES, 'q': learner.schedule.q, 'batch': 16}),
                    'schedule': canonical_digest({'segments': [s.horizon for s in learner.schedule.stages],
                                                   'reset_steps': learner.schedule.reset_steps, 'horizon': learner.schedule.horizon,
                                                   'warmup_ratio': .05, 'lr': 2e-4})}}
    if learner.reset_parent is not None:
        details['parent_identity'] = learner.reset_parent['sha256']
        details['lineage'] = [{'parent_sha256': learner.reset_parent['sha256'], 'parent_checkpoint': learner.reset_parent,
                              'completed_updates': learner.phase_start_step, 'from_phase': 'phase1', 'to_phase': 'phase2',
                              'transfer': 'retain_codec_reset_optimizer_lr_preserve_global_stream',
                              'stream_identity': metadata['stream_identity']}]
    reference = save_state(path, model=learner.model.codec, optimizer=learner.optimizer, scheduler=learner.scheduler,
                           scaler=learner.scaler, metadata=details, stream_state=stream)
    validated = open_state(reference, expected=details)
    if restore_state(validated, model=learner.model.codec, optimizer=learner.optimizer,
                     scheduler=learner.scheduler, scaler=learner.scaler) != stream:
        raise ValueError('stages checkpoint stream reload differs')
    return validated


def run_stages_training(model, *, plan, batches, metadata, output, guard, config=None, data_ids=None):
    if not plan.synthetic and (config is None or data_ids is None):
        raise ValueError('real trajectory requires resolved configuration and exact data membership')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if config is not None:
        from .config import save_config
        save_config(config, output / 'config.yaml')
    if data_ids is not None:
        write_manifest(output / 'data_ids.json', data_ids)
    learner = EncoderStagesLearner(model, plan=plan, metadata=metadata)
    learner.event_sink = AppendOnlyEventWriter(output / 'metrics.jsonl').append
    begun = time.monotonic()
    result = {'schema': 'encoder-stages-training-v1', 'status': 'incomplete', 'arm': plan.arm, 'width': plan.width,
              'prepared_identity': metadata['prepared_identity'], 'planned_updates': plan.horizon,
              'completed_updates': 0, 'checkpoints': [], 'phase_reset': None}
    write_manifest(output / 'campaign.json', result)
    try:
        for step in range(plan.horizon + 1):
            guard()
            if step in plan.checkpoint_steps:
                state = save_stages(learner, metadata, output / f'step_{step:06d}.pt')
                result['checkpoints'].append(asdict(state.reference))
                if step in plan.reset_steps:
                    learner.reset_phase(state)
                    reset = save_stages(learner, metadata, output / f'step_{step:06d}_reset.pt')
                    result['phase_reset'] = {'parent': asdict(state.reference), 'reset': asdict(reset.reference),
                                             'global_updates': step, 'phase_local_updates': 0}
            if step == plan.horizon:
                break
            learner.update([batches[step // 6]])
            if learner.completed % 600 == 0 or learner.completed == plan.horizon:
                result.update(completed_updates=learner.completed, elapsed_seconds=time.monotonic() - begun,
                              per_site=learner.exposure.per_site)
                write_manifest(output / 'campaign.json', result)
                print(json.dumps({'phase': 'training', 'arm': plan.arm, 'width': plan.width,
                                  'completed_updates': learner.completed, 'elapsed_seconds': result['elapsed_seconds']}), flush=True)
        counts = learner.exposure.finalize()
        events = read_events(output / 'metrics.jsonl')
        updates = [e for e in events if e['event_type'] == 'update']
        if len(updates) != plan.horizon or any(e['payload']['completed_step'] != i + 1 for i, e in enumerate(updates)):
            raise ValueError('complete scalar update log required before freeze')
        guard()
        freeze = {'schema': 'encoder-stages-freeze-v1', 'status': 'complete',
                  'prepared_identity': metadata['prepared_identity'], 'arm': plan.arm, 'width': plan.width,
                  'checkpoint': result['checkpoints'][-1], 'counts': counts,
                  'metrics': {'path': str(output / 'metrics.jsonl'), 'sha256': file_digest(output / 'metrics.jsonl')},
                  'final_panels': [[site, condition] for condition in CONDITIONS for site in SITES]}
        write_manifest(output / 'freeze.json', freeze)
        result.update(status='training_frozen', completed_updates=learner.completed, per_site=counts,
                      freeze_reference={'path': str(output / 'freeze.json'), 'sha256': file_digest(output / 'freeze.json')},
                      scalar_updates=len(updates), elapsed_seconds=time.monotonic() - begun)
        write_manifest(output / 'campaign.json', result)
        return result
    except Exception as error:
        result.update(completed_updates=learner.completed, elapsed_seconds=time.monotonic() - begun,
                      error={'type': type(error).__name__, 'message': str(error)})
        write_manifest(output / 'campaign.json', result)
        raise


def qualify_stages(model, *, plan, batches, metadata, output, guard):
    """Bounded changed-path diagnostics, exact restoration, and measured throughput."""
    from .sharing_qualification_guard import restore, snapshot, state_exact
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'incomplete', 'scientific_updates': 0, 'cycles': []}
    begun = time.monotonic()
    write_manifest(output / 'result.json', result)
    try:
        for arm in ('E', 'R'):
            for index in sorted(readiness_view_indices(batches)):
                selected = [batches[index], *batches[:index], *batches[index + 1:]]
                # Two distinct real native views, deliberately short diagnostic horizon.
                diagnostic = EncoderStagesPlan(tuple(batch_view(b) for b in selected[:2]),
                    plan.study_pairing_id + ':qualification', arm, synthetic=True, width=plan.width)
                learner = EncoderStagesLearner(model, plan=diagnostic, metadata=metadata)
                before = snapshot(learner)
                rows = []
                for site in SITES:
                    guard()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    started = time.monotonic()
                    learner.update([selected[0]])
                    torch.cuda.synchronize()
                    rows.append({'site': site, 'seconds': time.monotonic() - started,
                                 'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
                state = save_stages(learner, metadata, output / f'{arm}-view{index}.pt')
                guard()
                learner.reset_phase(state)
                reset = save_stages(learner, metadata, output / f'{arm}-view{index}-reset.pt')
                for site in SITES:
                    guard()
                    learner.update([selected[1]])
                save_stages(learner, metadata, output / f'{arm}-view{index}-terminal.pt')
                restore(learner, before)
                learner.phase_start_step = 0
                learner.reset_parent = None
                learner.identity = {**learner.identity, 'parent': None}
                if not state_exact(before, snapshot(learner)):
                    raise ValueError('qualification fresh-state restoration differs')
                result['cycles'].append({'arm': arm, 'view': index, 'sites': rows,
                                         'checkpoint': asdict(state.reference), 'reset_checkpoint': asdict(reset.reference),
                                         'phase_reset_and_stage_two_passed': True, 'fresh_state_restored_exact': True})
                write_manifest(output / 'result.json', result)
        result.update(status='pass', elapsed_seconds=time.monotonic() - begun)
        write_manifest(output / 'result.json', result)
        return result
    except Exception as error:
        result.update(error={'type': type(error).__name__, 'message': str(error)}, elapsed_seconds=time.monotonic() - begun)
        write_manifest(output / 'result.json', result)
        raise


class Original13TrainingState:
    """Integration-owned optimizer, reset and complete-state lifecycle.

    Routing supplies the actual learner via the M1 API. CPU mechanical fixtures
    may implement that API, but never count as universal-route acceptance.
    """

    def __init__(self, learner, *, plan, metadata, routing):
        from .encoder_stages_plan import ORIGINAL13_PROTOCOL
        self.learner, self.plan, self.routing = learner, plan, routing
        self.codec = routing.canonical_codec(learner.model)
        self.metadata = copy.deepcopy(metadata)
        if metadata.get('protocol_identity') != ORIGINAL13_PROTOCOL:
            raise ValueError('original13 protocol identity required')
        for key in ('source_identity', 'config_identity', 'data_identity', 'initialization_identity',
                    'stream_identity', 'prepared_identity', 'model_revision'):
            if not isinstance(metadata.get(key), str) or not metadata[key]:
                raise ValueError('missing original13 metadata: ' + key)
        if metadata['stream_identity'] != plan.input_identity:
            raise ValueError('original13 input identity differs')
        parameters = list(self.codec.parameters())
        if (len({id(p) for p in learner.parameters}) != len(learner.parameters)
                or {id(p) for p in parameters} != {id(p) for p in learner.parameters}):
            raise ValueError('optimizer requires exactly one canonical codec parameter owner')
        if state_dict_identity(routing.export_codec(learner.model)) != state_dict_identity(self.codec.state_dict()):
            raise ValueError('canonical weight export differs from codec body')
        if state_dict_identity(routing.export_codec(learner.model)) != metadata['initialization_identity']:
            raise ValueError('fresh original13 initializer identity differs')
        self.reset_parent = None
        self.ancestry = []
        self._new_optimizer()

    def _new_optimizer(self):
        from .sharing_schedule import sharing_lr_factor
        learner = self.learner
        stage = next(s for s in self.plan.stages if s.start == learner.phase_start_step)
        learner.optimizer = torch.optim.AdamW(learner.parameters, lr=2e-4, weight_decay=NATIVE16_WEIGHT_DECAY)
        learner.scheduler = torch.optim.lr_scheduler.LambdaLR(
            learner.optimizer, lambda index: sharing_lr_factor(index, stage.horizon))

    def save(self, path):
        from .encoder_stages_plan import original13_progress
        learner = self.learner
        progress = original13_progress(self.plan, learner.completed, learner.phase_start_step)
        details = {**self.metadata, 'original13_plan': self.plan.resolved(), 'plan_identity': self.plan.identity,
                   'input_identity': self.plan.input_identity, 'observation_identity': self.plan.observation_identity,
                   'completed_updates': learner.completed, 'phase_start_step': learner.phase_start_step,
                   'phase': progress['phase_id'], 'parent_identity': None if self.reset_parent is None
                   else self.reset_parent['sha256'], 'lineage': [] if self.reset_parent is None else
                   [{'parent_sha256': self.reset_parent['sha256'], 'parent_checkpoint': self.reset_parent,
                     'completed_updates': self.plan.reconstruction_stop, 'transfer': 'weights_only_reset_adam_lr'}],
                   'segment_ancestry': copy.deepcopy(self.ancestry),
                   'exposure': copy.deepcopy(learner.exposure.per_site),
                   'exposure_receipts': list(learner.exposure.receipts),
                   'valid_tokens': learner.valid_tokens, 'attempted': learner.attempted,
                   'route_state': copy.deepcopy(self.routing.observation_state(learner.model)),
                   'model_state_contract': 'original13-canonical-codec-v1'}
        reference = save_state(path, model=self.codec, optimizer=learner.optimizer, scheduler=learner.scheduler,
                               scaler=learner.scaler, metadata=details, stream_state=progress,
                               weight_export=lambda: self.routing.export_codec(learner.model))
        return open_state(reference, expected=details)

    def restore(self, reference):
        from .experiment_state import IDENTITIES
        expected = {**{k: self.metadata[k] for k in (*IDENTITIES, 'data_identity', 'prepared_identity',
                    'model_revision') if k != 'parent_identity'},
                    'plan_identity': self.plan.identity, 'input_identity': self.plan.input_identity,
                    'observation_identity': self.plan.observation_identity}
        expected.pop('parent_identity', None)
        # open_state requires a supplied parent; inspect it only after hash validation
        # via a compatibility open against the parent's value from trusted bytes.
        data = Path(reference.path).read_bytes()
        if len(data) != reference.size or file_digest(Path(reference.path)) != reference.sha256:
            raise ValueError('continuation checkpoint bytes differ')
        import io
        payload = torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)
        expected['parent_identity'] = payload['metadata']['parent_identity']
        state = open_state(reference, expected=expected)
        details = state.payload['metadata']
        learner = self.learner
        learner.phase_start_step = details['phase_start_step']
        self._new_optimizer()
        progress = restore_state(state, model=self.codec, optimizer=learner.optimizer,
                                 scheduler=learner.scheduler, scaler=learner.scaler,
                                 weight_load=lambda weights: self.routing.load_codec(learner.model, weights))
        self.routing.restore_runtime_state(learner.model, copy.deepcopy(details['route_state']))
        from .sharing_qualification_guard import state_exact
        from .experiment_state import capture_rng
        if (not state_exact(self.routing.observation_state(learner.model), details['route_state'])
                or not state_exact(capture_rng(), state.payload['rng'])):
            raise ValueError('continuation route/noise/RNG restoration differs')
        learner.completed = progress['completed_updates']
        learner.attempted = details['attempted']
        learner.offset = progress['offset']
        learner.valid_tokens = details['valid_tokens']
        learner.exposure.completed = learner.completed
        learner.exposure.per_site = copy.deepcopy(details['exposure'])
        learner.exposure.receipts = list(details['exposure_receipts'])
        self.reset_parent = None if not details['lineage'] else details['lineage'][-1]['parent_checkpoint']
        self.ancestry = [*details['segment_ancestry'], asdict(reference)]
        return state

    def reset(self, parent):
        learner = self.learner
        if (learner.failed or learner.completed != self.plan.reconstruction_stop
                or learner.phase_start_step != 0 or self.reset_parent is not None):
            raise ValueError('original13 permits exactly one boundary reset')
        verified = open_state(parent.reference, expected=parent.payload['metadata'])
        details = verified.payload['metadata']
        if (details['plan_identity'] != self.plan.identity or details['phase_start_step'] != 0
                or details['completed_updates'] != learner.completed
                or state_dict_identity(verified.payload['model']) != state_dict_identity(self.routing.export_codec(learner.model))):
            raise ValueError('reset requires exact own pre-reset state')
        self.reset_parent = asdict(verified.reference)
        learner.phase_start_step = self.plan.reconstruction_stop
        self._new_optimizer()


def run_original13_training(model, processor, *, plan, batches, metadata, observation_request,
                            output, guard, routing=None, observe=None, resume=None):
    """Fresh segment or explicit continuation; no automatic retry or model loading."""
    from . import original13_routing
    from .baseline_protocol import bind_batch_identity
    from .encoder_stages_evaluation import ObservationScoringError, observe_original13
    from .encoder_stages_plan import original13_progress
    from .experiment_state import capture_rng, restore_rng
    routing = original13_routing if routing is None else routing
    observe = observe_original13 if observe is None else observe
    if (len(batches) != plan.q or tuple(batch_view(b) for b in batches) != plan.views
            or canonical_digest(observation_request['panel']) != plan.observation_identity):
        raise ValueError('original13 verified inputs/observation panel differ')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = {'schema': 'original13-training-v1', 'status': 'incomplete', 'ratio': plan.ratio, 'width': plan.width,
              'plan_identity': plan.identity, 'completed_updates': 0, 'checkpoints': [],
              'observations': [{'step': step, 'state_role': role, 'status': 'pending',
                                'requires_explicit_completion': False} for step, role in plan.observations],
              'segment_parent': None if resume is None else asdict(resume)}
    write_manifest(output / 'campaign.json', result)
    learner = None
    try:
        learner = routing.create_original13_learner(model, plan=plan, metadata=metadata,
                                                   event_sink=AppendOnlyEventWriter(output / 'metrics.jsonl').append)
        state_machine = Original13TrainingState(learner, plan=plan, metadata=metadata, routing=routing)
        if resume is not None:
            restored = state_machine.restore(resume)
            for observation in result['observations']:
                if observation['step'] <= learner.completed:
                    observation.update(status='pending_prior_segment', requires_explicit_completion=True)
            if learner.completed == plan.reconstruction_stop and learner.phase_start_step == 0:
                state_machine.reset(restored)
                reset = state_machine.save(output / f'step_{learner.completed:06d}_post_reset.pt')
                result['checkpoints'].append({'role': 'post_reset', **asdict(reset.reference)})
        initial_completed = learner.completed
        bindings = [bind_batch_identity(b, expected_identity=v.view_sha256) for b, v in zip(batches, plan.views)]
        while True:
            guard()
            step = learner.completed
            for save_step, role in plan.checkpoint_states:
                if save_step != step or (role == 'post_reset') or (resume is not None and step == initial_completed):
                    continue
                state = state_machine.save(output / f'step_{step:06d}_{role}.pt')
                result['checkpoints'].append({'role': role, **asdict(state.reference)})
                if (step, role) in plan.observations:
                    from .sharing_qualification_guard import state_exact
                    saved_rng = capture_rng()
                    def training_state():
                        return copy.deepcopy({'weights': routing.export_codec(model),
                            'optimizer': learner.optimizer.state_dict(), 'scheduler': learner.scheduler.state_dict(),
                            'scaler': learner.scaler.state_dict() if learner.scaler is not None else None,
                            'counters': [learner.completed, learner.attempted, learner.offset, learner.valid_tokens,
                                         learner.phase_start_step, learner.failed],
                            'exposure': [learner.exposure.completed, learner.exposure.failed,
                                         learner.exposure.per_site, learner.exposure.receipts],
                            'grads': [None if p.grad is None else p.grad.detach().clone() for p in learner.parameters],
                            'route': routing.observation_state(model)})
                    saved_training = training_state()
                    request = {**observation_request, 'plan_identity': plan.identity, 'step': step,
                               'state_role': role, 'checkpoint_sha256': state.reference.sha256}
                    error = None
                    try:
                        receipt = observe(model, processor, state=state, request=request,
                                          output=output / f'observation_{step:06d}', guard=guard)
                        if receipt.get('status') != 'complete' or receipt.get('request_identity') != canonical_digest(request):
                            raise ValueError('observation receipt identity/status differs')
                    except ObservationScoringError as exc:
                        error = {'type': type(exc).__name__, 'message': str(exc)}
                    finally:
                        restore_rng(saved_rng)
                    # Consumer restores routes/noise; revalidate checkpoint and full
                    # training state before continuing even an isolated scoring error.
                    open_state(state.reference, expected=state.payload['metadata'])
                    if not state_exact(training_state(), saved_training) or not state_exact(capture_rng(), saved_rng):
                        raise ValueError('observation training state restoration differs')
                    row = next(r for r in result['observations'] if r['step'] == step)
                    row.update(request_identity=canonical_digest(request),
                        status='failed' if error else 'complete', error=error,
                        requires_explicit_completion=bool(error))
                    write_manifest(output / 'campaign.json', result)
                if role == 'pre_reset':
                    state_machine.reset(state)
                    reset = state_machine.save(output / f'step_{step:06d}_post_reset.pt')
                    result['checkpoints'].append({'role': 'post_reset', **asdict(reset.reference)})
            if step == plan.horizon:
                break
            expected = plan.update_at(step)
            used = [g['lr'] for g in learner.optimizer.param_groups]
            event = learner.update([batches[expected.site_local_index]],
                                   batch_identities=[bindings[expected.site_local_index]])
            if (learner.completed != step + 1 or learner.exposure.completed != learner.completed
                    or any(lr != 2e-4 * expected.lr_factor for lr in used)
                    or event['payload']['lr_used'] != used):
                raise ValueError('original13 update progress or consumed LR differs')
        counts = learner.exposure.finalize()
        events = [e for e in read_events(output / 'metrics.jsonl') if e['event_type'] == 'update']
        if [e['payload']['completed_step'] for e in events] != list(range(initial_completed + 1, plan.horizon + 1)):
            raise ValueError('original13 segment scalar log incomplete')
        terminal = next((c for c in result['checkpoints'] if c['role'] == 'terminal'), None)
        if terminal is None:
            raise ValueError('continuing an already-terminal state requires evaluation, not training')
        freeze = {'schema': 'original13-freeze-v1', 'status': 'training_frozen', 'plan_identity': plan.identity,
                  'checkpoint': {k: terminal[k] for k in ('path', 'sha256', 'size')}, 'counts': counts,
                  'progress': original13_progress(plan, learner.completed, learner.phase_start_step),
                  'segment_ancestry': state_machine.ancestry,
                  'metrics': {'path': str(output / 'metrics.jsonl'), 'sha256': file_digest(output / 'metrics.jsonl')}}
        write_manifest(output / 'freeze.json', freeze)
        result.update(status='training_frozen', completed_updates=learner.completed,
                      segment_updates=len(events), observations_status='pending' if any(
                          r['status'] != 'complete' for r in result['observations']) else 'complete',
                      freeze_reference={'path': str(output / 'freeze.json'), 'sha256': file_digest(output / 'freeze.json')})
        write_manifest(output / 'campaign.json', result)
        return result
    except Exception as exc:
        result.update(completed_updates=0 if learner is None else learner.completed,
                      error={'type': type(exc).__name__, 'message': str(exc)})
        write_manifest(output / 'campaign.json', result)
        raise
