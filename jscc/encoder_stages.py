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
from .baseline_protocol import objective_settings, read_prepared_batch, validate_task_receipt, write_manifest
from .config import load_config
from .encoder_pilot_plan import SITE_SPECS, SITES
from .encoder_shared_pilot import readiness_view_indices
from .encoder_stages_plan import ENCODER_STAGES_PROTOCOL, EncoderStagesPlan, build_stages_state
from .evaluation import evaluate_checkpoint, verify_observation_artifacts
from .experiment_records import AppendOnlyEventWriter, read_events
from .experiment_schedule import noise_namespace
from .experiment_state import CheckpointRef, open_state, restore_state, save_state
from .models.channel import AWGNChannel
from .presentation import PresentationSampler, derived_seed
from .sharing_preparation import _reference, source_inventory, state_dict_identity
from .sharing_protocol import SharingLearner, batch_view
from .sharing_schedule import NATIVE16_WEIGHT_DECAY

CONDITIONS = ('no_noise', -6, 0, 6, 18)


def stages_source_inventory():
    root = Path(__file__).resolve().parents[1]
    return {**source_inventory(root), **{name: file_digest(root / name) for name in
            ('scripts/encoder_stages.py', 'scripts/render_encoder_stages.py')}}


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


def open_stages_terminal(training_run, metadata, plan):
    root = Path(training_run).resolve()
    campaign = json.loads((root / 'campaign.json').read_text())
    freeze_path = root / 'freeze.json'
    if (campaign['status'] != 'training_frozen' or campaign['completed_updates'] != plan.horizon
            or campaign['prepared_identity'] != metadata['prepared_identity']
            or campaign['freeze_reference'] != {'path': str(freeze_path), 'sha256': file_digest(freeze_path)}):
        raise ValueError('own training trajectory is not completely frozen')
    freeze = json.loads(freeze_path.read_text())
    if (freeze['status'] != 'complete' or freeze['arm'] != plan.arm or freeze['width'] != plan.width
            or freeze['final_panels'] != [[site, condition] for condition in CONDITIONS for site in SITES]
            or file_digest(root / 'metrics.jsonl') != freeze['metrics']['sha256']):
        raise ValueError('terminal study plan or scalar log differs')
    expected = {k: v for k, v in metadata.items() if k not in {'phase', 'lineage'}}
    if plan.arm != 'A':
        reset = campaign['phase_reset']
        parent_ref = CheckpointRef(**reset['parent'])
        if Path(parent_ref.path).resolve() != root / f'step_{plan.horizon // 2:06d}.pt':
            raise ValueError('phase reset parent path differs from own training run')
        parent = open_state(parent_ref, expected=metadata)
        if parent.payload['metadata']['completed_updates'] != plan.horizon // 2:
            raise ValueError('phase reset parent did not complete stage one')
        expected['parent_identity'] = parent_ref.sha256
    state = open_state(CheckpointRef(**freeze['checkpoint']), expected=expected)
    if (state.payload['metadata']['completed_updates'] != plan.horizon
            or state.payload['metadata']['sharing']['arm'] != plan.arm
            or state.payload['metadata']['sharing']['width'] != plan.width):
        raise ValueError('terminal checkpoint differs from own arm/width')
    return state, campaign['freeze_reference']


def run_stages_evaluation(model, processor, *, state, freeze_reference, template, output, guard):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    details = state.payload['metadata']
    if template['conditions'] != list(CONDITIONS):
        raise ValueError('terminal five-condition template required')
    result = {'schema': 'encoder-stages-evaluation-v1', 'status': 'incomplete',
              'arm': details['sharing']['arm'], 'width': details['sharing']['width'],
              'prepared_identity': details['prepared_identity'],
              'checkpoint': asdict(state.reference), 'freeze_reference': freeze_reference, 'completed_panels': [],
              'interpretation': 'previously inspected validation, six trained encoder sites, one seed',
              'vanilla_execution': 'reuse compatible external evidence after identity matching'}
    begun = time.monotonic()
    write_manifest(output / 'result.json', result)
    try:
        for condition in CONDITIONS:
            for site in SITES:
                guard()
                request = {**template, 'schema': 'codec-observation-v1', 'conditions': [condition],
                    'checkpoint_sha256': state.reference.sha256, 'target_site': details['evaluation_sites'][site]['site'],
                    'target_role': 'trained', 'learner_kind': 'shared_encoder_v2', 'step': details['completed_updates'],
                    'site_step': details['sharing']['completed_per_site'][site], 'parent': details['parent_identity'],
                    'comparison': details['comparison_controls'], 'panel': f'final:{site}:{condition}',
                    'source': details['source_identity'], 'config': details['config_identity'],
                    'data': details['data_identity'], 'protocol': details['protocol_identity']}
                request.update(study_arm=details['sharing']['arm'], study_width=details['sharing']['width'],
                               prepared_identity=details['prepared_identity'])
                request['settings'] = {**request['settings'], 'evidence_mode': 'compact-v1'}
                path = output / f'{site}_{condition}'
                receipt = evaluate_checkpoint(state, request, model=model, processor=processor, output=path)
                validate_task_receipt(receipt, request)
                verify_observation_artifacts(receipt, path)
                guard()
                result['completed_panels'].append({'site': site, 'condition': condition,
                    'checkpoint': state.reference.sha256, 'completed_items': receipt['conditions'][0]['completed'],
                    'metrics': receipt['conditions'][0]['metrics'],
                    'observation': {'path': str(path / 'observation.json'), 'sha256': file_digest(path / 'observation.json')}})
                result['elapsed_seconds'] = time.monotonic() - begun
                write_manifest(output / 'result.json', result)
                print(json.dumps({'phase': 'evaluation', 'completed_codec_panels': len(result['completed_panels']),
                                  'elapsed_seconds': result['elapsed_seconds']}), flush=True)
        result.update(status='complete', elapsed_seconds=time.monotonic() - begun)
        write_manifest(output / 'result.json', result)
        return result
    except Exception as error:
        result.update(error={'type': type(error).__name__, 'message': str(error)}, elapsed_seconds=time.monotonic() - begun)
        write_manifest(output / 'result.json', result)
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
