"""Current shared state bodies; saved record meanings preserved."""
from __future__ import annotations

import copy
from .checkpoint_readers import _validate_original13_reference as _validate_original13_reference
from dataclasses import asdict
from pathlib import Path
import torch
from .experiment_state import open_state, restore_state
from .asset_identity import state_dict_identity

def recipe_runtime_state(model):
    """Idle observation hook for both released recipe channels."""
    from . import site_routing as routing
    from .models.channel import AWGNChannel, IdentityChannel
    routing.canonical_codec(model)
    if (model._routing_active or model._forward_active or model._transmission_active
            or model._payload_observer is not None):
        raise ValueError('recipe runtime requires an idle route')
    if type(model.channel) is AWGNChannel:
        return {'channel_type': 'awgn', 'idle': routing.observation_state(model)}
    if type(model.channel) is not IdentityChannel:
        raise ValueError('unsupported recipe state channel')
    idle = copy.deepcopy({'schema': 'original13-idle-runtime-v1', 'split': model.split,
        'fields': {key: getattr(model, key) for key in routing._RUNTIME_FIELDS},
        'modes': {name: module.training for name, module in model.named_modules()},
        'channel_stream': 'hidden', 'draw_summaries': [], 'memory_hook_count': len(model._memory_handles)})
    routing._validate_runtime_state(model, idle)
    return {'channel_type': 'identity', 'idle': idle}


def validate_recipe_runtime(model, state):
    from . import site_routing as routing
    current = recipe_runtime_state(model)
    if (not isinstance(state, dict) or set(state) != {'channel_type', 'idle'}
            or state['channel_type'] != current['channel_type']):
        raise ValueError('recipe runtime channel differs')
    routing._validate_runtime_state(model, state['idle'])
    if state['channel_type'] == 'identity' and (
            state['idle']['channel_stream'] != 'hidden' or state['idle']['draw_summaries'] != []):
        raise ValueError('identity channel cannot contain noise state')


def restore_recipe_runtime(model, state):
    from . import site_routing as routing
    validate_recipe_runtime(model, state)
    if state['channel_type'] == 'awgn':
        routing.restore_runtime_state(model, state['idle'])
    else:
        for key, value in state['idle']['fields'].items():
            setattr(model, key, copy.deepcopy(value))
        for name, module in model.named_modules():
            module.training = state['idle']['modes'][name]
        model.activation = model.reconstruction = None
        model.memory_activation = model.memory_reconstruction = None


def _recipe_plan(payload):
    from .shared_plan import RecipePlan
    from .shared_inputs import BatchView
    contract = payload['metadata']['recipe_plan']
    if set(contract) != {'description', 'views', 'study_pairing_id'}:
        raise ValueError('incomplete recipe plan')
    return RecipePlan(contract['description'], [BatchView(**v) for v in contract['views']],
                      contract['study_pairing_id'])


def _recipe_progress(plan, completed, phase_start):
    stage = next(s for s in plan.stages if s.start == phase_start)
    return {**plan.progress(completed, phase_start), 'offset': completed * 16,
            'phase_id': stage.phase_id, 'reset_count': int(stage.reset_optimizer)}


def _recipe_task_contract(plan):
    """Preserve HellaSwag state meaning; bind COCO as a distinct task domain."""
    task = plan.description['config']['task']
    if task == 'hellaswag':
        return {'protocol_identity': 'recipe-training-state-v1'}
    if task == 'coco':
        return {'protocol_identity': 'recipe-coco-training-state-v1', 'task': 'coco'}
    raise ValueError('unsupported recipe state task')


def validate_recipe_state(payload):
    """Complete recipe envelope validation at save/load, never on each update."""
    from .activation_replay import canonical_digest
    from .shared_plan import phase_lr_factor
    metadata = payload['metadata']
    plan = _recipe_plan(payload)
    completed, start = metadata['completed_updates'], metadata['phase_start_step']
    progress = _recipe_progress(plan, completed, start)
    training = plan.description['config']['training']
    stage = next(s for s in plan.stages if s.start == start)
    local = progress['phase_local_completed']
    factor = (phase_lr_factor(local, stage.horizon, training['warmup_ratio'], training['min_lr_ratio'])
              if local < stage.horizon else 0.)
    lr = training['lr'] * factor
    task_contract = _recipe_task_contract(plan)
    if (any(metadata.get(key) != value for key, value in task_contract.items())
            or ('task' in metadata and metadata['task'] != plan.description['config']['task'])
            or metadata['plan_identity'] != plan.identity
            or metadata['stream_identity'] != canonical_digest(metadata['recipe_plan'])
            or metadata['phase'] != stage.phase_id or payload['stream'] != progress
            or metadata['model_revision'] != plan.description['config']['model']['revision']
            or metadata['config_identity'] != canonical_digest(plan.description['config'])
            or metadata['model_state_contract'] != 'recipe-canonical-codec-v1'):
        raise ValueError('recipe plan, stream, phase or body differs')
    for key in ('data_identity', 'prepared_identity', 'observation_identity', 'noise_identity'):
        if not isinstance(metadata.get(key), str) or not metadata[key]:
            raise ValueError('missing recipe identity: ' + key)
    groups, scheduler = payload['optimizer']['param_groups'], payload['scheduler']
    if (not groups or bool(payload['optimizer']['state']) != bool(local)
            or scheduler.get('last_epoch') != local or scheduler.get('_step_count') != local + 1
            or scheduler.get('base_lrs') != [training['lr']] * len(groups)
            or scheduler.get('_last_lr') != [lr] * len(groups)
            or any(g.get('lr') != lr or g.get('initial_lr') != training['lr']
                   or g.get('weight_decay') != training['weight_decay']
                   or g.get('betas') != tuple(training['betas']) or g.get('eps') != training['eps']
                   or g.get('amsgrad') or g.get('maximize') or g.get('differentiable') for g in groups)):
        raise ValueError('recipe Adam/scheduler differs from phase progress')
    counts = progress['completed_per_site']
    exposure = {site: {'updates': count, 'sequences': 16 * count,
        **{field: sum(getattr(v, field) for v in plan.views[:count])
           for field in ('source_tokens', 'target_tokens', 'padded_tokens')}} for site, count in counts.items()}
    receipts = metadata['exposure_receipts']
    if (metadata['exposure'] != exposure or metadata['attempted'] != completed
            or metadata['valid_tokens'] != sum(e['source_tokens'] for e in exposure.values())
            or not isinstance(receipts, list) or len(receipts) != completed
            or any(not isinstance(r, str) or not r for r in receipts) or len(set(receipts)) != completed):
        raise ValueError('recipe exposure receipts differ')
    lineage = metadata['lineage']
    if stage.reset_optimizer:
        if (not isinstance(lineage, list) or len(lineage) != 1 or set(lineage[0]) != {
                'parent_sha256', 'parent_checkpoint', 'completed_updates', 'transfer'}
                or lineage[0]['completed_updates'] != start
                or lineage[0]['transfer'] != 'weights_only_reset_adam_lr'
                or lineage[0]['parent_sha256'] != metadata['parent_identity']
                or lineage[0]['parent_checkpoint']['sha256'] != metadata['parent_identity']):
            raise ValueError('recipe reset lineage differs')
        _validate_original13_reference(lineage[0]['parent_checkpoint'])
    elif lineage != [] or metadata['parent_identity'] is not None:
        raise ValueError('recipe pre-reset lineage differs')
    ancestry = metadata['segment_ancestry']
    if not isinstance(ancestry, list):
        raise ValueError('recipe segment ancestry must be explicit')
    for reference in ancestry:
        _validate_original13_reference(reference)
    if len({r['sha256'] for r in ancestry}) != len(ancestry):
        raise ValueError('recipe ancestry repeats')
    if (not isinstance(metadata['route_state'], dict)
            or set(metadata['route_state']) != {'channel_type', 'idle'}
            or metadata['route_state']['channel_type'] != plan.description['config']['channel']['type']):
        raise ValueError('recipe channel state differs')


class RecipeTrainingState:
    """State owner for RecipePlan; legacy original13 envelopes stay separate."""
    def __init__(self, learner, *, plan, metadata):
        from . import site_routing as routing
        from .activation_replay import canonical_digest
        from .shared_plan import RecipePlan
        if not isinstance(plan, RecipePlan) or learner.schedule is not plan:
            raise TypeError('recipe state requires the learners released RecipePlan')
        task_contract = _recipe_task_contract(plan)
        if learner.task != plan.description['config']['task']:
            raise ValueError('recipe learner task differs from resolved plan')
        if 'task' in metadata and metadata['task'] != learner.task:
            raise ValueError('recipe metadata task differs from resolved learner')
        if learner.completed or learner.failed or learner.phase_start_step:
            raise ValueError('recipe state must be attached to a fresh learner')
        self.learner, self.plan = learner, plan
        self.codec = routing.canonical_codec(learner.model)
        if ({id(p) for p in learner.parameters} != {id(p) for p in self.codec.parameters()}
                or len(learner.parameters) != len(list(self.codec.parameters()))):
            raise ValueError('recipe optimizer requires one canonical body')
        self.contract = {'description': plan.description, 'views': [asdict(v) for v in plan.views],
                         'study_pairing_id': plan.study_pairing_id}
        self.metadata = copy.deepcopy(metadata)
        for key in ('source_identity', 'data_identity', 'prepared_identity', 'observation_identity',
                    'initialization_identity', 'config_identity', 'model_revision'):
            if not isinstance(metadata.get(key), str) or not metadata[key]:
                raise ValueError('missing recipe metadata: ' + key)
        if (metadata['initialization_identity'] != state_dict_identity(routing.export_codec(learner.model))
                or metadata['config_identity'] != canonical_digest(plan.description['config'])
                or metadata['model_revision'] != plan.description['config']['model']['revision']):
            raise ValueError('recipe initializer/config/model identity differs')
        established = {**task_contract,
                       'stream_identity': canonical_digest(self.contract), 'noise_identity': learner.noise_identity}
        if any(key in metadata and metadata[key] != value for key, value in established.items()):
            raise ValueError('recipe stream/protocol/noise identity differs')
        if learner.identity != {'source': metadata['source_identity'], 'config': metadata['config_identity'],
                                'data': metadata['data_identity'], 'parent': None}:
            raise ValueError('recipe learner event identities differ')
        self.metadata.update(established)
        self.reset_parent = None
        self.ancestry = []
        recipe_runtime_state(learner.model)
        learner.optimizer, learner.scheduler = self._optimizer(0)

    def _optimizer(self, start):
        from .shared_training import build_recipe_scheduler
        training = self.plan.description['config']['training']
        optimizer = torch.optim.AdamW(self.learner.parameters, lr=training['lr'],
            weight_decay=training['weight_decay'], betas=tuple(training['betas']), eps=training['eps'])
        return optimizer, build_recipe_scheduler(optimizer, self.plan, start)

    def save(self, path):
        from .experiment_state import save_state
        from .site_routing import export_codec
        learner = self.learner
        if learner.failed or learner.exposure.failed:
            raise ValueError('cannot save a failed recipe learner')
        progress = _recipe_progress(self.plan, learner.completed, learner.phase_start_step)
        details = {**self.metadata, 'recipe_plan': self.contract, 'plan_identity': self.plan.identity,
            'completed_updates': learner.completed, 'phase_start_step': learner.phase_start_step,
            'phase': progress['phase_id'], 'parent_identity': None if self.reset_parent is None else self.reset_parent['sha256'],
            'lineage': [] if self.reset_parent is None else [{'parent_sha256': self.reset_parent['sha256'],
                'parent_checkpoint': self.reset_parent, 'completed_updates': learner.phase_start_step,
                'transfer': 'weights_only_reset_adam_lr'}], 'segment_ancestry': copy.deepcopy(self.ancestry),
            'exposure': copy.deepcopy(learner.exposure.per_site), 'exposure_receipts': list(learner.exposure.receipts),
            'valid_tokens': learner.valid_tokens, 'attempted': learner.attempted,
            'route_state': recipe_runtime_state(learner.model), 'model_state_contract': 'recipe-canonical-codec-v1'}
        reference = save_state(path, model=self.codec, optimizer=learner.optimizer, scheduler=learner.scheduler,
            scaler=learner.scaler, metadata=details, stream_state=progress,
            weight_export=lambda: export_codec(learner.model))
        return open_state(reference, expected=details)

    def _open(self, reference):
        import hashlib
        import io
        data = Path(reference.path).read_bytes()
        if len(data) != reference.size or hashlib.sha256(data).hexdigest() != reference.sha256:
            raise ValueError('recipe checkpoint bytes differ')
        payload = torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)
        expected = {**self.metadata, 'plan_identity': self.plan.identity,
                    'parent_identity': payload['metadata']['parent_identity']}
        return open_state(reference, expected=expected)

    def restore(self, reference):
        from .experiment_state import preflight_restore
        from .site_routing import load_codec
        from .qualification import state_exact
        from .experiment_state import capture_rng
        learner = self.learner
        if learner.failed or learner.exposure.failed:
            raise ValueError('failed recipe learner cannot continue')
        state = self._open(reference)
        details = state.payload['metadata']
        if reference.sha256 in {r['sha256'] for r in details['segment_ancestry']}:
            raise ValueError('recipe continuation repeats ancestry')
        optimizer, scheduler = self._optimizer(details['phase_start_step'])
        preflight_restore(state.payload, model=self.codec, optimizer=optimizer, scaler=learner.scaler)
        validate_recipe_runtime(learner.model, details['route_state'])
        progress = restore_state(state, model=self.codec, optimizer=optimizer, scheduler=scheduler,
            scaler=learner.scaler, weight_load=lambda weights: load_codec(learner.model, weights))
        restore_recipe_runtime(learner.model, details['route_state'])
        if (not state_exact(recipe_runtime_state(learner.model), details['route_state'])
                or not state_exact(capture_rng(), state.payload['rng'])):
            raise ValueError('recipe runtime/RNG restoration differs')
        learner.optimizer, learner.scheduler = optimizer, scheduler
        learner.phase_start_step = details['phase_start_step']
        learner.completed = learner.exposure.completed = progress['completed_updates']
        learner.offset = progress['offset']
        learner.attempted, learner.valid_tokens = details['attempted'], details['valid_tokens']
        learner.exposure.per_site = copy.deepcopy(details['exposure'])
        learner.exposure.receipts = list(details['exposure_receipts'])
        learner.optimizer.zero_grad(set_to_none=True)
        self.reset_parent = None if not details['lineage'] else details['lineage'][0]['parent_checkpoint']
        self.ancestry = [*details['segment_ancestry'], asdict(reference)]
        return state

    def reset(self, parent):
        from .experiment_state import capture_rng
        from .site_routing import export_codec
        from .qualification import state_exact
        learner = self.learner
        reset_stage = next((stage for stage in self.plan.stages if stage.reset_optimizer), None)
        if reset_stage is None:
            raise ValueError('recipe does not declare an optimizer reset')
        boundary = reset_stage.start
        if learner.failed or learner.completed != boundary or learner.phase_start_step != 0 or self.reset_parent is not None:
            raise ValueError('recipe permits exactly one boundary reset')
        verified = self._open(parent.reference)
        if (verified.payload['metadata']['phase_start_step'] != 0
                or verified.payload['metadata']['completed_updates'] != boundary
                or not state_exact(verified.payload['model'], export_codec(learner.model))
                or not state_exact(verified.payload['optimizer'], learner.optimizer.state_dict())
                or not state_exact(verified.payload['scheduler'], learner.scheduler.state_dict())
                or not state_exact(verified.payload['scaler'], learner.scaler.state_dict() if learner.scaler is not None else None)
                or not state_exact(verified.payload['rng'], capture_rng())
                or not state_exact(verified.payload['metadata']['route_state'], recipe_runtime_state(learner.model))
                or verified.payload['metadata']['attempted'] != learner.attempted
                or verified.payload['metadata']['valid_tokens'] != learner.valid_tokens
                or verified.payload['stream']['offset'] != learner.offset
                or verified.payload['metadata']['exposure'] != learner.exposure.per_site
                or verified.payload['metadata']['exposure_receipts'] != learner.exposure.receipts):
            raise ValueError('recipe reset requires exact own pre-reset state')
        optimizer, scheduler = self._optimizer(boundary)
        learner.optimizer, learner.scheduler = optimizer, scheduler
        learner.phase_start_step = boundary
        self.reset_parent = asdict(verified.reference)


def recipe_observation_snapshot(state):
    """Capture training-owned state for a nonadvancing observation, at idle."""
    from .experiment_state import capture_rng
    from .site_routing import export_codec
    learner = state.learner
    return copy.deepcopy({'weights': export_codec(learner.model),
        'optimizer': learner.optimizer.state_dict(), 'scheduler': learner.scheduler.state_dict(),
        'scaler': learner.scaler.state_dict() if learner.scaler is not None else None,
        'counters': [learner.completed, learner.attempted, learner.offset, learner.valid_tokens,
                     learner.phase_start_step, learner.failed],
        'exposure': [learner.exposure.completed, learner.exposure.failed,
                     learner.exposure.per_site, learner.exposure.receipts],
        'grads': [None if p.grad is None else p.grad.detach().clone() for p in learner.parameters],
        'runtime': recipe_runtime_state(learner.model), 'rng': capture_rng(),
        'reset_parent': state.reset_parent, 'ancestry': state.ancestry})


def restore_recipe_observation(state, snapshot):
    """Restore and verify even after scorer exceptions; restoration errors are fatal."""
    from .experiment_state import restore_rng
    from .site_routing import load_codec
    from .qualification import state_exact
    learner = state.learner
    validate_recipe_runtime(learner.model, snapshot['runtime'])
    load_codec(learner.model, snapshot['weights'])
    learner.optimizer.load_state_dict(copy.deepcopy(snapshot['optimizer']))
    learner.scheduler.load_state_dict(copy.deepcopy(snapshot['scheduler']))
    if learner.scaler is not None:
        learner.scaler.load_state_dict(copy.deepcopy(snapshot['scaler']))
    (learner.completed, learner.attempted, learner.offset, learner.valid_tokens,
     learner.phase_start_step, learner.failed) = snapshot['counters']
    (learner.exposure.completed, learner.exposure.failed, learner.exposure.per_site,
     learner.exposure.receipts) = copy.deepcopy(snapshot['exposure'])
    for parameter, grad in zip(learner.parameters, snapshot['grads']):
        parameter.grad = None if grad is None else grad.clone().to(parameter.device)
    restore_recipe_runtime(learner.model, snapshot['runtime'])
    restore_rng(snapshot['rng'])
    state.reset_parent, state.ancestry = copy.deepcopy(snapshot['reset_parent']), copy.deepcopy(snapshot['ancestry'])
    if not state_exact(recipe_observation_snapshot(state), snapshot):
        raise ValueError('recipe observation restoration differs')
