"""Original13 optimizer, phase reset and complete checkpoint state ownership."""
from dataclasses import asdict
import copy
from pathlib import Path

import torch

from .activation_replay import file_digest
from .experiment_state import open_state, restore_state
from .sharing_preparation import state_dict_identity
from .sharing_schedule import NATIVE16_WEIGHT_DECAY


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
        # Preserve the legacy checkpoint writer injection point.
        from .encoder_stages import save_state
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
        for key in ('source_commit', 'model_assets_identity'):
            if key in self.metadata:
                expected[key] = self.metadata[key]
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


def validate_original13_state(payload):
    from .encoder_stages_plan import ORIGINAL13_PROTOCOL, Original13Plan, original13_progress
    from .sharing_schedule import sharing_lr_factor
    metadata = payload['metadata']
    plan = Original13Plan.from_resolved(metadata['original13_plan'])
    progress = original13_progress(plan, metadata['completed_updates'], metadata['phase_start_step'])
    if (metadata['protocol_identity'] != ORIGINAL13_PROTOCOL or metadata['plan_identity'] != plan.identity
            or metadata['input_identity'] != plan.input_identity
            or metadata['observation_identity'] != plan.observation_identity
            or metadata['phase'] != progress['phase_id'] or payload['stream'] != progress):
        raise ValueError('original13 state plan, input, phase or stream differs')
    if bool(metadata['parent_identity']) != bool(progress['reset_count']):
        raise ValueError('original13 reset parent missing or unexpected')
    local = progress['phase_local_completed']
    if bool(payload['optimizer'].get('state')) != bool(local):
        raise ValueError('original13 optimizer state differs from phase progress')
    factor = sharing_lr_factor(local, next(s.horizon for s in plan.stages if s.start == metadata['phase_start_step']))
    if any(group['lr'] != 2e-4 * factor for group in payload['optimizer']['param_groups']):
        raise ValueError('original13 LR differs from phase-local schedule')
    groups = payload['optimizer']['param_groups']
    scheduler = payload['scheduler']
    if (scheduler.get('base_lrs') != [2e-4] * len(groups)
            or scheduler.get('_last_lr') != [2e-4 * factor] * len(groups)
            or scheduler.get('_step_count') != local + 1
            or any(g.get('initial_lr') != 2e-4 or g.get('weight_decay') != 0.002500001398541448
                   or g.get('betas') != (.9, .999) or g.get('eps') != 1e-8 or g.get('amsgrad')
                   or g.get('maximize') or g.get('differentiable') for g in groups)
            or not isinstance(metadata['route_state'], dict)):
        raise ValueError('original13 scheduler, Adam recipe or route state differs')
    counts = progress['completed_per_site']
    expected_exposure = {site: {'updates': count, 'sequences': 16 * count,
        **{field: sum(getattr(v, field) for v in plan.views[:count])
           for field in ('source_tokens', 'target_tokens', 'padded_tokens')}} for site, count in counts.items()}
    receipts = metadata['exposure_receipts']
    if (metadata['exposure'] != expected_exposure or metadata['attempted'] != metadata['completed_updates']
            or metadata['valid_tokens'] != sum(v['source_tokens'] for v in expected_exposure.values())
            or not isinstance(receipts, list) or len(receipts) != metadata['completed_updates']
            or any(not isinstance(r, str) or not r for r in receipts) or len(set(receipts)) != len(receipts)
            or not isinstance(metadata['segment_ancestry'], list)
            or metadata['model_state_contract'] != 'original13-canonical-codec-v1'):
        raise ValueError('original13 complete exposure, ancestry or body contract differs')
    if progress['reset_count']:
        entry = metadata['lineage'][-1]
        if (len(metadata['lineage']) != 1 or set(entry) != {
                'parent_sha256', 'parent_checkpoint', 'completed_updates', 'transfer'}
                or entry['completed_updates'] != plan.reconstruction_stop
                or entry['transfer'] != 'weights_only_reset_adam_lr'
                or entry['parent_checkpoint']['sha256'] != metadata['parent_identity']):
            raise ValueError('original13 reset lineage differs')
        _validate_original13_reference(entry['parent_checkpoint'])
    elif metadata['lineage'] != []:
        raise ValueError('original13 reconstruction state cannot have reset lineage')
    for reference in metadata['segment_ancestry']:
        _validate_original13_reference(reference)
    if len({r['sha256'] for r in metadata['segment_ancestry']}) != len(metadata['segment_ancestry']):
        raise ValueError('original13 segment ancestry repeats a checkpoint')



def _validate_original13_reference(reference):
    if (not isinstance(reference, dict) or set(reference) != {'path', 'sha256', 'size'}
            or not isinstance(reference['path'], str) or not reference['path'].strip()
            or not isinstance(reference['sha256'], str) or len(reference['sha256']) != 64
            or any(c not in '0123456789abcdef' for c in reference['sha256'])
            or type(reference['size']) is not int or reference['size'] <= 0):
        raise ValueError('original13 ancestry requires complete checkpoint references')


def recipe_runtime_state(model):
    """Idle observation hook for both released recipe channels."""
    from . import original13_routing as routing
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
    from . import original13_routing as routing
    current = recipe_runtime_state(model)
    if (not isinstance(state, dict) or set(state) != {'channel_type', 'idle'}
            or state['channel_type'] != current['channel_type']):
        raise ValueError('recipe runtime channel differs')
    routing._validate_runtime_state(model, state['idle'])
    if state['channel_type'] == 'identity' and (
            state['idle']['channel_stream'] != 'hidden' or state['idle']['draw_summaries'] != []):
        raise ValueError('identity channel cannot contain noise state')


def restore_recipe_runtime(model, state):
    from . import original13_routing as routing
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
    from .recipe_plan import RecipePlan
    from .sharing_schedule import BatchView
    contract = payload['metadata']['recipe_plan']
    if set(contract) != {'description', 'views', 'study_pairing_id'}:
        raise ValueError('incomplete recipe plan')
    return RecipePlan(contract['description'], [BatchView(**v) for v in contract['views']],
                      contract['study_pairing_id'])


def _recipe_progress(plan, completed, phase_start):
    stage = next(s for s in plan.stages if s.start == phase_start)
    return {**plan.progress(completed, phase_start), 'offset': completed * 16,
            'phase_id': stage.phase_id, 'reset_count': int(stage.reset_optimizer)}


def validate_recipe_state(payload):
    """Complete recipe envelope validation at save/load, never on each update."""
    from .activation_replay import canonical_digest
    from .recipe_plan import phase_lr_factor
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
    if (metadata['protocol_identity'] != 'recipe-training-state-v1'
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
        from . import original13_routing as routing
        from .activation_replay import canonical_digest
        from .recipe_plan import RecipePlan
        if not isinstance(plan, RecipePlan) or learner.schedule is not plan:
            raise TypeError('recipe state requires the learners released RecipePlan')
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
        established = {'protocol_identity': 'recipe-training-state-v1',
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
        from .original13_routing import build_recipe_scheduler
        training = self.plan.description['config']['training']
        optimizer = torch.optim.AdamW(self.learner.parameters, lr=training['lr'],
            weight_decay=training['weight_decay'], betas=tuple(training['betas']), eps=training['eps'])
        return optimizer, build_recipe_scheduler(optimizer, self.plan, start)

    def save(self, path):
        from .experiment_state import save_state
        from .original13_routing import export_codec
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
        from .original13_routing import load_codec
        from .sharing_qualification_guard import state_exact
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
        from .original13_routing import export_codec
        from .sharing_qualification_guard import state_exact
        learner = self.learner
        boundary = self.plan.stages[1].start
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
    from .original13_routing import export_codec
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
    from .original13_routing import load_codec
    from .sharing_qualification_guard import state_exact
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
