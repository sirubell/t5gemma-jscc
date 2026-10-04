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

