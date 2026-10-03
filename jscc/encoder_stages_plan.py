"""Matched six-site A/E/R trajectories with explicit optimizer-reset boundaries."""
from dataclasses import dataclass
import math

from .activation_replay import canonical_digest
from .encoder_pilot_plan import SITES, SITE_SPECS
from .experiment_schedule import NoiseKey
from .models.split_model import ResolvedSite
from .sharing_schedule import BatchView, ScheduledUpdate, sharing_lr_factor

ENCODER_STAGES_PROTOCOL = 'encoder-original6-native16-stages-v1'
WIDTHS = (512, 1152, 2304)
ARMS = ('A', 'E', 'R')
PEAK_LEARNING_RATE = 2e-4
TOTAL_SCIENTIFIC_UPDATES = len(WIDTHS) * len(ARMS) * 60000


@dataclass(frozen=True)
class EncoderStage:
    phase_id: str
    objective_kind: str
    start: int
    stop: int
    reset_optimizer: bool = False

    @property
    def horizon(self):
        return self.stop - self.start

    @property
    def objective_weights(self):
        return {'kl': 0.0 if self.objective_kind == 'reconstruction' else 1.0,
                'hidden': 0.1, 'memory': 0.0}

    @property
    def warmup_updates(self):
        return max(1, self.horizon // 20)


@dataclass(frozen=True)
class EncoderStagesPlan:
    views: tuple[BatchView, ...]
    study_pairing_id: str
    arm: str
    synthetic: bool = False
    width: int = 512
    protocol_id: str = ENCODER_STAGES_PROTOCOL
    trained_sites: tuple[str, ...] = SITES

    def __post_init__(self):
        if (self.protocol_id != ENCODER_STAGES_PROTOCOL or self.trained_sites != SITES
                or self.arm not in ARMS or type(self.synthetic) is not bool
                or not isinstance(self.study_pairing_id, str) or not self.study_pairing_id.strip()
                or not isinstance(self.views, tuple) or self.q < 2 or self.q % 2
                or (not self.synthetic and self.q != 10000)
                or any(not isinstance(v, BatchView) or v.sequences != 16 for v in self.views)
                or len({v.view_sha256 for v in self.views}) != self.q
                or type(self.width) is not int or self.width <= 0
                or (not self.synthetic and self.width not in WIDTHS)):
            raise ValueError('stages require six sites, A/E/R, distinct native16 q10000 views and approved width (CPU even q)')

    @property
    def q(self):
        return len(self.views)

    @property
    def batch_size(self):
        return 16

    @property
    def accumulation(self):
        return 1

    @property
    def name(self):
        return 'shared'

    @property
    def plan(self):
        return self

    @property
    def horizon(self):
        return 6 * self.q

    @property
    def boundary_steps(self):
        return (0, self.horizon // 2, self.horizon)

    @property
    def reset_steps(self):
        return () if self.arm == 'A' else (self.horizon // 2,)

    @property
    def stages(self) -> tuple[EncoderStage, ...]:
        if self.arm == 'A':
            return (EncoderStage('phase1', 'combined', 0, self.horizon),)
        midpoint = self.horizon // 2
        return (EncoderStage('phase1', 'reconstruction' if self.arm == 'R' else 'combined', 0, midpoint),
                EncoderStage('phase2', 'combined', midpoint, self.horizon, True))

    @property
    def checkpoint_steps(self):
        interval = 6 * max(1, self.q // 10)
        return tuple(sorted(set(range(0, self.horizon + 1, interval)) | set(self.boundary_steps)))

    @property
    def objective_steps(self):
        return ()

    @property
    def observations(self):
        return ()

    def stage_at(self, index):
        if type(index) is not int or not 0 <= index < self.horizon:
            raise ValueError('update outside fixed six-site stages horizon')
        return self.stages[0] if self.arm == 'A' or index < self.horizon // 2 else self.stages[1]

    def objective_kind_at(self, index):
        return self.stage_at(index).objective_kind

    def scheduler_index_at(self, index):
        return index - self.stage_at(index).start

    def scheduler_horizon_at(self, index):
        return self.stage_at(index).horizon

    def update_at(self, index):
        stage = self.stage_at(index)
        local, position = divmod(index, 6)
        site, view = SITES[position], self.views[local]
        key = NoiseKey(self.study_pairing_id, 'training', site, local, view.view_sha256,
                       view.layout_sha256, condition='uniform', draw_schema='runtime-awgn-replay-v2')
        return ScheduledUpdate('shared', index, site, local, view, key,
                               sharing_lr_factor(index - stage.start, stage.horizon))


def build_stages_state(plan, completed, *, phase_start_step=None):
    """Retain global stream/exposure while identifying either side of a reset.

    At the midpoint, start zero denotes the saved phase-one terminal checkpoint;
    start midpoint denotes the distinct fresh-optimizer phase-two checkpoint.
    """
    if type(completed) is not int or not 0 <= completed <= plan.horizon:
        raise ValueError('stages completed count outside horizon')
    if phase_start_step is None:
        phase_start_step = plan.stages[-1].start if completed >= plan.horizon // 2 else 0
    matches = [stage for stage in plan.stages if stage.start == phase_start_step
               and stage.start <= completed <= stage.stop]
    if type(phase_start_step) is not int or len(matches) != 1:
        raise ValueError('invalid phase start or completed count across reset boundary')
    stage = matches[0]
    counts = {site: completed // 6 + (position < completed % 6) for position, site in enumerate(SITES)}
    vectors = {field: [getattr(view, attr) for view in plan.views] for field, attr in
               [('source_valid_per_view', 'source_tokens'), ('target_valid_per_view', 'target_tokens'),
                ('padded_per_view', 'padded_tokens')]}
    next_site = SITES[completed % 6] if completed < plan.horizon else None
    sharing = {'schema': 'encoder-original6-stages-state-v1', 'synthetic': plan.synthetic,
               'learner': 'shared', 'arm': plan.arm, 'width': plan.width,
               'q': plan.q, 'batch_size': 16, 'accumulation': 1, 'trained_sites': list(SITES),
               'horizon': plan.horizon, 'protocol_id': ENCODER_STAGES_PROTOCOL,
               'study_pairing_id': plan.study_pairing_id, 'completed_updates': completed,
               'phase_id': stage.phase_id, 'phase_start_step': stage.start, 'phase_stop_step': stage.stop,
               'phase_local_completed': completed - stage.start,
               'objective_kind': stage.objective_kind, 'optimizer_reset_step': stage.start if stage.reset_optimizer else None,
               'ordered_views': [view.__dict__ for view in plan.views], **vectors,
               'completed_per_site': counts, 'next_site': next_site}
    for label, field in [('source_valid_per_site', 'source_valid_per_view'),
                         ('target_valid_per_site', 'target_valid_per_view'), ('padded_per_site', 'padded_per_view')]:
        sharing[label] = {site: sum(vectors[field][:count]) for site, count in counts.items()}
    return sharing, {'completed_updates': completed, 'offset': completed * 16,
                     'view_offsets': counts, 'next_site': next_site}


def validate_stages_state(payload):
    metadata = payload['metadata']
    sharing = metadata['sharing']
    plan = EncoderStagesPlan(tuple(BatchView(**v) for v in sharing['ordered_views']),
                             sharing['study_pairing_id'], sharing['arm'],
                             synthetic=sharing['synthetic'], width=sharing['width'])
    expected, stream = build_stages_state(plan, metadata['completed_updates'],
                                          phase_start_step=sharing['phase_start_step'])
    local = expected['phase_local_completed']
    if (payload['schema'] != 'experiment-state-v2' or payload['kind'] != 'shared_encoder_v2'
            or metadata['protocol_identity'] != ENCODER_STAGES_PROTOCOL
            or metadata['phase'] != expected['phase_id']
            or canonical_digest(sharing) != canonical_digest(expected) or payload['stream'] != stream
            or type(payload['scheduler']['last_epoch']) is not int or payload['scheduler']['last_epoch'] != local):
        raise ValueError('stages state counters, phase, protocol or horizon mismatch')
    moments = payload['optimizer'].get('state')
    if not isinstance(moments, dict) or (not moments) != (local == 0):
        raise ValueError('stages optimizer moments mismatch phase-local updates')
    for state in moments.values():
        try:
            step = float(state['step'])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('invalid stages optimizer step') from error
        if not math.isfinite(step) or step != local:
            raise ValueError('stages optimizer step mismatch phase-local updates')
    declarations = metadata['evaluation_sites']
    if set(declarations) != set(SITES):
        raise ValueError('stages requires exactly six declared trained sites')
    topologies = set()
    for name, declaration in declarations.items():
        site = ResolvedSite(**declaration['site'])
        path = ('encoder.layers.0.input' if name == 'enc_emb' else 'encoder.norm' if name == 'enc_fn'
                else f'encoder.layers.{site.index}')
        if (declaration['role'] != 'trained' or site.site_id != name or site.split != SITE_SPECS[name]
                or site.module_path != path or type(site.layer_count) is not int or site.layer_count < 20
                or type(site.hidden_dim) is not int or site.hidden_dim <= 0
                or not site.model_revision or site.model_revision != metadata['model_revision']):
            raise ValueError('stages resolved site declaration mismatch')
        topologies.add((site.model_revision, site.layer_count, site.hidden_dim))
    if len(topologies) != 1:
        raise ValueError('stages sites have inconsistent backbone topology')


ORIGINAL13_PROTOCOL = 'original13-native16-ratios-v1'
ORIGINAL13_SITES = ('enc_emb', 'enc_l4', 'enc_l9', 'enc_l14', 'enc_l19', 'enc_fn',
                    'dec_l0', 'dec_l4', 'dec_l8', 'dec_l12', 'dec_l16', 'dec_l20', 'dec_l24')
ORIGINAL13_RATIOS = (50, 75, 90)
OBSERVATION_SITES = ('enc_l9', 'enc_fn', 'dec_l0', 'dec_l24')


@dataclass(frozen=True)
class Original13Plan:
    views: tuple[BatchView, ...]
    study_pairing_id: str
    ratio: int
    width: int
    input_identity: str
    observation_identity: str
    synthetic: bool = False

    def __post_init__(self):
        if (type(self.ratio) is not int or self.ratio not in ORIGINAL13_RATIOS
                or type(self.width) is not int or self.width <= 0
                or type(self.synthetic) is not bool or not isinstance(self.views, tuple)
                or not self.views or self.q % 400
                or (not self.synthetic and (self.q != 10000 or self.width not in WIDTHS))
                or any(not isinstance(v, BatchView) or v.sequences != 16 for v in self.views)
                or len({v.view_sha256 for v in self.views}) != self.q
                or not isinstance(self.study_pairing_id, str) or not self.study_pairing_id.strip()):
            raise ValueError('original13 requires fixed ratios, width and distinct native16 complete-quarter views')
        for identity in (self.input_identity, self.observation_identity):
            if not isinstance(identity, str) or len(identity) != 64 or any(c not in '0123456789abcdef' for c in identity):
                raise ValueError('original13 input and observation require SHA256 identities')

    @property
    def q(self):
        return len(self.views)

    @property
    def plan(self):
        return self

    @property
    def name(self):
        return 'shared'

    @property
    def protocol_id(self):
        return ORIGINAL13_PROTOCOL

    @property
    def trained_sites(self):
        return ORIGINAL13_SITES

    @property
    def batch_size(self):
        return 16

    @property
    def accumulation(self):
        return 1

    @property
    def horizon(self):
        return 13 * self.q

    @property
    def reconstruction_stop(self):
        return self.horizon * self.ratio // 100

    @property
    def stages(self):
        stop = self.reconstruction_stop
        return (EncoderStage('reconstruction', 'reconstruction', 0, stop),
                EncoderStage('functional', 'combined', stop, self.horizon, True))

    @property
    def checkpoint_states(self):
        stop = self.reconstruction_stop
        functional = self.horizon - stop
        states = [(0, 'initialization')]
        states += [(stop * n // 4, 'pre_reset' if n == 4 else 'reconstruction') for n in range(1, 5)]
        states += [(stop, 'post_reset'), (stop + functional // 10, 'functional_observation')]
        states += [(stop + functional * n // 4, 'terminal' if n == 4 else 'functional') for n in range(1, 5)]
        return tuple(states)

    @property
    def observations(self):
        return ((self.reconstruction_stop, 'pre_reset'),
                (self.reconstruction_stop + (self.horizon - self.reconstruction_stop) // 10,
                 'functional_observation'))

    def stage_at(self, index):
        if type(index) is not int or not 0 <= index < self.horizon:
            raise ValueError('update outside original13 horizon')
        return self.stages[int(index >= self.reconstruction_stop)]

    def objective_at(self, index):
        stage = self.stage_at(index)
        decoder = ORIGINAL13_SITES[index % 13].startswith('dec_')
        return {'kl': float(stage.objective_kind == 'combined'),
                'hidden': .05 if decoder else .1, 'memory': .05 if decoder else 0.0}

    def update_at(self, index):
        stage = self.stage_at(index)
        local, position = divmod(index, 13)
        view, site = self.views[local], ORIGINAL13_SITES[position]
        key = NoiseKey(self.study_pairing_id, stage.objective_kind, site, local,
                       view.view_sha256, view.layout_sha256, condition='uniform',
                       draw_schema='original13-role-awgn-v1')
        return ScheduledUpdate('shared', index, site, local, view, key,
                               sharing_lr_factor(index - stage.start, stage.horizon))

    def resolved(self):
        """Versioned immutable plan; ratios never alter initializer/input identity."""
        return {'schema': ORIGINAL13_PROTOCOL, 'sites': list(ORIGINAL13_SITES),
                'roles': ['encoder_hidden', 'decoder_hidden', 'receiver_memory'],
                'ratio': self.ratio, 'width': self.width, 'synthetic': self.synthetic,
                'views': [v.__dict__ for v in self.views], 'study_pairing_id': self.study_pairing_id,
                'input_identity': self.input_identity, 'observation_identity': self.observation_identity,
                'batch_size': 16, 'accumulation': 1, 'quota_per_site': self.q, 'horizon': self.horizon,
                'stages': [{**s.__dict__, 'warmup_updates': s.warmup_updates,
                            'peak_lr': 2e-4, 'min_lr_ratio': 0} for s in self.stages],
                'objectives': {'encoder': {'hidden': .1, 'memory': 0},
                               'decoder': {'hidden': .05, 'memory': .05}, 'functional_kl': 1.0,
                               'denominators': {'kl': 'valid_target_tokens', 'hidden': 'valid_hidden_samples',
                                                'memory': 'valid_memory_samples'}},
                'checkpoint_states': [list(s) for s in self.checkpoint_states],
                'observations': [{'step': step, 'state': state, 'panel': self.observation_identity,
                                  'sites': list(OBSERVATION_SITES), 'conditions': ['no_noise', -6]}
                                 for step, state in self.observations],
                'terminal_requests': {'initial': ['no_noise', -6], 'supplemental_supported': [0, 6, 18],
                                      'supplemental_requested': False, 'items': 10042,
                                      'fewshot': 5, 'candidate_batch': 16, 'scoring_policy': 'fp32-v1'}}

    @property
    def identity(self):
        return canonical_digest(self.resolved())

    @classmethod
    def from_resolved(cls, value):
        plan = cls(tuple(BatchView(**v) for v in value['views']), value['study_pairing_id'],
                   value['ratio'], value['width'], value['input_identity'], value['observation_identity'],
                   value['synthetic'])
        if canonical_digest(value) != plan.identity:
            raise ValueError('unsupported or changed original13 plan')
        return plan


def original13_progress(plan, completed, phase_start):
    if type(completed) is not int or type(phase_start) is not int:
        raise ValueError('original13 progress requires integer counts')
    matches = [s for s in plan.stages if s.start == phase_start and s.start <= completed <= s.stop]
    if len(matches) != 1:
        raise ValueError('original13 progress crosses an unrecorded reset')
    phase = matches[0]
    counts = {site: completed // 13 + int(i < completed % 13) for i, site in enumerate(ORIGINAL13_SITES)}
    return {'completed_updates': completed, 'next_site': ORIGINAL13_SITES[completed % 13]
            if completed < plan.horizon else None, 'next_view': completed // 13,
            'offset': completed * 16, 'completed_per_site': counts,
            'phase_id': phase.phase_id, 'phase_start_step': phase.start,
            'phase_local_completed': completed - phase.start, 'reset_count': int(phase.reset_optimizer)}


def validate_original13_state(payload):
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
