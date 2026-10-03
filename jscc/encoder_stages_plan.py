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
