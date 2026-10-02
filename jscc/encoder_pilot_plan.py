"""Fixed original-six encoder pilot; distinct from the formal four-learner study."""
from dataclasses import dataclass

from .activation_replay import canonical_digest
from .experiment_schedule import CONDITIONS, NoiseKey
from .models.split_model import ResolvedSite
from .sharing_schedule import (ENCODER_PILOT_PROTOCOL, BatchView, ScheduledUpdate,
                               sharing_lr_factor)

SITES = ('enc_emb', 'enc_l4', 'enc_l9', 'enc_l14', 'enc_l19', 'enc_fn')
SITE_SPECS = {name: ({'stack': 'enc', 'where': 'after_embed'} if name == 'enc_emb' else
                    {'stack': 'enc', 'where': 'after_final_norm'} if name == 'enc_fn' else
                    {'stack': 'enc', 'where': 'after_layer', 'index': int(name[5:])})
              for name in SITES}


@dataclass(frozen=True)
class EncoderPilotPlan:
    views: tuple[BatchView, ...]
    study_pairing_id: str
    synthetic: bool = False
    protocol_id: str = ENCODER_PILOT_PROTOCOL
    trained_sites: tuple[str, ...] = SITES

    def __post_init__(self):
        if (self.protocol_id != ENCODER_PILOT_PROTOCOL or self.trained_sites != SITES
                or type(self.synthetic) is not bool or not self.study_pairing_id
                or not isinstance(self.views, tuple) or self.q not in ((4,) if self.synthetic else (400, 6400))
                or any(v.sequences != 16 for v in self.views)
                or len({v.view_sha256 for v in self.views}) != self.q):
            raise ValueError('pilot requires original six sites and distinct native16 q400/q6400 views (CPU q4)')

    @property
    def q(self):
        return len(self.views)

    @property
    def batch_size(self):
        return 16

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
    def checkpoint_steps(self):
        if self.q == 6400:
            return (0, 9600, 19200, 28800, 38400)
        return self.objective_steps

    @property
    def objective_steps(self):
        return (0, self.horizon // 2, self.horizon)

    def update_at(self, index):
        if type(index) is not int or not 0 <= index < self.horizon:
            raise ValueError('update outside fixed six-site pilot horizon')
        local, position = divmod(index, 6)
        site, view = SITES[position], self.views[local]
        key = NoiseKey(self.study_pairing_id, 'training', site, local, view.view_sha256,
                       view.layout_sha256, condition='uniform', draw_schema='runtime-awgn-replay-v2')
        return ScheduledUpdate('shared', index, site, local, view, key,
                               sharing_lr_factor(index, self.horizon))


def build_pilot_state(plan, completed):
    if type(completed) is not int or not 0 <= completed <= plan.horizon:
        raise ValueError('pilot completed count outside horizon')
    counts = {site: completed // 6 + (position < completed % 6) for position, site in enumerate(SITES)}
    vectors = {field: [getattr(view, attr) for view in plan.views] for field, attr in
               [('source_valid_per_view', 'source_tokens'), ('target_valid_per_view', 'target_tokens'),
                ('padded_per_view', 'padded_tokens')]}
    next_site = SITES[completed % 6] if completed < plan.horizon else None
    sharing = {'schema': 'encoder-original6-pilot-state-v1', 'synthetic': plan.synthetic,
               'learner': 'shared', 'q': plan.q, 'batch_size': 16, 'trained_sites': list(SITES),
               'horizon': plan.horizon, 'protocol_id': ENCODER_PILOT_PROTOCOL,
               'study_pairing_id': plan.study_pairing_id,
               'ordered_views': [view.__dict__ for view in plan.views], **vectors,
               'completed_per_site': counts, 'next_site': next_site}
    for label, field in [('source_valid_per_site', 'source_valid_per_view'),
                         ('target_valid_per_site', 'target_valid_per_view'), ('padded_per_site', 'padded_per_view')]:
        sharing[label] = {site: sum(vectors[field][:count]) for site, count in counts.items()}
    return sharing, {'completed_updates': completed, 'offset': completed * 16,
                     'view_offsets': counts, 'next_site': next_site}


def validate_pilot_state(payload):
    metadata = payload['metadata']
    sharing = metadata['sharing']
    plan = EncoderPilotPlan(tuple(BatchView(**v) for v in sharing['ordered_views']),
                            sharing['study_pairing_id'], synthetic=sharing['synthetic'])
    expected, stream = build_pilot_state(plan, metadata['completed_updates'])
    if (payload['schema'] != 'experiment-state-v2' or payload['kind'] != 'shared_encoder_v2'
            or metadata['protocol_identity'] != ENCODER_PILOT_PROTOCOL or metadata['phase'] != 'both'
            or canonical_digest(sharing) != canonical_digest(expected) or payload['stream'] != stream
            or type(payload['scheduler']['last_epoch']) is not int
            or payload['scheduler']['last_epoch'] != metadata['completed_updates']):
        raise ValueError('pilot state counters, protocol or horizon mismatch')
    declarations = metadata['evaluation_sites']
    if set(declarations) != set(SITES):
        raise ValueError('pilot requires exactly six declared trained sites')
    topologies = set()
    for name, declaration in declarations.items():
        site = ResolvedSite(**declaration['site'])
        path = ('encoder.layers.0.input' if name == 'enc_emb' else 'encoder.norm' if name == 'enc_fn'
                else f'encoder.layers.{site.index}')
        if (declaration['role'] != 'trained' or site.site_id != name or site.split != SITE_SPECS[name]
                or site.module_path != path or site.layer_count < 20 or site.hidden_dim <= 0
                or site.model_revision != metadata['model_revision']):
            raise ValueError('pilot resolved site declaration mismatch')
        topologies.add((site.model_revision, site.layer_count, site.hidden_dim))
    if len(topologies) != 1:
        raise ValueError('pilot sites have inconsistent backbone topology')


def final_panels():
    """All six no-noise first; caller inserts matched vanilla before SNR panels."""
    return tuple((site, condition) for condition in CONDITIONS for site in SITES)
