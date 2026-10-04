"""Ordered location/count contract; tensors and optimizer state belong to consumers."""
from dataclasses import dataclass
from copy import deepcopy
import math

from .activation_replay import canonical_digest
from .experiment_schedule import NoiseKey
from .sharing_schedule import BatchView, ScheduledUpdate


def phase_lr_factor(index, horizon, warmup_ratio, min_lr_ratio):
    warmup = max(1, int(horizon * warmup_ratio)) if warmup_ratio > 0 else 0
    if index < warmup:
        return index / warmup
    progress = min(1.0, (index - warmup) / max(1, horizon - warmup))
    return min_lr_ratio + (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2


@dataclass(frozen=True)
class RecipeStage:
    """Phase boundaries only; objectives and LR come from the recipe plan."""
    phase_id: str
    objective_kind: str
    start: int
    stop: int
    reset_optimizer: bool

    @property
    def horizon(self):
        return self.stop - self.start


@dataclass(frozen=True, init=False)
class RecipePlan:
    """Bind a resolved trajectory to exactly Q indexed native16 views.

    Repeated content is permitted. The input owner supplies immutable verified
    views in stream order; no file reads or byte hashing occur in update_at.
    """
    _description: dict
    views: tuple[BatchView, ...]
    study_pairing_id: str

    def __init__(self, description, views, study_pairing_id):
        if description.get('identity') != canonical_digest(
                {k: v for k, v in description.items() if k != 'identity'}):
            raise ValueError('changed resolved trajectory identity')
        object.__setattr__(self, '_description', deepcopy(description))
        object.__setattr__(self, 'views', tuple(views))
        object.__setattr__(self, 'study_pairing_id', study_pairing_id)
        if (len(self.views) != self.q or not self.study_pairing_id.strip()
                or any(not isinstance(v, BatchView) or v.sequences != 16 for v in self.views)):
            raise ValueError('recipe requires Q indexed native16 views and pairing identity')

    @property
    def description(self):
        return deepcopy(self._description)

    @property
    def q(self):
        return self._description['updates_per_location']

    @property
    def horizon(self):
        return self._description['counts']['optimizer_updates']

    @property
    def trained_sites(self):
        return tuple(self._description['locations'])

    @property
    def identity(self):
        return self._description['identity']

    @property
    def width(self):
        return self._description['width']

    @property
    def ratio(self):
        return self._description['ratio']

    @property
    def batch_size(self):
        return 16

    @property
    def accumulation(self):
        return 1

    @property
    def stages(self):
        return tuple(RecipeStage(p['phase_id'], p['objective_kind'], p['start'], p['stop'],
                                  p['reset_optimizer']) for p in self._description['phases'])

    def stage_at(self, index):
        if type(index) is not int or not 0 <= index < self.horizon:
            raise ValueError('update outside recipe horizon')
        return next(stage for stage in self.stages if stage.start <= index < stage.stop)

    def objective_at(self, index):
        update = self.update_at(index)
        weights = self._description['objectives']
        role = 'decoder' if update.site.startswith('dec_') else 'encoder'
        phase = next(p for p in self._description['phases'] if p['start'] <= index < p['stop'])
        return {**weights[role], 'kl': weights['functional_kl'] if phase['objective_kind'] == 'combined' else 0.0}

    def update_at(self, index):
        if type(index) is not int or not 0 <= index < self.horizon:
            raise ValueError('update outside recipe horizon')
        local, position = divmod(index, len(self.trained_sites))
        site, view = self.trained_sites[position], self.views[local]
        phase = next(p for p in self._description['phases'] if p['start'] <= index < p['stop'])
        key = NoiseKey(self.study_pairing_id, phase['objective_kind'], site, local,
                       view.view_sha256, view.layout_sha256, condition='uniform',
                       draw_schema='original13-role-awgn-v1')
        schedule = self._description['config']['training']
        return ScheduledUpdate('shared', index, site, local, view, key,
            phase_lr_factor(index - phase['start'], phase['stop'] - phase['start'],
                            schedule['warmup_ratio'], schedule['min_lr_ratio']))

    def progress(self, completed, phase_start):
        if (type(completed) is not int or type(phase_start) is not int
                or not any(p['start'] == phase_start and p['start'] <= completed <= p['stop']
                           for p in self._description['phases'])):
            raise ValueError('recipe progress crosses an unrecorded reset')
        n = len(self.trained_sites)
        return {'completed_updates': completed, 'next_view': completed // n,
                'next_site': self.trained_sites[completed % n] if completed < self.horizon else None,
                'completed_per_site': {site: completed // n + int(i < completed % n)
                                       for i, site in enumerate(self.trained_sites)},
                'phase_start_step': phase_start, 'phase_local_completed': completed - phase_start,
                'base_stream_presentations': ((completed + n - 1) // n) * self.batch_size,
                'site_presentations': completed * self.batch_size}
