"""Ordered location/count contract; tensors and optimizer state belong to consumers."""
from dataclasses import dataclass
from copy import deepcopy
import math

from .activation_replay import canonical_digest
from .experiment_schedule import NoiseKey
from .sharing_schedule import BatchView, ScheduledUpdate


def phase_lr_factor(index, horizon, warmup_ratio, min_lr_ratio):
    warmup = max(1, int(horizon * warmup_ratio))
    if index < warmup:
        return index / warmup
    progress = min(1.0, (index - warmup) / max(1, horizon - warmup))
    return min_lr_ratio + (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2


@dataclass(frozen=True)
class RecipePlan:
    """Bind a resolved trajectory to exactly Q indexed native16 views.

    Repeated content is permitted. The input owner supplies immutable verified
    views in stream order; no file reads or byte hashing occur in update_at.
    """
    description: dict
    views: tuple[BatchView, ...]
    study_pairing_id: str

    def __post_init__(self):
        if self.description.get('identity') != canonical_digest(
                {k: v for k, v in self.description.items() if k != 'identity'}):
            raise ValueError('changed resolved trajectory identity')
        object.__setattr__(self, 'description', deepcopy(self.description))
        if (len(self.views) != self.q or not self.study_pairing_id.strip()
                or any(not isinstance(v, BatchView) or v.sequences != 16 for v in self.views)):
            raise ValueError('recipe requires Q indexed native16 views and pairing identity')

    @property
    def q(self):
        return self.description['updates_per_location']

    @property
    def horizon(self):
        return self.description['counts']['optimizer_updates']

    @property
    def trained_sites(self):
        return tuple(self.description['locations'])

    @property
    def identity(self):
        return self.description['identity']

    @property
    def width(self):
        return self.description['width']

    @property
    def ratio(self):
        return self.description['ratio']

    @property
    def batch_size(self):
        return 16

    @property
    def accumulation(self):
        return 1

    def objective_at(self, index):
        update = self.update_at(index)
        weights = self.description['objectives']
        role = 'decoder' if update.site.startswith('dec_') else 'encoder'
        phase = next(p for p in self.description['phases'] if p['start'] <= index < p['stop'])
        return {**weights[role], 'kl': weights['functional_kl'] if phase['objective_kind'] == 'combined' else 0.0}

    def update_at(self, index):
        if type(index) is not int or not 0 <= index < self.horizon:
            raise ValueError('update outside recipe horizon')
        local, position = divmod(index, len(self.trained_sites))
        site, view = self.trained_sites[position], self.views[local]
        phase = next(p for p in self.description['phases'] if p['start'] <= index < p['stop'])
        key = NoiseKey(self.study_pairing_id, phase['objective_kind'], site, local,
                       view.view_sha256, view.layout_sha256, condition='uniform',
                       draw_schema='original13-role-awgn-v1')
        schedule = self.description['config']['training']
        return ScheduledUpdate('shared', index, site, local, view, key,
            phase_lr_factor(index - phase['start'], phase['stop'] - phase['start'],
                            schedule['warmup_ratio'], schedule['min_lr_ratio']))

    def progress(self, completed, phase_start):
        if (type(completed) is not int or type(phase_start) is not int
                or not any(p['start'] == phase_start and p['start'] <= completed <= p['stop']
                           for p in self.description['phases'])):
            raise ValueError('recipe progress crosses an unrecorded reset')
        n = len(self.trained_sites)
        return {'completed_updates': completed, 'next_view': completed // n,
                'next_site': self.trained_sites[completed % n] if completed < self.horizon else None,
                'completed_per_site': {site: completed // n + int(i < completed % n)
                                       for i, site in enumerate(self.trained_sites)},
                'phase_start_step': phase_start, 'phase_local_completed': completed - phase_start}
