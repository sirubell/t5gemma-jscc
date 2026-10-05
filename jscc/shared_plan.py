"""Ordered shared update, phase and exposure algebra."""
from __future__ import annotations

from dataclasses import dataclass
from .experiment_schedule import NoiseKey
from .shared_inputs import BatchView
from copy import deepcopy
import math
from .activation_replay import canonical_digest

# Transferred from sharing_schedule; serialized meanings remain unchanged.
@dataclass(frozen=True)
class ScheduledUpdate:
    learner: str
    global_index: int
    site: str
    site_local_index: int
    view: BatchView
    noise_key: NoiseKey
    lr_factor: float


class SharingExposureLedger:
    """Validate observed bindings against the next plan entry before counting it.

    Caller supplies the observed update using verified actual batch bindings and
    actual draw key. A failed/skipped attempt poisons this ledger; no retry or
    reinterpretation as successful completion is permitted.
    """

    def __init__(self, schedule, *, sites=("enc_l9", "enc_l19", "enc_fn")):
        self.schedule = schedule
        self.completed = 0
        self.failed = False
        self.receipts: list[str] = []
        self.per_site = {site: dict(updates=0, sequences=0, source_tokens=0,
                                    target_tokens=0, padded_tokens=0)
                         for site in sites}

    def record(self, update: ScheduledUpdate, *, receipt: str, status="completed"):
        if self.failed:
            raise ValueError("terminal exposure failure")
        try:
            if status != "completed":
                raise ValueError("failed or skipped update cannot count as completed")
            if not isinstance(receipt, str) or not receipt.strip() or receipt in self.receipts:
                raise ValueError("unique physical attempt receipt required")
            if update != self.schedule.update_at(self.completed):
                raise ValueError("observed update differs from ordered view/mask/layout/noise plan")
        except ValueError:
            self.failed = True
            raise
        self.receipts.append(receipt)
        self.completed += 1
        counts = self.per_site[update.site]
        counts["updates"] += 1
        for field in ("sequences", "source_tokens", "target_tokens", "padded_tokens"):
            counts[field] += getattr(update.view, field)

    def finalize(self):
        if self.failed or self.completed != self.schedule.horizon:
            raise ValueError("incomplete or failed exposure ledger")
        return {site: dict(counts) for site, counts in self.per_site.items()}


# Transferred from recipe_plan; serialized meanings remain unchanged.
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
        stage = self.stage_at(index)
        site = self.trained_sites[index % len(self.trained_sites)]
        weights = self._description['objectives']
        role = 'decoder' if site.startswith('dec_') else 'encoder'
        return {**weights[role], 'kl': weights['functional_kl'] if stage.objective_kind == 'combined' else 0.0}

    def update_at(self, index):
        stage = self.stage_at(index)
        local, position = divmod(index, len(self.trained_sites))
        site, view = self.trained_sites[position], self.views[local]
        key = NoiseKey(self.study_pairing_id, stage.objective_kind, site, local,
                       view.view_sha256, view.layout_sha256, condition='uniform',
                       draw_schema='original13-role-awgn-v1')
        schedule = self._description['config']['training']
        return ScheduledUpdate('shared', index, site, local, view, key,
            phase_lr_factor(index - stage.start, stage.horizon,
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

