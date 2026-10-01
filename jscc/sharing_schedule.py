"""Finite encoder-sharing schedule and exact ordered exposure reconciliation.

Batch bindings must come from verified complete input views (including masks and
padded layouts). This module neither reads tensors nor authorizes execution.
Indices are zero based; saved checkpoint steps count completed updates.
"""

from dataclasses import dataclass
import math

from .experiment_schedule import CONDITIONS, NoiseKey

TRAINED_SITES = ("enc_l9", "enc_l19", "enc_fn")
HELDOUT_SITE = "enc_l14"
DEPTH_PROTOCOL = "sharing-dn-q400-effective64-v1"


@dataclass(frozen=True)
class BatchView:
    view_sha256: str
    mask_sha256: str
    layout_sha256: str
    sequences: int
    source_tokens: int
    target_tokens: int
    padded_tokens: int

    def __post_init__(self):
        for digest in (self.view_sha256, self.mask_sha256, self.layout_sha256):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("view, mask and layout require SHA-256 bindings")
        if any(type(n) is not int or n <= 0 for n in (
            self.sequences, self.source_tokens, self.target_tokens, self.padded_tokens
        )) or self.padded_tokens < self.source_tokens:
            raise ValueError("invalid complete-sequence exposure")


def sharing_lr_factor(index: int, horizon: int) -> float:
    """LR used by update index; horizon itself denotes the terminal next LR."""
    if type(horizon) is not int or horizon < 4:
        raise ValueError("horizon must be at least four")
    if type(index) is not int or not 0 <= index <= horizon:
        raise ValueError("schedule index outside finite horizon")
    warmup = max(1, horizon // 20)
    return index / warmup if index < warmup else 0.5 * (
        1 + math.cos(math.pi * (index - warmup) / (horizon - warmup))
    )


@dataclass(frozen=True)
class ScheduledUpdate:
    learner: str
    global_index: int
    site: str
    site_local_index: int
    view: BatchView
    noise_key: NoiseKey
    lr_factor: float


@dataclass(frozen=True)
class Observation:
    purpose: str
    learner: str
    site: str
    step: int
    stage: str
    conditions: tuple = CONDITIONS


@dataclass(frozen=True)
class SharingPlan:
    views: tuple[BatchView, ...]
    study_pairing_id: str
    synthetic: bool = False
    layer_count: int = 26
    protocol_id: str | None = None

    def __post_init__(self):
        if not isinstance(self.views, tuple) or not self.views:
            raise ValueError("a frozen ordered view tuple is required")
        if type(self.synthetic) is not bool or not self.study_pairing_id.strip():
            raise ValueError("explicit synthetic marker and study identity required")
        if type(self.layer_count) is not int or self.layer_count < 20:
            raise ValueError("pinned layer count must contain every declared site")
        if self.q < 4 or self.q % 4:
            raise ValueError("quota must have four distinct integral quarters")
        if len({v.sequences for v in self.views}) != 1:
            raise ValueError("effective batch size must be constant")
        if not self.synthetic:
            quota = 400 if self.protocol_id == DEPTH_PROTOCOL else 200
            if self.q != quota or self.batch_size != 64:
                raise ValueError("production requires q200 or explicit q400 depth protocol and effective batch64")
            if quota == 400 and len({view.view_sha256 for view in self.views}) != 400:
                raise ValueError("q400 requires400 distinct ordered views; cycling q200 is forbidden")

    @property
    def q(self):
        return len(self.views)

    @property
    def batch_size(self):
        return self.views[0].sequences

    @property
    def raw_final_diagnostic(self):
        return f"enc_l{self.layer_count - 1}"

    @property
    def learners(self):
        return tuple(f"specialist_{site}" for site in TRAINED_SITES) + ("shared",)

    def learner(self, name: str):
        return LearnerSchedule(name, self)

    @property
    def saved_states(self):
        return (("common_initialization", 0),) + tuple(
            (name, step) for name in self.learners
            for step in self.learner(name).checkpoint_steps
        )

    @property
    def observations(self):
        result = []
        for purpose in ("task", "objective"):
            for site in (*TRAINED_SITES, HELDOUT_SITE):
                result.append(Observation(purpose, "common_initialization", site, 0,
                                          "post_freeze" if site == HELDOUT_SITE else "pre_freeze"))
            for name in self.learners:
                schedule = self.learner(name)
                steps = schedule.assessment_steps if purpose == "task" else schedule.checkpoint_steps
                sites = (*TRAINED_SITES, HELDOUT_SITE) if name == "shared" else (name.removeprefix("specialist_"),)
                for site in sites:
                    for step in steps:
                        result.append(Observation(purpose, name, site, step,
                                                  "post_freeze" if site == HELDOUT_SITE else "pre_freeze"))
        return tuple(result)

    @property
    def condition_panels(self):
        return {purpose: sum(len(o.conditions) for o in self.observations if o.purpose == purpose)
                for purpose in ("task", "objective")}

    @property
    def additional_vanilla_panels(self):
        return 1


@dataclass(frozen=True)
class LearnerSchedule:
    name: str
    plan: SharingPlan

    def __post_init__(self):
        if self.name not in self.plan.learners:
            raise ValueError("undeclared learner or heldout/diagnostic training")

    @property
    def q(self):
        return self.plan.q

    @property
    def batch_size(self):
        return self.plan.batch_size

    @property
    def horizon(self):
        return self.q * (3 if self.name == "shared" else 1)

    @property
    def checkpoint_steps(self):
        return tuple(self.horizon * fraction // 4 for fraction in (1, 2, 3, 4))

    @property
    def assessment_steps(self):
        return (self.horizon // 2, self.horizon)

    def update_at(self, index: int) -> ScheduledUpdate:
        if type(index) is not int or not 0 <= index < self.horizon:
            raise ValueError("update outside finite horizon")
        if self.name == "shared":
            local, position = divmod(index, 3)
            site = TRAINED_SITES[(local % 3 + position) % 3]
        else:
            local, site = index, self.name.removeprefix("specialist_")
        view = self.plan.views[local]
        key = NoiseKey(self.plan.study_pairing_id, "training", site, local,
                       view.view_sha256, view.layout_sha256,
                       condition="uniform", draw_schema="runtime-awgn-replay-v2")
        return ScheduledUpdate(self.name, index, site, local, view, key,
                               sharing_lr_factor(index, self.horizon))


class SharingExposureLedger:
    """Validate observed bindings against the next plan entry before counting it.

    Caller supplies the observed update using verified actual batch bindings and
    actual draw key. A failed/skipped attempt poisons this ledger; no retry or
    reinterpretation as successful completion is permitted.
    """

    def __init__(self, schedule: LearnerSchedule):
        self.schedule = schedule
        self.completed = 0
        self.failed = False
        self.receipts: list[str] = []
        self.per_site = {site: dict(updates=0, sequences=0, source_tokens=0,
                                    target_tokens=0, padded_tokens=0)
                         for site in TRAINED_SITES}

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
