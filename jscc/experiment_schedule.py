"""Finite, offline baseline plans and independently keyed random streams."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math

import torch

CELLS = ("D-none", "D-LN", "R-none", "R-LN")
CONDITIONS = ("no_noise", -6, 0, 6, 12, 18)


@dataclass(frozen=True)
class Segment:
    cell: str
    strategy: str
    start: int
    stop: int
    checkpoints: tuple[int, ...]
    assessments: tuple[int, ...]
    parent: str | None = None


@dataclass(frozen=True)
class BaselinePlan:
    segments: tuple[Segment, ...]
    owner_approval: str | None = None

    @property
    def physical_updates(self):
        return sum(s.stop - s.start for s in self.segments)

    @property
    def task_assessments(self):
        return sum(len(s.assessments) for s in self.segments)

    @property
    def condition_panels(self):
        return self.task_assessments * len(CONDITIONS)

    @property
    def development_item_assessments(self):
        return self.condition_panels * 256


def compile_baseline_plan(selected_cells=(), owner_approval=None):
    """Compile physical operations, reusing simultaneous arms and exact prefixes.

    An approval is a recorded owner decision identifier, never launch authorization.
    Initialization task observations are reused within a selected cell.
    """
    selected = tuple(selected_cells)
    if len(set(selected)) != len(selected) or len(selected) > 2:
        raise ValueError("select at most two distinct cells")
    if any(cell not in CELLS for cell in selected):
        raise ValueError("unknown baseline cell")
    if selected and (not isinstance(owner_approval, str) or not owner_approval.strip()):
        raise ValueError("conditional strategies require a recorded owner approval")
    segments = [
        Segment(c, "both", 0, 400, (100, 200, 300, 400), (0, 200, 400)) for c in CELLS
    ]
    for c in selected:
        parent = f"{c}/reconstruction-prefix/200"
        segments.extend(
            (
                Segment(
                    c, "reconstruction-prefix", 0, 200, (50, 100, 150, 200), (100, 200)
                ),
                Segment(c, "reconstruction", 200, 400, (300, 400), (400,), parent),
                Segment(
                    c, "staged", 200, 400, (250, 300, 350, 400), (300, 400), parent
                ),
            )
        )
    return BaselinePlan(tuple(segments), owner_approval)


def baseline_lr_factor(step):
    if type(step) is not int or not 0 <= step <= 400:
        raise ValueError("baseline schedule is finite: 0..400")
    return step / 20 if step < 20 else 0.5 * (1 + math.cos(math.pi * (step - 20) / 380))


def build_baseline_scheduler(optimizer):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, baseline_lr_factor)


@dataclass(frozen=True)
class NoiseKey:
    study_pairing_id: str
    purpose: str
    site: str
    site_local_batch: int
    sequence_view_identity: str
    microbatch_layout: str
    channel_stream: str = "hidden"
    condition: str = "train"
    draw_schema: str = "cpu-torch-fp32-v1"


def noise_namespace(key: NoiseKey):
    """Stable key identity excluding architecture, checkpoint, phase and global step."""
    return hashlib.sha256(json.dumps(asdict(key), sort_keys=True).encode()).hexdigest()


def paired_noise(key: NoiseKey, shape):
    """CPU float32 draws: identical keys/shapes pair, without global RNG changes."""
    if key.purpose not in {"training", "evaluation", "validation", "acquisition"}:
        raise ValueError("unknown noise purpose")
    if key.draw_schema != "cpu-torch-fp32-v1" or key.site_local_batch < 0:
        raise ValueError("invalid draw schema or site-local batch")
    if len(shape) < 1 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("noise shape must be nonempty and positive")
    namespace = noise_namespace(key)
    seed = int(namespace[:16], 16) % (2**63)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if key.condition == "train":
        if key.purpose != "training":
            raise ValueError("training condition requires training purpose")
        snr = torch.rand(shape[0], generator=generator) * 24 - 6
    elif key.condition == "no_noise":
        snr = torch.full((shape[0],), float("inf"))
    else:
        condition = float(key.condition)
        if condition not in CONDITIONS:
            raise ValueError("unplanned condition")
        snr = torch.full((shape[0],), condition)
    noise = torch.randn(shape, generator=generator)
    if key.condition == "no_noise":
        noise.zero_()
    digest = hashlib.sha256(snr.numpy().tobytes() + noise.numpy().tobytes()).hexdigest()
    return (
        snr,
        noise,
        {
            "key": asdict(key),
            "namespace": namespace,
            "seed": seed,
            "runtime": torch.__version__,
            "algorithm": "torch-cpu-generator",
            "draw_sha256": digest,
        },
    )


@dataclass
class ExposureLedger:
    requested: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    sequence_presentations: int = 0
    source_valid_tokens: int = 0
    target_valid_tokens: int = 0
    receipts: list = field(default_factory=list)

    def record(self, *, status, sequences, source_tokens, target_tokens, receipt):
        if status not in {"completed", "nonfinite", "skipped"}:
            raise ValueError("unknown update status")
        if any(
            type(n) is not int or n < 0
            for n in (sequences, source_tokens, target_tokens)
        ):
            raise ValueError("exposure counts must be nonnegative integers")
        if not receipt or any(r["receipt"] == receipt for r in self.receipts):
            raise ValueError("a unique physical attempt receipt is required")
        self.requested += 1
        self.completed += status == "completed"
        self.failed += status == "nonfinite"
        self.skipped += status == "skipped"
        self.sequence_presentations += sequences
        self.source_valid_tokens += source_tokens
        self.target_valid_tokens += target_tokens
        self.receipts.append(
            dict(
                status=status,
                sequences=sequences,
                source_tokens=source_tokens,
                target_tokens=target_tokens,
                receipt=receipt,
            )
        )
