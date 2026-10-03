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


class AllocationStopped(RuntimeError):
    """The cumulative budget or a terminal attempt prevents further commands."""


class AllocationLedger:
    """Durable device-allocation accounting, independent of GPU-active time.

    Poll ``check`` while allocated; the execution host must additionally enforce
    hard allocation limits. This CPU ledger does not kill a process or claim
    interruption recovery. Existing files are never resumed or overwritten.
    Reserve and duration measurements are caller-supplied preparation evidence.
    """

    def __init__(
        self,
        path,
        campaign_id,
        cap_device_seconds,
        *,
        prior_device_seconds: float | int = 0,
        prior_command_ids=(),
        clock=None,
    ):
        from pathlib import Path
        import time

        self.path = Path(path)
        self.clock = clock or time.monotonic
        self._validate_seconds(cap_device_seconds, positive=True)
        self._validate_seconds(prior_device_seconds)
        if not campaign_id or prior_device_seconds > cap_device_seconds:
            raise ValueError("invalid campaign or prior allocation")
        if len(set(prior_command_ids)) != len(prior_command_ids) or any(
            not isinstance(command, str) or not command.strip() for command in prior_command_ids
        ):
            raise ValueError("unique prior command identities required")
        self.state: dict = dict(
            schema=1,
            campaign_id=campaign_id,
            cap_device_seconds=cap_device_seconds,
            prior_device_seconds=prior_device_seconds,
            prior_command_ids=list(prior_command_ids),
            allocations=[],
            terminal_reason=None,
        )
        self._last_time = self.clock()
        self._validate_seconds(self._last_time)
        # Exclusive creation prevents ambiguous replay of prior attempts.
        with self.path.open("x") as output:
            self._write(output)

    @staticmethod
    def _validate_seconds(value, positive=False):
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or value < 0
            or (positive and value == 0)
        ):
            raise ValueError("seconds must be finite and nonnegative")

    def _write(self, output):
        import os

        json.dump(self.state, output, sort_keys=True, allow_nan=False)
        output.flush()
        os.fsync(output.fileno())

    def _persist(self):
        import os
        import tempfile

        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=self.path.parent, delete=False
            ) as output:
                temporary = output.name
                self._write(output)
            os.replace(temporary, self.path)
            descriptor = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except Exception:
            self.state["terminal_reason"] = "record_write_failure"
            raise
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    def _accrue(self):
        now = self.clock()
        if not math.isfinite(now) or now < self._last_time:
            self.state["terminal_reason"] = "invalid_monotonic_clock"
            self._persist()
            raise AllocationStopped("invalid monotonic clock")
        self._last_time = now
        for allocation in self.state["allocations"]:
            if allocation["status"] == "allocated":
                allocation["elapsed_seconds"] = now - allocation["started_monotonic"]
                allocation["charged_device_seconds"] = (
                    allocation["elapsed_seconds"] * allocation["devices"]
                )
        return now

    def snapshot(self):
        """Return a detached accounting view; use check to sample active time."""
        result = json.loads(json.dumps(self.state))
        result["charged_device_seconds"] = result["prior_device_seconds"] + sum(
            a["charged_device_seconds"] for a in result["allocations"]
        )
        result["remaining_device_seconds"] = max(
            0, result["cap_device_seconds"] - result["charged_device_seconds"]
        )
        return result

    def check(self):
        self._accrue()
        active = [a for a in self.state["allocations"] if a["status"] == "allocated"]
        if any(a["elapsed_seconds"] >= a["max_duration_seconds"] for a in active):
            self.state["terminal_reason"] = "allocation_deadline"
        view = self.snapshot()
        if view["charged_device_seconds"] >= view["cap_device_seconds"]:
            self.state["terminal_reason"] = "cumulative_deadline"
        if active and view["remaining_device_seconds"] < max(
            a["mandatory_reserve_device_seconds"] for a in active
        ):
            self.state["terminal_reason"] = "mandatory_reserve_exhausted"
        self._persist()
        if self.state["terminal_reason"]:
            raise AllocationStopped(self.state["terminal_reason"])
        return self.snapshot()

    def start(
        self,
        command_id,
        *,
        stage,
        devices,
        max_duration_seconds,
        mandatory_reserve_device_seconds,
        measurement_receipt,
        identities=None,
    ):
        self.check()
        self._validate_seconds(max_duration_seconds, positive=True)
        self._validate_seconds(mandatory_reserve_device_seconds)
        if type(devices) is not int or devices <= 0:
            raise ValueError("positive physical device count required")
        if not command_id or not stage or not measurement_receipt:
            raise ValueError("command, stage and measured preparation receipt required")
        if not isinstance(identities, dict) or any(
            not isinstance(identities.get(key), str) or not identities[key].strip()
            for key in ("source", "config", "input")
        ):
            raise ValueError("exact source/config/input identity references required")
        if command_id in self.state["prior_command_ids"] or any(a["command_id"] == command_id for a in self.state["allocations"]):
            raise AllocationStopped("duplicate command; automatic retry prohibited")
        committed = sum(
            (a["max_duration_seconds"] - a["elapsed_seconds"]) * a["devices"]
            for a in self.state["allocations"]
            if a["status"] == "allocated"
        )
        reserve = max(
            [mandatory_reserve_device_seconds]
            + [
                a["mandatory_reserve_device_seconds"]
                for a in self.state["allocations"]
                if a["status"] == "allocated"
            ]
        )
        if (
            committed + devices * max_duration_seconds + reserve
            > self.snapshot()["remaining_device_seconds"]
        ):
            self.state["terminal_reason"] = "insufficient_measured_reserve"
            self._persist()
            raise AllocationStopped(self.state["terminal_reason"])
        allocation = dict(
            command_id=command_id,
            stage=stage,
            devices=devices,
            max_duration_seconds=max_duration_seconds,
            mandatory_reserve_device_seconds=mandatory_reserve_device_seconds,
            measurement_receipt=measurement_receipt,
            identities=identities or {},
            started_monotonic=self._last_time,
            stopped_monotonic=None,
            elapsed_seconds=0,
            charged_device_seconds=0,
            status="allocated",
        )
        self.state["allocations"].append(allocation)
        self._persist()
        return json.loads(json.dumps(allocation))

    def stop(self, command_id, *, status="completed"):
        if status not in {"completed", "failed", "interrupted"}:
            raise ValueError("invalid terminal allocation status")
        failure = None
        try:
            self.check()
        except AllocationStopped as error:
            failure = error
        matches = [
            a
            for a in self.state["allocations"]
            if a["command_id"] == command_id and a["status"] == "allocated"
        ]
        if len(matches) != 1:
            raise AllocationStopped("unknown or already stopped allocation")
        matches[0]["status"] = status if failure is None else "interrupted"
        matches[0]["stopped_monotonic"] = self._last_time
        if status != "completed":
            self.state["terminal_reason"] = self.state["terminal_reason"] or status
        self._persist()
        if failure:
            raise failure
        return self.snapshot()
