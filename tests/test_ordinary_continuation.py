"""Actual tiny ordinary optimizer paths; no full weights, downloads or scheduler."""
import json
import random
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from datasets import Dataset

from jscc import ordinary_continuation, ordinary_runs, training
from jscc.config import load_config
from jscc.data import TaskData
from jscc.data.coco import CaptionDataset
from jscc.data.hellaswag import Collator
from jscc.data.continuation import OrdinaryCursor
from jscc.models.channel import AWGNChannel
from test_coco import coco_model
from test_core import toy_model

REPO = Path(__file__).parents[1]


def read(path):
    return json.loads(path.read_text())


def assert_state_equal(actual, expected):
    if torch.is_tensor(actual):
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_state_equal(left, right)
    else:
        assert actual == expected


class CaptionProcessor:
    name_or_path = "tiny-pinned-processor"

    @property
    def tokenizer(self):
        return self

    def __call__(self, target=None, *, images=None, text=None, **kwargs):
        if images is not None:
            pixels = torch.tensor(np.array(images[-1]), dtype=torch.float32).permute(2, 0, 1) / 255
            return {"input_ids": torch.tensor([[2, 31, 31, 31, 31, 4]]),
                    "attention_mask": torch.ones(1, 6, dtype=torch.long),
                    "pixel_values": pixels}
        assert target is not None
        tokens = [5 + int(target.split("-")[0]), 16 + int(target.split("-")[1])]
        return {"input_ids": torch.tensor([tokens]), "attention_mask": torch.ones(1, 2, dtype=torch.long)}


class StatefulChannel(AWGNChannel):
    """Supported non-tensor extra state plus global NumPy/Python randomness."""
    supports_exact_continuation = True

    def __init__(self):
        super().__init__()
        self.calls = 0

    def get_extra_state(self):
        return {"calls": self.calls}

    def set_extra_state(self, state):
        self.calls = state["calls"]

    def forward(self, z, snr_db):
        self.calls += 1
        gain = 1 + (random.random() + np.random.random()) * .01
        return super().forward(z, snr_db) * gain


def task_fixture(tmp_path, monkeypatch, task, *, max_steps=9, checkpoint=3, accumulation=2,
                 custom_channel=False):
    config = load_config(REPO / f"configs/tasks/{task}.yaml")
    config["model"].update(device="cpu", dtype="float32")
    config["data"]["num_workers"] = 0
    config["training"].update(max_steps=max_steps, schedule_steps=max_steps, batch_size=2,
        gradient_accumulation=accumulation, eval_every=2, selection_steps=[1, checkpoint, max_steps],
        min_steps=0, patience=None, save_steps=[checkpoint, max_steps], log_every=1,
        validation_batches=1, record_coco_presentations=False, streamed_backward=True,
        deterministic_algorithms=True)
    for key in ("expected_attention_backend", "presentation_stream", "paired_randomness", "feature_summary"):
        config["training"].pop(key, None)
    processor = CaptionProcessor() if task == "coco" else None
    version = ["pinned-tiny-rows-v1"]
    init_changed = [False]
    def build(_):
        model = coco_model() if task == "coco" else toy_model("dec")
        if custom_channel:
            model.channel = StatefulChannel()
        elif task == "coco":
            model.channel = AWGNChannel()
        if init_changed[0]:
            with torch.no_grad():
                next(model.codec.parameters()).add_(1)
        return processor, model
    monkeypatch.setattr(training, "build_model", build)
    def data(_, __, saved_ids=None):
        if task == "coco":
            rows = [{"image": Image.new("RGB", (8, 8), (index * 30, 80, 140)),
                     "answer": [f"{index}-{caption}" for caption in range(3)],
                     "file_name": f"COCO_fixture_{index:012d}.jpg"} for index in range(7)]
            train = CaptionDataset(rows, processor, [], [], config["data"], True)
            selection = CaptionDataset(rows[:2], processor, [], [], config["data"], False)
        else:
            rows = [{"input_ids": [1, index + 2] + ([3] if index % 2 else []),
                     "attention_mask": [1] * (3 if index % 2 else 2),
                     "label_ids": [index + 5, 8] if index % 2 else [index + 5],
                     "row_id": index} for index in range(7)]
            train = Dataset.from_list(rows)
            selection = Dataset.from_list(rows[:2])
        if isinstance(train, CaptionDataset):
            train.rows = type("Rows", (list,), {"_fingerprint": version[0]})(train.rows)
        else:
            train._fingerprint = version[0]
        collate = None if task == "coco" else Collator(0)
        ids = {"fixture": list(range(7)), "validation_rows": [0, 1]}
        return TaskData(train=DataLoader(cast(TorchDataset, train), batch_size=2, shuffle=True, collate_fn=collate),
                        validation=DataLoader(cast(TorchDataset, selection), batch_size=2, collate_fn=collate), ids=ids)
    monkeypatch.setattr(training, "load_data", data)
    return config, version, init_changed


def scientific_rows(attempt):
    rows = [json.loads(line) for line in (attempt / "metrics.jsonl").read_text().splitlines()]
    keys = ("step", "phase", "loss", "kl", "nmse", "lr", "presentations",
            "sampled_snr_mean", "sampled_snr_min", "sampled_snr_max")
    return [{key: row[key] for key in keys if key in row} for row in rows]


@pytest.mark.parametrize("task", ["coco", "hellaswag"])
@pytest.mark.parametrize("checkpoint", [2, 3])
def test_exact_interrupted_task_crosses_cycles_and_observations(tmp_path, monkeypatch, task, checkpoint):
    config, _, _ = task_fixture(tmp_path, monkeypatch, task, checkpoint=checkpoint,
                                custom_channel=task == "hellaswag")
    config["training"]["feature_summary"] = {"selection_count": 2, "steps": [0, checkpoint, 9]}
    baseline = ordinary_runs.train_task(config, tmp_path / "baseline")
    baseline_attempt = baseline / "attempts/training-0001"
    original = training.append_metrics
    def interrupt(path, row):
        original(path, row)
        if row.get("phase") == "train" and row["step"] == checkpoint + 2:
            raise KeyboardInterrupt("unsaved optimizer work")
    monkeypatch.setattr(training, "append_metrics", interrupt)
    root = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        ordinary_runs.train_task(config, root)
    parent = root / "attempts/training-0001"
    durable = parent / f"step_{checkpoint:06d}.pt"
    before = {p: p.read_bytes() for p in parent.iterdir() if p.is_file()}
    saved, _ = ordinary_continuation.load_checkpoint(durable)
    assert saved["step"] == checkpoint
    monkeypatch.setattr(training, "append_metrics", original)
    ordinary_runs.train_task(config, root, resume=durable, attempt_id="0002")
    child = root / "attempts/training-0002"
    completion = read(child / "completion.json")
    assert completion["optimizer_updates"] == 9 and completion["attempt_optimizer_updates"] == 9 - checkpoint
    assert completion["attempt_presentations"] == completion["presentations"] - saved["continuation"]["counters"]["presentations"]
    assert [json.loads(line)["step"] for line in (child / "feature_summary.jsonl").read_text().splitlines()] == [9]
    request = read(child / "attempt.json")["request"]
    assert request["next_optimizer_step"] == checkpoint + 1
    assert request["parent_attempt"] == str(parent)
    assert read(parent / "attempt.json")["status"] == "interrupted"
    assert all(p.read_bytes() == payload for p, payload in before.items())
    expected = torch.load(baseline_attempt / "last.pt", weights_only=True)
    actual = torch.load(child / "last.pt", weights_only=True)
    for key in ("codec", "memory_codec", "channel", "optimizer", "scheduler", "scaler",
                "step", "best", "bad_evaluations", "python_rng", "torch_rng", "cuda_rng"):
        assert_state_equal(actual[key], expected[key])
    for key in ("cursor", "counters", "completed_updates", "rng", "best_step"):
        assert_state_equal(actual["continuation"][key], expected["continuation"][key])
    assert actual["continuation"]["cursor"]["cycle"] >= 3
    assert scientific_rows(child) == [row for row in scientific_rows(baseline_attempt) if row["step"] > checkpoint]
    with pytest.raises(FileExistsError):
        ordinary_runs.train_task(config, root, resume=durable, attempt_id="0002")


@pytest.mark.parametrize("task", ["coco", "hellaswag"])
def test_real_1000_checkpoint_unsaved_1500_resumes_1001(tmp_path, monkeypatch, task):
    config, _, _ = task_fixture(tmp_path, monkeypatch, task, max_steps=1502, checkpoint=1000, accumulation=1)
    config["training"].update(log_every=1000, selection_steps=[1000, 1502])
    baseline = ordinary_runs.train_task(config, tmp_path / "baseline")
    original = training.batch_losses
    calls = [0]
    def interrupt(*args, **kwargs):
        # Interrupted at the next batch after1500 actual optimizer updates.
        if args[0].training:
            calls[0] += 1
            if calls[0] == 1501:
                raise KeyboardInterrupt("after unsaved1500")
        return original(*args, **kwargs)
    monkeypatch.setattr(training, "batch_losses", interrupt)
    root = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        ordinary_runs.train_task(config, root)
    assert calls[0] == 1501
    parent = root / "attempts/training-0001"
    state, _ = ordinary_continuation.load_checkpoint(parent / "last.pt")
    assert state["step"] == 1000 and state["continuation"]["completed_updates"] == 1000
    immutable = (parent / "last.pt").read_bytes()
    monkeypatch.setattr(training, "batch_losses", original)
    ordinary_runs.train_task(config, root, resume=parent / "last.pt", attempt_id="0002")
    child = root / "attempts/training-0002"
    assert read(child / "attempt.json")["request"]["next_optimizer_step"] == 1001
    assert (parent / "last.pt").read_bytes() == immutable
    actual = torch.load(child / "last.pt", weights_only=True)
    expected = torch.load(baseline / "attempts/training-0001/last.pt", weights_only=True)
    for key in ("codec", "memory_codec", "optimizer", "scheduler", "scaler", "step"):
        assert_state_equal(actual[key], expected[key])
    for key in ("cursor", "counters", "rng"):
        assert_state_equal(actual["continuation"][key], expected["continuation"][key])


@pytest.mark.parametrize("field", ["source", "runtime", "config", "data", "initializer", "cursor"])
def test_incompatible_or_missing_state_is_rejected(tmp_path, monkeypatch, field):
    config, version, initializer = task_fixture(tmp_path, monkeypatch, "hellaswag")
    root = ordinary_runs.train_task(config, tmp_path / "root")
    checkpoint = root / "attempts/training-0001/step_000003.pt"
    if field in ("source", "runtime", "cursor"):
        state = torch.load(checkpoint, weights_only=True)
        if field == "cursor":
            state["continuation"]["cursor"] = None
        else:
            state["continuation"]["compatibility"][field] = {"different": True}
        # A structurally published checkpoint can belong to another run contract.
        ordinary_continuation.atomic_torch_save(state, checkpoint)
        ordinary_continuation.publish_checkpoint(checkpoint, 3)
    elif field == "config":
        config["training"]["lr"] *= 2
    elif field == "data":
        version[0] = "different-pinned-data"
    else:
        initializer[0] = True
    before = checkpoint.read_bytes()
    with pytest.raises(ValueError, match="(incompatible|missing/unsupported)"):
        ordinary_runs.train_task(config, root, resume=checkpoint, attempt_id="0002")
    assert checkpoint.read_bytes() == before
    assert read(root / "attempts/training-0001/attempt.json")["status"] == "completed"
    if (root / "attempts/training-0002").exists():
        assert read(root / "attempts/training-0002/attempt.json")["status"] == "failed"
    assert not (root / ".training-update").exists()


def test_partial_tampered_legacy_and_foreign_checkpoints_reject(tmp_path, monkeypatch):
    config, _, _ = task_fixture(tmp_path, monkeypatch, "hellaswag")
    root = ordinary_runs.train_task(config, tmp_path / "root")
    parent = root / "attempts/training-0001"
    checkpoint = parent / "step_000003.pt"
    payload = checkpoint.read_bytes()
    checkpoint.write_bytes(payload[:len(payload) // 2])
    with pytest.raises(ValueError, match="bytes changed"):
        ordinary_runs.train_task(config, root, resume=checkpoint)
    checkpoint.write_bytes(payload)
    checkpoint.with_suffix(".pt.json").unlink()
    with pytest.raises(ValueError, match="durable checkpoint receipt"):
        ordinary_runs.train_task(config, root, resume=checkpoint)
    state = torch.load(checkpoint, weights_only=True)
    state.pop("continuation")
    ordinary_continuation.atomic_torch_save(state, checkpoint)
    ordinary_continuation.publish_checkpoint(checkpoint, 3)
    with pytest.raises(ValueError, match="cursor or training state"):
        ordinary_runs.train_task(config, root, resume=checkpoint)
    other = ordinary_runs.train_task(config, tmp_path / "other")
    with pytest.raises(ValueError, match="this bound case"):
        ordinary_runs.train_task(config, root, resume=other / "attempts/training-0001/step_000003.pt")


def test_failed_atomic_save_preserves_previous_bytes(tmp_path, monkeypatch):
    checkpoint = tmp_path / "last.pt"
    ordinary_continuation.atomic_torch_save({"step": 1}, checkpoint)
    before = checkpoint.read_bytes()
    def fail(payload, stream):
        stream.write(b"partial failed write")
        raise OSError("fixture disk failure")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk failure"):
        ordinary_continuation.atomic_torch_save({"step": 2}, checkpoint)
    assert checkpoint.read_bytes() == before
    assert list(tmp_path.iterdir()) == [checkpoint]


def test_failed_receipt_never_publishes_new_last_checkpoint(tmp_path, monkeypatch):
    config, _, _ = task_fixture(tmp_path, monkeypatch, "hellaswag", checkpoint=3)
    original = ordinary_continuation.publish_checkpoint
    def fail(path, step):
        if step == 3:
            raise OSError("receipt fsync failed")
        return original(path, step)
    monkeypatch.setattr(ordinary_continuation, "publish_checkpoint", fail)
    root = tmp_path / "root"
    with pytest.raises(OSError, match="receipt fsync"):
        ordinary_runs.train_task(config, root)
    parent = root / "attempts/training-0001"
    state, _ = ordinary_continuation.load_checkpoint(parent / "last.pt")
    assert state["step"] == 1
    assert not (parent / "checkpoint_000003.pt.json").exists()
    with pytest.raises(ValueError, match="durable checkpoint receipt"):
        ordinary_continuation.load_checkpoint(parent / "checkpoint_000003.pt")


def test_cursor_preserves_fresh_shuffle_short_tail_and_rng():
    from jscc.runtime import seed_everything
    rows = list(range(7))
    seed_everything(81)
    baseline = DataLoader(cast(TorchDataset, rows), batch_size=2, shuffle=True)
    expected = [batch.tolist() for batch in baseline]
    expected_rng = ordinary_continuation.capture_rng()
    seed_everything(81)
    cursor = OrdinaryCursor(DataLoader(cast(TorchDataset, rows), batch_size=2, shuffle=True))
    actual = []
    for batch in cursor.loader:
        cursor.consumed()
        actual.append(batch.tolist())
    assert actual == expected and len(actual[-1]) == 1
    assert_state_equal(ordinary_continuation.capture_rng(), expected_rng)
    bad = cursor.state_dict()
    bad["next_batch"] = 100
    with pytest.raises(ValueError, match="position/order"):
        OrdinaryCursor(DataLoader(cast(TorchDataset, rows), batch_size=2, shuffle=True), bad)
    with pytest.raises(ValueError, match="zero-worker"):
        OrdinaryCursor(DataLoader(cast(TorchDataset, rows), batch_size=2, shuffle=True, num_workers=1))


def test_observation_restores_custom_state_rng_and_mode_on_error():
    from jscc.runtime import seed_everything
    seed_everything(81)
    model = toy_model("enc", channel=StatefulChannel()).train()
    before = ordinary_continuation.capture_rng()
    with pytest.raises(RuntimeError, match="observer"):
        with ordinary_continuation.isolated_observation(model):
            model.eval()
            model.channel(torch.ones(2, 3, 4), 3.)
            raise RuntimeError("observer failed")
    assert model.training and model.channel.calls == 0
    assert_state_equal(ordinary_continuation.capture_rng(), before)
    cast(Any, model.channel).supports_exact_continuation = False
    assert "declare" in (ordinary_continuation.channel_support(model) or "")


def test_resume_retains_checkpoint_time_best_even_after_later_unsaved_improvement(tmp_path, monkeypatch):
    config, _, _ = task_fixture(tmp_path, monkeypatch, "hellaswag", checkpoint=3)
    config["training"]["selection_steps"] = [1, 3, 5, 9]
    original_append, original_validate = training.append_metrics, training.validate
    current_step = [0]
    def append(path, row):
        original_append(path, row)
        if row.get("phase") == "train":
            current_step[0] = row["step"]
            if row["step"] == 6:
                raise KeyboardInterrupt("after later best")
    def validate(*args, **kwargs):
        metrics = original_validate(*args, **kwargs)
        metrics[config["training"]["monitor"]] = {1: 1., 3: 2., 5: .5, 9: .4}[current_step[0]]
        return metrics
    monkeypatch.setattr(training, "append_metrics", append)
    monkeypatch.setattr(training, "validate", validate)
    root = tmp_path / "root"
    with pytest.raises(KeyboardInterrupt):
        ordinary_runs.train_task(config, root)
    parent = root / "attempts/training-0001"
    # last is now5, but the explicitly selected earlier3 checkpoint owns best1.
    checkpoint = parent / "step_000003.pt"
    saved, _ = ordinary_continuation.load_checkpoint(checkpoint)
    assert saved["continuation"]["best_step"] == 1
    assert torch.load(parent / "best.pt", weights_only=True)["step"] == 5
    def resumed_append(path, row):
        original_append(path, row)
        if row.get("phase") == "train":
            current_step[0] = row["step"]
            if row["step"] == 4:
                assert torch.load(path.parent / "best.pt", weights_only=True)["step"] == 1
    monkeypatch.setattr(training, "append_metrics", resumed_append)
    ordinary_runs.train_task(config, root, resume=checkpoint)
    assert checkpoint.is_file()
    child = root / "attempts/training-resume-0001-3"
    assert read(child / "attempt.json")["status"] == "completed"


@pytest.mark.parametrize("task", ["coco", "hellaswag"])
def test_mid_accumulation_interruption_discards_unsaved_gradients(tmp_path, monkeypatch, task):
    config, _, _ = task_fixture(tmp_path, monkeypatch, task, checkpoint=3)
    baseline = ordinary_runs.train_task(config, tmp_path / "baseline")
    original = training.batch_losses
    calls = [0]
    def interrupt(*args, **kwargs):
        if args[0].training:
            calls[0] += 1
            if calls[0] == 8:  # step4's second microbatch; first has already backpropagated.
                raise KeyboardInterrupt("mid effective batch")
        return original(*args, **kwargs)
    monkeypatch.setattr(training, "batch_losses", interrupt)
    root = tmp_path / "root"
    with pytest.raises(KeyboardInterrupt):
        ordinary_runs.train_task(config, root)
    monkeypatch.setattr(training, "batch_losses", original)
    ordinary_runs.train_task(config, root, resume=root / "attempts/training-0001/last.pt")
    actual = torch.load(root / "attempts/training-resume-0001-3/last.pt", weights_only=True)
    expected = torch.load(baseline / "attempts/training-0001/last.pt", weights_only=True)
    for key in ("codec", "memory_codec", "optimizer", "scheduler", "scaler", "torch_rng", "python_rng"):
        assert_state_equal(actual[key], expected[key])
    for key in ("cursor", "counters", "rng"):
        assert_state_equal(actual["continuation"][key], expected["continuation"][key])


@pytest.mark.parametrize("change", ["nan-best", "negative-bad", "fractional-bad", "invalid-best-step", "missing-counters"])
def test_published_malformed_selection_and_counters_reject(tmp_path, monkeypatch, change):
    config, _, _ = task_fixture(tmp_path, monkeypatch, "hellaswag")
    root = ordinary_runs.train_task(config, tmp_path / "root")
    checkpoint = root / "attempts/training-0001/step_000003.pt"
    state = torch.load(checkpoint, weights_only=True)
    if change == "nan-best":
        state["best"] = float("nan")
    elif change == "negative-bad":
        state["bad_evaluations"] = -1
    elif change == "fractional-bad":
        state["bad_evaluations"] = 1.5
    elif change == "invalid-best-step":
        state["continuation"]["best_step"] = 4
    else:
        state["continuation"]["counters"] = {}
    ordinary_continuation.atomic_torch_save(state, checkpoint)
    ordinary_continuation.publish_checkpoint(checkpoint, 3)
    with pytest.raises(ValueError, match="(selection state|counters/RNG)"):
        ordinary_runs.train_task(config, root, resume=checkpoint)


def test_terminal_patience_state_rejects_even_before_loop_exit(tmp_path, monkeypatch):
    config, _, _ = task_fixture(tmp_path, monkeypatch, "hellaswag")
    config["training"].update(patience=1, selection_steps=[1, 3, 5, 9])
    original = training.validate
    def constant(*args, **kwargs):
        metrics = original(*args, **kwargs)
        metrics[config["training"]["monitor"]] = 1.
        return metrics
    monkeypatch.setattr(training, "validate", constant)
    root = ordinary_runs.train_task(config, tmp_path / "root")
    parent = root / "attempts/training-0001"
    assert read(parent / "completion.json")["step"] == 3
    for status in ("completed", "interrupted"):
        record = read(parent / "attempt.json")
        record["status"] = status  # Fault fixture: crash between durable save and policy loop exit.
        (parent / "attempt.json").write_text(json.dumps(record))
        with pytest.raises(ValueError, match="terminal early-stop"):
            ordinary_runs.train_task(config, root, resume=parent / "last.pt")
    # Earlier state remains resumable even though the parent later reached patience.
    ordinary_runs.train_task(config, root, resume=parent / "checkpoint_000001.pt")
    child = root / "attempts/training-resume-0001-1"
    assert read(child / "completion.json")["step"] == 3


def install_tiny_evaluation(monkeypatch, config):
    from jscc import evaluation
    monkeypatch.setattr(evaluation, "build_model", training.build_model)
    monkeypatch.setattr(evaluation, "load_data", lambda *args, **kwargs: training.load_data(*args))
    def score(model, processor, *args):
        data = training.load_data(config, processor)
        sample = next(iter(data.validation))
        loss = training.batch_losses(model, sample, config["training"], None)["loss"].item()
        return {"fixture_loss": loss, "num_samples": 2}
    monkeypatch.setattr(evaluation, "evaluate_hellaswag", score)


def evaluation_config(config):
    config["evaluation"].pop("attention_backend", None)
    config["evaluation"].update(snrs=["no_noise"], vanilla=False, noise_seed=0, num_samples=None)


def test_partial_child_supplemental_checkpoint_is_selected(tmp_path, monkeypatch):
    config, _, _ = task_fixture(tmp_path, monkeypatch, "hellaswag")
    evaluation_config(config)
    config["training"]["max_minutes"] = 1.e-12
    install_tiny_evaluation(monkeypatch, config)
    root = ordinary_runs.train_task(config, tmp_path / "root")
    parent = root / "attempts/training-0001"
    assert read(parent / "completion.json")["step"] == 1
    ordinary_runs.train_task(config, root, resume=parent / "last.pt", attempt_id="0002")
    child = root / "attempts/training-0002"
    assert read(child / "completion.json")["step"] == 2
    assert read(root / "training.json")["path"] == str(child)
    output = ordinary_runs.evaluate_run(root, "last.pt", expected_step=2)
    assert read(output / "results.json")["optimizer_step"] == 2
    assert read(output / "attempt.json")["request"]["checkpoint"]["path"] == str(child / "last.pt")


def test_partial_child_preserves_previous_default_checkpoint_ancestry(tmp_path, monkeypatch):
    from types import SimpleNamespace
    config, _, _ = task_fixture(tmp_path, monkeypatch, "hellaswag")
    evaluation_config(config)
    config["training"]["max_minutes"] = 1.e-12
    install_tiny_evaluation(monkeypatch, config)
    original_time = training.time
    clock = [0.]
    def monotonic():
        clock[0] += 1.e-16
        return clock[0]
    # Isolate policy clock only; optimizer/model work remains real CPU work.
    monkeypatch.setattr(training, "time", SimpleNamespace(monotonic=monotonic))
    root = ordinary_runs.train_task(config, tmp_path / "root")
    original_target = read(root / "checkpoint.json")
    parent = root / "attempts/training-0001"
    monkeypatch.setattr(training, "time", original_time)
    ordinary_runs.train_task(config, root, resume=parent / "step_000003.pt", attempt_id="0002")
    assert read(root / "attempts/training-0002/attempt.json")["status"] == "partial"
    assert read(root / "checkpoint.json") == original_target
    output = ordinary_runs.evaluate_run(root)
    assert read(output / "attempt.json")["request"]["checkpoint"] == original_target
    assert read(output / "results.json")["checkpoint"] == original_target["path"]


def test_split_root_publication_keeps_default_target_consistent_and_can_reconcile(tmp_path, monkeypatch):
    config, _, _ = task_fixture(tmp_path, monkeypatch, "hellaswag")
    evaluation_config(config)
    install_tiny_evaluation(monkeypatch, config)
    root = ordinary_runs.train_task(config, tmp_path / "root")
    original = ordinary_runs.write_json_atomic
    def fail(path, payload):
        if path == root / "training.json":
            raise OSError("training pointer publication failed")
        return original(path, payload)
    monkeypatch.setattr(ordinary_runs, "write_json_atomic", fail)
    with pytest.raises(OSError, match="pointer publication"):
        ordinary_runs.train_task(config, root, resume=root / "attempts/training-0001/step_000003.pt", attempt_id="0002")
    child = root / "attempts/training-0002"
    assert read(child / "attempt.json")["status"] == "completed"
    assert read(child / "publication.json")["status"] == "pending"
    assert read(root / "training.json")["path"].endswith("0001")
    assert read(root / "checkpoint.json")["path"].startswith(str(child))
    # The authoritative default target derives its own valid training ancestry.
    output = ordinary_runs.evaluate_run(root)
    assert read(output / "results.json")["checkpoint"].startswith(str(child))
    monkeypatch.setattr(ordinary_runs, "write_json_atomic", original)
    ordinary_runs.publish_continuation(child)
    assert read(root / "training.json")["path"] == str(child)
    assert read(child / "publication.json")["status"] == "published"
    with pytest.raises(ValueError, match="no pending"):
        ordinary_runs.publish_continuation(child)


def test_new_recorded_cursor_state_cannot_silently_use_legacy_restart(tmp_path, monkeypatch):
    config, _, _ = task_fixture(tmp_path, monkeypatch, "hellaswag")
    root = ordinary_runs.train_task(config, tmp_path / "root")
    with pytest.raises(ValueError, match="bound --output root"):
        training.train(config, resume=root / "attempts/training-0001/step_000003.pt")
