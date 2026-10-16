import json
import signal
from contextlib import nullcontext

import pytest
import torch

from jscc.train import Trainer, teacher_pass
from tests.helpers import image_batch, text_batch, tiny_model


class FakeData:
    device = torch.device("cpu")

    def train_batch(self, k):
        return text_batch(seed=100 + k % 7)

    def selection_batches(self):
        return [text_batch(seed=1), text_batch(seed=2)]

    def model_inputs(self, batch, model):
        return dict(batch), batch["labels"]

    def autocast(self, model):
        return nullcontext()


def config(sites, recon=False, max_updates=40, patience=2):
    es = {"every": 4, "patience": patience, "min_delta": 0.0}
    return {"seed": 0, "channel": {"type": "awgn", "train_noise": True, "train_snr_range": [-6, 18]},
            "training": {"sites": sites, "lr": 1e-2, "weight_decay": 0.0, "grad_clip": 1.0,
                         "max_updates": max_updates, "early_stopping": es, "log_every": 1,
                         "loss_weights": {"kl": 1.0, "encoder": {"hidden": 0.1},
                                          "decoder": {"hidden": 0.05, "memory": 0.05}},
                         "reconstruction": {"enabled": recon, "lr": 1e-2, "max_updates": 8, "early_stopping": es},
                         "validation": {"conditions": ["no_noise", -6], "seed": 0}}}


def test_round_robin_and_validation(tmp_path):
    t = Trainer(tiny_model(), FakeData(), config(["enc_l1", "dec_l0"], max_updates=8, patience=99), tmp_path)
    stops = t.train()
    assert stops == {"distillation": "max_updates"}
    assert t.site_updates == {"enc_l1": 4, "dec_l0": 4}
    lines = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    vals = [line for line in lines if line["type"] == "validation"]
    assert len(vals) == 2 and set(vals[0]["per_site"]) == {"enc_l1|no_noise", "enc_l1|-6", "dec_l0|no_noise", "dec_l0|-6"}
    assert (tmp_path / "best_distillation.pt").exists() and (tmp_path / "last.pt").exists()


def test_reconstruction_then_distillation(tmp_path):
    t = Trainer(tiny_model(), FakeData(), config(["enc_l1"], recon=True, max_updates=8, patience=99), tmp_path)
    assert t.train() == {"reconstruction": "max_updates", "distillation": "max_updates"}


def test_early_stopping_triggers(tmp_path):
    cfg = config(["enc_l1"], max_updates=10_000, patience=1)
    cfg["training"]["lr"] = 0.0  # nothing improves after the first validation
    t = Trainer(tiny_model(), FakeData(), cfg, tmp_path)
    assert t.train() == {"distillation": "early_stopping"} and t.update == 8


def test_resume_matches_uninterrupted(tmp_path):
    cfg = config(["enc_l1", "enc_fn"], max_updates=16, patience=99)
    full = Trainer(tiny_model(), FakeData(), cfg, tmp_path / "a")
    full.train()
    part = Trainer(tiny_model(), FakeData(), {**cfg, "training": {**cfg["training"], "max_updates": 8}}, tmp_path / "b")
    part.train()
    resumed = Trainer(tiny_model(), FakeData(), cfg, tmp_path / "b")
    resumed.load(tmp_path / "b" / "last.pt")
    resumed.phase_index = 0
    resumed.train()
    for a, b in zip(full.model.codec.state_dict().values(), resumed.model.codec.state_dict().values()):
        assert torch.allclose(a, b, atol=1e-6)


class Rebuild(Trainer):
    """The pre-memo code path: batch, inputs and teacher rebuilt at every update, no prefetch."""

    def _prepared(self, k):
        inputs, labels = self.data.model_inputs(self.data.train_batch(k), self.model)
        logits, prefix = teacher_pass(self.model, self.data, inputs, labels, self.sites)
        return inputs, labels, logits if self.phase == "distillation" else None, prefix


def records(run):
    lines = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    return [{k: v for k, v in r.items() if k != "time"} for r in lines]


def assert_same_state(a, b):
    for x, y in zip(a.model.communication_state()["codec"].values(), b.model.communication_state()["codec"].values()):
        assert torch.equal(x, y)


def test_shared_batch_reuse_is_exact(tmp_path):
    cfg = config(["enc_l1", "enc_fn", "dec_l0"], recon=True, max_updates=24, patience=99)
    calls = []

    class Counting(FakeData):
        def train_batch(self, k):
            calls.append(k)
            return super().train_batch(k)

    new = Trainer(tiny_model(), Counting(), cfg, tmp_path / "new")
    new.train()
    old = Rebuild(tiny_model(), FakeData(), cfg, tmp_path / "old")
    old.train()
    assert records(new.run) == records(old.run)
    assert_same_state(new, old)
    # each batch built once per phase, in order, plus one prefetched ahead: reconstruction uses
    # batches 0-2, distillation continues each site's count with batches 2-10
    assert calls == [0, 1, 2, 3, *range(2, 12)]


def _sigterm_after(trainer_cls, update):
    class Interrupted(trainer_cls):
        def step(self):
            record = super().step()
            if self.total_updates == update:
                signal.raise_signal(signal.SIGTERM)
            return record
    return Interrupted


def test_sigterm_resume_matches_uninterrupted(tmp_path):
    cfg = config(["enc_l1", "enc_fn", "dec_l0"], max_updates=24, patience=99)
    full = Trainer(tiny_model(), FakeData(), cfg, tmp_path / "a")
    full.train()
    part = _sigterm_after(Trainer, 10)(tiny_model(), FakeData(), cfg, tmp_path / "b")  # mid-batch, no validation
    with pytest.raises(KeyboardInterrupt):
        part.train()
    assert part.total_updates == 10 and signal.getsignal(signal.SIGTERM) is not part._request_stop
    resumed = Trainer(tiny_model(), FakeData(), cfg, tmp_path / "b")
    resumed.load(tmp_path / "b" / "last.pt")
    resumed.train()
    assert records(resumed.run) == records(full.run)
    assert_same_state(resumed, full)


def test_sigterm_during_validation_reruns_it(tmp_path):
    cfg = config(["enc_l1", "dec_l0"], max_updates=16, patience=99)
    full = Trainer(tiny_model(), FakeData(), cfg, tmp_path / "a")
    full.train()

    class SignalInValidation(FakeData):
        def selection_batches(self):  # pyright: ignore[reportIncompatibleMethodOverride]
            batches = super().selection_batches()
            yield batches[0]
            signal.raise_signal(signal.SIGTERM)
            yield from batches[1:]

    part = Trainer(tiny_model(), SignalInValidation(), cfg, tmp_path / "b")
    with pytest.raises(KeyboardInterrupt):
        part.train()
    assert part.update == 4 and part.history == []  # first validation aborted, not recorded
    resumed = Trainer(tiny_model(), FakeData(), cfg, tmp_path / "b")
    resumed.load(tmp_path / "b" / "last.pt")
    resumed.train()
    assert records(resumed.run) == records(full.run)
    assert_same_state(resumed, full)


def test_resume_after_hard_kill_drops_later_records(tmp_path):
    cfg = config(["enc_l1", "dec_l0"], max_updates=16, patience=99)
    full = Trainer(tiny_model(), FakeData(), cfg, tmp_path / "a")
    full.train()

    class Killed(Trainer):
        def step(self):
            if self.total_updates == 6:  # last.pt is from the update-4 validation; updates 5-6 are logged
                raise RuntimeError("killed")
            return super().step()

    with pytest.raises(RuntimeError):
        Killed(tiny_model(), FakeData(), cfg, tmp_path / "b").train()
    resumed = Trainer(tiny_model(), FakeData(), cfg, tmp_path / "b")
    resumed.load(tmp_path / "b" / "last.pt")
    resumed.train()
    assert records(resumed.run) == records(full.run)


def test_best_checkpoint_on_every_new_minimum(tmp_path, monkeypatch):
    means = iter([1.0, 0.99, 0.98, 0.985, 0.97])
    monkeypatch.setattr("jscc.train.validate", lambda *a, **k: {"per_site": {}, "mean": next(means)})
    cfg = config(["enc_l1"], max_updates=100, patience=4)
    cfg["training"]["early_stopping"]["min_delta"] = 0.05  # only the first validation counts for patience
    t = Trainer(tiny_model(), FakeData(), cfg, tmp_path)
    assert t.train() == {"distillation": "early_stopping"}
    vals = [r for r in records(tmp_path) if r["type"] == "validation"]
    assert [v["new_best"] for v in vals] == [True, True, True, False, True]
    assert [v["improved"] for v in vals] == [True, False, False, False, False] and vals[-1]["bad"] == 4
    best = torch.load(tmp_path / "best_distillation.pt", weights_only=False)
    assert best["update"] == 20 and best["best"] == 0.97 and t.best == 0.97 and t.reference == 1.0


class ImageData(FakeData):
    def train_batch(self, k):
        return image_batch(seed=100 + k % 7)

    def selection_batches(self):
        return [image_batch(seed=1), image_batch(seed=2)]


@pytest.mark.parametrize("data", [FakeData, ImageData])
def test_student_from_prefix_matches_full_forward(tmp_path, monkeypatch, data):
    cfg = config(["enc_emb", "enc_l1", "enc_l2", "enc_fn", "dec_l0", "dec_l1"], recon=True, max_updates=24, patience=99)
    new = Trainer(tiny_model(), data(), cfg, tmp_path / "new")
    new.train()
    monkeypatch.setattr("jscc.model.SplitModel.from_prefix", lambda self, inputs, states: nullcontext(inputs))
    full = Trainer(tiny_model(), data(), cfg, tmp_path / "full")
    full.train()
    assert records(new.run) == records(full.run)
    assert_same_state(new, full)
