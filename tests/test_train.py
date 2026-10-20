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


def config(sites, recon=False, max_updates=40, patience: int | None = 2):
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


def test_normalized_selection_keeps_its_own_best(tmp_path, monkeypatch):
    # cell "b" dominates the raw mean while cell "a" gets worse: the two minima differ
    cells = iter([{"a|x": 1.0, "b|x": 10.0}, {"a|x": 2.0, "b|x": 8.0}, {"a|x": 0.5, "b|x": 9.0},
                  {"a|x": 3.0, "b|x": 6.0}])
    monkeypatch.setattr("jscc.train.validate",
                        lambda *a, **k: (lambda per: {"per_site": per, "mean": sum(per.values()) / 2})(next(cells)))
    cfg = config(["enc_l1"], max_updates=16, patience=None)
    cfg["training"]["selection"] = {"reference": {"a|x": 0.5, "b|x": 5.0}}
    t = Trainer(tiny_model(), FakeData(), cfg, tmp_path)
    assert t.train() == {"distillation": "max_updates"}
    vals = [r for r in records(tmp_path) if r["type"] == "validation"]
    assert [v["normalized"] for v in vals] == [2.0, 2.8, 1.4, 3.6]  # mean of (a / 0.5, b / 5)
    assert [v["new_best_normalized"] for v in vals] == [True, False, True, False]
    assert [v["new_best"] for v in vals] == [True, True, True, True]
    raw = torch.load(tmp_path / "best_distillation.pt", weights_only=False)
    norm = torch.load(tmp_path / "best_distillation_normalized.pt", weights_only=False)
    assert (raw["update"], norm["update"]) == (16, 12) and norm["best_normalized"] == 1.4


def test_continue_early_stopped_run_to_fixed_budget(tmp_path):
    """The shared13 continuation: resume an early-stopped run from its best checkpoint with early stopping
    off, a larger max_updates and selection.reference; it follows the same trajectory as a run that never
    stopped (same data order, LR and losses)."""
    import shutil
    sites = ["enc_l1", "enc_fn", "dec_l0"]
    stopped = config(sites, max_updates=60, patience=2)
    stopped["training"]["lr"] = 0.1  # validation means 0.00348, 0.00350, 0.00364, 0.00279, ...
    a = Trainer(tiny_model(), FakeData(), stopped, tmp_path / "a")
    assert a.train() == {"distillation": "early_stopping"} and a.total_updates == 12
    best_update = torch.load(tmp_path / "a" / "best_distillation.pt", weights_only=False)["update"]
    assert best_update == 4  # like the B2 run, the best checkpoint is before the stop

    budget = config(sites, max_updates=30, patience=None)
    budget["training"]["lr"] = 0.1
    full = Trainer(tiny_model(), FakeData(), budget, tmp_path / "full")
    assert full.train() == {"distillation": "max_updates"}

    shutil.copytree(tmp_path / "a", tmp_path / "c")
    shutil.copy(tmp_path / "c" / "best_distillation.pt", tmp_path / "c" / "last.pt")
    first = [r for r in records(tmp_path / "a") if r["type"] == "validation" and r["update"] == best_update][0]
    reference = {k: 2 * v for k, v in first["per_site"].items()}
    cont_cfg = {**budget, "training": {**budget["training"], "selection": {"reference": reference}}}
    c = Trainer(tiny_model(), FakeData(), cont_cfg, tmp_path / "c")
    c.load(tmp_path / "c" / "last.pt")
    assert c.best_normalized == 0.5  # seeded from the resumed checkpoint's own validation
    assert c.train() == {"distillation": "max_updates"} and c.total_updates == 30
    assert c.site_updates == full.site_updates == {s: 10 for s in sites}

    extra = ("normalized", "new_best_normalized")
    assert [{k: v for k, v in r.items() if k not in extra} for r in records(c.run)] == records(full.run)
    assert all("normalized" in r for r in records(c.run) if r["type"] == "validation" and r["update"] > best_update)
    assert_same_state(c, full)  # both hold the raw-best weights after training
    for name in ("last.pt", "best_distillation.pt"):
        x, y = (torch.load(run / name, weights_only=False) for run in (c.run, full.run))
        assert x["update"] == y["update"] and all(
            torch.equal(p, q) for p, q in zip(x["communication"]["codec"].values(), y["communication"]["codec"].values()))
    assert torch.load(c.run / "last.pt", weights_only=False)["update"] == 30  # final weights, not the best ones
    norm = torch.load(c.run / "best_distillation_normalized.pt", weights_only=False)
    vals = [r for r in records(c.run) if r["type"] == "validation" and r["update"] > best_update]
    assert norm["best_normalized"] == min([0.5] + [v["normalized"] for v in vals])


def test_loss_normalization_scales_each_site_kl(tmp_path):
    # site references 0.1 (enc_l1, mean of its cells) and 0.4 (dec_l0): factors 1/0.1 and 1/0.4 scaled to mean 1
    reference = {"enc_l1|no_noise": 0.05, "enc_l1|-6": 0.15, "dec_l0|no_noise": 0.3, "dec_l0|-6": 0.5}
    cfg = config(["enc_l1", "dec_l0"], max_updates=4, patience=99)
    cfg["training"]["loss_normalization"] = {"reference": reference}
    t = Trainer(tiny_model(), FakeData(), cfg, tmp_path)
    assert t.kl_scale == pytest.approx({"enc_l1": 1.6, "dec_l0": 0.4})
    t.train()
    updates = [r for r in records(tmp_path) if r["type"] == "update"]
    assert {r["site"] for r in updates} == {"enc_l1", "dec_l0"}
    for r in updates:
        hidden = 0.1 * r["nmse_hidden"] if r["site"] == "enc_l1" else 0.05 * (r["nmse_hidden"] + r["nmse_memory"])
        assert r["loss"] == pytest.approx(t.kl_scale[r["site"]] * r["kl"] + hidden, rel=1e-5)
    assert Trainer(tiny_model(), FakeData(), config(["enc_l1"]), tmp_path / "p").kl_scale == {"enc_l1": 1.0}


def test_early_stop_point_kept_while_training_on(tmp_path, monkeypatch):
    means = iter([1.0, 0.99, 0.98, 0.985, 0.97, 0.96])
    monkeypatch.setattr("jscc.train.validate", lambda *a, **k: {"per_site": {}, "mean": next(means)})
    cfg = config(["enc_l1"], max_updates=24, patience=2)
    cfg["training"]["early_stopping"].update(min_delta=0.05, stop=False)  # the rule fires at update 12
    t = Trainer(tiny_model(), FakeData(), cfg, tmp_path)
    assert t.train() == {"distillation": "max_updates"}
    marks = [r for r in records(tmp_path) if r["type"] == "early_stop_point"]
    assert [(m["update"], m["best"]) for m in marks] == [(12, 0.98)]
    early = torch.load(tmp_path / "best_distillation_early_stop.pt", weights_only=False)
    best = torch.load(tmp_path / "best_distillation.pt", weights_only=False)
    assert (early["update"], early["best"]) == (12, 0.98) and (best["update"], best["best"]) == (24, 0.96)
