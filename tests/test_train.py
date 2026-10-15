import json
from contextlib import nullcontext

import torch

from jscc.train import Trainer
from tests.helpers import text_batch, tiny_model


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
