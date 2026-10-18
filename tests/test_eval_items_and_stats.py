"""Per-item evaluation records, evaluate --out, site_norm_sides and the site-statistics collector."""
import json

import numpy as np
import pytest
import torch
import yaml

import experiment
from jscc import evaluate
from jscc.model import wrap
from jscc.site_stats import collect
from tests.helpers import CHANNEL, CODEC, text_batch, tiny_backbone, tiny_model
from tests.test_cli import patched, write_config  # noqa: F401 (fixture)
from tests.test_train import FakeData


class FakeScorer:
    calls: list = []

    def __init__(self, config, processor):
        self.per_item = None

    def __call__(self, model, condition, output):
        FakeScorer.calls.append((model.site, condition))
        self.per_item = np.array([1, 0, int(condition == "no_noise")], dtype=np.int8)
        return {"acc_norm": float(self.per_item.mean())}


def test_evaluate_saves_per_item_records_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "HellaSwagScorer", FakeScorer)
    FakeScorer.calls = []
    config = {"task": "hellaswag", "evaluation": {"conditions": ["vanilla", "no_noise", -6]}}
    model = tiny_model()
    results = evaluate.evaluate(model, None, config, ["enc_l1", "dec_l0"], tmp_path)
    items = dict(np.load(tmp_path / "items.npz"))
    assert set(items) == set(results) == {"vanilla", "enc_l1|no_noise", "enc_l1|-6", "dec_l0|no_noise", "dec_l0|-6"}
    assert items["enc_l1|no_noise"].tolist() == [1, 0, 1] and items["dec_l0|-6"].tolist() == [1, 0, 0]
    assert all(results[k]["acc_norm"] == items[k].mean() for k in items)
    FakeScorer.calls = []
    config["evaluation"]["conditions"] = ["no_noise", -6, -10]  # only the new panels run; old items kept
    evaluate.evaluate(model, None, config, ["enc_l1", "dec_l0"], tmp_path)
    assert FakeScorer.calls == [("enc_l1", "-10"), ("dec_l0", "-10")]
    assert len(np.load(tmp_path / "items.npz").files) == 7


def test_evaluate_out_leaves_the_run_untouched(tmp_path, patched, monkeypatch):  # noqa: F811
    path = write_config(tmp_path)
    assert experiment.main(["train", str(path), "--name", "r"]) == 0
    run = tmp_path / "runs" / "r"
    before = {p.name: p.read_bytes() for p in run.iterdir() if p.is_file()}
    index = (tmp_path / "runs" / "INDEX.md").read_text()
    seen = []

    def fake_evaluate(model, processor, config, sites, output, data=None):
        seen.append((config["evaluation"]["conditions"], sites, output))
        return {f"{s}|{c}": {"acc_norm": 0.5} for s in sites for c in config["evaluation"]["conditions"]}
    monkeypatch.setattr("jscc.evaluate.evaluate", fake_evaluate)
    out = tmp_path / "v1" / "r"
    assert experiment.main(["evaluate", str(run), "--conditions", "-10", "0", "2.5", "no_noise", "--out", str(out)]) == 0
    assert seen == [([-10, 0, 2.5, "no_noise"], ["enc_l1", "dec_l0"], out)]
    assert yaml.safe_load((out / "config.yaml").read_text())["evaluation"]["conditions"] == [-10, 0, 2.5, "no_noise"]
    assert json.loads((out / "source.json").read_text())["run"] == str(run)
    assert {p.name: p.read_bytes() for p in run.iterdir() if p.is_file()} == before
    assert (tmp_path / "runs" / "INDEX.md").read_text() == index and not (run / "eval").exists()
    with pytest.raises(SystemExit):
        experiment.main(["evaluate", str(run), "--conditions", "-10"])  # --conditions needs --out


@pytest.mark.parametrize("sides", ["input", "output"])
def test_site_norm_sides(tmp_path, sides):
    g = torch.Generator().manual_seed(0)
    stats = {s: {"mean": torch.randn(16, generator=g), "std": torch.rand(16, generator=g) + 0.5} for s in ("enc_l1",)}
    torch.save({"stats": stats}, tmp_path / "stats.pt")
    config = {"codec": {**CODEC, "site_norm": "fixed", "site_norm_sides": sides, "site_stats": str(tmp_path / "stats.pt")},
              "channel": dict(CHANNEL), "model": {}, "training": {"sites": ["enc_l1"]}}
    model = wrap(tiny_backbone(), config).set_site("enc_l1")
    seen = {}
    model.codec.encoder.register_forward_hook(lambda m, args, out: seen.__setitem__("in", args[0]))
    model.codec.output_norm.register_forward_hook(lambda m, args, out: seen.__setitem__("out", out))
    batch = text_batch()
    with model.transmission(None, encoder_mask=batch["attention_mask"]):
        model(**batch)
    s, h, r = stats["enc_l1"], model.activation, model.reconstruction
    assert h is not None and r is not None
    standardized = torch.allclose(seen["in"].float(), (h.float() - s["mean"]) / s["std"], atol=1e-5)
    restored = torch.allclose(r.float(), seen["out"].float() * s["std"] + s["mean"], atol=1e-4)
    assert (standardized, restored) == ((True, False) if sides == "input" else (False, True))
    assert torch.allclose(seen["in"].float(), h.float()) == (sides == "output")


def test_site_stats_match_the_codec_inputs():
    model = tiny_model()
    data = FakeData()
    stats = collect(model, data, ["enc_l1", "enc_fn", "dec_l0"], n_batches=2)
    for site in ("enc_l1", "dec_l0"):
        model.set_site(site)
        rows = []
        for k in range(2):
            batch = data.train_batch(k)
            enc, dec = batch["attention_mask"].bool(), batch["labels"] != -100
            with model.transmission(bypass=True, encoder_mask=batch["attention_mask"], decoder_mask=dec):
                model(**batch)
            assert model.activation is not None
            rows.append(model.activation[dec if site.startswith("dec_") else enc].double())
        flat = torch.cat(rows)
        assert stats[site]["tokens"] == flat.shape[0]
        assert torch.allclose(stats[site]["mean"], flat.mean(0).float(), atol=1e-5)
        assert torch.allclose(stats[site]["std"], flat.std(0, unbiased=False).float(), atol=1e-4)
    assert torch.equal(stats["memory"]["mean"], stats["enc_fn"]["mean"])  # the memory is the enc_fn output
