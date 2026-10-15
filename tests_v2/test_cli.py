import json

import pytest
import yaml

import experiment
from tests_v2.helpers import tiny_backbone
from tests_v2.test_train import FakeData, config as train_config
from jscc.model import wrap


def write_config(tmp_path, **training):
    cfg = train_config(["enc_l1", "dec_l0"], max_updates=8, patience=99)
    cfg["training"].update(batch_size=2, **training)
    cfg.update(task="hellaswag", run={"name": "t", "root": str(tmp_path / "runs")},
               model={"device": "cpu"}, codec={"architecture": "linear", "bottleneck_dim": 8},
               evaluation={"after_training": False})
    cfg["channel"]["normalize_power"] = True
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(experiment, "_load_model", lambda config: (None, wrap(tiny_backbone(), config)))
    monkeypatch.setattr("jscc.tasks.load_task_data", lambda config, processor: FakeData())


def test_train_writes_summary_and_index(tmp_path, patched):
    assert experiment.main(["train", str(write_config(tmp_path)), "--name", "r1"]) == 0
    run = tmp_path / "runs" / "r1"
    summary = json.loads((run / "summary.json").read_text())
    assert summary["status"] == "trained" and summary["training"]["updates"] == 8
    assert summary["source"]["commit"] and (run / "SUMMARY.md").exists() and (run / "config.yaml").exists()
    index = (tmp_path / "runs" / "INDEX.md").read_text()
    assert "| r1 | train | started |" in index and "| r1 | train | done |" in index


def test_failure_is_recorded(tmp_path, patched, monkeypatch):
    def boom(self):
        raise FloatingPointError("nonfinite loss at update 3")
    monkeypatch.setattr("jscc.train.Trainer.step", boom)
    with pytest.raises(FloatingPointError):
        experiment.main(["train", str(write_config(tmp_path)), "--name", "r2"])
    run = tmp_path / "runs" / "r2"
    assert json.loads((run / "summary.json").read_text())["status"] == "failed"
    assert "nonfinite loss" in (run / "SUMMARY.md").read_text()
    assert "| r2 | train | failed |" in (tmp_path / "runs" / "INDEX.md").read_text()
