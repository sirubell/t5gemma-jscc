"""Study planning is CPU-only; subprocess execution is mocked for routing tests."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from jscc.config import load_config, save_config
from jscc.studies import expand_study, export_study
from jscc.study_task import run_task


CONFIGS = Path(__file__).parents[1] / "configs"


@pytest.mark.parametrize("name,count", [("baseline", 2), ("splits", 26), ("bottleneck", 6), ("smoke_h200", 2)])
def test_supplied_studies_expand_for_both_tasks(name, count):
    plan = expand_study(CONFIGS / "studies" / f"{name}.yaml")
    assert len(plan.runs) == count
    assert {run.task for run in plan.runs} == {"coco", "hellaswag"}
    assert all(run.config["codec"]["snr_film"] is False for run in plan.runs)
    coco = [run for run in plan.runs if run.task == "coco"]
    hs = [run for run in plan.runs if run.task == "hellaswag"]
    for left, right in zip(coco, hs):
        assert (left.experiment, left.seed) == (right.experiment, right.seed)
        for section in ("model", "split", "codec", "channel"):
            assert left.config[section] == right.config[section]


def test_filter_and_variant_configs_are_independent():
    path = CONFIGS / "studies" / "bottleneck.yaml"
    plan = expand_study(path, task="coco")
    assert len(plan.runs) == 3
    assert [run.config["codec"]["bottleneck_dim"] for run in plan.runs] == [256, 512, 1024]
    plan.runs[0].config["codec"]["hidden_dim"] = 17
    assert plan.runs[1].config["codec"]["hidden_dim"] == 1152
    assert load_config(CONFIGS / "tasks/coco.yaml")["codec"]["bottleneck_dim"] == 512


def test_split_replacement_and_order_are_deterministic():
    path = CONFIGS / "studies/splits.yaml"
    first = expand_study(path)
    second = expand_study(path)
    assert first == second
    splits = {run.experiment: run.config["split"] for run in first.runs}
    assert splits["enc_fn"] == {"stack": "enc", "where": "after_final_norm"}
    assert splits["dec_l24"] == {"stack": "dec", "where": "after_layer", "index": 24}


def test_seed_expansion(tmp_path):
    spec = yaml.safe_load((CONFIGS / "studies/baseline.yaml").read_text())
    spec["task_configs"] = [str(CONFIGS / "tasks/coco.yaml"), str(CONFIGS / "tasks/hellaswag.yaml")]
    spec["seeds"] = [0, 7, 11]
    path = tmp_path / "seeds.yaml"
    save_config(spec, path)
    plan = expand_study(path)
    assert len(plan.runs) == 6
    assert [run.seed for run in plan.runs] == [0, 7, 11, 0, 7, 11]
    assert len({run.config["run"]["name"] for run in plan.runs}) == 6


@pytest.mark.parametrize("bad,match", [
    ({"seeds": [0, 0]}, "unique"),
    ({"experiments": [{"name": "../escape"}]}, "Experiment name"),
    ({"experiments": [{"name": "x"}, {"name": "x"}]}, "unique"),
    ({"experiments": [{"name": "x", "model_overrides": {"codec": {"botleneck_dim": 1}}}]}, "Unknown codec"),
    ({"experiments": [{"name": "x", "model_overrides": {"training": {"max_steps": 1}}}]}, "Unknown model"),
    ({"experiments": [{"name": "x", "model_overrides": {"split": {"stack": "enc", "where": "after_layer"}}}]}, "index"),
])
def test_invalid_study_fails_before_export(tmp_path, bad, match):
    spec = yaml.safe_load((CONFIGS / "studies/baseline.yaml").read_text())
    spec["task_configs"] = [str(CONFIGS / "tasks/coco.yaml")]
    spec.update(bad)
    path = tmp_path / "bad.yaml"
    save_config(spec, path)
    with pytest.raises(ValueError, match=match):
        expand_study(path)


def test_export_is_resolved_portable_and_does_not_overwrite(tmp_path):
    plan = expand_study(CONFIGS / "studies/baseline.yaml")
    original = copy.deepcopy(plan)
    output = tmp_path / "prepared"
    manifest = export_study(plan, output)
    assert plan == original
    assert not (output / "runs").exists()  # Export never trains.
    moved = tmp_path / "another-host"
    shutil.move(str(output), moved)
    data = json.loads((moved / manifest.name).read_text())
    assert len(data["runs"]) == 2
    for entry in data["runs"]:
        path = moved / entry["config"]
        raw = yaml.safe_load(path.read_text())
        assert "model_config" not in raw
        assert load_config(path)["run"]["output_dir"] == str(moved / "runs")
        assert not (moved / entry["run_link"]).exists()
    with pytest.raises(FileExistsError):
        export_study(plan, moved)


def test_train_evaluate_pairing_uses_the_same_index(tmp_path, monkeypatch):
    manifest = export_study(expand_study(CONFIGS / "studies/baseline.yaml"), tmp_path / "plan")
    entries = json.loads(manifest.read_text())["runs"]
    calls = []
    trained = tmp_path / "actual-hellaswag-run"

    def invoke(command, **kwargs):
        calls.append(command)
        assert kwargs["check"] is True
        if "--run-path-file" in command:
            Path(command[command.index("--run-path-file") + 1]).write_text(str(trained) + "\n")

    monkeypatch.setattr(subprocess, "run", invoke)
    run_task(manifest, 1, "train")
    run_task(manifest, 1, "evaluate", checkpoint="last.pt")
    assert calls[0][calls[0].index("--config") + 1] == str(manifest.parent / entries[1]["config"])
    assert calls[1][calls[1].index("--run") + 1] == str(trained)
    assert calls[1][-1] == "last.pt"
    assert not (manifest.parent / entries[0]["run_link"]).exists()
    with pytest.raises(FileExistsError):
        run_task(manifest, 1, "train")
    with pytest.raises(FileNotFoundError):
        run_task(manifest, 0, "evaluate")
    with pytest.raises(ValueError, match="out of range"):
        run_task(manifest, -1, "train")


def test_failed_training_does_not_create_a_success_link(tmp_path, monkeypatch):
    manifest = export_study(expand_study(CONFIGS / "studies/baseline.yaml"), tmp_path / "plan")

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        run_task(manifest, 0, "train")
    assert not (manifest.parent / "links/0000.txt").exists()


def test_cli_preview_has_no_execution_or_filesystem_output(tmp_path, monkeypatch, capsys):
    import study

    def unexpected(*args, **kwargs):
        pytest.fail("Preview must not launch commands")

    monkeypatch.setattr(subprocess, "run", unexpected)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["study.py", "--config", str(CONFIGS / "studies/splits.yaml"), "--task", "coco"])
    study.main()
    assert "13 training configurations; no jobs started" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []
