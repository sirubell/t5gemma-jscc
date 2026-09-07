"""Shared design composition, local runtime overrides and saved-config independence."""
from pathlib import Path

import pytest
import yaml

from jscc.config import load_config, save_config


def setup_recipes(tmp_path):
    source = Path(__file__).parents[1] / "configs"
    design = yaml.safe_load((source / "model.yaml").read_text())
    profiles = tmp_path / "models"
    tasks = tmp_path / "tasks"
    profiles.mkdir()
    tasks.mkdir()
    model_path = profiles / "shared.yaml"
    save_config(design, model_path)
    for task in ("coco", "hellaswag"):
        recipe = yaml.safe_load((source / "tasks" / f"{task}.yaml").read_text())
        recipe["model_config"] = "../models/shared.yaml"
        recipe["run"]["output_dir"] = "../runs"
        save_config(recipe, tasks / f"{task}.yaml")
    return model_path, tasks


def test_one_design_change_reaches_both_tasks(tmp_path):
    model_path, tasks = setup_recipes(tmp_path)
    design = yaml.safe_load(model_path.read_text())
    design["codec"].update(bottleneck_dim=256, snr_film=True)
    save_config(design, model_path)
    for task in ("coco", "hellaswag"):
        resolved = load_config(tasks / f"{task}.yaml")
        assert resolved["task"] == task
        assert resolved["codec"] == design["codec"]
        assert "model_config" not in resolved
        assert Path(resolved["run"]["output_dir"]) == tmp_path / "runs"


def test_runtime_override_does_not_modify_the_shared_file(tmp_path):
    model_path, tasks = setup_recipes(tmp_path)
    original = model_path.read_bytes()
    task_path = tasks / "coco.yaml"
    task = yaml.safe_load(task_path.read_text())
    task["runtime"] = {"device": "cpu", "dtype": "float32"}
    save_config(task, task_path)
    resolved = load_config(task_path)
    other = load_config(tasks / "hellaswag.yaml")
    assert resolved["model"]["device"] == "cpu"
    assert resolved["model"]["dtype"] == "float32"
    assert other["model"]["device"] == "cuda"
    assert other["model"]["dtype"] == "bfloat16"
    assert resolved["codec"] == other["codec"]
    assert "runtime" not in resolved
    assert model_path.read_bytes() == original


def test_resolved_legacy_config_survives_without_model_file(tmp_path):
    model_path, tasks = setup_recipes(tmp_path)
    resolved = load_config(tasks / "coco.yaml")
    # Historical flat configs retain their original architecture and FiLM setting.
    resolved["split"] = {"stack": "dec", "where": "after_layer", "index": 24}
    resolved["codec"]["snr_film"] = True
    saved = tmp_path / "run-config.yaml"
    save_config(resolved, saved)
    model_path.unlink()
    assert load_config(saved) == resolved


def test_task_cannot_override_model_design(tmp_path):
    _, tasks = setup_recipes(tmp_path)
    path = tasks / "coco.yaml"
    task = yaml.safe_load(path.read_text())
    task["codec"] = {"snr_film": True}
    save_config(task, path)
    with pytest.raises(ValueError, match="not in the task YAML"):
        load_config(path)


def test_runtime_only_accepts_execution_settings(tmp_path):
    _, tasks = setup_recipes(tmp_path)
    path = tasks / "hellaswag.yaml"
    task = yaml.safe_load(path.read_text())
    task["runtime"] = {"snr_film": True}
    save_config(task, path)
    with pytest.raises(ValueError, match="device and dtype"):
        load_config(path)
