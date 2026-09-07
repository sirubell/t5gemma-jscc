"""Expand a study into independent, fully resolved experiment configurations."""
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
from typing import Any

import yaml

from .config import load_config, save_config, validate_config


@dataclass
class PlannedRun:
    task: str
    experiment: str
    seed: int
    config: dict[str, Any]


@dataclass
class StudyPlan:
    name: str
    runs: list[PlannedRun]


def _name(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError(f"{label} must use letters, digits, underscores or hyphens")
    return value


def _override_design(config, overrides):
    for section, values in overrides.items():
        if section not in {"model", "split", "codec", "channel"}:
            raise ValueError(f"Unknown model override section: {section}")
        allowed = {"stack", "where", "index"} if section == "split" else set(config[section])
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown {section} override fields: {sorted(unknown)}")
        if section == "split":
            # Replace the split as one unit so an old layer index cannot leak into it.
            if values.get("stack") not in {"enc", "dec"} or values.get("where") not in {
                "after_embed", "before_first_layer", "after_layer", "after_final_norm"
            }:
                raise ValueError("A split override requires a valid stack and where")
            if values["where"] == "after_layer" and (
                type(values.get("index")) is not int or values["index"] < 0
            ):
                raise ValueError("An after_layer split requires a non-negative integer index")
            config[section] = copy.deepcopy(values)
        else:
            config[section].update(copy.deepcopy(values))


def expand_study(path, task=None):
    path = Path(path).resolve()
    spec = yaml.safe_load(path.read_text())
    unknown = set(spec) - {"name", "task_configs", "seeds", "experiments"}
    if unknown:
        raise ValueError(f"Unknown study fields: {sorted(unknown)}")
    name = _name(spec["name"], "Study name")
    seeds = spec["seeds"]
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty list of unique non-negative integers")
    experiments = spec["experiments"]
    if not experiments:
        raise ValueError("experiments must not be empty")
    names = []
    for experiment in experiments:
        if set(experiment) - {"name", "model_overrides"}:
            raise ValueError("Each experiment supports name and model_overrides only")
        names.append(_name(experiment["name"], "Experiment name"))
    if len(names) != len(set(names)):
        raise ValueError("Experiment names must be unique")
    bases = [load_config(path.parent / filename) for filename in spec["task_configs"]]
    tasks = [base["task"] for base in bases]
    if len(tasks) != len(set(tasks)):
        raise ValueError("Use one task recipe per task in a study")
    runs = []
    for base in bases:
        if task is not None and base["task"] != task:
            continue
        for experiment in experiments:
            for seed in seeds:
                config = copy.deepcopy(base)
                _override_design(config, experiment.get("model_overrides", {}))
                config["seed"] = seed
                config["run"]["name"] = f"{name}-{base['task']}-{experiment['name']}-s{seed}"
                validate_config(config)
                runs.append(PlannedRun(base["task"], experiment["name"], seed, config))
    if not runs:
        raise ValueError("No matching task recipes in this study")
    return StudyPlan(name, runs)


def _source_state():
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def export_study(plan, output):
    """Write a new portable plan directory; never train or submit jobs."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "configs").mkdir()
    (output / "links").mkdir()
    entries = []
    for index, run in enumerate(plan.runs):
        config = copy.deepcopy(run.config)
        # The prepared directory can be moved to another host before execution.
        config["run"]["output_dir"] = "../runs"
        relative = f"configs/{index:04d}-{config['run']['name']}.yaml"
        save_config(config, output / relative)
        entries.append({"index": index, "task": run.task, "experiment": run.experiment,
                        "seed": run.seed, "config": relative, "run_link": f"links/{index:04d}.txt"})
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps({"version": 1, "study": plan.name,
                                   "source": _source_state(), "runs": entries}, indent=2) + "\n")
    return manifest
