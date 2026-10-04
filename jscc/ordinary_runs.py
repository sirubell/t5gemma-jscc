"""Proportional logical roots and retained ordinary-task execution attempts."""
import copy
import hashlib
import importlib.metadata
import json
import platform
import re
import time
from pathlib import Path

import torch
import yaml

from .config import save_config
from .evaluation_policy import file_digest
from .experiment_records import write_json_atomic
from .runtime import source_state

SCHEMA = "experiment-run-v1"


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("case/attempt identifier must be a plain nonempty name")
    return value


def _read(path):
    return json.loads(Path(path).read_text())


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def runtime_identity():
    root = Path(__file__).resolve().parents[1]
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name)
                         for name in ("torch", "transformers", "datasets")},
            "uv_lock_sha256": file_digest(root / "uv.lock")}


def reserve_root(output, *, kind="ordinary-task", context=None):
    """Atomically claim a fresh logical root. Existing roots are never retries."""
    root = Path(output).expanduser().absolute()
    root.mkdir(parents=True, exist_ok=False)
    write_json_atomic(root / "experiment.json", {"schema": SCHEMA, "kind": kind,
                      "root": str(root.resolve()), "context": context or {},
                      "source": source_state(), "runtime": runtime_identity()})
    return root.resolve()


def bind_case(root, config, *, authored_config=None, case_id=None):
    """Freeze effective settings at the singleton root or a fresh named child."""
    root = Path(root).resolve()
    if _read(root / "experiment.json")["schema"] != SCHEMA:
        raise ValueError("unsupported experiment root schema")
    case = root
    if case_id is not None:
        case = root / "cases" / _name(case_id)
        case.mkdir(parents=True, exist_ok=False)
    # Exclusive reservation also prevents a second singleton binding.
    with (case / "case.json").open("x"):
        pass
    effective = copy.deepcopy(config)
    context = []
    if authored_config is not None:
        authored = Path(authored_config).resolve()
        text = authored.read_text()
        context.append({"path": str(authored), "text": text, "sha256": file_digest(authored)})
        reference = yaml.safe_load(text).get("model_config")
        if reference is not None:
            design = (authored.parent / Path(reference).expanduser()).resolve()
            context.append({"path": str(design), "text": design.read_text(), "sha256": file_digest(design)})
    save_config(effective, case / "config.yaml")
    write_json_atomic(case / "case.json", {"schema": SCHEMA, "case_id": case_id,
                      "experiment": str(root), "config_sha256": file_digest(case / "config.yaml"),
                      "effective_identity": _digest(effective), "authored": context,
                      "inputs": {"model": effective["model"], "data": effective["data"],
                                 "initializer": {"seed": effective["seed"], "codec": effective["codec"]}},
                      "budget": {"optimizer_updates": effective["training"]["max_steps"],
                                 "batch_size": effective["training"]["batch_size"],
                                 "gradient_accumulation": effective["training"]["gradient_accumulation"]}})
    return case


def bound_config(case):
    """Read the bound complete config, never reload authored references."""
    case = Path(case).resolve()
    record = _read(case / "case.json")
    if record["schema"] != SCHEMA or file_digest(case / "config.yaml") != record["config_sha256"]:
        raise ValueError("bound effective configuration changed or has unsupported schema")
    return yaml.safe_load((case / "config.yaml").read_text())


def reserve_attempt(case, stage, *, attempt_id="0001", request=None):
    """Reserve one explicitly identified stage attempt before invoking its engine."""
    if stage not in ("training", "evaluation", "collection"):
        raise ValueError("unsupported execution stage")
    bound_config(case)
    output = Path(case).resolve() / ("evaluations" if stage == "evaluation" else "attempts") / f"{stage}-{_name(attempt_id)}"
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "attempt.json", {"schema": SCHEMA, "case": str(Path(case).resolve()),
                      "stage": stage, "attempt_id": attempt_id, "status": "running",
                      "source": source_state(), "runtime": runtime_identity(),
                      "request": request or {}})
    return output


def validate_reserved_attempt(output, stage):
    """Protect the engine seam against reuse of an already populated attempt."""
    output = Path(output).resolve()
    record = _read(output / "attempt.json")
    if (record.get("schema") != SCHEMA or record.get("stage") != stage
            or record.get("status") != "running" or (output / "config.yaml").exists()):
        raise ValueError("engine requires a fresh reserved attempt for its stage")
    config = bound_config(record["case"])
    # This one-way reservation excludes concurrent engine calls and remains on failure.
    (output / ".engine-started").mkdir(exist_ok=False)
    return config


def finish_attempt(output, status, *, details=None):
    if status not in ("completed", "failed", "interrupted", "partial"):
        raise ValueError("unsupported attempt outcome")
    path = Path(output) / "attempt.json"
    record = _read(path)
    if record["status"] != "running":
        raise ValueError("attempt already has a final outcome")
    record.update(status=status, details=details or {})
    write_json_atomic(path, record)


def _checkpoint(path, *, role, expected_config):
    if not path.is_file():
        raise ValueError(f"missing eligible checkpoint: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("config") != expected_config or type(state.get("step")) is not int or state["step"] < 1:
        raise ValueError("checkpoint configuration/step differs from its bound case")
    return {"path": str(path.resolve()), "role": role, "step": state["step"],
            "sha256": file_digest(path), "size_bytes": path.stat().st_size}


def train_task(config, output, *, authored_config=None, resume=None):
    """Train one ordinary task at exactly output; no implicit retry or resume."""
    if resume is not None:
        raise ValueError("exact ordinary-task resume is not released; iterator restoration is required")
    if config.get("task") not in ("coco", "hellaswag") or config.get("schema"):
        raise ValueError("ordinary roots support COCO/HellaSwag task configs only")
    root = reserve_root(output)
    effective = copy.deepcopy(config)
    effective["run"]["output_dir"] = str(root)
    case = bind_case(root, effective, authored_config=authored_config)
    effective = bound_config(case)
    attempt = reserve_attempt(case, "training")
    write_json_atomic(case / "training.json", {"path": str(attempt)})
    started = time.perf_counter()
    from .training import train
    try:
        run = train(effective, run_directory=attempt)
        if Path(run).resolve() != attempt:
            raise ValueError("training engine returned a different reserved output")
        completion = _read(attempt / "completion.json")
        # Legacy ordinary best-selection and fixed-presentation full-budget policy.
        full_budget = bool(effective["training"].get("presentation_stream"))
        eligible = (completion.get("status") == "FULL_BUDGET_COMPLETED" if full_budget
                    else (completion["reason"] == "early_stopping" or
                          (completion["reason"] == "max_steps" and completion["optimizer_updates"]
                           == effective["training"]["max_steps"])))
        target = None
        if eligible:
            target = _checkpoint(attempt / "best.pt", role="ordinary-best", expected_config=effective)
        if target is not None:
            write_json_atomic(case / "checkpoint.json", target)
        finish_attempt(attempt, "completed" if eligible else "partial",
                       details={"completion": completion, "checkpoint": target,
                                "input_record": str(attempt / "data_ids.json"),
                                "runtime_record": str(attempt / "run.json"),
                                "elapsed_seconds": time.perf_counter() - started})
    except BaseException as error:
        finish_attempt(attempt, "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                       details={"error_type": type(error).__name__, "message": str(error),
                                "elapsed_seconds": time.perf_counter() - started})
        raise
    return root


def is_recorded_root(path):
    return (Path(path) / "experiment.json").is_file() or (Path(path) / "case.json").is_file()


def evaluate_run(root, checkpoint_name=None, overrides_path=None, expected_step=None, *, attempt_id="0001"):
    """Evaluate the recorded target, or an explicit supplemental checkpoint."""
    case = Path(root).resolve()
    config = bound_config(case)
    if expected_step is not None and checkpoint_name is None:
        raise ValueError("expected-step requires an explicit checkpoint")
    training_run = Path(_read(case / "training.json")["path"])
    target = _read(case / "checkpoint.json") if checkpoint_name is None else None
    training_record = _read(training_run / "attempt.json")
    if training_record["status"] not in ("completed", "partial"):
        raise ValueError("training attempt has no eligible completed outcome")
    if checkpoint_name is not None:
        _name(checkpoint_name)
        target = _checkpoint(training_run / checkpoint_name, role="supplemental", expected_config=config)
    elif target is None or _checkpoint(Path(target["path"]), role=target["role"], expected_config=config) != target:
        raise ValueError("recorded checkpoint identity changed")
    assert target is not None
    if expected_step is not None and target["step"] != expected_step:
        raise ValueError("checkpoint does not match expected optimizer step")
    overrides = yaml.safe_load(Path(overrides_path).read_text()) if overrides_path else None
    request = {"checkpoint": target, "role": "default" if checkpoint_name is None and overrides is None else "supplemental",
               "overrides": overrides}
    attempt = reserve_attempt(case, "evaluation", attempt_id=attempt_id, request=request)
    started = time.perf_counter()
    from .evaluation import evaluate
    try:
        # Freeze override bytes too; evaluation never rereads the author's YAML.
        frozen_overrides = None
        if overrides is not None:
            frozen_overrides = attempt / "overrides.yaml"
            save_config(overrides, frozen_overrides)
        output = evaluate(training_run, Path(target["path"]).name, frozen_overrides,
                          expected_step=target["step"], run_directory=attempt,
                          expected_config=config, expected_checkpoint=target)
        if Path(output).resolve() != attempt:
            raise ValueError("evaluation engine returned a different reserved output")
        finish_attempt(attempt, "completed", details={"results": str(attempt / "results.json"),
                       "elapsed_seconds": time.perf_counter() - started})
    except BaseException as error:
        finish_attempt(attempt, "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                       details={"error_type": type(error).__name__, "message": str(error),
                                "elapsed_seconds": time.perf_counter() - started})
        raise
    return attempt
