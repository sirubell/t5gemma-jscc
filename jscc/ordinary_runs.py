"""Proportional logical roots and retained ordinary-task execution attempts."""
import copy
import time
from pathlib import Path

import torch
import yaml

from .config import save_config
from .evaluation_policy import file_digest
from .experiment_records import write_json_atomic
from .runtime import source_state























def _checkpoint(path, *, role, expected_config):
    if not path.is_file():
        raise ValueError(f"missing eligible checkpoint: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("config") != expected_config or type(state.get("step")) is not int or state["step"] < 1:
        raise ValueError("checkpoint configuration/step differs from its bound case")
    return {"path": str(path.resolve()), "role": role, "step": state["step"],
            "sha256": file_digest(path), "size_bytes": path.stat().st_size}


def prepare_task(config, output, *, authored_config=None):
    """Reserve and bind an ordinary root without loading model/data or training."""
    root = reserve_root(output)
    effective = copy.deepcopy(config)
    effective['run']['output_dir'] = str(root)
    bind_case(root, effective, authored_config=authored_config)
    write_json_atomic(root/'prepared.json', {'schema': 'ordinary-prepared-v1', 'config': effective})
    return root


def train_task(config, output, *, authored_config=None, resume=None, attempt_id=None):
    """Train one ordinary task at exactly output; no implicit retry or resume."""
    if resume is not None:
        return _continue_task(config, output, resume, attempt_id=attempt_id)
    if config.get("task") not in ("coco", "hellaswag") or config.get("schema"):
        raise ValueError("ordinary roots support COCO/HellaSwag task configs only")
    root = Path(output).resolve()
    if root.exists():
        effective = copy.deepcopy(config)
        effective['run']['output_dir'] = str(root)
        if (not (root/'prepared.json').is_file() or (root/'training.json').exists()
                or _read(root/'prepared.json') != {'schema': 'ordinary-prepared-v1', 'config': effective}
                or bound_config(root) != effective):
            raise FileExistsError('occupied root is not this exact unconsumed ordinary preparation')
    else:
        root = prepare_task(config, root, authored_config=authored_config)
    case = root
    effective = bound_config(case)
    attempt = reserve_attempt(case, "training", attempt_id=attempt_id or "0001")
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


def _continue_task(config, output, resume, *, attempt_id=None):
    """Resume explicitly selected durable ordinary state in a retained new attempt."""
    from .ordinary_continuation import load_checkpoint
    case = Path(output).resolve()
    effective = copy.deepcopy(config)
    effective["run"]["output_dir"] = str(case)
    saved = bound_config(case)
    if saved != effective:
        raise ValueError("incompatible ordinary continuation config")
    path = Path(resume).resolve()
    parent = _read(path.parent / "attempt.json")
    if (parent.get("case") != str(case) or parent.get("stage") != "training"
            or parent.get("status") not in ("completed", "partial", "interrupted", "failed")
            or path.parent.parent != case / "attempts"):
        raise ValueError("continuation requires a finalized training attempt in this bound case")
    state, target = load_checkpoint(path)
    identity = state["continuation"]["compatibility"]
    for key, current in (("source", source_state()), ("runtime", runtime_identity()), ("config", saved)):
        if identity.get(key) != current:
            raise ValueError(f"incompatible ordinary continuation {key}")
    if state["step"] >= saved["training"]["max_steps"]:
        raise ValueError("checkpoint has already reached max_steps")
    lock = case / ".training-update"
    lock.mkdir(exist_ok=False)
    attempt = None
    started = time.perf_counter()
    try:
        attempt = reserve_attempt(case, "training", attempt_id=attempt_id or
                                  f"resume-{parent['attempt_id']}-{state['step']}",
                                  request={"resume": target, "parent_attempt": str(path.parent),
                                           "next_optimizer_step": state["step"] + 1})
        from .training import train
        run = train(saved, resume=path, run_directory=attempt, expected_resume=target)
        if Path(run).resolve() != attempt:
            raise ValueError("training engine returned a different reserved output")
        completion = _read(attempt / "completion.json")
        eligible = (completion["reason"] == "early_stopping" or
                    (completion["reason"] == "max_steps" and completion["optimizer_updates"]
                     == saved["training"]["max_steps"]))
        target = _checkpoint(attempt / "best.pt", role="ordinary-best", expected_config=saved) if eligible else None
        finish_attempt(attempt, "completed" if eligible else "partial", details={
            "completion": completion, "checkpoint": target, "parent_attempt": str(path.parent),
            "resumed_checkpoint": _read(attempt / "attempt.json")["request"]["resume"],
            "input_record": str(attempt / "data_ids.json"), "runtime_record": str(attempt / "run.json"),
            "elapsed_seconds": time.perf_counter() - started})
        write_json_atomic(attempt / "publication.json", {
            "schema": "ordinary-continuation-publication-v1", "status": "pending",
            "training": {"path": str(attempt)}, "checkpoint": target,
            "previous_training": _read(case / "training.json"),
            "previous_checkpoint": _read(case / "checkpoint.json") if (case / "checkpoint.json").exists() else None})
        _publish_continuation(attempt)
    except BaseException as error:
        if attempt is not None and _read(attempt / "attempt.json")["status"] == "running":
            finish_attempt(attempt, "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                           details={"error_type": type(error).__name__, "message": str(error),
                                    "elapsed_seconds": time.perf_counter() - started})
        raise
    finally:
        lock.rmdir()
    return case


def _publish_continuation(attempt):
    """Complete the saved root publication plan; never repeat optimizer work."""
    record = _read(attempt / "attempt.json")
    if record["stage"] != "training" or record["status"] not in ("completed", "partial"):
        raise ValueError("continuation publication requires a finalized scientific outcome")
    if record["source"] != source_state() or record["runtime"] != runtime_identity():
        raise ValueError("incompatible continuation publication source/runtime")
    case = Path(record["case"])
    plan = _read(attempt / "publication.json")
    if plan.get("schema") != "ordinary-continuation-publication-v1" or plan.get("status") != "pending":
        raise ValueError("no pending continuation publication")
    if plan["training"] != {"path": str(attempt)}:
        raise ValueError("continuation publication attempt identity changed")
    target = plan["checkpoint"]
    current_training = _read(case / "training.json")
    current_target = _read(case / "checkpoint.json") if (case / "checkpoint.json").exists() else None
    if (current_training not in (plan["previous_training"], plan["training"])
            or current_target not in (plan["previous_checkpoint"], target)):
        raise ValueError("continuation publication superseded by another attempt")
    if target is not None:
        if Path(target["path"]).parent != attempt or _checkpoint(Path(target["path"]), role=target["role"],
                                                                 expected_config=bound_config(case)) != target:
            raise ValueError("continuation publication checkpoint identity changed")
        write_json_atomic(case / "checkpoint.json", target)
    # Even a partial child becomes the current supplemental training attempt.
    # A prior eligible default keeps its own independently recorded ancestry.
    write_json_atomic(case / "training.json", plan["training"])
    plan["status"] = "published"
    write_json_atomic(attempt / "publication.json", plan)


def publish_continuation(attempt):
    """Explicitly reconcile pending root records from a completed continuation."""
    attempt = Path(attempt).resolve()
    case = Path(_read(attempt / "attempt.json")["case"]).resolve()
    if attempt.parent != case / "attempts":
        raise ValueError("continuation publication requires this case's training attempt")
    lock = case / ".training-update"
    lock.mkdir(exist_ok=False)
    try:
        _publish_continuation(attempt)
    finally:
        lock.rmdir()
    return case

def is_recorded_root(path):
    return (Path(path) / "experiment.json").is_file() or (Path(path) / "case.json").is_file()


def evaluate_run(root, checkpoint_name=None, overrides_path=None, expected_step=None, *, attempt_id="0001"):
    """Evaluate the recorded target, or an explicit supplemental checkpoint."""
    case = Path(root).resolve()
    config = bound_config(case)
    if expected_step is not None and checkpoint_name is None:
        raise ValueError("expected-step requires an explicit checkpoint")
    target = _read(case / "checkpoint.json") if checkpoint_name is None else None
    training_run = (Path(target["path"]).parent if target is not None
                    else Path(_read(case / "training.json")["path"]))
    if training_run.parent != case / "attempts":
        raise ValueError("ordinary checkpoint has incompatible training-attempt ancestry")
    training_record = _read(training_run / "attempt.json")
    if training_record.get("case") != str(case) or training_record.get("stage") != "training":
        raise ValueError("ordinary checkpoint has incompatible training-attempt ancestry")
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

# Transitional providers until stages100/101 caller closure.
from .run_records import _name as _name  # noqa: F401, E402
from .run_records import _read as _read  # noqa: F401, E402
from .run_records import _digest as _digest  # noqa: F401, E402
from .run_records import runtime_identity as runtime_identity  # noqa: F401, E402
from .run_records import reserve_root as reserve_root  # noqa: F401, E402
from .run_records import bind_case as bind_case  # noqa: F401, E402
from .run_records import bound_config as bound_config  # noqa: F401, E402
from .run_records import reserve_attempt as reserve_attempt  # noqa: F401, E402
from .run_records import validate_reserved_attempt as validate_reserved_attempt  # noqa: F401, E402
from .run_records import finish_attempt as finish_attempt  # noqa: F401, E402
from .run_records import SCHEMA as SCHEMA  # noqa: F401, E402
