"""Current run records bodies; saved record meanings preserved."""
from __future__ import annotations

import re
import json
from pathlib import Path
import hashlib
import importlib.metadata
import platform
from .evaluation_policy import file_digest
from .experiment_records import write_json_atomic
from .runtime import source_state
import copy
import yaml
from .config import save_config

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

