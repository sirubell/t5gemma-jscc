"""Bounded D-N16x4 diagnostic entry; never issues production qualification."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from dataclasses import asdict
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from .activation_replay import canonical_digest, file_digest
from .baseline_protocol import bind_batch_identity, read_prepared_batch
from .sharing_accumulation import partition_policy
from .sharing_controller import (
    binding_identity,
    _cleanup_group,
    _put,
    _read_ref,
    _watchdog,
    output_bytes,
)
from .sharing_preparation import load_prepared, state_dict_identity

GUARD_SHA256 = "bc7865910b5696ee4f1b70fdc8b62dad3b671a8e6d687e0f2b691fdcb35d3bf1"
SITES = ("enc_l9", "enc_l19", "enc_fn")
CONDITIONS = ["no_noise", -6, 0, 6, 12, 18]
FINISH_BEFORE_EPOCH = datetime(2026, 10, 2, 10, 0, tzinfo=timezone.utc).timestamp()


def _require(value, message):
    if not value:
        raise ValueError(message)


def validate_contract(path):
    path = Path(path).resolve()
    contract = json.loads(path.read_text())
    _require(
        contract.get("schema") == "sharing-no-norm-qualification-v1",
        "qualification contract required",
    )
    _read_ref(contract["prepared"])
    manifest = load_prepared(contract["prepared"]["path"])
    config = manifest["resolved_config"]
    _require(
        not manifest["synthetic_cpu"]
        and manifest["cell"] == "D-N"
        and manifest["selected_bottleneck_dim"] in (512, 1152, 2304)
        and manifest["execution_partition"] == partition_policy(16)
        and config["codec"]["architecture"] == "direct_affine"
        and config["codec"]["layernorm"] == "none",
        "exact D-N width/native16x4 package required",
    )
    _require(
        config["model"].get("device") == "cuda"
        and config["model"].get("dtype") == "bfloat16"
        and config["model"].get("numerical_policy") == "native"
        and config["model"].get("attn_implementation") == "sdpa"
        and config["model"].get("local_files_only") is True,
        "target qualification requires local CUDA/BF16/native/SDPA model",
    )
    _require(
        contract.get("binding") == binding_identity(manifest),
        "exact diagnostic binding mismatch",
    )
    _require(
        contract.get("diagnostic_only") is True
        and contract.get("automatic_retry") is False
        and contract.get("sole_owner") == "task-7",
        "task-7 diagnostic-only/no-retry admission required",
    )
    cap = contract.get("hard_cap_seconds")
    _require(
        type(cap) is int and 20 <= cap <= min(1200, manifest["hard_cap_seconds"]),
        "explicit bounded cap required",
    )
    _require(
        type(contract.get("output_byte_cap")) is int
        and contract["output_byte_cap"] > 0,
        "explicit output byte bound required",
    )
    _require(
        Path(contract["allocation_root"]).resolve() == path.parent,
        "contract allocation root mismatch",
    )
    _require(
        isinstance(contract.get("run_id"), str)
        and contract["run_id"]
        and all(c.isalnum() or c in "-_" for c in contract["run_id"]),
        "safe run id required",
    )
    _require(
        file_digest(Path(__file__).with_name("sharing_qualification_guard.py"))
        == GUARD_SHA256,
        "reviewed numerical guard bytes changed",
    )
    _require(
        manifest["source_inventory"].get("scripts/sharing_qualification.py")
        == file_digest(
            Path(__file__).resolve().parents[1] / "scripts/sharing_qualification.py"
        ),
        "qualification entry must be in prepared source inventory",
    )
    _require(
        manifest["task_template"]["settings"]["batch_size"] == 16
        and manifest["task_template"]["conditions"] == CONDITIONS,
        "exact task16/six conditions required",
    )
    _require(
        Path(contract["hf_home"]).is_absolute(), "absolute pinned HF cache required"
    )
    deadline = contract.get("finish_before_epoch")
    _require(
        type(deadline) in (int, float)
        and math.isfinite(deadline)
        and deadline <= FINISH_BEFORE_EPOCH,
        "explicit deadline no later than Oct2 18:00 Taipei required",
    )
    _read_ref(contract["expected_panel"])
    return contract, manifest


def _assets(contract, manifest):
    """Hash exact local model/tokenizer/data assets on the actual execution host."""
    root = Path(manifest["resolved_config"]["model"]["name"]).resolve()
    _require(
        root.is_absolute() and root.is_dir(), "local pinned model directory required"
    )
    assets = contract.get("target_assets")
    required = {
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "hellaswag-train.arrow",
        "hellaswag-validation.arrow",
    }
    _require(
        isinstance(assets, dict) and set(assets) == required,
        "complete pinned model/tokenizer/data assets required",
    )
    for name, ref in assets.items():
        asset = Path(ref["path"])
        _require(
            asset.is_absolute() and file_digest(asset) == ref["sha256"],
            "target asset mismatch: " + name,
        )
        if not name.endswith(".arrow"):
            _require(
                asset.resolve() == root / name,
                "model asset outside exact configured snapshot",
            )
    expected = contract.get("runtime_versions")
    _require(
        isinstance(expected, dict)
        and set(expected) == {"torch", "transformers", "datasets", "lm-eval"},
        "pinned runtime versions required",
    )
    actual = {name: importlib.metadata.version(name) for name in expected}
    _require(actual == expected, "target runtime versions differ")
    return {"assets_identity": canonical_digest(assets), "runtime_versions": actual}


def preflight(path):
    import torch

    _require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and not torch.cuda.is_initialized(),
        "preflight requires CUDA hidden and uninitialized",
    )
    contract, manifest = validate_contract(path)
    checked = _assets(contract, manifest)
    _require(not torch.cuda.is_initialized(), "CPU preflight initialized CUDA")
    receipt = {
        "schema": "sharing-qualification-target-preflight-v1",
        "status": "cpu_verified",
        "contract_sha256": file_digest(Path(path)),
        "binding": contract["binding"],
        "hostname": socket.gethostname(),
        "python": str(Path(sys.executable).resolve()),
        "cuda_initialized": False,
        "full_model_constructed": False,
        **checked,
    }
    with (Path(contract["allocation_root"]) / "target-preflight.json").open(
        "x"
    ) as stream:
        json.dump(receipt, stream, indent=2)
    return receipt


def _admission(contract):
    from .sharing_admission import validate_overnight_admission

    return validate_overnight_admission(contract, diagnostic_only=True)


def _target_receipt(path, contract, manifest):
    receipt = json.loads(
        (Path(contract["allocation_root"]) / "target-preflight.json").read_text()
    )
    _require(
        receipt.get("status") == "cpu_verified"
        and receipt.get("cuda_initialized") is False
        and receipt.get("full_model_constructed") is False
        and receipt.get("contract_sha256") == file_digest(Path(path))
        and receipt.get("binding") == contract["binding"]
        and receipt.get("hostname") == socket.gethostname()
        and receipt.get("python") == str(Path(sys.executable).resolve()),
        "actual target CPU preflight required",
    )
    actual = _assets(contract, manifest)
    _require(
        all(receipt.get(k) == v for k, v in actual.items()),
        "target assets/runtime changed after preflight",
    )


def preview(path):
    contract, _ = validate_contract(path)
    entry = Path(__file__).resolve().parents[1] / "scripts/sharing_qualification.py"
    return {
        "launched": False,
        "admission_checked": False,
        "diagnostic_only": True,
        "command": [
            "/usr/bin/timeout",
            "--signal=TERM",
            "--kill-after=2s",
            str(contract["hard_cap_seconds"] - 2),
            sys.executable,
            "-B",
            str(entry),
            "--controller",
            str(Path(path).resolve()),
        ],
    }


def controller(path):
    started = time.monotonic()
    contract, manifest = validate_contract(path)
    _watchdog(contract)
    _target_receipt(path, contract, manifest)
    _admission(contract)
    _require(
        time.time() + contract["hard_cap_seconds"] <= contract["finish_before_epoch"],
        "insufficient remaining authorized window",
    )
    root = Path(contract["allocation_root"])
    lease_path = Path(contract["device_lease"])
    _require(
        lease_path.is_absolute() and lease_path.is_file(),
        "existing absolute shared lease required",
    )
    claim = lease_path.parent / (contract["run_id"] + ".attempt.json")
    with claim.open("x") as stream:
        json.dump(
            {
                "run_id": contract["run_id"],
                "binding": contract["binding"],
                "controller_pid": os.getpid(),
                "contract_sha256": file_digest(Path(path)),
            },
            stream,
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.link(claim, root / "attempt.json")
    report = {
        "diagnostic_only": True,
        "production_qualified": False,
        "status": "starting",
        "run_id": contract["run_id"],
        "automatic_retry": False,
        "binding": contract["binding"],
        "controller_pid": os.getpid(),
        "deadline_monotonic": started + contract["hard_cap_seconds"] - 10,
    }
    lease = None
    child = None
    stopped = False
    previous = {}

    def stop(*_):
        nonlocal stopped
        stopped = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        previous[sig] = signal.signal(sig, stop)
    try:
        lease = lease_path.open("r+")
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _require(
            not subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid",
                    "--format=csv,noheader,nounits",
                ],
                timeout=3,
            ).strip(),
            "GPU busy; zero-work stop",
        )
        report.update(lease_held=True, lease_fd=lease.fileno(), status="admitted")
        _put(root / "active-allocation.json", report)
        env = dict(
            os.environ,
            HF_HUB_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="8",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
            HF_HOME=contract["hf_home"],
            HF_DATASETS_CACHE=contract["hf_home"] + "/datasets",
            HUGGINGFACE_HUB_CACHE=contract["hf_home"] + "/hub",
        )
        entry = Path(__file__).resolve().parents[1] / "scripts/sharing_qualification.py"
        with (root / "worker.log").open("x") as log:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    str(entry),
                    "--worker",
                    str(Path(path).resolve()),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
            )
            while child.poll() is None:
                if (
                    stopped
                    or time.time() >= contract["finish_before_epoch"]
                    or time.monotonic() >= report["deadline_monotonic"]
                    or output_bytes(root) > contract["output_byte_cap"]
                ):
                    raise TimeoutError("diagnostic signal/deadline/output bound")
                time.sleep(0.1)
        _require(child.returncode == 0, "diagnostic worker failed")
        _require(
            output_bytes(root) <= contract["output_byte_cap"],
            "post-exit output byte cap exceeded",
        )
        result = json.loads((root / "result.json").read_text())
        validate_result(result, contract)
        report.update(status="terminal", exit_code=0)
        return report
    except BaseException as error:
        report.update(status="incomplete_no_retry", error=str(error))
        raise
    finally:
        errors = []
        try:
            try:
                if child is not None:
                    if child.poll() is None:
                        child.kill()
                    child.wait(timeout=2)
            except BaseException as error:
                errors.append(str(error))
            try:
                _cleanup_group()
            except BaseException as error:
                errors.append(str(error))
            if output_bytes(root) > contract["output_byte_cap"]:
                errors.append("final output byte cap exceeded")
            if errors:
                report.update(status="cleanup_failed", cleanup_errors=errors)
        finally:
            elapsed = time.monotonic() - started
            report.update(
                elapsed_seconds=elapsed,
                charged_device_seconds=math.ceil(elapsed),
                cleanup_verified=not errors,
            )
            try:
                _put(root / "allocation.json", report)
            finally:
                try:
                    if lease is not None:
                        lease.close()
                finally:
                    for sig, handler in previous.items():
                        signal.signal(sig, handler)
        if errors:
            raise RuntimeError("diagnostic cleanup failed; inspect durable accounting")


def validate_result(result, contract):
    _require(
        result.get("status") == "diagnostic_completed"
        and result.get("completed_updates") == 24
        and result.get("diagnostic_only") is True
        and result.get("production_qualified") is False
        and result.get("binding") == contract["binding"]
        and set(result.get("sites", {})) == set(SITES),
        "complete exact24 diagnostic receipt required",
    )
    for record in result["sites"].values():
        guard = record.get("guard", {})
        objective = record.get("objective", {})
        tasks = record.get("task", [])
        _require(
            record.get("status") == "diagnostic_completed"
            and guard.get("status") == "passed"
            and guard.get("physical_updates") == 5
            and guard.get("checkpoint", {}).get("exact") is True
            and guard.get("restored_second_update_exact") is True
            and len(record.get("stress_timings", [])) == 3
            and objective.get("sequences") == 128
            and len(objective.get("rows", [])) == 6
            and [row.get("condition") for row in objective["rows"]] == CONDITIONS
            and all(
                row.get("completed") == 128 and row.get("failed") == 0
                for row in objective["rows"]
            )
            and [row.get("condition") for row in tasks] == CONDITIONS
            and all(row.get("candidate_forward_requests") == 1024 for row in tasks),
            "all three guards/reloads/full128objective/full256task panels required",
        )


def worker(path):
    """All admission and actual-target checks precede any model construction."""
    contract, manifest = validate_contract(path)
    _target_receipt(path, contract, manifest)
    _admission(contract)
    root = Path(contract["allocation_root"])
    active = json.loads((root / "active-allocation.json").read_text())
    attempt = json.loads((root / "attempt.json").read_text())
    parent = os.getppid()
    _require(
        active.get("controller_pid") == parent == attempt.get("controller_pid")
        and active.get("binding") == attempt.get("binding") == contract["binding"]
        and attempt.get("contract_sha256") == file_digest(Path(path))
        and active.get("lease_held") is True
        and active.get("status") == "admitted",
        "live admitted controller required",
    )
    proc = Path(f"/proc/{parent}")
    watchdog = int((proc / "stat").read_text().split(") ", 1)[1].split()[1])
    expected = [
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=2s",
        str(contract["hard_cap_seconds"] - 2),
    ]
    _require(
        Path(f"/proc/{watchdog}/cmdline").read_bytes().decode().split("\0")[:4]
        == expected
        and os.getpgrp() == watchdog,
        "external watchdog absent",
    )
    _require(
        (proc / "fd" / str(active["lease_fd"])).resolve()
        == Path(contract["device_lease"]).resolve(),
        "controller lease descriptor mismatch",
    )
    with Path(contract["device_lease"]).open("r+") as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise ValueError("controller lease is not exclusively held")

    def guard():
        _require(os.getppid() == parent, "controller disappeared")
        if (
            time.time() >= contract["finish_before_epoch"]
            or time.monotonic() >= active["deadline_monotonic"]
            or output_bytes(root) > contract["output_byte_cap"]
        ):
            raise TimeoutError("diagnostic resource bound")

    guard()
    return run_diagnostic(
        manifest,
        Path(contract["prepared"]["path"]).parent,
        root,
        guard,
        expected_panel=_read_ref(contract["expected_panel"]),
    )


def _verify_panel(directory, expected):
    for name, key in [
        ("documents", "sample_id"),
        ("prompts", "prompt_id"),
        ("fewshots", "fewshot_id"),
    ]:
        rows = [
            json.loads(line)
            for line in (directory / (name + ".jsonl")).read_text().splitlines()
        ]
        target = expected[name]
        _require(
            len({r[key] for r in rows}) == len(rows)
            and len({r[key] for r in target}) == len(target)
            and {r[key]: r for r in rows} == {r[key]: r for r in target},
            "task panel identity mismatch: " + name,
        )
        if name == "documents":
            _require(
                [r[key] for r in rows] == [r[key] for r in target],
                "task document order mismatch",
            )


def run_diagnostic(manifest, package_root, output, guard, *, expected_panel):
    """Actual reviewed numerical guard and panels; tiny CPU fixtures use this core."""
    import torch
    from .models.split_model import build_model
    from .models.channel import AWGNChannel
    from .runtime import configure_training_determinism, seed_everything
    from .sharing_protocol import SharingLearner, batch_view
    from .sharing_schedule import SharingPlan
    from .sharing_qualification_guard import same_shape_guard, snapshot, restore
    from .experiment_state import isolated_rng
    from .evaluation import evaluate_hellaswag

    config = copy.deepcopy(manifest["resolved_config"])
    guard()
    configure_training_determinism(config["training"])
    seed_everything(0)
    if not manifest["synthetic_cpu"]:
        _require(
            torch.cuda.device_count() == 1
            and torch.cuda.get_device_name() == "NVIDIA GeForce RTX 5090",
            "one RTX5090 required",
        )
    processor, model = build_model(config)
    if not manifest["synthetic_cpu"]:
        parameter = next(model.base.parameters())
        _require(
            parameter.device.type == "cuda"
            and parameter.dtype == torch.bfloat16
            and model.base.config.decoder._attn_implementation == "sdpa",
            "constructed backbone must actually be CUDA/BF16/SDPA",
        )
    channel = model.channel
    if not isinstance(channel, AWGNChannel):
        raise ValueError("qualification requires AWGN replay channel")
    model.codec.load_state_dict(manifest["initialization_state"], strict=True)
    _require(
        state_dict_identity(model.codec.state_dict())
        == manifest["initialization"]["state_identity"],
        "actual model initialization mismatch",
    )
    batches = [read_prepared_batch(ref, package_root) for ref in manifest["updates"]]
    indexes = [
        0,
        max(
            range(len(batches)),
            key=lambda i: int(batches[i]["attention_mask"].sum(1).max()),
        ),
        max(
            range(len(batches)),
            key=lambda i: int((batches[i]["labels"] != -100).sum(1).max()),
        ),
    ]
    order = list(dict.fromkeys(indexes)) + [
        i for i in range(len(batches)) if i not in indexes
    ]
    _require(
        len(set(indexes)) == 3,
        "three distinct stress views required for diagnostic schedule",
    )
    plan = SharingPlan(
        tuple(batch_view(batches[i]) for i in order),
        manifest["study_pairing_id"] + ":qualification-only",
        synthetic=manifest["synthetic_cpu"],
    )
    parents = [batches[i] for i in indexes]
    validation = [
        read_prepared_batch(ref, package_root) for ref in manifest["validation"]
    ]
    template = manifest["task_template"]
    settings = copy.deepcopy(template["settings"])
    settings.update(
        noise_seed=template["noise"]["seed"],
        noise_namespace=template["noise"]["namespace"],
    )
    result = {
        "status": "incomplete_no_retry",
        "diagnostic_only": True,
        "production_qualified": False,
        "completed_updates": 0,
        "sites": {},
        "stress_production_indices": indexes,
        "diagnostic_order": order,
        "binding": binding_identity(manifest),
        "guard_sha256": GUARD_SHA256,
    }

    def save():
        _put(output / "result.json", result)

    def checked_update():
        guard()
        _require(result["completed_updates"] < 24, "diagnostic update budget exhausted")

    def sync():
        if not manifest["synthetic_cpu"]:
            torch.cuda.synchronize()

    def memory():
        return (
            {}
            if manifest["synthetic_cpu"]
            else {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        )

    handle = model.base.register_forward_pre_hook(lambda *_: guard())
    try:
        for site in SITES:
            model.codec.load_state_dict(manifest["initialization_state"], strict=True)
            kwargs = dict(
                schedule=plan.learner("specialist_" + site),
                run_id="qualification-" + site,
                identity={
                    "source": manifest["source_identity"],
                    "config": manifest["config_identity"],
                    "data": canonical_digest(manifest["data_ids"]),
                    "parent": None,
                },
                model_revision=manifest["model_revision"],
                microbatch_size=16,
            )
            reference = SharingLearner(model, enc_fn_reuse=False, **kwargs)
            candidate = SharingLearner(model, enc_fn_reuse=site == "enc_fn", **kwargs)
            zero = snapshot(candidate)
            metadata = {
                key: manifest[key]
                for key in (
                    "source_identity",
                    "config_identity",
                    "model_revision",
                    "recipe_identity",
                    "architecture_decision",
                )
            }
            metadata.update(
                initialization_identity=manifest["initialization"]["state_identity"],
                stream_identity=canonical_digest([asdict(v) for v in plan.views]),
                protocol_identity=manifest["protocol_id"],
            )
            record = {}
            result["sites"][site] = record
            record["guard"] = same_shape_guard(
                reference,
                candidate,
                parents[:2],
                checkpoint_path=output / (site + "-checkpoint.pt"),
                metadata=metadata,
                before_update=checked_update,
            )
            result["completed_updates"] += 5
            record["stress_timings"] = []
            for label, parent_batch in zip(
                ("representative", "longest_source", "longest_target"), parents
            ):
                checked_update()
                sync()
                if not manifest["synthetic_cpu"]:
                    torch.cuda.reset_peak_memory_stats()
                started = time.monotonic()
                event = candidate.update(
                    [parent_batch], batch_identities=[bind_batch_identity(parent_batch)]
                )
                sync()
                result["completed_updates"] += 1
                record["stress_timings"].append(
                    {
                        "label": label,
                        "seconds": time.monotonic() - started,
                        "exposure": event["payload"]["exposure"],
                        **memory(),
                    }
                )
                save()
            restore(candidate, zero)
            guard()
            sync()
            started = time.monotonic()
            objectives = candidate.validate_site(validation, site)
            sync()
            expected_count = sum(len(batch["labels"]) for batch in validation)
            _require(
                len(objectives) == 6
                and all(
                    row["completed"] == expected_count and row["failed"] == 0
                    for row in objectives
                ),
                "objective panel incomplete",
            )
            record["objective"] = {
                "sequences": expected_count,
                "seconds": time.monotonic() - started,
                "rows": objectives,
                **memory(),
            }
            restore(candidate, zero)
            model.eval()
            directory = output / ("task-" + site)
            directory.mkdir()
            record["task"] = []
            for condition in CONDITIONS:
                guard()
                sync()
                started = time.monotonic()
                with (
                    model.at_site(
                        candidate.sites[site],
                        purpose="evaluate",
                        authorization={"sites": {site: "trained"}},
                    ),
                    isolated_rng(settings["noise_seed"]),
                    channel.replay(settings["noise_seed"], settings["noise_namespace"]),
                    model.transmission(
                        None if condition == "no_noise" else float(condition)
                    ),
                ):
                    metrics = evaluate_hellaswag(
                        model,
                        processor,
                        settings,
                        directory,
                        condition,
                        template["data_settings"],
                    )
                sync()
                _verify_panel(directory, expected_panel)
                _require(
                    metrics["candidate_forward_requests"]
                    == 4 * template["expected_items"],
                    "candidate task request count differs",
                )
                record["task"].append(
                    {
                        "condition": condition,
                        "seconds": time.monotonic() - started,
                        "candidate_forward_requests": metrics[
                            "candidate_forward_requests"
                        ],
                        **memory(),
                    }
                )
                save()
            restore(candidate, zero)
            record["status"] = "diagnostic_completed"
            save()
        _require(
            result["completed_updates"] == 24, "exact24 diagnostic updates required"
        )
        result["status"] = "diagnostic_completed"
        return result
    finally:
        handle.remove()
        save()
