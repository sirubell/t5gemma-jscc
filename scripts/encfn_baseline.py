#!/usr/bin/env python3
"""Preview the bounded plan or execute one explicitly prepared baseline segment."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jscc.experiment_schedule import compile_baseline_plan


def _execute(manifest_path, output, resource_guard: Any = None):
    import copy
    import torch
    from jscc.activation_replay import canonical_digest, file_digest
    from jscc.baseline_protocol import (BaselineLearner, read_prepared_batch, run_baseline,
                                        open_baseline_replay, read_replay_batch)
    from jscc.config import load_config
    from jscc.data import load_data
    from jscc.evaluation import evaluate_checkpoint
    from jscc.experiment_state import CheckpointRef, open_state
    from jscc.models.split_model import build_model, resolve_encoder_site
    from jscc.runtime import configure_training_determinism, seed_everything

    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "prepared-encfn-baseline-v2":
        raise ValueError("explicit versioned prepared manifest required")
    config_ref = manifest["config"]
    config_path = root / config_ref["path"]
    if file_digest(config_path) != config_ref["sha256"]:
        raise ValueError("prepared config checksum mismatch")
    config = load_config(config_path)
    native_c = manifest.get("native64_c")
    if native_c is not None:
        if native_c != {"schema": "formal-native64-c-v1", "same_shape_guard": True} or resource_guard is None:
            raise ValueError("native64+C requires the approved externally bounded formal wrapper")
        if (config["model"].get("numerical_policy") != "native" or
                config["training"]["batch_size"] != 64 or config["training"]["gradient_accumulation"] != 1):
            raise ValueError("formal C requires actual native64 without accumulation")
    if (config["split"] != {"stack": "enc", "where": "after_final_norm"}
            or config["codec"]["bottleneck_dim"] != 512 or config["seed"] != 0
            or config["training"]["max_steps"] != 400):
        raise ValueError("prepared config differs from enc_fn/B512/seed0/400 protocol")
    training = config["training"]
    if (training["lr"] != 2e-4 or training["weight_decay"] != 0.01 or training["grad_clip"] != 1.0
            or training["batch_size"] * training["gradient_accumulation"] != 64):
        raise ValueError("prepared optimizer/effective batch differs from accepted baseline")
    if (config["codec"]["snr_film"] or config["codec"]["dropout"] != 0
            or config["channel"]["type"] != "awgn" or not config["channel"]["normalize_power"]):
        raise ValueError("baseline requires dropout0/FiLMoff/masked normalized AWGN")
    if config["codec"]["architecture"] == "residual_mlp" and (
            config["codec"]["n_res_blocks"] != 2 or config["codec"]["hidden_dim"] != 1152
            or config["codec"]["activation"] != "gelu"):
        raise ValueError("baseline residual architecture requires two H1152/GELU blocks")
    if config["task"] != "hellaswag":
        raise ValueError("initial scientific baseline is HellaSwag; COCO uses separate protocol")
    plan = compile_baseline_plan(manifest.get("selected_cells", ()), manifest.get("owner_approval"))
    segments = [s for s in plan.segments if s.cell == manifest["cell"] and s.strategy == manifest["strategy"]]
    if len(segments) != 1:
        raise ValueError("segment is not in the explicitly authorized plan")
    segment = segments[0]
    cell_settings = {"D-none": ("direct_affine", "none"), "D-LN": ("direct_outer_ln", "both"),
                     "R-none": ("residual_mlp", "none"), "R-LN": ("residual_mlp", "both")}
    if (config["codec"]["architecture"], config["codec"]["layernorm"]) != cell_settings[segment.cell]:
        raise ValueError("prepared cell topology mismatch")
    # Preparation binds actual executed source bytes, not Git's dirty flag.
    source_root = Path(__file__).resolve().parents[1]
    inventory = manifest["source_inventory"]
    expected_paths = {str(p.relative_to(source_root)) for p in (source_root / "jscc").rglob("*.py")}
    expected_paths |= {"scripts/encfn_baseline.py", "uv.lock", "pyproject.toml"}
    if set(inventory) != expected_paths:
        raise ValueError("learner source inventory must cover jscc, runner and dependencies")
    for name, digest in inventory.items():
        if file_digest(source_root / name) != digest:
            raise ValueError(f"executed source differs from preparation: {name}")
    metadata = copy.deepcopy(manifest["state_metadata"])
    if metadata["source_identity"] != canonical_digest(inventory):
        raise ValueError("state source identity differs from executed inventory")
    if metadata["config_identity"] != canonical_digest(config):
        raise ValueError("state config identity differs from resolved config")
    if metadata["stream_identity"] != canonical_digest(manifest["updates"]):
        raise ValueError("ordered input stream identity mismatch")
    if len(manifest["updates"]) != 400:
        raise ValueError("complete 400-update stream must be prepared")
    import zipfile
    archive = manifest["source_archive"]
    archive_path = root / archive["path"]
    if file_digest(archive_path) != archive["sha256"]:
        raise ValueError("retained source archive checksum mismatch")
    with zipfile.ZipFile(archive_path) as bundle:
        import hashlib
        for name, digest in inventory.items():
            if hashlib.sha256(bundle.read(name)).hexdigest() != digest:
                raise ValueError("retained source bytes differ from executed inventory")
    metadata["source_archive"] = {"path": str(archive_path), "sha256": archive["sha256"]}
    if metadata["data_identity"] != canonical_digest(manifest["data_ids"]):
        raise ValueError("state data identity differs from prepared data IDs")
    if resource_guard is not None:
        resource_guard()
    metadata["execution_determinism"] = configure_training_determinism(config["training"])
    seed_everything(0)
    processor, model = build_model(config)
    site = resolve_encoder_site(model.base, config["split"], config["model"]["revision"])
    metadata.update(site=asdict(site), model_state_contract="codec-only-stateless-channel-v1",
                    config=config, data_ids=manifest["data_ids"],
                    prepared_manifest={"path": str(manifest_path), "sha256": file_digest(manifest_path)})
    initial_ref = manifest["initialization"]
    initial_path = root / initial_ref["path"]
    if file_digest(initial_path) != initial_ref["sha256"] or metadata["initialization_identity"] != initial_ref["sha256"]:
        raise ValueError("immutable initialization checksum mismatch")
    model.codec.load_state_dict(torch.load(initial_path, weights_only=True, map_location="cpu"), strict=True)
    metadata["retained_inputs"] = {
        "source.zip": {"path": str(archive_path), "sha256": archive["sha256"]},
        "prepared.json": {"path": str(manifest_path), "sha256": file_digest(manifest_path)},
        "initialization.pt": {"path": str(initial_path), "sha256": initial_ref["sha256"]}}
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("run output must be fresh")
    from jscc.native64_baseline import Native64Learner, PreparedUpdates, same_shape_guard, evaluate_vanilla_panel
    learner_class = Native64Learner if native_c else BaselineLearner
    learner = learner_class(model, run_id=manifest["run_id"], task=config["task"],
        identity={"source": metadata["source_identity"], "config": metadata["config_identity"],
                  "data": metadata["data_identity"], "parent": metadata["parent_identity"]},
        pairing_id=manifest["pairing_id"], lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"], grad_clip=config["training"]["grad_clip"],
        event_sink=None,
        audit_policy="sparse_first_final" if native_c else "full")
    if native_c:
        same_shape_guard(learner, manifest["updates"], root,
                         output.parent / "acceptance" / manifest["cell"], resource_guard)
        learner.event_sink = getattr(resource_guard, "observe_event", None)
    replay = {}
    if segment.strategy != "both":
        for role, declaration in manifest["replays"].items():
            requirement = declaration["requirement"]
            backbone = manifest["capture_backbone"]
            if backbone["model_revision"] != config["model"]["revision"]:
                raise ValueError("capture backbone differs from executed model revision")
            replay[role] = open_baseline_replay(root / declaration["path"], requirement,
                source_identity=metadata["source_identity"], role=role, site=site,
                backbone=backbone, dtype=next(model.base.parameters()).dtype)
    def read_many(references, kind, role):
        if kind == "combined":
            return [read_prepared_batch(ref, root) for ref in references]
        return [read_replay_batch(replay[role], ref) for ref in references]
    def updates(index, kind):
        return read_many(manifest["updates"][index], kind, "optimization")
    def validation(kind):
        batches = read_many(manifest["objective_validation"], kind, "objective_validation")
        if sum(len(b["labels"]) for b in batches) != 128:
            raise ValueError("objective validation requires exact prepared 128 sequences")
        return batches
    from jscc.baseline_protocol import comparison_refs
    kind = "combined" if segment.strategy in {"both", "staged"} else "local"
    controls = comparison_refs(learner, metadata, kind)
    # Bind the actual resolved backbone site, as in canonical runtime acceptance.
    request = {**manifest["task_request"], "target_site": asdict(site)}
    if request["comparison"] != controls:
        raise ValueError("prepared task comparison differs from actual config/state")
    if request["expected_items"] != 256:
        raise ValueError("development assessment requires exact 256-item panel")
    task_data = load_data(config, processor, manifest["data_ids"], for_training=False) if config["task"] == "coco" else None
    def assess(state, directory):
        actual = {**request, "checkpoint_sha256": state.reference.sha256,
                  "step": state.payload["metadata"]["completed_updates"],
                  "parent": state.payload["metadata"]["parent_identity"],
                  "comparison": state.payload["metadata"]["comparison_controls"]}
        return evaluate_checkpoint(state, actual, model=model, processor=processor, data=task_data, output=directory)
    parent = None
    if segment.start:
        ref = dict(manifest["parent_checkpoint"])
        ref["path"] = str(root / ref["path"])
        parent = open_state(CheckpointRef(**ref), expected=manifest["parent_expected"])
    reused = {}
    for step, declaration in manifest.get("reused_assessments", {}).items():
        checkpoint = dict(declaration["checkpoint"])
        checkpoint["path"] = str(root / checkpoint["path"])
        reused_state = open_state(CheckpointRef(**checkpoint), expected=declaration["expected"])
        receipt_path = root / declaration["receipt"]["path"]
        if file_digest(receipt_path) != declaration["receipt"]["sha256"]:
            raise ValueError("reused assessment receipt checksum mismatch")
        reused[int(step)] = {"state": reused_state, "receipt": json.loads(receipt_path.read_text()), "root": receipt_path.parent}
    if native_c and manifest["cell"] == "D-none":
        assert resource_guard is not None
        import time
        from jscc.experiment_state import isolated_rng
        resource_guard()
        begin = time.monotonic()
        with isolated_rng(config["seed"]):
            evaluate_vanilla_panel(model, processor, request, output.parent / "vanilla")
        resource_guard.vanilla_complete(time.monotonic() - begin)
    def run(actual_updates):
        return run_baseline(learner, output=output, metadata=metadata, update_batches=actual_updates,
                            validation_batches=validation, assess=assess, segment=segment, parent=parent,
                            reused_assessments=reused, task_request=request, resource_guard=resource_guard,
                            allocation_required=resource_guard is not None,
                            on_output_created=getattr(resource_guard, "on_output_created", None))
    if native_c:
        with PreparedUpdates(learner, manifest["updates"], root) as prepared_updates:
            return run(prepared_updates)
    return run(updates)


def execute(manifest_path, output):
    """Charge the entire prepared command before model construction on a GPU."""
    from jscc.activation_replay import file_digest
    from jscc.config import load_config
    from jscc.experiment_schedule import AllocationLedger
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("run output must be fresh")
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    config_path = manifest_path.parent / manifest["config"]["path"]
    if file_digest(config_path) != manifest["config"]["sha256"]:
        raise ValueError("prepared config checksum mismatch")
    config = load_config(config_path)
    if config["model"]["device"] == "cpu":
        return _execute(manifest_path, output)
    allocation = manifest["allocation"]
    if allocation["cap_device_seconds"] != 7200:
        raise ValueError("baseline whole-pilot cap is exactly two aggregate GPU hours")
    measurement = manifest_path.parent / allocation["measurement"]["path"]
    if file_digest(measurement) != allocation["measurement"]["sha256"]:
        raise ValueError("allocation measurements differ from retained receipt")
    measured = json.loads(measurement.read_text())
    for key in ("max_duration_seconds", "mandatory_reserve_device_seconds", "devices", "identities"):
        if allocation[key] != measured[key]:
            raise ValueError("allocation differs from measured preparation")
    identities = allocation["identities"]
    if identities != {"source": manifest["state_metadata"]["source_identity"],
                      "config": manifest["state_metadata"]["config_identity"],
                      "input": manifest["state_metadata"]["stream_identity"]}:
        raise ValueError("allocation identities differ from prepared command")
    journal = Path(allocation["campaign_journal"])
    if not journal.is_absolute() or measured["campaign_journal"] != str(journal):
        raise ValueError("stable absolute campaign journal must be bound in preparation")
    reservation = Path(str(journal) + ".claim")
    reservation.mkdir()  # A surviving reservation requires explicit recovery, never replay.
    from jscc.experiment_records import write_json_atomic
    terminal_written = False
    output_ownership = {"created": False}
    started = False
    ledger = None
    prior_hash = None
    def publish(status, error=None):
        nonlocal terminal_written
        write_json_atomic(journal, {"schema": "baseline-campaign-journal-v1",
            "campaign_id": allocation["campaign_id"], "command_id": allocation["command_id"],
            "status": status, "prior_sha256": prior_hash, "output": str(Path(output).resolve()),
            "ledger": ledger.snapshot() if ledger is not None else None,
            "error": None if error is None else {"type": type(error).__name__, "message": str(error)}})
        terminal_written = status in {"complete", "failed"}
    try:
        prior = 0.0
        prior_commands = []
        reference = allocation.get("prior_ledger")
        if journal.exists():
            if reference is None:
                raise ValueError("existing campaign requires exact current prior journal reference")
            path = (manifest_path.parent / reference["path"]).resolve()
            if path != journal.resolve() or file_digest(journal) != reference["sha256"]:
                raise ValueError("prior campaign journal identity is missing or stale")
            prior_hash = reference["sha256"]
            previous_journal = json.loads(journal.read_text())
            if previous_journal["status"] != "complete" or previous_journal["campaign_id"] != allocation["campaign_id"]:
                raise ValueError("failed or active campaign cannot automatically continue")
            previous = previous_journal["ledger"]
            if previous["campaign_id"] != allocation["campaign_id"] or previous["cap_device_seconds"] != 7200:
                raise ValueError("prior ledger campaign/cap mismatch")
            if previous["terminal_reason"] or any(a["status"] != "completed" for a in previous["allocations"]):
                raise ValueError("failed or active campaign cannot automatically continue")
            prior = previous["charged_device_seconds"]
            prior_commands = previous.get("prior_command_ids", []) + [a["command_id"] for a in previous["allocations"]]
            if allocation["command_id"] in prior_commands:
                raise ValueError("duplicate campaign command; automatic retry prohibited")
        elif reference is not None:
            raise ValueError("prior campaign journal is missing")
        started = True
        publish("reserved")
        ledger = AllocationLedger(Path(str(output) + ".allocation.json"), allocation["campaign_id"], 7200,
                                  prior_device_seconds=prior, prior_command_ids=prior_commands)
        ledger.start(allocation["command_id"], stage=allocation["stage"], devices=allocation["devices"],
                     max_duration_seconds=allocation["max_duration_seconds"],
                     mandatory_reserve_device_seconds=allocation["mandatory_reserve_device_seconds"],
                     measurement_receipt=allocation["measurement"]["sha256"], identities=identities)
        publish("running")
        def guard():
            ledger.check()
            publish("running")
        setattr(guard, "on_output_created", lambda: output_ownership.update(created=True))
        result = _execute(manifest_path, output, guard)
        ledger.stop(allocation["command_id"], status="completed")
        publish("pending_records")
        status = bind_allocation_result(output, ledger.snapshot())
        if status["status"] != "complete":
            raise RuntimeError("allocation-bound records remain incomplete")
        publish("complete")
        return {**result, **status}
    except BaseException as error:
        if started:
            try:
                if ledger is not None and any(a["status"] == "allocated" for a in ledger.snapshot()["allocations"]):
                    ledger.stop(allocation["command_id"], status="failed")
            finally:
                publish("failed", error)
                if ledger is not None and output_ownership["created"]:
                    try:
                        bind_allocation_result(output, ledger.snapshot(), error)
                    except Exception:
                        # Required allocation artifact/receipt remains absent or incomplete;
                        # the failed campaign journal prevents continuation independently.
                        pass
        raise
    finally:
        if not started or terminal_written:
            reservation.rmdir()
        # Ambiguous journal writes intentionally retain the exclusive reservation.



def bind_allocation_result(output, receipt, failure=None):
    """Include physical allocation outcome in the same recomputed completion."""
    from datetime import datetime, timezone
    from jscc.experiment_records import (append_event, artifact_ref, completion_status,
                                        measure, read_events, write_json_atomic)
    output = Path(output)
    write_json_atomic(output / "allocation.json", receipt)
    events = read_events(output / "metrics.jsonl") if (output / "metrics.jsonl").exists() else []
    if not events:
        # No successful record production means no claim of completed work.
        status = {"status": "incomplete", "error": str(failure)}
        write_json_atomic(output / "completion.json", status)
        return status
    template = {key: value for key, value in events[-1].items() if key not in {"event_type", "payload"}}
    template["timestamp"] = datetime.now(timezone.utc).isoformat()
    seconds = sum(a["charged_device_seconds"] for a in receipt["allocations"])
    append_event(output / "metrics.jsonl", {**template, "event_type": "cost", "payload": {
        "category": "gpu_allocation", "seconds": measure(seconds), "attribution": "first_use", "site_id": "enc_fn",
        "scope": "aggregate_physical_device_seconds_current_command; prior separately in allocation.json",
        "concurrency": "explicit_ledger", "device_count": sum(a["devices"] for a in receipt["allocations"])}})
    if failure is not None or receipt["terminal_reason"]:
        append_event(output / "metrics.jsonl", {**template, "event_type": "failure", "payload": {
            "reason": str(failure or receipt["terminal_reason"]), "reference": "allocation.json"}})
    manifest = json.loads((output / "manifest.json").read_text())
    if "allocation.json" not in manifest["expected_artifacts"]:
        manifest["expected_artifacts"].append("allocation.json")
    write_json_atomic(output / "manifest.json", manifest)
    inventory = {"schema": "experiment-records-v1", "mode": "tensor-complete", "omitted_tensors": [],
                 "artifacts": [artifact_ref(path, output, "tensor" if path.suffix == ".pt" else "metadata")
                               for path in sorted(output.rglob("*")) if path.is_file()
                               and path.name not in {"inventory.json", "completion.json"}]}
    write_json_atomic(output / "inventory.json", inventory)
    status = completion_status(manifest, read_events(output / "metrics.jsonl"), inventory, output)
    write_json_atomic(output / "completion.json", status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparisons", type=Path, help="write exact contrasts for a complete four-cell campaign")
    parser.add_argument("--execute", type=Path, help="prepared immutable manifest; otherwise preview only")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--selected-cell", action="append", default=[])
    parser.add_argument("--owner-approval")
    args = parser.parse_args()
    if args.comparisons:
        if args.execute or args.output or args.selected_cell or args.owner_approval:
            parser.error("--comparisons cannot be combined with execution or preview arguments")
        from jscc.baseline_protocol import write_baseline_comparisons
        result = write_baseline_comparisons(args.comparisons)
    elif args.execute:
        if args.output is None:
            parser.error("--execute requires a fresh --output directory")
        result = execute(args.execute, args.output)
    else:
        plan = compile_baseline_plan(args.selected_cell, args.owner_approval)
        result = {"schema": "baseline-plan-v2", **asdict(plan), "physical_updates": plan.physical_updates,
                  "task_assessments": plan.task_assessments, "condition_panels": plan.condition_panels,
                  "development_item_assessments": plan.development_item_assessments,
                  "launch_authorization": False}
        if args.output:
            from jscc.baseline_protocol import write_manifest
            if args.output.exists():
                raise FileExistsError(args.output)
            write_manifest(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
