#!/usr/bin/env python3
"""Preview the bounded plan or execute one explicitly prepared baseline segment."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jscc.experiment_schedule import compile_baseline_plan


def execute(manifest_path, output):
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
    from jscc.runtime import seed_everything

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
    if (config["split"] != {"stack": "enc", "where": "after_final_norm"}
            or config["codec"]["bottleneck_dim"] != 512 or config["seed"] != 0
            or config["training"]["max_steps"] != 400):
        raise ValueError("prepared config differs from enc_fn/B512/seed0/400 protocol")
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
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("run output must be fresh")
    def event_sink(event):
        with (output / "metrics.jsonl").open("a") as destination:
            destination.write(json.dumps(event, allow_nan=False) + "\n")
    learner = BaselineLearner(model, run_id=manifest["run_id"], task=config["task"],
        identity={"source": metadata["source_identity"], "config": metadata["config_identity"],
                  "data": metadata["data_identity"], "parent": metadata["parent_identity"]},
        pairing_id=manifest["pairing_id"], lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"], grad_clip=config["training"]["grad_clip"],
        event_sink=event_sink)
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
    request = manifest["task_request"]
    if request["expected_items"] != 256:
        raise ValueError("development assessment requires exact 256-item panel")
    task_data = load_data(config, processor, manifest["data_ids"], for_training=False) if config["task"] == "coco" else None
    def assess(state, directory):
        actual = {**request, "checkpoint_sha256": state.reference.sha256,
                  "step": state.payload["metadata"]["completed_updates"],
                  "parent": state.payload["metadata"]["parent_identity"]}
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
        reused[int(step)] = {"state": reused_state, "receipt": json.loads(receipt_path.read_text())}
    result = run_baseline(learner, output=output, metadata=metadata, update_batches=updates,
                         validation_batches=validation, assess=assess, segment=segment, parent=parent, reused_assessments=reused, task_request=request)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", type=Path, help="prepared immutable manifest; otherwise preview only")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--selected-cell", action="append", default=[])
    parser.add_argument("--owner-approval")
    args = parser.parse_args()
    if args.execute:
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
