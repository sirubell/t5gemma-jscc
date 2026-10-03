"""Create deterministic synthetic experiment records without model imports."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from jscc import experiment_records as records  # noqa: E402


def measure(value, reason=None):
    return {"value": value, "reason": reason}


def event(kind, run, phase, payload, task="hellaswag"):
    return {
        "schema": "experiment-records-v1", "event_type": kind,
        "run_id": run, "phase_id": phase, "task": task,
        "timestamp": "2026-09-30T00:00:00+00:00",
        "units": {"time": "seconds", "memory": "bytes", "snr": "dB", "coordinates": "real"},
        "identity": {"source": "synthetic-source", "config": f"synthetic-config-{run}", "data": "synthetic-data", "parent": "synthetic-initialization"},
        "payload": payload,
    }


def update(run, phase, step, final=12, site="enc_fn", task="hellaswag"):
    local = phase == "local"
    raw_r = 0.7 / (step + 1)
    raw_k = 0.5 / (step + 1)
    k = {"raw": measure(None, "unmeasured"), "weighted": measure(None, "unmeasured"), "numerator": measure(None, "unmeasured"), "denominator": 0} if local else {"raw": measure(raw_k), "weighted": measure(raw_k), "numerator": measure(raw_k * 16), "denominator": 16}
    return event("update", run, phase, {
        "site_id": site, "site_step": step, "sweep": step, "phase_start_step": 0, "attempted_step": step, "completed_step": step, "final_step": final,
        "exposure": {"source_tokens": 32, "target_tokens": 16, "sequences": 4, "padded_tokens": 40, "valid_tokens": 32, "cumulative_valid_tokens": 32 * step, "latent_width": 8, "hidden_coordinates": 256, "hidden_coordinates_allocated": 320, "memory_coordinates": measure(None, "not_applicable"), "memory_coordinates_allocated": measure(None,"not_applicable")},
        "objective": {"kind": "local" if local else "combined", "total": measure(0.1 * raw_r + (0 if local else raw_k)), "components": {"R": {"raw": measure(raw_r), "weighted": measure(0.1 * raw_r), "numerator": measure(raw_r * 4), "denominator": 4}, "K": k}},
        "lr_used": [0.0 if step == 1 else 0.001 * (final-step+1)/final], "lr_next": [0.001 * (final-step)/final],
        "gradient": {"pre_clip": measure(1.4/step), "post_clip": measure(min(1.0, 1.4/step)), "clip_threshold": measure(1.0), "clipped": step == 1},
        "nonfinite": False, "skipped": False,
        "update_l2": measure(0.002 * step) if step in (1, final) or step % 10 == 0 else measure(None, "not_sampled"),
        "snr": {"condition": "uniform[-6,18]", "min": measure(-5.0), "mean": measure(6.0), "max": measure(17.0), "draw_ref": f"synthetic-{site}-{step}"},
        "timing": {"step_seconds": measure(0.25), "elapsed_seconds": measure(0.25*step), "data_cache_seconds": measure(0.05), "throughput": measure(128.0), "throughput_denominator": "source_valid_tokens", "scope": "host_wall_async_no_cuda_sync"},
        "memory": {"allocated_bytes": measure(None, "cpu_only"), "reserved_bytes": measure(None, "cpu_only"), "reset_scope": "phase"},
    }, task)


def observation(run, phase, site, kind, step, *, task="hellaswag", incomplete=False):
    conditions = ["vanilla"] if kind == "vanilla" else records.CONDITIONS
    request = {
        "source": "synthetic-source", "config": f"synthetic-config-{run}", "data": "synthetic-data",
        "parent": "synthetic-initialization", "checkpoint": f"synthetic-{run}-{step}",
        "panel": f"synthetic-{task}-panel", "site": site,
        "role": "heldout" if site == "enc_l14" else "trained", "task": task,
        "protocol": "synthetic-protocol", "noise": "synthetic-fixed-noise", "scorer": "synthetic-scorer",
        "layout": "synthetic-full-sequences", "precision": "float32", "backend": "cpu",
        "conditions": conditions, "expected_items": 4, "learner_kind": kind, "step": step, "site_step": step, "purpose": "task", "objective_kind": None,
        "comparison": {"architecture": run if run.startswith("baseline-") else "synthetic-architecture", "initialization": run if run.startswith("baseline-") else "synthetic-initial-tensors", "training_data": "synthetic-training-data", "objective": "synthetic-combined" if phase != "local" else "synthetic-local", "exposure": "synthetic-q12-per-site", "schedule": "synthetic-site-cosine"},
        "details": {"prompt": "synthetic-prompt", "native_policy": "reference-prefix", "draws": "synthetic-draws", "outputs": "synthetic-output-binding"},
    }
    rows = []
    for index, condition in enumerate(conditions):
        failed = incomplete and index == len(conditions) - 1
        items = []
        for item in range(3 if failed else 4):
            normalized = int((item + index + (kind == "shared") + (run == "baseline-direct")) % 4 != 0)
            raw = int((item + index) % 3 != 0)
            row = {"item_id": str(item), "source_id": f"family-{item//2}", "normalized_correct": normalized, "raw_correct": raw,
                   "prediction": str(item % 4), "tokens": [1,2,3], "score": normalized}
            if task == "coco":
                row.update({"caption_raw": "a synthetic object .", "caption_clean": "a synthetic object.", "cider": float(item + 1), "eos": True, "cap_hit": False})
            items.append(row)
        metrics = {"normalized_accuracy": measure(sum(i["normalized_correct"] for i in items)/len(items)), "raw_accuracy": measure(sum(i["raw_correct"] for i in items)/len(items))}
        if task == "coco":
            metrics = {"cider": measure(sum(i["cider"] for i in items)/len(items))}
        rows.append({"condition": condition, "requested": 4, "completed": len(items), "failed": int(failed), "failure_reason": "synthetic item failure" if failed else None, "denominator": len(items), "metrics": metrics, "items": items, "objectives": {}})
    payload = {"request": request, "identity": records.observation_identity(request), "status": "incomplete" if incomplete else "complete", "conditions": rows, "reuse": None}
    return event("observation", run, phase, payload, task)


def objective_observation(run, phase, step, task):
    row = observation(run, phase, "enc_fn", "specialist", step, task=task)
    payload = row["payload"]
    payload["request"]["purpose"] = "objective"
    payload["request"]["objective_kind"] = "local" if phase == "local" else "combined"
    payload["request"]["panel"] = "synthetic-objective-selection-panel"
    payload["identity"] = records.observation_identity(payload["request"])
    components = update(run, phase, max(1,step), task=task)["payload"]["objective"]["components"]
    for condition in payload["conditions"]:
        condition["metrics"] = {}
        condition["objectives"] = components
    return row


def build(output: Path):
    output.mkdir(parents=True, exist_ok=False)
    run_specs = [("shared", "combined", "hellaswag"), ("specialist", "combined", "hellaswag"), ("local", "local", "hellaswag"), ("coco", "combined", "coco"), ("staged", "local", "hellaswag"), ("baseline-residual", "combined", "hellaswag"), ("baseline-direct", "combined", "hellaswag")]
    terminal = {}
    for run, phase, task in run_specs:
        directory = output / run
        directory.mkdir()
        total = 36 if run == "shared" else 12
        events = [update(run, phase, step, final=total, task=task) for step in range(1,total+1)]
        if run == "shared":
            sites = ("enc_l9", "enc_l19", "enc_fn")
            for row in events:
                step = row["payload"]["attempted_step"]
                sweep = (step-1)//3
                row["payload"]["site_id"] = sites[((step-1)%3+sweep)%3]
                row["payload"]["site_step"] = sweep+1
                row["payload"]["sweep"] = sweep+1
        phases = [{"phase_id": phase, "start_step": 0, "start_valid_tokens": 0, "updates": total}]
        if run == "staged":
            for step in range(13,25):
                row = update(run, "combined", step, final=24, task=task)
                row["payload"]["phase_start_step"] = 12
                row["payload"]["update_l2"] = measure(0.002*step) if step in (13,20,24) else measure(None,"not_sampled")
                events.append(row)
            phases.append({"phase_id": "combined", "start_step": 12, "start_valid_tokens": 384, "updates": 12})
            for step in (15,18,21,24):
                events.append(objective_observation(run, "combined", step, task))
            for step in (18,24):
                events.append(observation(run, "combined", "enc_fn", "specialist", step))
        if run in ("shared", "specialist"):
            sites = ["enc_l9", "enc_l19", "enc_fn"] + (["enc_l14"] if run == "shared" else [])
            for site in sites:
                for step in (0,6,12):
                    row = observation(run, phase, site, "initialization" if step == 0 else run, step*3 if run == "shared" else step)
                    row["payload"]["request"]["site_step"] = step
                    row["payload"]["identity"] = records.observation_identity(row["payload"]["request"])
                    events.append(row)
            if run == "shared":
                vanilla = observation(run, phase, "bypass", "vanilla", 0)
                request = vanilla["payload"]["request"]
                request["role"] = "diagnostic"
                request["noise"] = "not_applicable"
                request["details"]["draws"] = "not_applicable"
                request["checkpoint"] = "synthetic-frozen-backbone"
                request["comparison"] = dict.fromkeys(request["comparison"], "not_applicable")
                vanilla["payload"]["identity"] = records.observation_identity(request)
                events.append(vanilla)
        else:
            for step in (0,6,12):
                events.append(observation(run, phase, "enc_fn", "initialization" if step == 0 else "specialist", step, task=task, incomplete=run == "coco" and step == 12))
        for step in (3,6,9,12):
            row = objective_observation(run, phase, step*3 if run == "shared" else step, task)
            row["payload"]["request"]["site_step"] = step
            if run == "shared":
                row["payload"]["request"]["learner_kind"] = "shared"
            row["payload"]["identity"] = records.observation_identity(row["payload"]["request"])
            events.append(row)
        for attribution in ("first_use", "reuse"):
            for site in ("enc_l9", "enc_l19", "enc_fn"):
                for category in ("setup", "capture", "train", "validation", "task_scoring", "checkpoint_io", "record_io", "command_wall", "gpu_allocation"):
                    events.append(event("cost", run, phase, {"category": category, "seconds": measure(1.25 if attribution == "first_use" else 0.5), "attribution": attribution, "site_id": site, "scope": "per_site", "concurrency": "sequential", "device_count": 0}, task))
        if run == "specialist":
            for attribution in ("first_use", "reuse"):
                events.append(event("cost", run, phase, {"category": "train", "seconds": measure(3.75 if attribution == "first_use" else 1.5), "attribution": attribution, "site_id": "enc_l9+enc_l19+enc_fn", "scope": "bank_total", "concurrency": "sequential", "device_count": 0}, task))
        if run == "coco":
            events.append(event("failure", run, phase, {"reason": "synthetic missing terminal item", "reference": "synthetic-failure-receipt"}, task))
        tensor_sites = ("enc_l9", "enc_l19", "enc_fn") if run == "specialist" else ("enc_fn",)
        tensor_paths = []
        for site in tensor_sites:
            name = f"synthetic-{site}.bin"
            (directory / name).write_bytes(site.encode().ljust(8, b".")[:8])
            tensor_paths.append(name)
            events.append(event("footprint", run, phase, {"scope": "shared" if run == "shared" else "per_site", "site_id": site, "parameter_count": measure(16 if run == "baseline-direct" else 64), "checkpoint_bytes": measure(8), "checkpoint_refs": [name]}, task))
        if run == "specialist":
            events.append(event("footprint", run, phase, {"scope": "bank_total", "site_id": "enc_l9+enc_l19+enc_fn", "parameter_count": measure(192), "checkpoint_bytes": measure(24), "checkpoint_refs": tensor_paths}, task))
        for row in events:
            if run.startswith("baseline-") and row["event_type"] == "observation" and row["payload"]["request"]["purpose"] == "task" and row["payload"]["request"]["step"] == 12:
                terminal[run] = row["payload"]["identity"]
            records.append_event(directory / "metrics.jsonl", row)
        (directory / "source.txt").write_text("Synthetic scalar-only source fixture; no trained tensors.\n")
        inventory = {"schema": records.SCHEMA, "mode": "metadata-only", "artifacts": [records.artifact_ref(directory / name, directory, "metadata") for name in ("metrics.jsonl", "source.txt")] + [records.artifact_ref(directory / name, directory, "tensor") for name in tensor_paths], "omitted_tensors": []}
        manifest = {"schema": records.SCHEMA, "run_id": run, "phases": phases, "expected_observations": [e["payload"]["identity"] for e in events if e["event_type"] == "observation"], "expected_artifacts": ["metrics.jsonl", "source.txt"]}
        records.write_json_atomic(directory / "inventory.json", inventory)
        records.write_json_atomic(directory / "manifest.json", manifest)
    comparisons = {"schema": "experiment-comparisons-v1", "contrasts": [{"id": "synthetic-direct-minus-residual-terminal", "left_observation": terminal["baseline-direct"], "right_observation": terminal["baseline-residual"], "metrics": ["normalized_correct", "raw_correct"], "allowed_control_differences": ["architecture", "initialization"]}]}
    records.write_json_atomic(output / "comparisons.json", comparisons)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(build(args.output))
