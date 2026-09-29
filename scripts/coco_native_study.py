"""Prepare a COCO fixed-step study and timing estimates; never submit GPU jobs.

Run with ``uv run --locked --no-sync python -m scripts.coco_native_study``.

Measurements are an explicit reviewed summary of preflight evidence, not a
quality-based selection. All thirteen routes need passed execution checks with any numerical limitations
explicitly reviewed and preserved, and every timing proxy
must name its measured representative. Formal execution requires later approval.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

from jscc.config import save_config, validate_config
from jscc.studies import StudyPlan, expand_study, export_study

SPLITS = ("enc_emb", "enc_l4", "enc_l9", "enc_l14", "enc_l19", "enc_fn",
          "dec_l0", "dec_l4", "dec_l8", "dec_l12", "dec_l16", "dec_l20", "dec_l24")
CONDITIONS = ["no_noise", -6, 6, 18]
PRESENTATIONS = 192_000


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def positive(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return value


def integer(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def selected_indices(receipt):
    indices = receipt.get("selected_indices", list(range(13)))
    if (not isinstance(indices, list) or not indices
            or any(type(index) is not int or index not in range(13) for index in indices)
            or indices != sorted(set(indices)) or 5 not in indices):
        raise ValueError("selected_indices must be sorted unique canonical indices including enc_fn index5")
    return indices


def validate_measurements(receipt):
    """Validate the normalized, evidence-bound preflight review contract."""
    selected_indices(receipt)
    if receipt.get("version") != "coco-native-measurements-v1" or receipt.get("status") != "PASS":
        raise ValueError("A passed coco-native-measurements-v1 review is required")
    if receipt.get("selection_basis") != "capacity_and_throughput_before_quality":
        raise ValueError("Batch selection must precede quality observations")
    if set(receipt.get("routes", {})) != set(SPLITS):
        raise ValueError("Exactly thirteen route correctness receipts are required")
    for split, row in receipt["routes"].items():
        if row.get("status") == "PASS_EXECUTION_WITH_NUMERICAL_LIMITATION":
            limits = row.get("numerical_limitations")
            review = row.get("numerical_review_file")
            if row.get("execution_checks_passed") is not True or not isinstance(limits, list) or not limits or not all(limits):
                raise ValueError(f"{split}: numerical limitations require passed execution checks and a nonempty limitations list")
            if not isinstance(review, str) or not review or review not in receipt.get("evidence_files", {}):
                raise ValueError(f"{split}: numerical limitations require an evidence-bound independent review")
        elif row.get("status") != "PASS":
            raise ValueError("All routes need passed execution checks or reviewed numerical limitations")
        elif row.get("numerical_limitations"):
            raise ValueError(f"{split}: numerical limitations must not be hidden under PASS")
    batch = integer(receipt["training_batch_size"], "training batch")
    accumulation = integer(receipt["gradient_accumulation"], "accumulation")
    integer(receipt["evaluation_batch_size"], "evaluation batch")
    presentations = integer(receipt.get("presentations_per_run", PRESENTATIONS), "presentations per run")
    if presentations % (batch * accumulation):
        raise ValueError("Effective batch must divide the chosen presentations per run")
    if not receipt.get("evidence_files") or not receipt.get("source_files"):
        raise ValueError("Preflight raw evidence and executed source bindings are required")
    for split, row in receipt["routes"].items():
        if row["timing_representative"] not in SPLITS:
            raise ValueError(f"Unknown timing representative for {split}")
        for key in ("seconds_per_update", "generation_seconds_per_image", "selection_seconds"):
            positive(row[key], f"{split}.{key}")
        if row.get("timing_kind") not in {"measured", "conservative_proxy"}:
            raise ValueError("Timing must distinguish measurements from proxies")
    for row in receipt["routes"].values():
        representative = receipt["routes"][row["timing_representative"]]
        if representative["timing_kind"] != "measured":
            raise ValueError("Timing proxies must point to an actually measured route")
    positive(receipt["vanilla_generation_seconds_per_image"], "vanilla generation")
    return presentations // (batch * accumulation)


def numerical_limitations(receipt):
    """Carry raw numerical exceptions into the same later approval card."""
    return [{"split": split, "status": row["status"],
             "numerical_limitations": copy.deepcopy(row["numerical_limitations"]),
             "numerical_review_file": row["numerical_review_file"]}
            for split, row in receipt["routes"].items()
            if row["status"] == "PASS_EXECUTION_WITH_NUMERICAL_LIMITATION"]


def verify_bindings(receipt, evidence_root, source_root):
    for key, root in (("evidence_files", evidence_root), ("source_files", source_root)):
        root = Path(root).resolve()
        for relative, expected in receipt[key].items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or digest(path) != expected:
                raise ValueError(f"Changed or outside-root binding: {key}/{relative}")


def estimate(receipt, updates, samples=2000, factor=1.5):
    """Include measured selection and conservative startup/checkpoint allowance.

    Times are planning estimates, not reserved runtime or convergence claims.
    Four-slot lower bound ignores queueing and scheduling differences.
    """
    positive(factor, "timing factor")
    if factor < 1:
        raise ValueError("Timing safety factor cannot be less than one")
    rows = []
    for index in selected_indices(receipt):
        split = SPLITS[index]
        timing = receipt["routes"][split]
        train = factor * (updates * timing["seconds_per_update"] + timing["selection_seconds"]) + 600
        evaluation = factor * len(CONDITIONS) * samples * timing["generation_seconds_per_image"] + 600
        rows.append({"split": split, "timing_kind": timing["timing_kind"],
                     "timing_representative": timing["timing_representative"],
                     "train_seconds": train, "evaluation_seconds": evaluation,
                     "train_job_minutes": math.ceil(train / 60),
                     "evaluation_job_minutes": math.ceil(evaluation / 60)})
    vanilla = factor * samples * receipt["vanilla_generation_seconds_per_image"] + 600
    total = sum(row["train_seconds"] + row["evaluation_seconds"] for row in rows) + vanilla
    return {"timing_hardware": receipt.get("hardware", "UNSPECIFIED; no H200 speed claim"),
            "cross_hardware_scaling_applied": False, "safety_factor": factor, "startup_checkpoint_allowance_seconds_per_job": 600,
            "rows": rows, "vanilla_seconds": vanilla, "total_gpu_hours": total / 3600,
            "four_gpu_no_queue_lower_bound_hours": max(total / 4,
                max(row["train_seconds"] + row["evaluation_seconds"] for row in rows)) / 3600,
            "all_jobs_within_48h": all(max(row["train_seconds"], row["evaluation_seconds"]) <= 172800
                                         for row in rows) and vanilla <= 172800,
            "limitations": "Short preflight extrapolation on the recorded hardware only; no WS-to-H200 conversion. Queueing, late-training variability and shared I/O excluded."}


def scenarios(receipt):
    """Compare exposure/evaluation choices without selecting one from quality.

    Lower exposures are explicit alternatives, not silently shortened baselines.
    This function also accepts representative-only timing summaries; its caller
    must retain their incomplete correctness coverage and must not dispatch.
    """
    effective = integer(receipt["training_batch_size"], "batch") * integer(
        receipt["gradient_accumulation"], "accumulation")
    result = []
    for presentations in (64_000, 128_000, 192_000):
        if presentations % effective:
            continue
        for images in (512, 2000):
            updates = presentations // effective
            result.append({"presentations_per_run": presentations,
                           "role": "legacy_nominal_exposure" if presentations == 192_000 else "reduced_exposure_alternative",
                           "effective_batch": effective, "updates": updates,
                           "images_per_panel": images, "quality_equivalence_claimed": False,
                           "execution_authorized": False,
                           "estimates": estimate(receipt, updates, images)})
    return result


def prepare(measurements, source_root, evidence_root, output, study_spec=None):
    receipt = json.loads(Path(measurements).read_text())
    updates = validate_measurements(receipt)
    verify_bindings(receipt, evidence_root, source_root)
    if study_spec is None:
        study_spec = Path(__file__).resolve().parents[1] / "configs/studies/splits.yaml"
    base = expand_study(study_spec, task="coco")
    if tuple(run.experiment for run in base.runs) != SPLITS:
        raise ValueError("The study must contain the canonical thirteen REF routes in order")
    for run in base.runs:
        config = run.config
        if run.seed != 0 or config["codec"]["snr_film"] or not config["channel"]["normalize_power"]:
            raise ValueError("Native baseline requires seed0, FiLM off and retained power normalization")
        expected_norm = "post" if run.experiment == "enc_emb" else "both" if run.experiment == "enc_fn" else "none"
        if config["codec"]["layernorm"] != expected_norm or config["codec"].get("memory", {}).get("layernorm") != "both":
            raise ValueError("REF main/memory outer LayerNorm policy changed")
        training = config["training"]
        if integer(config["data"]["num_train"], "training row count") % receipt["training_batch_size"]:
            raise ValueError("Microbatch must divide data.num_train; larger batches require an explicit full-batch sampling policy before approval")
        # A graceful timer stop is partial evidence, not successful fixed-budget training.
        training.pop("max_minutes", None)
        training.update(batch_size=receipt["training_batch_size"],
                        gradient_accumulation=receipt["gradient_accumulation"],
                        max_steps=updates, schedule_steps=updates, eval_every=updates,
                        selection_steps=[updates], save_steps=[updates], patience=None,
                        streamed_backward=True, valid_only_kl=True)
        config["evaluation"].update(snrs=CONDITIONS, vanilla=False, mode="codec_only",
                                    batch_size=receipt["evaluation_batch_size"])
        config["run"]["name"] = f"coco-native-ref13-{run.experiment}-s0"
        validate_config(config)
    samples = base.runs[0].config["evaluation"]["num_samples"]
    timing = estimate(receipt, updates, samples)
    manifest = export_study(StudyPlan("coco-native-ref13", base.runs), output)
    output = manifest.parent
    data = json.loads(manifest.read_text())
    for row in data["runs"]:
        row["evaluation_link"] = f"links/{row['index']:04d}-evaluation.txt"
    manifest.write_text(json.dumps(data, indent=2) + "\n")
    vanilla = copy.deepcopy(base.runs[0].config["evaluation"])
    vanilla.update(mode="vanilla_only", vanilla=True)
    save_config(vanilla, output / "vanilla.yaml")
    (output / "preflight-review.json").write_text(json.dumps(receipt, indent=2) + "\n")
    card = {"version": 1, "execution_authorized": False,
            "numerical_limitations": numerical_limitations(receipt), "accept_numerical_limitations": False,
            "status": "AWAITING_TIMING_REVIEW_AND_RENEWED_APPROVAL", "source_root": str(Path(source_root).resolve()),
            "approval_reference": None, "site_four_gpu_cap_verified": False,
            "selected_indices": selected_indices(receipt),
            "runs": len(selected_indices(receipt)), "panels": 4 * len(selected_indices(receipt)) + 1,
            "presentations_per_run": receipt.get("presentations_per_run", PRESENTATIONS),
            "updates": updates, "horizon": updates, "conditions": CONDITIONS,
            "training_batch_size": receipt["training_batch_size"],
            "gradient_accumulation": receipt["gradient_accumulation"],
            "evaluation_batch_size": receipt["evaluation_batch_size"], "images_per_panel": samples,
            "concurrency": "one GPU/job; site account cap four GPUs must be verified before future dispatch",
            "fixed_checkpoint": f"step_{updates:06d}.pt", "estimates": timing,
            "decision_scenarios": scenarios(receipt),
            "selection_policy": {"candidate": "final_only", "locked": False,
                "historical_policy": "periodic_validation",
                "note": "Prospective timing scenario; selection cadence and its cost require a decision before formal configuration freeze."},
            "training_execution": {"streamed_backward": True, "valid_only_kl": True},
            "graph": f"{len(selected_indices(receipt))} selected train array tasks; same-index aftercorr evaluations; one shared vanilla after enc_fn train",
            "submission": "Dry-run commands only; execution requires frozen renewed approval and verified site concurrency cap."}
    (output / "APPROVAL-CARD.json").write_text(json.dumps(card, indent=2) + "\n")
    bindings = {str(path.relative_to(output)): digest(path) for path in sorted(output.rglob("*")) if path.is_file()}
    (output / "PREPARED-FILES.json").write_text(json.dumps(bindings, indent=2) + "\n")
    return card


def verify_plan(directory, expected_manifest_hash=None, require_authorized=False):
    """Bind frozen source/config to the approval supplied to each job."""
    directory = Path(directory).resolve()
    manifest_path = directory / "PREPARED-FILES.json"
    if expected_manifest_hash is not None and digest(manifest_path) != expected_manifest_hash:
        raise ValueError("Prepared manifest differs from the submitted frozen hash")
    bindings = json.loads(manifest_path.read_text())
    required = {"manifest.json", "APPROVAL-CARD.json", "preflight-review.json", "vanilla.yaml"}
    if not required.issubset(bindings):
        raise ValueError("Prepared manifest omits a required input")
    for relative, expected in bindings.items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory) or digest(path) != expected:
            raise ValueError(f"Changed prepared input: {relative}")
    card = json.loads((directory / "APPROVAL-CARD.json").read_text())
    receipt = json.loads((directory / "preflight-review.json").read_text())
    updates = validate_measurements(receipt)
    manifest = json.loads((directory / "manifest.json").read_text())
    if tuple(row["experiment"] for row in manifest["runs"]) != SPLITS:
        raise ValueError("Incorrect thirteen-route manifest")
    if any(row["config"] not in bindings for row in manifest["runs"]):
        raise ValueError("Prepared manifest omits an experiment config")
    if (card["presentations_per_run"] != receipt.get("presentations_per_run", PRESENTATIONS)
            or card["updates"] != updates or card["horizon"] != updates
            or card["fixed_checkpoint"] != f"step_{updates:06d}.pt"):
        raise ValueError("Fixed exposure/checkpoint policy changed")
    indices = selected_indices(receipt)
    if (card.get("selected_indices", list(range(13))) != indices or card["runs"] != len(indices)
            or card["panels"] != 4 * len(indices) + 1
            or [row["split"] for row in card["estimates"]["rows"]] != [SPLITS[index] for index in indices]):
        raise ValueError("Approved selected scope or timing rows differ from the reviewed receipt")
    limits = numerical_limitations(receipt)
    if card.get("numerical_limitations", []) != limits:
        raise ValueError("Approval card must preserve all reviewed numerical limitations")
    if require_authorized:
        if limits and card.get("accept_numerical_limitations") is not True:
            raise ValueError("Formal approval must explicitly accept the recorded numerical limitations")
        if card.get("execution_authorized") is not True or not card.get("approval_reference"):
            raise ValueError("Formal COCO execution is not authorized")
        if card.get("selection_policy", {}).get("locked") is not True:
            raise ValueError("Selection cadence must be approved and locked")
        if card.get("site_four_gpu_cap_verified") is not True:
            raise ValueError("Account-wide four GPU cap must be verified before array execution")
        if not card["estimates"]["all_jobs_within_48h"]:
            raise ValueError("A proposed job exceeds the 48-hour limit")
    return card, receipt, manifest


def dry_run_commands(directory):
    """Return argv templates only. This function never invokes sbatch."""
    directory = Path(directory).resolve()
    card, _, _ = verify_plan(directory)
    timing = card["estimates"]
    if not timing["all_jobs_within_48h"]:
        raise ValueError("Revise the study before submission: estimated job exceeds 48 hours")
    source_root = Path(card["source_root"])
    wrapper = str(source_root / "scripts/slurm_coco_native.sh")
    frozen_hash = digest(directory / "PREPARED-FILES.json")
    common = ["sbatch", "--parsable", "--partition=h200q", "--account=shaoyulien", "--qos=h200q_1g",
              "--gres=gpu:1", "--cpus-per-task=16", "--mem=174080M", "--no-requeue",
              "--kill-on-invalid-dep=yes", f"--chdir={source_root}",
              f"--output={directory}/logs/%A_%a.out", f"--error={directory}/logs/%A_%a.err"]
    commands = []
    for phase, minutes, dependency in (
        ("train", max(row["train_job_minutes"] for row in timing["rows"]), None),
        ("evaluate", max(row["evaluation_job_minutes"] for row in timing["rows"]), "aftercorr:<TRAIN_ARRAY_JOB_ID>"),
        ("vanilla", math.ceil(timing["vanilla_seconds"] / 60), "afterok:<TRAIN_ARRAY_JOB_ID>_5"),
    ):
        argv = common + [f"--job-name=coco-{phase}", f"--time={minutes}"]
        if phase != "vanilla":
            indices = card.get("selected_indices", list(range(13)))
            array = "0-12" if indices == list(range(13)) else ",".join(map(str, indices))
            argv += [f"--array={array}%4"]
        if dependency:
            argv += [f"--dependency={dependency}"]
        argv += [wrapper, phase, str(directory), frozen_hash]
        commands.append({"phase": phase, "argv_template": argv})
    return {"execution_authorized": card["execution_authorized"], "submitted": False,
            "requires": ["renewed approval", "freeze approved card and all hashes", "create logs directory",
                         "verify account GPU cap four", "substitute returned train array job ID"],
            "concurrency_note": "Array throttles are per array, not aggregate. Verified account cap enforces aggregate four.",
            "commands": commands}


def verify_training_completion(directory, manifest, index, card):
    """COCO has no HellaSwag presentation-stream status; inspect actual fields."""
    import torch
    directory = Path(directory)
    run_text = (directory / manifest["runs"][index]["run_link"]).read_text().strip()
    if not run_text:
        raise ValueError("Completed training link is empty")
    run = Path(run_text)
    completion = json.loads((run / "completion.json").read_text())
    expected = {"reason": "max_steps", "step": card["updates"],
                "optimizer_updates": card["updates"], "presentations": card["presentations_per_run"]}
    for key, value in expected.items():
        if completion.get(key) != value:
            raise ValueError(f"Incomplete COCO training: {key}={completion.get(key)!r}, expected {value!r}")
    checkpoint = run / card["fixed_checkpoint"]
    if not checkpoint.is_file():
        raise ValueError("Final fixed-step checkpoint is missing")
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    if type(state.get("step")) is not int or state["step"] != card["updates"]:
        raise ValueError("Saved checkpoint step differs from fixed training budget")
    return {"run": str(run), "completion": completion,
            "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint)}


def execute(directory, phase, index, expected_manifest_hash):
    """Run one already-scheduled phase; do not allocate or submit resources."""
    directory = Path(directory).resolve()
    card, receipt, manifest = verify_plan(directory, expected_manifest_hash, require_authorized=True)
    source_root = Path(card["source_root"]).resolve()
    if source_root != Path(__file__).resolve().parents[1]:
        raise ValueError("Run this helper from the approved frozen source root")
    required_source = {"train.py", "evaluate.py", "jscc/training.py", "jscc/evaluation.py",
                       "jscc/study_task.py", "scripts/coco_native_study.py", "scripts/slurm_coco_native.sh", "pyproject.toml", "uv.lock"}
    required_source.update(str(path.relative_to(source_root)) for path in (source_root / "jscc").rglob("*.py"))
    if not required_source.issubset(receipt["source_files"]):
        raise ValueError("Executed-source bindings omit required entrypoints")
    for relative, expected in receipt["source_files"].items():
        path = (source_root / relative).resolve()
        if not path.is_relative_to(source_root) or digest(path) != expected:
            raise ValueError(f"Changed approved source: {relative}")
    if phase not in {"train", "evaluate", "vanilla"}:
        raise ValueError("Unknown execution phase")
    if type(index) is not int or index not in selected_indices(receipt) or (phase == "vanilla" and index != 5):
        raise ValueError("Invalid or unselected index; shared vanilla uses enc_fn index5 only")
    claims = directory / "execution"
    claims.mkdir(exist_ok=True)
    claim = claims / f"{phase}-{index:04d}.json"
    # Exclusive creation preserves failed attempts and forbids silent replay.
    with claim.open("x") as stream:
        json.dump({"status": "STARTED", "phase": phase, "index": index,
                   "prepared_manifest_sha256": expected_manifest_hash}, stream, indent=2)
    try:
        completed_training = None
        if phase != "train":
            completed_training = verify_training_completion(directory, manifest, index, card)
        if phase == "vanilla":
            assert completed_training is not None
            run = completed_training["run"]
            subprocess.run([sys.executable, str(source_root / "evaluate.py"), "--run", run,
                "--checkpoint", card["fixed_checkpoint"], "--expected-step", str(card["updates"]),
                "--config", str(directory / "vanilla.yaml"),
                "--output-path-file", str(directory / "links/vanilla.txt")], cwd=source_root, check=True)
        else:
            from jscc.study_task import run_task
            kwargs = ({"checkpoint": card["fixed_checkpoint"], "expected_step": card["updates"]}
                      if phase == "evaluate" else {})
            run_task(directory / "manifest.json", index, phase, **kwargs)
            if phase == "train":
                completed_training = verify_training_completion(directory, manifest, index, card)
    except Exception as error:
        claim.write_text(json.dumps({"status": "FAILED", "phase": phase, "index": index,
                                     "error": repr(error), "prepared_manifest_sha256": expected_manifest_hash}, indent=2) + "\n")
        raise
    claim.write_text(json.dumps({"status": "COMPLETED", "phase": phase, "index": index,
                                 "verified_training": completed_training,
                                 "prepared_manifest_sha256": expected_manifest_hash}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--measurements", required=True)
    prep.add_argument("--source-root", required=True)
    prep.add_argument("--evidence-root", required=True)
    prep.add_argument("--output", required=True)
    dry = sub.add_parser("dry-run")
    dry.add_argument("--plan", required=True)
    run = sub.add_parser("execute")
    run.add_argument("phase", choices=("train", "evaluate", "vanilla"))
    run.add_argument("--plan", required=True)
    run.add_argument("--index", type=int, required=True)
    run.add_argument("--prepared-sha256", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.measurements, args.source_root, args.evidence_root, args.output)
    elif args.command == "dry-run":
        result = dry_run_commands(args.plan)
    else:
        execute(args.plan, args.phase, args.index, args.prepared_sha256)
        return
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
