"""Explicit HellaSwag enc_l9 A/B/C/D preparation and bounded phase execution.

No command schedules jobs. ``check`` and ``plan`` read recipes only. Capture,
local, and functional are separate opt-in commands with exact update budgets.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jscc.config import load_config


def check_recipe(config: dict) -> dict:
    settings = config["training"]
    if config["task"] != "hellaswag" or config["split"] != {"stack": "enc", "where": "after_layer", "index": 9}:
        raise ValueError("two-stage pilot requires HellaSwag enc_l9")
    stream = settings.get("presentation_stream")
    if not stream or stream.get("policy") != "epoch-permutations-v1":
        raise ValueError("pilot requires a fixed presentation stream")
    updates = settings["max_steps"]
    presentations = updates * settings["batch_size"] * settings["gradient_accumulation"]
    if stream["total_presentations"] != presentations:
        raise ValueError("declared presentation count differs from update budget")
    if config["channel"]["type"] != "awgn" or config["codec"].get("snr_film", False):
        raise ValueError("pilot requires AWGN and FiLM off")
    if settings.get("paired_randomness", {}).get("policy") != "step-microbatch-stream-v1":
        raise ValueError("pilot requires explicit paired randomness")
    if settings.get("patience", 0):
        raise ValueError("fixed pilot must complete the declared terminal step")
    return {"updates": updates, "presentations": presentations,
            "batch_size": settings["batch_size"],
            "accumulation": settings["gradient_accumulation"],
            "schedule_steps": settings.get("schedule_steps", updates),
            "selection_steps": settings.get("selection_steps"),
            "presentation_seed": stream["seed"],
            "presentation_start": stream.get("start_presentation", 0),
            "noise_seed": settings["paired_randomness"]["seed"]}


def check_plan(paths: dict[str, str]) -> dict:
    from jscc.local_reconstruction import parent_recipe_sha256
    configs = {arm: load_config(path) for arm, path in paths.items()}
    budgets = {arm: check_recipe(config) for arm, config in configs.items()}
    a, b, c, d = (configs[key] for key in "ABCD")
    if budgets["A"]["updates"] != budgets["B"]["updates"] + budgets["C"]["updates"]:
        raise ValueError("A updates must equal B local plus C functional updates")
    if budgets["C"] != budgets["D"]:
        raise ValueError("C/D phase-two budgets, batches, schedule or presentation/noise seeds differ")
    if budgets["A"]["presentation_start"] != 0 or budgets["B"]["presentation_start"] != 0:
        raise ValueError("A/B presentation streams must start at zero")
    if budgets["C"]["presentation_start"] != budgets["B"]["presentations"]:
        raise ValueError("C/D must begin after B's exact presentation count")
    if budgets["A"]["presentation_seed"] != budgets["B"]["presentation_seed"] or budgets["A"]["presentation_seed"] != budgets["C"]["presentation_seed"]:
        raise ValueError("A/B/C/D must use the same presentation permutation seed")
    c_training, d_training = copy.deepcopy(c["training"]), copy.deepcopy(d["training"])
    c_training.pop("phase_transfer", None)
    d_training.pop("phase_transfer", None)
    if c_training != d_training:
        raise ValueError("C/D functional training, LR, warmup, objective or selection policies differ")
    if any(a["evaluation"] != other["evaluation"] for other in (b, c, d)):
        raise ValueError("A/B/C/D must use identical task evaluation settings")
    for other in (b, c, d):
        for key in ("task", "protocol", "model", "split", "codec", "channel", "data", "seed"):
            if key == "codec":
                left, right = copy.deepcopy(a[key]), copy.deepcopy(other[key])
                left.pop("input_dim", None)
                right.pop("input_dim", None)
            else:
                left, right = a[key], other[key]
            if left != right:
                raise ValueError(f"A/B/C/D {key} recipes differ")
    declared = c["training"].get("phase_transfer")
    if not isinstance(declared, dict) or declared.get("parent_recipe_sha256") != parent_recipe_sha256(b) or declared.get("parent_step") != b["training"]["max_steps"]:
        raise ValueError("C declaration does not bind the planned B recipe and terminal step")
    return {"arms": budgets, "comparison": "A and C match nominal updates; D matches C phase two",
            "cost_accounting": "C inherits B capture and local cost once"}


def prepare_plan(base_path: str, output_dir: str | Path, *, total_updates: int,
                 local_updates: int, functional_updates: int,
                 functional_lr: float | None = None) -> dict:
    """Export a coherent four-arm plan; all budget numbers are caller supplied."""
    if min(total_updates, local_updates, functional_updates) < 1 or total_updates != local_updates + functional_updates:
        raise ValueError("require positive explicit U, L and F with U = L + F")
    base = load_config(base_path)
    from jscc.local_reconstruction import parent_recipe_sha256
    check_recipe(base)
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("plan directory must be fresh and empty")
    output.mkdir(parents=True, exist_ok=True)
    batch = base["training"]["batch_size"] * base["training"]["gradient_accumulation"]
    arm_paths = {}
    b_digest = None
    for arm, updates, start in (("A", total_updates, 0), ("B", local_updates, 0),
                                ("C", functional_updates, local_updates * batch),
                                ("D", functional_updates, local_updates * batch)):
        config = copy.deepcopy(base)
        settings = config["training"]
        settings["max_steps"] = settings["schedule_steps"] = updates
        settings["eval_every"] = updates
        settings["min_steps"] = updates
        settings["patience"] = None
        settings["save_steps"] = [updates]
        settings["selection_steps"] = [updates]
        settings["feature_summary"] = None
        settings["presentation_stream"]["total_presentations"] = updates * batch
        settings["presentation_stream"]["start_presentation"] = start
        settings["paired_randomness"]["audit_steps"] = [1, updates]
        if arm in ("C", "D") and functional_lr is not None:
            if not math.isfinite(functional_lr) or functional_lr <= 0:
                raise ValueError("functional LR must be finite and positive")
            settings["lr"] = functional_lr
        if arm == "B":
            b_digest = parent_recipe_sha256(config)
        if arm == "C":
            assert b_digest is not None
            settings["phase_transfer"] = {"parent_recipe_sha256": b_digest,
                                          "parent_step": local_updates}
        config["run"]["name"] = f"hellaswag-enc_l9-{arm.lower()}"
        config["run"]["output_dir"] = str(output / "runs")
        path = output / f"{arm.lower()}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        arm_paths[arm] = str(path)
    result = check_plan(arm_paths)
    result["recipes"] = arm_paths
    result["base_recipe"] = str(Path(base_path).resolve())
    result["execution"] = "none; use explicit capture/local/functional commands with --max-seconds"
    (output / "plan.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="read one recipe without loading a model")
    check.add_argument("--config", required=True)
    plan = commands.add_parser("plan", help="read all four recipes without loading a model")
    for arm in "ABCD":
        plan.add_argument(f"--{arm.lower()}", required=True)
    prepare = commands.add_parser("prepare", help="export a fresh A/B/C/D plan without loading a model")
    prepare.add_argument("--base", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--u", type=int, required=True, help="A total functional updates")
    prepare.add_argument("--l", type=int, required=True, help="B local updates")
    prepare.add_argument("--f", type=int, required=True, help="C/D functional updates")
    prepare.add_argument("--functional-lr", type=float, help="shared fresh phase-two LR; defaults to base recipe")
    capture = commands.add_parser("capture", help="capture full encoder sequences for B")
    capture.add_argument("--config", required=True)
    capture.add_argument("--cache-dir", required=True)
    capture.add_argument("--max-seconds", type=float, required=True, help="soft wall cap including setup")
    capture.add_argument("--max-cache-bytes", type=int, required=True,
                         help="maximum accepted serialized activation-shard bytes")
    local = commands.add_parser("local", help="execute B local reconstruction")
    local.add_argument("--config", required=True)
    local.add_argument("--cache-dir", required=True)
    local.add_argument("--max-seconds", type=float, required=True, help="soft wall cap including setup")
    functional = commands.add_parser("functional", help="execute one bounded A, C or D functional phase")
    functional.add_argument("--config", required=True)
    functional.add_argument("--arm", choices=("A", "C", "D"), required=True)
    functional.add_argument("--parent", help="terminal B checkpoint, required for C")
    functional.add_argument("--max-seconds", type=float, required=True, help="soft wall cap including setup")
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps(check_plan({arm: getattr(args, arm.lower()) for arm in "ABCD"}), indent=2))
        return
    if args.command == "prepare":
        print(json.dumps(prepare_plan(args.base, args.output_dir, total_updates=args.u,
                                      local_updates=args.l, functional_updates=args.f,
                                      functional_lr=args.functional_lr), indent=2))
        return
    if args.command != "check" and (not math.isfinite(args.max_seconds) or args.max_seconds <= 0):
        raise ValueError("--max-seconds must be finite and positive")
    execution_started = time.monotonic() if args.command != "check" else None
    deadline = execution_started + args.max_seconds if execution_started is not None else None
    config = load_config(args.config)
    budget = check_recipe(config)
    if args.command == "check":
        print(json.dumps(budget, indent=2))
        return
    if args.command == "capture":
        if args.max_cache_bytes < 1:
            raise ValueError("--max-cache-bytes must be positive")
        from jscc.activation_replay import write_cache
        from jscc.data import load_data
        from jscc.models.split_model import build_model
        from jscc.runtime import configure_training_determinism, seed_everything
        configure_training_determinism(config["training"])
        seed_everything(config["seed"])
        tokenizer, model = build_model(config)
        input_dim = int(config["codec"]["input_dim"])
        max_tokens = max(config["data"]["max_length"],
                         config["data"].get("prompt_policy", {}).get("source_max_length", 0))
        dtype_bytes = next(model.base.parameters()).element_size()
        upper_activation_bytes = budget["presentations"] * max_tokens * input_dim * dtype_bytes
        print(json.dumps({"activation_upper_bound_bytes": upper_activation_bytes,
                          "max_cache_bytes": args.max_cache_bytes,
                          "note": "upper bound excludes masks, tokens and serialization overhead"}), flush=True)
        data = load_data(config, tokenizer)
        manifest = write_cache(args.cache_dir, config, data.ids, data.train, model,
                               prompt_evidence=getattr(data, "prompt_evidence", None), deadline=deadline,
                               execution_started=execution_started,
                               max_cache_bytes=args.max_cache_bytes)
        print(json.dumps({"cache": str(Path(args.cache_dir).resolve()),
                          "identity_sha256": manifest["identity_sha256"], **budget}, indent=2))
    elif args.command == "local":
        from jscc.local_reconstruction import run_local
        print(run_local(config, args.cache_dir, deadline=deadline))
    else:
        if (args.arm == "C") != bool(args.parent):
            raise ValueError("C requires --parent; A and D must start fresh")
        from jscc.training import train
        print(train(config, initial_checkpoint=args.parent, deadline=deadline))


if __name__ == "__main__":
    main()
