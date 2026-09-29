"""Isolated MAIN_WEIGHT_EARLY_SCREEN_V1 evaluator; no production defaults change.

Fixtures: root/fixtures/encoded-requests.jsonl, 256 documents x four candidates.
All scored forwards, including failed attempts and controls, reserve a shared
flock-protected 16,384-request ledger before execution. No optimizer is created.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.frozen_prompt_gap import digest, read, rows, sha, write
from scripts.experiments.evening_eval import compare, score
from scripts.evaluate_prompt_alignment import compact_row, verify_tokenization

LIMIT = 16384
TOLERANCE = {"atol": 5e-4, "rtol": 1e-5}


def require(value, message):
    if not value:
        raise ValueError(message)


def append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()


class GlobalBudget:
    """A single append-only budget shared by both GPU lanes (POSIX flock)."""
    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / "request-ledger.jsonl"
        self.local_used = 0

    def authorized(self, new_job=False):
        auth = read(self.root / "AUTHORIZATION.json")
        require(auth.get("execution_authorized") is True and
                auth.get("plan_id") == "MAIN_WEIGHT_EARLY_SCREEN_V1", "Missing suite authorization")
        key = "stop_new_gpu_at_unix" if new_job else (
            "gpu_work_stop_at_unix" if "gpu_work_stop_at_unix" in auth else "hard_stop_at_unix")
        require(time.time() < auth[key], f"Suite deadline reached: {key}")

    def reserve(self, count, cell, phase):
        require(count == 64, "Only exact64 candidate batches are authorized")
        self.authorized()
        with self.path.open("a+") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.seek(0)
            previous = [json.loads(line) for line in stream]
            used = sum(row["count"] for row in previous)
            require(used + count <= LIMIT, "Global candidate request budget exhausted")
            stream.seek(0, 2)
            stream.write(json.dumps({"cell": cell, "phase": phase, "count": count,
                                     "cumulative": used + count, "pid": os.getpid(),
                                     "reserved_unix": time.time()}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            self.local_used += count


def validate_fixture(root):
    path = root / "fixtures/encoded-requests.jsonl"
    encoded = rows(path)
    require(len(encoded) == 1024, "Require exactly 1024 candidate requests")
    ids = [encoded[i]["sample_id"] for i in range(0, 1024, 4)]
    require(len(set(ids)) == 256, "Require 256 unique documents")
    for i in range(0, 1024, 4):
        group = encoded[i:i + 4]
        require([r["candidate"] for r in group] == list(range(4)), "Candidate order changed")
        require(len({r["sample_id"] for r in group}) == 1, "Mixed document candidates")
        require(len({(r["gold"], r["source_id"], r["prompt_sha256"]) for r in group}) == 1,
                "Document metadata disagrees")
        for row in group:
            require(row.get("shot", 5) == 5 and row["gold"] in range(4), "Five-shot/gold mismatch")
            require(0 < len(row["context_token_ids"]) <= 2048 and
                    0 < len(row["continuation_token_ids"]) <= 2048, "Token cap violated")
            require(row["denominator"] > 0, "Invalid character denominator")
    return encoded, sha(path)


def payload_check(production, wrapper):
    checks = {key: production[key] == wrapper[key] for key in ("allocated", "valid")}
    return {"passed": all(checks.values()), "checks": checks,
            "production": {key: production[key] for key in checks},
            "wrapper": {key: wrapper[key] for key in checks}}


def run(args):
    import torch
    from lm_eval.models.huggingface import HFLM
    from jscc.evaluation_policy import scoring_kwargs
    from jscc.harness_payload import payload_harness_class
    from jscc.models.split_model import build_model
    from jscc.runtime import configure_training_determinism

    root = args.root
    budget = GlobalBudget(root)
    budget.authorized(new_job=True)
    encoded, fixture_hash = validate_fixture(root)
    require(sha(args.checkpoint) == args.checkpoint_sha256, "Checkpoint bytes differ from inventory")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]
    require(state["step"] == args.step, "Wrong internal checkpoint step")
    require(config["task"] == "hellaswag" and config["model"]["dtype"] == "bfloat16", "Wrong task/dtype")
    require(config["split"] == {"stack": args.route[:3], "where": "after_layer", "index": int(args.route.split("_l")[1])}, "Wrong route")
    require(config["channel"]["normalize_power"] is True, "Power must remain enabled")
    require(args.route == "dec_l20" and args.step in {500, 1000}, "Screen route/step mismatch")
    require(args.arm in {"W005", "W05", "W5"}, "Unknown screen arm")
    expected = {"kl": 1.0, "hidden": {"W005": .05, "W05": .5, "W5": 5.0}[args.arm], "memory": .05}
    require(config["training"]["loss_weights"] == expected, "Checkpoint loss weights mismatch")
    require(not args.vanilla or (args.arm == "W005" and args.step == 1000), "Single vanilla owner mismatch")
    output = root / "results" / f"{args.arm}_{args.step:06d}"
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write(output / "tolerances.json", {**TOLERANCE, "frozen_before_scores": True})
    write(output / "config.json", config)
    receipt = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": args.checkpoint_sha256,
               "checkpoint_step": args.step, "fixture_sha256": fixture_hash, "measurement": "MEASURED"}
    completed = []
    try:
        configure_training_determinism({"deterministic_algorithms": True})
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        tokenizer, model = build_model(copy.deepcopy(config))
        model.load_communication_state(state)
        state.clear()
        model.eval().requires_grad_(False)
        require(next(model.base.parameters()).device.type == "cuda", "GPU execution requires assigned CUDA device")
        require(model.codec.film is None and (model.memory_codec is None or model.memory_codec.film is None), "FiLM must be off")
        require(all(p.dtype == torch.float32 for p in model.codec.parameters()), "Codec parameters must remain FP32")
        require(getattr(model.base.config.decoder, "_attn_implementation", None) == "sdpa", "Expected formal SDPA backend")
        verify_tokenization(tokenizer, encoded)
        cls = payload_harness_class(HFLM)
        adapter = cls(communication_model=model, pretrained=model.base, tokenizer=tokenizer,
                      backend="seq2seq", batch_size=64, max_length=2048,
                      **scoring_kwargs({"scoring_policy": "fp32-v1"}, model))
        native_dir = root / "fixtures" / "native_batches"
        native_dir.mkdir(exist_ok=True)

        def batch_score(batch, cell, phase, policies=None, seed=None):
            values, evidence, native = score(adapter, model, batch, budget, cell, phase, policies, seed)
            append(output / "batches.jsonl", evidence)
            # Unique deterministic path; equal fixtures must agree across routes.
            target = native_dir / f"{digest(native['request_keys'])}.json"
            with target.open("a+") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                stream.seek(0)
                old = stream.read()
                require(not old or json.loads(old) == native, "Native preparation/packing changed")
                if not old:
                    stream.write(json.dumps(native, allow_nan=False) + "\n")
            return values, evidence

        def panel(cell, experiment, policies=None, seed=None, phase="core"):
            scores, per_request, events = [], [], []
            for start in range(0, 1024, 64):
                values, evidence = batch_score(encoded[start:start + 64], cell, phase, policies, seed)
                scores.extend(values)
                events.append(evidence)
                for i in range(64):
                    payload = {e["stream"]: {"allocated": e["allocated_per_request"][i],
                                             "valid": e["valid_per_request"][i], "policy": e["policy"],
                                             "noise": e.get("noise", [None] * 64)[i]}
                               for e in evidence["events"]}
                    for stream in ("hidden", "memory"):
                        if stream not in payload:
                            codec = model.codec if stream == "hidden" else model.memory_codec
                            length_key = "continuation_token_ids" if stream == "hidden" and model.split["stack"] == "dec" else "context_token_ids"
                            count = (len(encoded[start + i][length_key]) * codec.config["bottleneck_dim"]
                                     if codec is not None and evidence["allocated"][stream] else 0)
                            payload[stream] = {"allocated": evidence["allocated"][stream] // 64,
                                               "valid": count, "policy": "production" if count else "bypassed_or_absent",
                                               "noise": None}
                    per_request.append(payload)
            result = []
            for start in range(0, 1024, 4):
                item = compact_row(encoded[start:start + 4], scores[start:start + 4], args.route, 5, receipt, fixture_hash)
                if cell == "vanilla":
                    item.update(route="vanilla", checkpoint=None, checkpoint_sha256=None,
                                checkpoint_step=None, base_model=config["model"])
                item.update(experiment=experiment, arm=args.arm, case=cell, noise_seed=seed,
                            main_noise_enabled=bool(policies and policies["hidden"] == "A"),
                            memory_noise_enabled=bool(policies and policies["memory"] == "A"),
                            main_snr_db=-6.0 if policies and policies["hidden"] == "A" else None,
                            memory_snr_db=-6.0 if policies and policies["memory"] == "A" else None,
                            override_semantics="O bypasses codec/power/channel at receiver boundary; C retains codec/power with noise disabled; A retains codec/power with AWGN",
                            stream_policy=policies, privileged_oracle=bool(policies and "O" in policies.values()),
                            payload_coordinates=per_request[start:start + 4])
                result.append(item)
                append(output / f"{cell}.jsonl", item)
            completed.append(cell)
            write(output / "progress.json", {"completed": completed, "requests": budget.local_used})
            print(f"{args.route} step{args.step} {cell} COMPLETE requests={budget.local_used}", flush=True)
            return scores, events, result

        reference, reference_evidence = batch_score(encoded[:64], "production", "reference_self_replay")
        replay, _ = batch_score(encoded[:64], "production", "reference_self_replay")
        control = compare(reference, replay)
        write(output / "reference-self-replay.json", control)
        require(control["passed"], "Production reference self-replay failed")
        values, events, _ = panel("no_noise", "MAIN_WEIGHT_EARLY_SCREEN_V1", {"hidden": "C", "memory": "C"})
        control = compare(reference, values[:64])
        write(output / "CC-control.json", control)
        require(control["passed"], "No-intervention wrapper differs from production")
        payload_control = payload_check(reference_evidence, events[0])
        write(output / "payload-control.json", payload_control)
        require(payload_control["passed"], "No-intervention payload differs from production")
        if args.step == 1000:
            noisy, _, _ = panel("awgn_-6_20260921", "MAIN_WEIGHT_EARLY_SCREEN_V1", {"hidden": "A", "memory": "A"}, 20260921)
            replay, _ = batch_score(encoded[:64], "awgn_-6_20260921", "noise_replay", {"hidden": "A", "memory": "A"}, 20260921)
            check = compare(noisy[:64], replay)
            write(output / "noise-replay.json", check)
            require(check["passed"], "Noisy replay failed")
        if args.vanilla:
            panel("vanilla", "MAIN_WEIGHT_EARLY_SCREEN_V1")
        require(sha(args.checkpoint) == args.checkpoint_sha256, "Checkpoint mutated during evaluation")
        require(sha(root / "fixtures/encoded-requests.jsonl") == fixture_hash, "Fixture mutated during evaluation")
        write(output / "completion.json", {**receipt, "status": "COMPLETED", "completed": completed,
              "candidate_requests": budget.local_used, "optimizer_updates": 0,
              "oracle_control": None, "arm": args.arm,
              "elapsed_seconds_including_model_load": time.monotonic() - started,
              "runner_sha256": sha(__file__)})
    except BaseException as exc:
        write(output / "failure.json", {"type": type(exc).__name__, "message": str(exc),
              "candidate_requests": budget.local_used, "completed": completed,
              "elapsed_seconds": time.monotonic() - started})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--route", choices=("dec_l0", "dec_l20", "enc_l9"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--step", type=int, choices=(500, 1000, 5000, 10000), required=True)
    parser.add_argument("--arm", choices=("W005", "W05", "W5"), required=True)
    parser.add_argument("--vanilla", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
