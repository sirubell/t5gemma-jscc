"""Isolated EVENING_SMALL_SUITE_V1 E1/E2/E4 runner; no production defaults change.

Fixtures: root/fixtures/encoded-requests.jsonl, 256 documents x four candidates.
All scored forwards, including failed attempts and controls, reserve a shared
flock-protected 49,152-request ledger before execution. No optimizer is created.
"""
import argparse
from contextlib import contextmanager
import copy
import fcntl
import json
import os
from pathlib import Path
import sys
import time
from types import MethodType
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.frozen_prompt_gap import digest, read, rows, sha, tensor_batch, write
from scripts.evaluate_prompt_alignment import compact_row, verify_tokenization

LIMIT = 49152
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
                auth.get("plan_id") == "EVENING_SMALL_SUITE_V1", "Missing suite authorization")
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


def replay_epsilon(z, mask, request_keys, seed, stream):
    """Valid coordinates are independent of padding, companions and other streams.

    Separate deterministic draws for valid/padding coordinates preserve physical
    AWGN on all allocated coordinates without allowing padding to advance payload
    RNG. Per-request stream keys are reconstructable from saved seed and IDs.
    """
    import torch
    from jscc.presentation import tensor_digest
    require(mask.shape == z.shape[:2], "Expected canonical [batch,tokens] mask")
    require(len(request_keys) == len(z), "Request/noise identity mismatch")
    epsilon = torch.empty_like(z)
    evidence = []
    for i, request in enumerate(request_keys):
        valid = mask[i].bool()
        for label, positions in (("valid", valid), ("padding", ~valid)):
            key = ["evening-valid-coordinates-v1", seed, stream, request, label]
            derived = int(digest(key)[:16], 16) % (2**63 - 1)
            generator = torch.Generator(device=z.device).manual_seed(derived)
            noise = torch.randn((int(positions.sum()), z.shape[-1]), device=z.device,
                                dtype=z.dtype, generator=generator)
            epsilon[i, positions] = noise
        evidence.append({"request": request, "stream": stream, "seed": seed,
                         "valid_epsilon_sha256": tensor_digest(epsilon[i, valid]),
                         "dtype": str(z.dtype), "device_type": z.device.type,
                         "valid_shape": [int(valid.sum()), z.shape[-1]],
                         "allocated_shape": list(z[i].shape)})
    return epsilon, evidence


class Intervention:
    """Intercept only the production transmission seam; preserve all model hooks."""
    def __init__(self, model, policies, seed, request_keys):
        self.model, self.policies, self.seed, self.request_keys = model, policies, seed, request_keys
        self.events = []
        self.active = None

    @contextmanager
    def install(self):
        from jscc.models.channel import AWGNChannel, _token_mask
        from jscc.presentation import tensor_digest
        model = self.model
        require(type(model.channel) is AWGNChannel, "Only production AWGN is supported")
        require(model.codec.film is None and
                (model.memory_codec is None or model.memory_codec.film is None), "FiLM must remain off")
        original_transmit, original_channel = model._transmit, model.channel.forward

        def channel(_channel, z, snr):
            event = self.active
            if event is None:
                raise ValueError("Channel outside stream scope")
            event["channel_calls"] += 1
            if event["noise_enabled"]:
                epsilon, proof = replay_epsilon(z, event.pop("_mask"), self.request_keys, self.seed, event["stream"])
                event["noise"] = proof
                return z + epsilon * (10.0 ** (6.0 / 20.0))
            event.pop("_mask", None)
            return original_channel(z, None)

        def transmit(_model, hidden, codec, stream, valid_mask=None, *, token_wise=False,
                     mask_representation=None):
            policy = self.policies[stream]
            require(policy in {"C", "O", "A"}, "Unknown stream intervention")
            mask = _token_mask(hidden, valid_mask, representation=mask_representation)
            if mask is None:
                raise ValueError("Missing stream payload mask")
            width = codec.config["bottleneck_dim"]
            event = {"stream": stream, "policy": policy, "codec_calls": 0,
                     "channel_calls": 0, "oracle_calls": int(policy == "O"),
                     "noise_enabled": policy == "A", "snr_db": -6.0 if policy == "A" else None,
                     "input_sha256": tensor_digest(hidden),
                     "allocated_per_request": [0 if policy == "O" else hidden.shape[1] * width] * len(hidden),
                     "valid_per_request": [0 if policy == "O" else int(m.sum()) * width for m in mask],
                     "_mask": mask}
            self.events.append(event)
            if policy == "O":
                event.pop("_mask")
                return hidden
            event["codec_calls"] += 1
            self.active = event
            try:
                return original_transmit(hidden, codec, stream, valid_mask, token_wise=token_wise,
                                         mask_representation=mask_representation)
            finally:
                self.active = None

        model._transmit = MethodType(transmit, model)
        model.channel.forward = MethodType(channel, model.channel)
        try:
            yield self
        finally:
            model._transmit = original_transmit
            model.channel.forward = original_channel


def compare(reference, candidate):
    import torch
    a, b = torch.tensor(reference, dtype=torch.float64), torch.tensor(candidate, dtype=torch.float64)
    require(a.shape == b.shape, "Replay score shape mismatch")
    return {"passed": bool(torch.allclose(a, b, atol=TOLERANCE["atol"], rtol=TOLERANCE["rtol"])),
            "maximum_absolute_difference": float((a - b).abs().max()), **TOLERANCE}


def score(adapter, model, batch, budget, cell, phase, policies=None, seed=None):
    import torch
    from jscc.harness_payload import ContinuationLengths
    from jscc.presentation import tensor_digest
    device = next(model.base.parameters()).device
    tensors = tensor_batch(batch, device)
    adapter._payload_lengths = ContinuationLengths([
        ((r["context"], r["continuation"]), r["context_token_ids"], r["continuation_token_ids"])
        for r in batch], 2048)
    native = model.base.prepare_decoder_input_ids_from_labels(tensors["labels"])
    native_observed = []

    def capture(_module, args, kwargs):
        actual = kwargs.get("input_ids", args[0] if args else None)
        if actual is None or not torch.equal(actual, native):
            raise ValueError("Native decoder input mismatch")
        cache = kwargs.get("past_key_values")
        require(cache is None or cache.get_seq_length() == 0, "Reused decoder cache")
        native_observed.append({"input_ids": actual.tolist(),
                                "attention_mask": None if kwargs.get("attention_mask") is None
                                else kwargs["attention_mask"].tolist(),
                                "position_ids": None if kwargs.get("position_ids") is None
                                else kwargs["position_ids"].tolist(),
                                "cache_initial_length": 0})

    hook = model.base.get_decoder().register_forward_pre_hook(capture, with_kwargs=True)
    intervention = Intervention(model, policies, seed, [[r["sample_id"], r["candidate"]] for r in batch]) if policies else None
    from contextlib import nullcontext
    started = time.monotonic()
    try:
        budget.reserve(64, cell, phase)
        with torch.inference_mode(), model.transmission(None, bypass=cell == "vanilla"), (
            intervention.install() if intervention else nullcontext()
        ):
            logits = adapter._model_call(tensors["input_ids"], attn_mask=tensors["attention_mask"], labels=tensors["labels"])
            require(logits.dtype == next(model.base.parameters()).dtype, "Forward dtype mismatch")
            logp = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
            # Preserve production's sum over only the actual continuation.
            values = torch.stack([
                logp[i, :len(row["continuation_token_ids"])]
                .gather(-1, tensors["labels"][i, :len(row["continuation_token_ids"]), None])
                .sum(dtype=torch.float32) for i, row in enumerate(batch)
            ])
            require(bool(torch.isfinite(values).all()), "Nonfinite candidate scores")
            values = values.tolist()
        require(len(native_observed) == 1, "Expected one decoder call per batch")
        if intervention:
            expected_streams = {"hidden", "memory"} if model.memory_codec is not None else {"hidden"}
            require(len(intervention.events) == len(expected_streams)
                    and {e["stream"] for e in intervention.events} == expected_streams,
                    "Actual transmission branch counts changed")
            for event in intervention.events:
                oracle = event["policy"] == "O"
                require(event["codec_calls"] == event["channel_calls"] == int(not oracle)
                        and event["oracle_calls"] == int(oracle), "Unexpected codec/channel/oracle calls")
                require(sum(event["allocated_per_request"]) == model.channel_uses_allocated[event["stream"]]
                        and sum(event["valid_per_request"]) == model.valid_payload_counts[event["stream"]],
                        "Per-request payload differs from measured production counters")
        evidence = {"cell": cell, "phase": phase, "request_keys": [[r["sample_id"], r["candidate"]] for r in batch],
                    "scores": values, "elapsed_seconds": time.monotonic() - started,
                    "allocated": dict(model.channel_uses_allocated), "valid": dict(model.valid_payload_counts),
                    "events": intervention.events if intervention else [],
                    "native_sha256": tensor_digest(native)}
        native_record: dict[str, Any] = {k: v.tolist() for k, v in tensors.items()}
        native_record.update(native_decoder=native_observed[0], request_keys=evidence["request_keys"],
                             rule="HFLM seq2seq labels; native BOS2 preparation; right PAD0; fresh cache each forward")
        return values, evidence, native_record
    finally:
        hook.remove()
        adapter._payload_lengths = None


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
    require(args.mode != "decoder-final" or (args.step == 10000 and args.route.startswith("dec")), "Wrong E1/E2 route/step")
    require(not args.vanilla or (args.route == "dec_l0" and args.mode == "decoder-final"), "Shared vanilla only on dec_l0")
    require(args.mode != "curve" or args.route in {"enc_l9", "dec_l20"}, "Wrong E4 route")
    family = config.get("experiment", {}).get("codec_family")
    require(args.mode != "pilot" or (args.route == "enc_l9" and args.step == 500 and
            family in {"affine_core", "shallow_nonlinear", "current_residual"}), "Wrong pilot family/step")
    suffix = f"_{family}" if args.mode == "pilot" else ""
    output = root / "results" / f"{args.route}_{args.step:06d}{suffix}"
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
        if args.mode == "pilot":
            from scripts.experiments.evening_pilot import experimental_build_model
            tokenizer, model = experimental_build_model(copy.deepcopy(config))
        else:
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
                item.update(experiment=experiment, case=cell, noise_seed=seed,
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

        reference, _ = batch_score(encoded[:64], "production", "reference_self_replay")
        replay, _ = batch_score(encoded[:64], "production", "reference_self_replay")
        control = compare(reference, replay)
        write(output / "reference-self-replay.json", control)
        require(control["passed"], "Production reference self-replay failed")
        if args.mode in {"curve", "pilot"}:
            values, _, _ = panel("no_noise", "E5" if args.mode == "pilot" else "E4",
                                 {"hidden": "C", "memory": "C"})
            control = compare(reference, values[:64])
            write(output / "CC-control.json", control)
            require(control["passed"], "Wrapper no-noise differs from production")
            if args.mode == "pilot":
                panel("awgn_-6_20260921", "E5", {"hidden": "A", "memory": "C"}, 20260921)
        else:
            if args.vanilla:
                vanilla, _, vanilla_rows = panel("vanilla", "E1")
                write(root / "results/vanilla.json", {"fixture_sha256": fixture_hash,
                      "model_config": config["model"], "scores": vanilla, "rows": vanilla_rows})
            production, _, _ = panel("production", "CONTROL", phase="production_control")
            cc, cc_events, _ = panel("CC", "E1", {"hidden": "C", "memory": "C"})
            cc_control = compare(production, cc)
            write(output / "CC-control.json", cc_control)
            require(cc_control["passed"], "Wrapper CC differs from production")
            for case in ("OC", "CO", "OO"):
                values, events, _ = panel(case, "E1", {"hidden": case[0], "memory": case[1]})
                # Main input must be untouched by receiver-side memory changes.
                for base_batch, changed_batch in zip(cc_events, events, strict=True):
                    base_hash = next(e["input_sha256"] for e in base_batch["events"] if e["stream"] == "hidden")
                    changed_hash = next(e["input_sha256"] for e in changed_batch["events"] if e["stream"] == "hidden")
                    require(base_hash == changed_hash, "Receiver intervention changed transmitter main input")
                if case == "OO":
                    write(output / "OO-scores.json", {"fixture_sha256": fixture_hash, "model_config": config["model"], "scores": values})
                    shared = root / "results/vanilla.json"
                    if shared.exists():
                        baseline = read(shared)
                        require(baseline["fixture_sha256"] == fixture_hash and baseline["model_config"] == config["model"], "Vanilla identity mismatch")
                        oo_control = compare(baseline["scores"], values)
                        write(output / "OO-control.json", oo_control)
                        # Keep E2 independently executable; failed oracle is explicitly invalid.
                    else:
                        write(output / "OO-control.json", {"passed": None, "reason": "Shared vanilla pending; E1 mechanism interpretation blocked"})
            for seed in (20260921, 20260922):
                proofs = {}
                for case in ("AN", "NA", "AA"):
                    policies = {"hidden": "A" if case[0] == "A" else "C",
                                "memory": "A" if case[1] == "A" else "C"}
                    values, events, _ = panel(f"{case}_{seed}", "E2", policies, seed)
                    proofs[case] = {e["stream"]: [n["valid_epsilon_sha256"] for b in events for e2 in b["events"]
                                                if e2["stream"] == e["stream"] for n in e2.get("noise", [])]
                                    for e in events[0]["events"] if e["noise_enabled"]}
                    if case == "AA":
                        replay, _ = batch_score(encoded[:64], f"{case}_{seed}", "noise_replay", policies, seed)
                        check = compare(values[:64], replay)
                        write(output / f"noise-replay-{seed}.json", check)
                        require(check["passed"], "Noise self-replay failed")
                require(proofs["AN"]["hidden"] == proofs["AA"]["hidden"] and
                        proofs["NA"]["memory"] == proofs["AA"]["memory"], "Noise stream pairing failed")
                write(output / f"noise-pairing-{seed}.json", {"passed": True, "valid_epsilon_hashes": proofs})
        require(sha(args.checkpoint) == args.checkpoint_sha256, "Checkpoint mutated during evaluation")
        require(sha(root / "fixtures/encoded-requests.jsonl") == fixture_hash, "Fixture mutated during evaluation")
        write(output / "completion.json", {**receipt, "status": "COMPLETED", "completed": completed,
              "candidate_requests": budget.local_used, "optimizer_updates": 0,
              "oracle_control": read(output / "OO-control.json") if args.mode == "decoder-final" else None,
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
    parser.add_argument("--mode", choices=("decoder-final", "curve", "pilot"), required=True)
    parser.add_argument("--vanilla", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
