"""Bounded, artifact-bound replay of previously visited COCO generation states."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from . import coco_diagnostic as endpoint
from .data.coco import caption_prompt
from .runtime import autocast_for

PROTOCOL = "coco-visited-trajectory-v1"
MAX_RUN_SECONDS = 1200
MAX_ARTIFACT_BYTES = 2 * 1024**3
EVIDENCE_FILES = ("identity.json", "status.json", "items.jsonl")


def selected_states(tokens: list[int], eos: int, newline: list[int], cap: int = 64) -> dict[str, Any]:
    """Prefix lengths index next-token decisions; BOS is retained, EOS never is."""
    if len(tokens) < 2 or len(tokens) > cap + 1 or not newline:
        raise ValueError("Invalid trajectory length or empty newline encoding")
    generated = tokens[1:]
    if eos in generated[:-1] or (generated[-1] != eos and len(generated) != cap):
        raise ValueError("Trajectory continues after EOS or stops before its cap")
    events: dict[str, int] = {"terminal_decision": len(tokens) - 1}
    first = next((i for i in range(1, len(tokens) - len(newline) + 1)
                  if tokens[i:i + len(newline)] == newline), None)
    after = None if first is None else first + len(newline)
    if first is not None:
        events["before_first_newline"] = first
        if after is not None and after < len(tokens):
            events["after_first_newline"] = after
    return {"events": events, "prefix_lengths": sorted(set(events.values())),
            "newline_present": first is not None,
            "after_newline_state": "missing_no_newline" if first is None else
                "unvisited_after_final_token" if after == len(tokens) else "visited",
            "truncated": generated[-1] != eos}


def read_evidence(directory: Path) -> dict[str, Any]:
    status = endpoint.read_json(directory / "status.json")
    identity = endpoint.read_json(directory / "identity.json")
    if status.get("status") != "complete" or status.get("completed") != 1024:
        raise ValueError("Prior diagnostic is not the complete 1024-request screen")
    if endpoint.sha256(directory / "identity.json") != status["manifest_sha256"]:
        raise ValueError("Prior identity bytes differ from completion receipt")
    rows = [json.loads(line) for line in (directory / "items.jsonl").read_text().splitlines()]
    endpoints = [row for row in rows if row.get("status") == "complete" and row.get("request_kind") == "endpoint"]
    token_contracts = {(row["eos_token_id"], tuple(row["newline_token_ids"])) for row in endpoints}
    if len(endpoints) != 768 or len(token_contracts) != 1:
        raise ValueError("Missing or inconsistent prior tokenizer contracts")
    eos, newline = token_contracts.pop()
    trajectories = [row for row in rows if row.get("status") == "complete" and
                    row.get("request_kind") == "generation" and row.get("query_control") == "matched"]
    keys = {(row["image_id"], row["model_mode"]) for row in trajectories}
    ids = list(dict.fromkeys(row["image_id"] for row in trajectories))
    if len(trajectories) != 128 or len(ids) != 32 or keys != {(i, m) for i in ids for m in endpoint.MODES}:
        raise ValueError("Require exactly 32 images by four matched modes")
    for row in trajectories:
        if row["checkpoint_sha256"] != identity["modes"][row["model_mode"]]["checkpoint_sha256"]:
            raise ValueError("Prior checkpoint identity mismatch")
        row["selection"] = selected_states(row["generated_token_ids"], eos, list(newline))
    return {"trajectories": trajectories, "eos": eos, "newline": list(newline), "identity": identity,
            "evidence_sha256": {name: endpoint.sha256(directory / name) for name in EVIDENCE_FILES}}


def workload(evidence: dict[str, Any]) -> dict[str, int]:
    rows = evidence["trajectories"]
    return {"trajectories": len(rows),
            "selected_states": sum(len(r["selection"]["prefix_lengths"]) for r in rows),
            "logical_requests": sum((1 + len(r["selection"]["prefix_lengths"])) *
                                    (1 if r["model_mode"] == "vanilla" else 2) for r in rows),
            "decoder_calls": sum((len(r["generated_token_ids"]) - 1 + len(r["selection"]["prefix_lengths"])) *
                                 (1 if r["model_mode"] == "vanilla" else 2) for r in rows),
            "encoder_calls": sum((1 + len(r["selection"]["prefix_lengths"])) *
                                 (1 if r["model_mode"] == "vanilla" else 2) for r in rows),
            "missing_newline": sum(not r["selection"]["newline_present"] for r in rows),
            "training_updates": 0}


def candidate(repo: Path, evidence_directory: Path) -> dict[str, Any]:
    evidence = read_evidence(evidence_directory)
    source = endpoint.source_digest(repo)
    source["scripts/coco_trajectory_diagnostic.py"] = endpoint.sha256(repo / "scripts/coco_trajectory_diagnostic.py")
    return {"protocol": PROTOCOL, "evidence_sha256": evidence["evidence_sha256"],
            "source_files": source, "workload": workload(evidence),
            "runtime_packages": endpoint.runtime_packages(), "max_artifact_bytes": MAX_ARTIFACT_BYTES, "max_run_seconds": MAX_RUN_SECONDS,
            "status": "prepared_not_executed"}


def finite(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.detach().float().cpu()
    if logits.ndim != 1 or not bool(torch.isfinite(logits).all()):
        raise ValueError("Expected finite full-vocabulary vector")
    return logits


def logit_metrics(logits: torch.Tensor, eos: int, newline: list[int]) -> dict[str, Any]:
    logits = finite(logits)
    other = logits.clone()
    other[eos] = -torch.inf
    top = torch.topk(logits, min(5, logits.numel()))
    return {**endpoint.score_logits(logits, eos, newline),
            "eos_logit_margin": float(logits[eos] - other.max()),
            "top1_tie_count": int((logits == logits.max()).sum()),
            "top5_token_ids": top.indices.tolist(), "top5_logits": top.values.tolist()}


def compare(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    left, right = finite(left), finite(right)
    if left.shape != right.shape:
        raise ValueError("Vocabulary shapes differ")
    return {"max_abs_logit_delta": float((left - right).abs().max()),
            "kl_left_to_right": endpoint.teacher_student_kl(left, right),
            "top1_equal": int(left.argmax()) == int(right.argmax())}


class ForcedTrajectory(LogitsProcessor):
    """Preserve native cache preparation, forcing only the known next token."""
    def __init__(self, tokens: list[int]):
        self.tokens = tokens
        self.greedy_tokens: list[int] = []

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        index = input_ids.shape[1]
        if input_ids[0].tolist() != self.tokens[:index] or index >= len(self.tokens):
            raise ValueError("Native generation prefix differs from frozen trajectory")
        finite(scores[0])
        self.greedy_tokens.append(int(scores[0].argmax()))
        forced = torch.full_like(scores, -torch.inf)
        forced[:, self.tokens[index]] = 0
        return forced


def cached_replay(model, inputs, tokens: list[int], lengths: list[int], *, vanilla: bool, budget):
    source = endpoint.diagnostic_inputs(model, inputs)
    captured: dict[int, torch.Tensor] = {}
    steps = 0
    def capture(_module, _args, result):
        nonlocal steps
        budget.check_time()
        steps += 1
        raw = finite(result.logits[0, -1])
        if steps in lengths:
            captured[steps] = raw
    forcing = ForcedTrajectory(tokens)
    hook = model.base.register_forward_hook(capture)
    try:
        with torch.no_grad(), autocast_for(model), model.observe_payload() as events, model.transmission(
                None, bypass=vanilla, encoder_mask=source["attention_mask"]):
            generated = model.generate(**source, max_new_tokens=len(tokens) - 1, do_sample=False, num_beams=1,
                use_cache=True, logits_processor=LogitsProcessorList([forcing]))
    finally:
        hook.remove()
    if generated[0].tolist() != tokens or set(captured) != set(lengths) or steps != len(tokens) - 1:
        raise ValueError("Native cached replay did not cover the frozen trajectory")
    memory_events = sum(event["stream"] == "memory" for event in events)
    expected_memory = int(not vanilla and model.memory_codec is not None)
    if memory_events != expected_memory:
        raise ValueError("Cached receiver-memory transmission count differs from one-per-trajectory contract")
    return captured, forcing.greedy_tokens, memory_events


def run(plan: dict[str, Any], evidence_directory: Path, manifest: dict[str, Any], output: Path,
        max_seconds: float, reserve_seconds: float) -> dict[str, Any]:
    """Execute only after endpoint.verify_plan has checked real checkpoint bytes."""
    if max_seconds > MAX_RUN_SECONDS:
        raise ValueError("Run deadline exceeds the prepared 1200-second ceiling")
    evidence = read_evidence(evidence_directory)
    if manifest["evidence_sha256"] != evidence["evidence_sha256"]:
        raise ValueError("Evidence bytes changed after preparation")
    if list(plan["selection"]["image_ids"]) != list(dict.fromkeys(r["image_id"] for r in evidence["trajectories"])):
        raise ValueError("Current selection differs from prior trajectories")
    for mode in endpoint.MODES:
        if plan["modes"][mode]["checkpoint_sha256"] != evidence["identity"]["modes"][mode]["checkpoint_sha256"]:
            raise ValueError("Real checkpoint differs from prior screen")
    for key in ("HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE"):
        os.environ[key] = "1"
    budget = endpoint.RequestBudget(max_seconds, reserve_seconds)
    journal = endpoint.Journal(output)
    expected = workload(evidence)
    status: dict[str, Any] = {"protocol": PROTOCOL, "status": "running", "expected": expected,
                            "training_updates": 0, "artifact_bytes": 0, "manifest": manifest, "endpoint_manifest_sha256": plan["manifest_sha256"]}
    def save():
        status.update(attempts=budget.attempts, completed=budget.completed,
                      decoder_calls_observed=budget.decoder_calls, encoder_calls_observed=budget.encoder_calls,
                      elapsed_seconds=budget.now() - budget.start)
        temporary = output / "status.tmp"
        temporary.write_text(json.dumps(status, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(output / "status.json")
    def request(request_id, operation):
        if budget.attempts >= expected["logical_requests"]:
            raise RuntimeError("Trajectory logical request ceiling reached")
        attempt = budget.begin()
        journal.append({"request_id": request_id, "status": "partial", "attempt": attempt})
        try:
            result = operation()
            buffer = io.BytesIO()
            torch.save({"request_id": request_id, "result": result}, buffer)
            size = buffer.tell()
            if status["artifact_bytes"] + size > MAX_ARTIFACT_BYTES:
                raise RuntimeError("Retained raw artifact storage ceiling reached")
            artifact = output / f"{request_id}.pt"
            with artifact.open("xb") as stream:
                stream.write(buffer.getbuffer())
                stream.flush()
                os.fsync(stream.fileno())
            status["artifact_bytes"] += size
        except Exception as error:
            journal.append({"request_id": request_id, "status": "failed", "attempt": attempt,
                            "error": str(error)})
            raise
        budget.completed += 1
        journal.append({"request_id": request_id, "status": "complete", "attempt": attempt,
                        "artifact": artifact.name, "artifact_sha256": endpoint.sha256(artifact),
                        "artifact_bytes": size})
        save()
        return result
    handles = []
    def count_encoder(*_):
        budget.check_time()
        budget.encoder_calls += 1
        if budget.encoder_calls > expected["encoder_calls"]:
            raise RuntimeError("Encoder call ceiling exceeded")
    def count_decoder(*_):
        budget.check_time()
        budget.decoder_calls += 1
        if budget.decoder_calls > expected["decoder_calls"]:
            raise RuntimeError("Decoder call ceiling exceeded")
    save()
    try:
        with endpoint.hard_alarm(max_seconds):
            rows = endpoint.load_rows_offline(plan["modes"]["vanilla"]["config"], plan["ids"], plan["selection"])
            for mode in endpoint.MODES:
                budget.check_time()
                processor, model = endpoint.load_mode(plan["modes"][mode], mode)
                handles = [model.base.get_encoder().register_forward_pre_hook(count_encoder),
                           model.base.get_decoder().register_forward_pre_hook(count_decoder)]
                tokenizer = processor.tokenizer
                if tokenizer.eos_token_id != evidence["eos"] or tokenizer("\n", add_special_tokens=False)["input_ids"] != evidence["newline"]:
                    raise ValueError("Tokenizer contract changed")
                for row in (r for r in evidence["trajectories"] if r["model_mode"] == mode):
                    budget.check_time()
                    demos = plan["selection"]["demo_ids"]
                    processed = processor(images=[rows[i]["image"].convert("RGB") for i in [*demos, row["image_id"]]],
                        text=caption_prompt([rows[i]["answer"][0] for i in demos]), return_tensors="pt", padding=True, truncation=False)
                    inputs = {key: processed[key] for key in ("input_ids", "attention_mask", "pixel_values")}
                    hashes = {key: endpoint.tensor_sha256(value) for key, value in inputs.items()}
                    if hashes != row["input_tensor_sha256"]:
                        raise ValueError("Processed matched inputs differ from prior screen")
                    tokens, selection = row["generated_token_ids"], row["selection"]
                    lengths = selection["prefix_lengths"]
                    native = model.base.prepare_decoder_input_ids_from_labels(labels=torch.tensor([[evidence["eos"]]]))
                    if native[0, 0].item() != tokens[0]:
                        raise ValueError("Native decoder BOS changed")
                    key = f'{row["image_id"]}-{mode}'
                    journal.append({"kind": "trajectory_start", "image_id": row["image_id"], "model_mode": mode,
                        "generated_token_ids": tokens, "selection": selection, "input_tensor_sha256": hashes,
                        "checkpoint_sha256": row["checkpoint_sha256"], "prior_request_id": row["request_id"]})
                    results = {}
                    for role in (("student",) if mode == "vanilla" else ("student", "teacher")):
                        bypass = mode == "vanilla" or role == "teacher"
                        cached, greedy, memory = request(f"{key}-{role}-cached", lambda model=model: cached_replay(
                            model, inputs, tokens, lengths, vanilla=bypass, budget=budget))
                        full = {}
                        for length in lengths:
                            decoder = torch.tensor([tokens[:length]], dtype=torch.long)
                            full[length] = request(f"{key}-{role}-full-{length}", lambda model=model: finite(
                                endpoint.forward_endpoint(model, inputs, decoder, vanilla=bypass)))
                        results[role] = {"cached": cached, "full": full, "greedy": greedy, "memory_events": memory}
                    if mode == "vanilla":
                        results["teacher"] = results["student"]
                    states = []
                    for length in lengths:
                        student, teacher = results["student"], results["teacher"]
                        states.append({"prefix_length": length, "decoder_input_ids": tokens[:length],
                            "events": [name for name, value in selection["events"].items() if value == length],
                            "observed_next_token": tokens[length],
                            "metrics": {f"{role}_{path}": logit_metrics(results[role][path][length], evidence["eos"], evidence["newline"])
                                        for role in ("student", "teacher") for path in ("cached", "full")},
                            "student_cache_full": compare(student["cached"][length], student["full"][length]),
                            "teacher_cache_full": compare(teacher["cached"][length], teacher["full"][length]),
                            "teacher_student_cached": compare(teacher["cached"][length], student["cached"][length]),
                            "teacher_student_full": compare(teacher["full"][length], student["full"][length])})
                    journal.append({"kind": "trajectory", "image_id": row["image_id"], "model_mode": mode,
                        "selection": selection, "generated_token_ids": tokens, "raw_caption": row["raw_caption"],
                        "repeated_lines": row["repeated_lines"], "trigram_repetition": row["trigram_repetition"],
                        "student_native_greedy_reproduces_record": results["student"]["greedy"] == tokens[1:],
                        "states": states})
                for handle in handles:
                    handle.remove()
                handles = []
                del model, processor
            if (budget.completed, budget.decoder_calls, budget.encoder_calls) != (
                    expected["logical_requests"], expected["decoder_calls"], expected["encoder_calls"]):
                raise ValueError("Completed workload does not match frozen counts")
            status["status"] = "complete"
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        for handle in handles:
            handle.remove()
        save()
        journal.close()
    return status
