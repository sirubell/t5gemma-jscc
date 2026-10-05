"""Offline COCO trajectory readers and reusable cache/full comparison helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from . import coco_diagnostic as endpoint
from .runtime import autocast_for

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
