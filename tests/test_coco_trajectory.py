"""Deterministic CPU tests; no pretrained downloads or accelerator work."""
from pathlib import Path

import pytest
import torch

from model_helpers import tiny_backbone
from jscc.coco_diagnostic import RequestBudget, forward_endpoint
from jscc.coco_trajectory import (
    ForcedTrajectory, cached_replay, compare, finite, logit_metrics, read_evidence,
    selected_states, workload,
)
from jscc.models.channel import AWGNChannel
from jscc.models.codec import Codec
from jscc.models.split_model import SplitModel


def test_states_stop_before_eos_and_deduplicate_events():
    result = selected_states([2, 8, 1], 1, [8])
    assert result["events"] == {"before_first_newline": 1, "after_first_newline": 2, "terminal_decision": 2}
    assert result["prefix_lengths"] == [1, 2]
    assert result["after_newline_state"] == "visited"


def test_missing_newline_and_capped_last_token_are_explicit():
    assert selected_states([2, 4, 1], 1, [8])["after_newline_state"] == "missing_no_newline"
    result = selected_states([2, 4, 8], 1, [8], cap=2)
    assert result["events"] == {"terminal_decision": 2, "before_first_newline": 2}
    assert result["after_newline_state"] == "unvisited_after_final_token"
    assert result["truncated"]


def test_multitoken_newline_selects_before_entire_sequence():
    assert selected_states([2, 7, 8, 1], 1, [7, 8])["prefix_lengths"] == [1, 3]


@pytest.mark.parametrize("tokens", [[2, 1, 3], [2, 5], [2]])
def test_rejects_unvisited_or_incomplete_trajectories(tokens):
    with pytest.raises(ValueError):
        selected_states(tokens, 1, [8])


def test_finite_and_ties_are_not_argmax_counts():
    with pytest.raises(ValueError):
        finite(torch.tensor([0., float("nan")]))
    values = logit_metrics(torch.tensor([2., 2., 0.]), 1, [2])
    assert values["eos_rank"] == 1
    assert values["next_token_top1"] == 0
    assert values["top1_tie_count"] == 2
    assert values["eos_logit_margin"] == 0
    assert compare(torch.tensor([1., 2.]), torch.tensor([1., 2.]))["max_abs_logit_delta"] == 0


def test_forcing_records_unforced_greedy_and_rejects_wrong_prefix():
    forcing = ForcedTrajectory([2, 4, 1])
    scores = forcing(torch.tensor([[2]]), torch.tensor([[1., 2., 3., 4., 0.]]))
    assert scores.argmax().item() == 4
    assert forcing.greedy_tokens == [3]
    with pytest.raises(ValueError):
        forcing(torch.tensor([[2, 3]]), torch.zeros(1, 5))


@pytest.mark.parametrize("stack,vanilla", [("enc", True), ("enc", False), ("dec", False)])
def test_native_cache_full_same_prefix_and_receiver_memory(stack, vanilla):
    torch.manual_seed(0)
    model = SplitModel(tiny_backbone(), Codec(16, {"hidden_dim": 16, "bottleneck_dim": 8, "activation": "gelu", "layernorm": "none", "snr_film": False, "n_res_blocks": 0}), AWGNChannel(),
                       {"stack": stack, "where": "after_layer", "index": 0}, {"normalize_power": True, "clean_film_snr": 18.0})
    model.eval()
    inputs = {"input_ids": torch.tensor([[2, 5, 6]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
    tokens = [2, 4, 5, 1]
    cached, greedy, memory = cached_replay(model, inputs, tokens, [1, 2, 3], vanilla=vanilla,
                                         budget=RequestBudget(30, 0))
    assert len(greedy) == 3
    assert memory == int(stack == "dec" and not vanilla)
    for length in [1, 2, 3]:
        full = forward_endpoint(model, inputs, torch.tensor([tokens[:length]]), vanilla=vanilla)
        # Native tiny model in FP32; this is a numerical path test, not BF16 acceptance.
        torch.testing.assert_close(cached[length], full, atol=2e-5, rtol=2e-5)


def test_complete_observed_matrix_and_exact_call_budget():
    root = Path(__file__).resolve().parents[1]
    evidence_path = root / "runs/observed/20260929-coco-job-18387"
    if not evidence_path.exists():
        pytest.skip("Private historical evidence is not included in portable source")
    evidence = read_evidence(evidence_path)
    counts = workload(evidence)
    assert counts["trajectories"] == 128
    assert counts["selected_states"] <= 384
    assert counts["logical_requests"] <= 896
    assert counts["decoder_calls"] <= 15008
    assert counts["training_updates"] == 0




@pytest.fixture
def saved_screen(tmp_path):
    """Small receipt matrix, independent of private scientific artifacts."""
    import json
    from jscc.coco_diagnostic import MODES, sha256

    identity = {"modes": {mode: {"checkpoint_sha256": str(index) * 64}
                          for index, mode in enumerate(MODES)}}
    (tmp_path / "identity.json").write_text(json.dumps(identity))
    (tmp_path / "status.json").write_text(json.dumps({
        "status": "complete", "completed": 1024,
        "manifest_sha256": sha256(tmp_path / "identity.json")}))
    endpoints = [{"status": "complete", "request_kind": "endpoint",
                  "eos_token_id": 1, "newline_token_ids": [7, 8]} for _ in range(768)]
    trajectories = [{"status": "complete", "request_kind": "generation",
                     "query_control": "matched", "image_id": image_id, "model_mode": mode,
                     "checkpoint_sha256": identity["modes"][mode]["checkpoint_sha256"],
                     "generated_token_ids": [2, 7, 8, 1]}
                    for image_id in range(32) for mode in MODES]
    rows = endpoints + trajectories
    (tmp_path / "items.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return tmp_path


def test_saved_reader_preserves_exact_matrix_hashes_and_workload(saved_screen):
    from jscc.coco_diagnostic import sha256

    evidence = read_evidence(saved_screen)
    assert evidence["eos"] == 1 and evidence["newline"] == [7, 8]
    assert evidence["evidence_sha256"] == {name: sha256(saved_screen / name)
        for name in ("identity.json", "status.json", "items.jsonl")}
    assert workload(evidence) == {"trajectories": 128, "selected_states": 256,
        "logical_requests": 672, "decoder_calls": 1120, "encoder_calls": 672,
        "missing_newline": 0, "training_updates": 0}
    assert evidence["trajectories"][0]["selection"]["prefix_lengths"] == [1, 3]


@pytest.mark.parametrize("mutation,message", [
    ("incomplete", "complete 1024"),
    ("identity", "identity bytes"),
    ("missing_endpoint", "tokenizer contracts"),
    ("tokenizer", "tokenizer contracts"),
    ("missing_generation", "32 images by four"),
    ("duplicate_mode", "32 images by four"),
    ("checkpoint", "checkpoint identity"),
])
def test_saved_reader_rejects_changed_or_incomplete_receipts(saved_screen, mutation, message):
    import json

    if mutation == "incomplete":
        path = saved_screen / "status.json"
        value = json.loads(path.read_text())
        value["completed"] = 1023
        path.write_text(json.dumps(value))
    elif mutation == "identity":
        path = saved_screen / "identity.json"
        path.write_text(path.read_text() + "\n")
    else:
        path = saved_screen / "items.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if mutation == "missing_endpoint":
            rows.pop(0)
        elif mutation == "tokenizer":
            rows[0]["newline_token_ids"] = [8]
        elif mutation == "missing_generation":
            rows.pop()
        elif mutation == "duplicate_mode":
            rows[-1]["model_mode"] = "enc_fn"
        elif mutation == "checkpoint":
            rows[-1]["checkpoint_sha256"] = "changed"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match=message):
        read_evidence(saved_screen)
