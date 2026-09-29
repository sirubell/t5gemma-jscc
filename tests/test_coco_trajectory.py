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


@pytest.mark.parametrize("failure_stage", ["cached", "full"])
def test_failed_request_retains_attempt_artifact_and_failed_status(tmp_path, monkeypatch, failure_stage):
    from jscc import coco_trajectory as trajectory
    from jscc import coco_diagnostic as endpoint
    from PIL import Image
    torch.manual_seed(0)
    model = SplitModel(tiny_backbone(), Codec(16, {"hidden_dim": 16, "bottleneck_dim": 8,
        "activation": "gelu", "layernorm": "none", "snr_film": False, "n_res_blocks": 0}),
        AWGNChannel(), {"stack": "enc", "where": "after_layer", "index": 0},
        {"normalize_power": True, "clean_film_snr": 18.0}).eval()
    inputs = {"input_ids": torch.tensor([[2, 5]]), "attention_mask": torch.ones(1, 2, dtype=torch.long),
              "pixel_values": torch.empty((0, 3, 8, 8))}
    class Tokenizer:
        eos_token_id = 1
        def __call__(self, *args, **kwargs):
            return {"input_ids": [8]}
    class Processor:
        tokenizer = Tokenizer()
        def __call__(self, **kwargs):
            return inputs
    row = {"image_id": 1, "model_mode": "vanilla", "generated_token_ids": [2, 4, 1],
           "selection": selected_states([2, 4, 1], 1, [8]), "checkpoint_sha256": "digest",
           "request_id": "original", "input_tensor_sha256": {k: endpoint.tensor_sha256(v) for k, v in inputs.items()}}
    evidence = {"trajectories": [row], "evidence_sha256": {}, "eos": 1, "newline": [8],
                "identity": {"modes": {m: {"checkpoint_sha256": "digest"} for m in endpoint.MODES}}}
    monkeypatch.setattr(trajectory, "read_evidence", lambda *_: evidence)
    monkeypatch.setattr(endpoint, "load_rows_offline", lambda *_: {1: {"image": Image.new("RGB", (8, 8)), "answer": ["caption"]}})
    monkeypatch.setattr(endpoint, "load_mode", lambda *_: (Processor(), model))
    def fail(*args, **kwargs):
        raise ValueError("injected nonfinite logits")
    if failure_stage == "cached":
        monkeypatch.setattr(trajectory, "cached_replay", fail)
    else:
        monkeypatch.setattr(trajectory, "cached_replay", lambda *args, **kwargs: ({2: torch.zeros(32)}, [4, 1], 0))
        monkeypatch.setattr(endpoint, "forward_endpoint", fail)
    plan = {"selection": {"image_ids": [1], "demo_ids": [1] * 4}, "ids": {}, "manifest_sha256": "test",
            "modes": {m: {"checkpoint_sha256": "digest", "config": {}} for m in endpoint.MODES}}
    result = trajectory.run(plan, tmp_path, {"evidence_sha256": {}}, tmp_path / "output", 30, 0)
    assert result["status"] == "failed"
    assert result["attempts"] == (1 if failure_stage == "cached" else 2)
    assert result["completed"] == (0 if failure_stage == "cached" else 1)
    import json
    records = [json.loads(line) for line in (tmp_path / "output/items.jsonl").read_text().splitlines()]
    assert [r["status"] for r in records if "status" in r] == (
        ["partial", "failed"] if failure_stage == "cached" else ["partial", "complete", "partial", "failed"])
    if failure_stage == "full":
        complete = next(r for r in records if r.get("status") == "complete")
        artifact = tmp_path / "output" / complete["artifact"]
        assert endpoint.sha256(artifact) == complete["artifact_sha256"]
        saved = torch.load(artifact, weights_only=True)
        assert saved["request_id"] == complete["request_id"]
        assert torch.isfinite(saved["result"][0][2]).all()
    assert json.loads((tmp_path / "output/status.json").read_text())["status"] == "failed"
    assert not model.base._forward_hooks
