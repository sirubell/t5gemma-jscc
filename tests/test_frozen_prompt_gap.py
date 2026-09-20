import json
import pytest
import torch
from scripts.frozen_prompt_gap import RequestBudget, check_replay, tensor_batch


def fixture_batch():
    return [
        {
            "context_token_ids": [2, 0, 7][: 1 + i % 3],
            "continuation_token_ids": [0, 4][: 1 + i % 2],
            "denominator": 10 + i,
        }
        for i in range(64)
    ]


def test_budget_counts_before_forward_and_refuses_overflow(tmp_path):
    p = tmp_path / "ledger.jsonl"
    b = RequestBudget(p, maximum=128)
    b.reserve(64, "fixture", "warmup")
    b.reserve(64, "fixture", "core")
    with pytest.raises(RuntimeError):
        b.reserve(64, "fixture", "replay")
    assert b.used == 128 and len(p.read_text().splitlines()) == 2
    assert json.loads(p.read_text().splitlines()[-1])["cumulative"] == 128


def test_tensor_masks_use_lengths_not_pad_values():
    t = tensor_batch(fixture_batch(), torch.device("cpu"))
    assert t["input_ids"].shape == (64, 3) and t["labels"].shape == (64, 2)
    assert t["decoder_payload_mask"][0].tolist() == [True, False]
    assert t["labels"][0].tolist() == [0, 0]
    assert t["attention_mask"][2].tolist() == [1, 1, 1]
    with pytest.raises(ValueError):
        tensor_batch(fixture_batch()[:63], torch.device("cpu"))


def test_batch_padding_matches_pinned_hflm():
    from lm_eval.models.utils_hf import pad_and_concat

    rows = fixture_batch()
    t = tensor_batch(rows, torch.device("cpu"))
    for key, field in [
        ("input_ids", "context_token_ids"),
        ("labels", "continuation_token_ids"),
    ]:
        seqs = [torch.tensor(r[field], dtype=torch.long) for r in rows]
        expected = pad_and_concat(max(len(s) for s in seqs), seqs)
        torch.testing.assert_close(t[key], expected)


def test_replay_checks_raw_and_normalized_scores():
    r = fixture_batch()
    check_replay([-3.0] * 64, [-3.0] * 64, r)
    with pytest.raises(AssertionError):
        check_replay([-3.0] * 64, [-2.0] * 64, r)
