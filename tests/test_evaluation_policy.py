"""Production HFLM scoring, compact evidence, and shared vanilla CPU checks."""
from typing import Any, cast

import pytest
import torch

from jscc.evaluation_policy import compact_sample, evaluation_conditions, scoring_kwargs
from jscc.harness_payload import payload_harness_class
from jscc.models.channel import AWGNChannel
from jscc.models.codec import Codec
from jscc.models.split_model import SplitModel
from model_helpers import tiny_backbone
from test_harness_payload import fixture_model


@pytest.mark.parametrize("stack", ["enc", "dec"])
def test_real_adapter_fp32_sum_from_same_bf16_logits(stack):
    """The returned HFLM score, not an unused alternate calculation, is FP32."""
    from lm_eval.api.registry import get_model
    tokenizer, model = fixture_model()
    if stack == "enc":
        model = SplitModel(tiny_backbone(), Codec(16, model.codec.config), AWGNChannel(),
                           {"stack": "enc", "where": "after_final_norm"},
                           {"normalize_power": True, "clean_film_snr": 18.0}).eval()
    cls = payload_harness_class(cast(type[Any], get_model("hf")))
    adapter = cls(communication_model=model, pretrained=model.base, tokenizer=tokenizer,
                  backend="seq2seq", batch_size=64, max_length=16,
                  record_requests=True, **scoring_kwargs({"scoring_policy": "fp32-v1"}, model))
    requests = [((f"c{i}", f"a{j}"), [2, 4] + [4] * (i % 3) + [1], [4] * (j + 1) + [0, 1])
                for i in range(16) for j in range(4)]
    expected = {}
    original = adapter._model_call

    def same_logits(inps, attn_mask=None, labels=None):
        assert attn_mask is not None and labels is not None
        logits = original(inps, attn_mask=attn_mask, labels=labels).bfloat16()
        # A CPU fixture records one independent reference from these exact logits.
        for inp, mask, target, value in zip(inps, attn_mask, labels, logits, strict=True):
            context = tuple(inp[mask.bool()].tolist())
            for request, ctx, cont in requests:
                if context == tuple(ctx) and target[:len(cont)].tolist() == cont:
                    selected = value[:len(cont)].float().log_softmax(-1).gather(
                        -1, torch.tensor(cont)[:, None])
                    expected[request] = (float(selected.sum(dtype=torch.float32)),
                                         bool((value[:len(cont)].argmax(-1) == torch.tensor(cont)).all()))
        return logits

    adapter._model_call = same_logits
    with model.transmission(None):
        actual = adapter._loglikelihood_tokens(requests, disable_tqdm=True)
        counts = model.valid_payload_counts
        assert counts["hidden"] > 0
        if stack == "dec":
            assert counts["hidden"] == sum(len(cont) for _, _, cont in requests) * 4
            assert counts["memory"] > 0
    assert actual == [expected[request] for request, _, _ in requests]
    assert len(adapter.request_evidence) == 64
    with model.transmission(None, bypass=True):
        vanilla = adapter._loglikelihood_tokens(requests, disable_tqdm=True)
        assert model.valid_payload_counts == {"hidden": 0, "memory": 0}
        assert model.channel_uses_allocated == {"hidden": 0, "memory": 0}
    assert all(torch.isfinite(torch.tensor(score)) for score, _ in vanilla)


def test_vanilla_scores_identical_across_routes():
    from lm_eval.api.registry import get_model
    tokenizer, dec = fixture_model()
    base = tiny_backbone()
    base.load_state_dict(dec.base.state_dict())
    enc = SplitModel(base, Codec(16, dec.codec.config), AWGNChannel(),
                     {"stack": "enc", "where": "after_final_norm"},
                     {"normalize_power": True, "clean_film_snr": 18.0}).eval()
    cls = payload_harness_class(cast(type[Any], get_model("hf")))
    requests = [(("c", "a"), [2, 4, 1], [4, 1])]
    scores = []
    for model in (enc, dec):
        adapter = cls(communication_model=model, pretrained=model.base, tokenizer=tokenizer,
                      backend="seq2seq", batch_size=64, max_length=16, softmax_dtype=torch.float32)
        with model.transmission(None, bypass=True):
            scores.append(adapter._loglikelihood_tokens(requests, disable_tqdm=True))
            assert sum(model.valid_payload_counts.values()) == 0
    assert scores[0] == scores[1]


def test_compact_gold_margin_ranking_and_char_denominator():
    sample = {"doc_id": 343, "doc": {"choices": ["a", "bb", "ccc", "dddd"],
                                    "gold": 1, "source_id": "original-source"},
              "filtered_resps": [(-2, False), (-3, False), (-6, False), (-9, False)]}
    row = compact_sample(sample, "no_noise", "identity", "noise")
    assert row["raw_prediction"] == 0 and row["normalized_prediction"] == 1
    assert row["raw_gold_margin"] == -1 and row["normalized_gold_margin"] == .5
    assert row["raw_top1_top2_gap"] == 1
    assert row["source_id"] == "original-source"
    sample["filtered_resps"][0] = (float("nan"), False)
    with pytest.raises(ValueError, match="Non-finite"):
        compact_sample(sample, "no_noise", "identity", "noise")


def test_policy_defaults_and_vanilla_only():
    tokenizer, model = fixture_model()
    assert scoring_kwargs({}, model) == {}
    settings = {"snrs": ["no_noise", -6], "vanilla": True}
    assert evaluation_conditions(settings) == ["no_noise", -6, "vanilla"]
    assert evaluation_conditions({**settings, "mode": "vanilla_only"}) == ["vanilla"]
    assert evaluation_conditions({**settings, "mode": "codec_only"}) == ["no_noise", -6]
    with pytest.raises(ValueError, match="Unknown scoring"):
        scoring_kwargs({"scoring_policy": "fp64"}, model)


def test_real_evaluation_entry_compact_fewshots_identity_and_payload(monkeypatch, tmp_path):
    """Run real TaskManager/simple_evaluate with an in-memory dataset, no network."""
    import json
    from datasets import Dataset, DatasetDict
    from jscc.evaluation import evaluate_hellaswag

    def rows(count):
        return [{"ctx_a": "x", "ctx_b": "x", "activity_label": "x",
                 "endings": ["x", "x x", "x x x", "x x x x"], "label": str(i % 4),
                 "source_id": f"original~{i}", "ind": i} for i in range(count)]
    raw = DatasetDict({"train": Dataset.from_list(rows(8)), "validation": Dataset.from_list(rows(16))})
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: raw)
    tokenizer, model = fixture_model()
    settings = {"batch_size": 64, "num_fewshot": 5, "num_samples": 16,
                "scoring_policy": "fp32-v1", "evidence_mode": "compact-v1", "noise_seed": 0}
    with model.transmission(None):
        result = evaluate_hellaswag(model, tokenizer, settings, tmp_path, "no_noise",
                                   {"name": "fixture", "revision": "fixed"})
    with model.transmission(None, bypass=True):
        vanilla = evaluate_hellaswag(model, tokenizer, settings, tmp_path, "vanilla",
                                    {"name": "fixture", "revision": "fixed"})
    assert result["evaluation_identity"] == vanilla["evaluation_identity"]
    assert result["compact_num_samples"] == 16
    prompts = [json.loads(line) for line in (tmp_path / "prompts.jsonl").read_text().splitlines()]
    assert all(len(prompt["fewshot_ids"]) == 5 for prompt in prompts)
    assert (tmp_path / "fewshots.jsonl").is_file()
    compact = [json.loads(line) for line in (tmp_path / "compact_vanilla.jsonl").read_text().splitlines()]
    assert all(sum(value.values()) == 0 for row in compact for value in row["valid_payload_coordinates"])
    assert compact[0]["source_id"] == "original~0"
    original_identity = result["evaluation_identity"]
    settings["batch_size"] = 32
    with model.transmission(None, bypass=True):
        changed = evaluate_hellaswag(model, tokenizer, settings, tmp_path / "changed", "vanilla",
                                    {"name": "fixture", "revision": "fixed"})
    assert changed["evaluation_identity"] != original_identity


def test_bounded_scoring_cli_helper_on_cpu():
    from lm_eval.api.registry import get_model
    from scripts.outer_ln_scoring_check import check
    tokenizer, model = fixture_model()
    adapter = cast(Any, get_model("hf"))(pretrained=model.base, tokenizer=tokenizer,
                                       backend="seq2seq", batch_size=64)
    rows = []
    for i in range(16):
        pairs = [("x", " x" * (j + 1)) for j in range(4)]
        tokens = [adapter._encode_pair(*pair) for pair in pairs]
        rows.append({"doc_id": i, "exact_candidate_arguments": pairs,
                     "candidate_tokens": [{"context": context, "target": target} for context, target in tokens]})
    result = check(model, tokenizer, rows, bypass=True)
    assert result["candidate_requests"] == 64
    assert result["scores"] == result["same_logits_fp32_scores"]
    assert result["payload"] == {"hidden": 0, "memory": 0}
