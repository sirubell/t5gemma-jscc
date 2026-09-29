"""Per-image COCO evidence: variable padding, EOS and offline score replay."""
import json
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from jscc.evaluation import coco_generation_inputs, coco_payload_records, evaluate_coco
from test_coco import coco_model


def test_payload_observer_exact_unequal_lengths_and_no_numeric_change():
    torch.manual_seed(5)
    model = coco_model().eval()
    hidden = torch.randn(2, 3, 16)
    mask = torch.tensor([[1, 1, 1], [1, 0, 0]])
    with torch.no_grad():
        plain = model._transmit(hidden, model.codec, "hidden", mask)
        with model.observe_payload() as events:
            observed = model._transmit(hidden, model.codec, "hidden", mask)
    torch.testing.assert_close(plain, observed, rtol=0, atol=0)
    assert events == [{"stream": "hidden", "shape": [2, 3, 8],
                       "allocated": [24, 24], "mask_valid": [24, 8]}]
    assert model._payload_observer is None


def test_payload_eos_not_equal_batch_division():
    events = [{"stream": "memory", "shape": [2, 3, 8],
               "allocated": [24, 24], "mask_valid": [24, 8]}]
    events += [{"stream": "hidden", "shape": [2, 1, 8],
                "allocated": [8, 8], "mask_valid": [8, 8]} for _ in range(3)]
    rows = coco_payload_records(events, [1, 3], "dec")
    assert rows[0]["mask_valid"]["hidden"] == rows[1]["mask_valid"]["hidden"] == 24
    assert rows[0]["generation_valid"] == {"hidden": 8, "memory": 24}
    assert rows[1]["generation_valid"] == {"hidden": 24, "memory": 8}
    events[-1]["shape"][1] = 2
    assert coco_payload_records(events, [1, 3], "dec")[0]["generation_valid"] is None


class Image:
    def convert(self, mode):
        assert mode == "RGB"
        return self


class Processor:
    def __init__(self):
        self.tokenizer: Any = SimpleNamespace(eos_token_id=1, name_or_path="fixture")

    def __call__(self, **kwargs):
        assert len(kwargs["images"]) == len(kwargs["text"]) == 2
        assert all(len(images) == 5 for images in kwargs["images"])
        assert kwargs["padding"] is True
        assert kwargs["truncation"] is False
        return {"input_ids": torch.tensor([[2, 3, 4], [2, 3, 0]]),
                "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
                "pixel_values": torch.ones(10, 3, 2, 2)}


def test_generation_uses_processor_native_batch_padding():
    data = SimpleNamespace(demo_images=[Image()] * 4, demo_captions=["demo"] * 4)
    values = coco_generation_inputs(Processor(), data, [{"image": Image()}] * 2,
                                    "cpu", torch.bfloat16)
    assert values["pixel_values"].dtype == torch.bfloat16
    assert values["input_ids"].dtype == torch.int64
    assert values["attention_mask"].sum(1).tolist() == [3, 2]


def test_coco_exports_references_ptb_tokens_and_exact_payload(monkeypatch, tmp_path):
    class Tokenizer:
        eos_token_id = 1
        name_or_path = "fixture"

        def decode(self, tokens, skip_special_tokens):
            return "a cat. another sentence"

        def __call__(self, text, add_special_tokens):
            return {"input_ids": list(range(len(text.split()) + 1))}

    processor = Processor()
    processor.tokenizer = Tokenizer()
    model = coco_model().eval()
    def generate(**kwargs):
        assert model._payload_observer is not None
        model._payload_observer.append({"stream": "hidden", "shape": [2, 3, 8],
                                        "allocated": [24, 24], "mask_valid": [24, 16]})
        return torch.tensor([[2, 5, 1, 0], [2, 5, 6, 7]])
    monkeypatch.setattr(model, "generate", generate)
    def score(refs, captions):
        assert refs[1] == [{"caption": "first ref"}, {"caption": "other ref"}]
        return .5, [.25, .75], {key: ["a cat", "a pet"] for key in refs}, {key: ["a cat"] for key in refs}
    monkeypatch.setattr("jscc.evaluation.score_coco_captions", score)
    data = SimpleNamespace(demo_images=[Image()] * 4, demo_captions=["demo"] * 4,
                           ids={"demo_ids": [9, 10, 11, 12]},
                           report=[{"image": Image(), "file_name": f"COCO_val2014_{i:012d}.jpg",
                                    "answer": ["First Ref", "Other Ref"]} for i in (1, 2)])
    metrics = evaluate_coco(model, processor, data, {"batch_size": 2, "max_new_tokens": 3},
                            tmp_path, "no_noise")
    rows = [json.loads(line) for line in (tmp_path / "captions_no_noise.jsonl").read_text().splitlines()]
    assert metrics["cider"] == .5 and metrics["truncated_rate"] == .5
    assert rows[0]["references"] == ["First Ref", "Other Ref"]
    assert rows[0]["generated_tokens_through_eos"] == 2
    assert rows[1]["payload"]["mask_valid"]["hidden"] == 16
    assert rows[0]["ptb_references"] == ["a cat", "a pet"]
    metadata = json.loads((tmp_path / "coco_evidence_no_noise.json").read_text())
    assert metadata["batches"][0]["image_ids"] == [1, 2]
    assert metadata["generation"]["use_cache"] is True


def test_offline_cider_recompute_matches_saved_per_image(tmp_path):
    from pycocoevalcap.cider.cider import Cider
    from scripts.recompute_coco import recompute
    refs = {1: ["a cat sleeps", "a cat naps"], 2: ["a dog runs", "the dog walks"]}
    predictions = {1: ["a cat sleeps"], 2: ["the dog runs"]}
    score, scores = Cider().compute_score(refs, predictions)
    path = tmp_path / "captions.jsonl"
    path.write_text("\n".join(json.dumps({"image_id": image_id, "ptb_references": refs[image_id],
                                         "ptb_caption": predictions[image_id], "cider": float(value)})
                              for image_id, value in zip(refs, scores)))
    result = recompute(path)
    assert result["cider"] == pytest.approx(score)
    assert result["maximum_saved_score_difference"] == 0
