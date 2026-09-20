"""CPU-only prompt policy fidelity, leakage boundaries and unchanged targets."""
import random
from typing import Any, cast

from datasets import Dataset, DatasetDict
import pytest

from jscc.data.hellaswag import load_data
from jscc.data.hellaswag_prompts import PromptBuilder


class Tokenizer:
    pad_token_id = 0
    truncation_side = "right"

    def __call__(self, texts, truncation=False, max_length=None):
        tokens = [[2] + [ord(char) + 3 for char in text] + [1] for text in texts]
        if truncation:
            tokens = [row[:max_length] for row in tokens]
        return {"input_ids": tokens, "attention_mask": [[1] * len(row) for row in tokens]}


def fixture():
    docs = [{"ctx": f"raw {i} " + "x" * 2200, "activity_label": "Activity [artifact]",
             "ctx_a": f"Person {i} [title] enters" + " y" * 300, "ctx_b": "tHE Room",
             "endings": ["wrong", f" right {i} [artifact]  ending "], "label": "1",
             "source_id": str(i // 2)} for i in range(12)]
    raw = DatasetDict({"train": Dataset.from_list(docs)})
    ids = {"train_rows": list(range(10)), "validation_rows": [10, 11], "validation_split": "train"}
    policy = {"version": "prompt-alignment-v1", "mode": "five_shot", "seed": 20260920,
              "source_max_length": 2048}
    config = {"data": {"name": "fixture", "revision": "fixed", "max_length": 512,
                       "num_workers": 0, "prompt_policy": policy},
              "model": {"device": "cpu"}, "training": {"batch_size": 2}}
    return raw, ids, config


def test_pinned_formatter_and_demo_fidelity():
    from lm_eval.tasks.hellaswag.utils import process_docs
    raw, ids, config = fixture()
    builder = PromptBuilder(raw["train"], ids, config["data"]["prompt_policy"])
    expected = process_docs(raw["train"])
    for index, doc in enumerate(raw["train"]):
        assert builder.format(doc) == (expected[index]["query"], expected[index]["choices"])
        text, demos = builder.build(doc, index)
        assert len(set(demos)) == 5
        assert set(demos) <= set(ids["train_rows"])
        assert index not in demos
        assert all(cast(Any, raw["train"][demo])["source_id"] != cast(Any, doc)["source_id"] for demo in demos)
        assert text == "\n\n".join([*(expected[key]["query"] + " " + expected[key]["choices"][1]
                                      for key in demos), expected[index]["query"]])


def test_prompt_deterministic_without_global_rng_effect():
    raw, ids, config = fixture()
    builder = PromptBuilder(raw["train"], ids, config["data"]["prompt_policy"])
    before = random.getstate()
    first = builder.build(raw["train"][3], 3)
    assert random.getstate() == before
    random.Random(42).random()
    builder.build(raw["train"][4], 4)
    assert first == builder.build(raw["train"][3], 3)


def test_loader_targets_default_path_caps_and_evidence(monkeypatch):
    raw, ids, config = fixture()
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: raw)
    tokenizer = Tokenizer()
    policy = config["data"].pop("prompt_policy")
    default = load_data(config, tokenizer, ids)
    assert not hasattr(default, "prompt_evidence")
    assert default.train.dataset[0]["input_ids"] == tokenizer([raw["train"][0]["ctx"]], True, 512)["input_ids"][0]
    outputs = []
    for mode in ("raw", "five_shot"):
        config["data"]["prompt_policy"] = {**policy, "mode": mode}
        loaded = load_data(config, tokenizer, ids)
        outputs.append(loaded)
        builder = PromptBuilder(raw["train"], ids, config["data"]["prompt_policy"])
        for name in ("train", "validation"):
            dataset = getattr(loaded, name).dataset
            for row in dataset:
                row_id = row["prompt_row_id"]
                text, _ = builder.build(raw["train"][row_id], row_id)
                assert row["input_ids"] == tokenizer([text])["input_ids"][0][-2048:]
                assert row["label_ids"] == tokenizer([raw["train"][row_id]["endings"][1]], True, 512)["input_ids"][0]
        assert getattr(loaded, "prompt_evidence")["train"]["count"] == 10
        assert getattr(loaded, "prompt_evidence")["selection"]["count"] == 2
        assert loaded.ids == ids
        batch = next(iter(loaded.train))
        assert set(batch) == {"input_ids", "attention_mask", "labels"}
    assert all(getattr(output, "prompt_evidence")["train"]["truncated_count"] == 10 for output in outputs)
    assert tokenizer.truncation_side == "right"


def test_rejects_selection_overlap_and_insufficient_demo_pool():
    raw, ids, config = fixture()
    policy = config["data"]["prompt_policy"]
    with pytest.raises(ValueError, match="exclude selection"):
        PromptBuilder(raw["train"], {**ids, "validation_rows": [1]}, policy)
    builder = PromptBuilder(raw["train"], {**ids, "train_rows": [0, 1, 2, 3, 4]}, policy)
    with pytest.raises(ValueError, match="five eligible"):
        builder.build(raw["train"][0], 0)


def test_paired_presentation_stream_unchanged(monkeypatch):
    raw, ids, config = fixture()
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: raw)
    config["training"].update({"max_steps": 6, "gradient_accumulation": 1,
                              "presentation_stream": {"policy": "epoch-permutations-v1",
                                                      "total_presentations": 12, "seed": 7}})
    orders = []
    for mode in ("raw", "five_shot"):
        config["data"]["prompt_policy"]["mode"] = mode
        loaded = load_data(config, Tokenizer(), ids)
        orders.append([row_id for batch in loaded.train for row_id in batch["row_ids"].tolist()])
    assert orders[0] == orders[1]
    assert len(orders[0]) == 12
