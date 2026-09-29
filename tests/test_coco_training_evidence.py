"""COCO evidence is observational: caption sampling and model inputs stay identical."""
import random
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch.utils.data import default_collate

from jscc.data.coco import CaptionDataset, coco_input_evidence
from jscc.runtime import model_inputs


class Processor:
    @property
    def tokenizer(self):
        return self

    def __call__(self, text=None, **kwargs):
        if "images" in kwargs:
            return {"input_ids": torch.tensor([[7, 8, 0]]),
                    "attention_mask": torch.tensor([[1, 1, 0]]),
                    "pixel_values": torch.ones(1, 3, 2, 2)}
        assert text is not None
        if not kwargs.get("truncation"):
            return {"input_ids": torch.tensor([[ord(text[0]), 1]])}
        return {"input_ids": torch.tensor([[ord(text[0]), 1, 0]]),
                "attention_mask": torch.tensor([[1, 1, 0]])}


def dataset(record=False, training=True):
    row = {"answer": ["alpha", "beta"], "file_name": "COCO_val2014_000000000042.jpg",
           "image": SimpleNamespace(convert=lambda mode: object())}
    return CaptionDataset([row], Processor(), [], [],
                          {"max_prompt_length": 3, "max_target_length": 3}, training,
                          record_presentations=record)


def test_evidence_does_not_change_caption_sampling_or_model_inputs():
    random.seed(12)
    plain = default_collate([dataset()[0] for _ in range(6)])
    next_plain = random.random()
    random.seed(12)
    recorded = default_collate([dataset(True)[0] for _ in range(6)])
    assert random.random() == next_plain
    assert set(plain) == {"input_ids", "attention_mask", "pixel_values", "labels"}
    for key in plain:
        assert torch.equal(plain[key], recorded[key])
    base = SimpleNamespace(parameters=lambda: iter([torch.zeros(1)]),
                           config=SimpleNamespace(pad_token_id=0, decoder_start_token_id=2))
    model = SimpleNamespace(base=base)
    kwargs, labels = model_inputs(plain, model)
    observed_kwargs, observed_labels = model_inputs(recorded, model)
    assert torch.equal(labels, observed_labels)
    for key, value in kwargs.items():
        if torch.is_tensor(value):
            assert torch.equal(value, observed_kwargs[key])
        else:
            assert value == observed_kwargs[key]
    rng_state = random.getstate()
    evidence = coco_input_evidence(recorded)
    assert random.getstate() == rng_state
    assert evidence["image_ids"] == [42] * 6
    assert evidence["target_token_ids"] == recorded["labels"].tolist()
    assert evidence["target_text"] == recorded["coco_target_text"]
    assert evidence["target_untruncated_lengths"] == [2] * 6
    assert evidence["target_truncated"] == [False] * 6
    assert evidence["source_valid_lengths"] == [2] * 6
    assert len(set(evidence["source_valid_sha256"])) == 1


def test_selection_targets_and_source_hash_are_exact():
    batch = default_collate([dataset(True, training=False)[0]])
    before = coco_input_evidence(batch)
    assert before["target_token_ids"] == [[ord("a"), 1, -100]]
    batch["input_ids"][0, 2] = 999  # Masked padding does not enter the valid-token hash.
    assert coco_input_evidence(batch) == before
    batch["input_ids"][0, 0] += 1
    assert coco_input_evidence(batch)["source_valid_sha256"] != before["source_valid_sha256"]


def test_requested_evidence_requires_image_identity():
    with pytest.raises(KeyError, match="coco_image_id"):
        coco_input_evidence(default_collate([dataset()[0]]))


def test_training_and_selection_evidence_preserve_final_weights(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    from jscc import training
    from jscc.config import load_config
    from jscc.data import TaskData
    from test_core import batch, toy_model

    config = load_config(Path(__file__).parents[1] / "configs/tasks/coco.yaml")
    config["run"].update(output_dir=str(tmp_path), name="coco-evidence-test")
    config["model"].update(device="cpu", dtype="float32")
    config["training"].update(max_steps=1, eval_every=1, gradient_accumulation=2,
                              validation_snrs=["no_noise", 6], validation_batches=1,
                              patience=None, save_steps=[], log_every=1)
    monkeypatch.setattr(training, "build_model", lambda _: (None, toy_model("enc")))
    def load(config, *_):
        item: dict[str, Any] = batch()
        if config["training"]["record_coco_presentations"]:
            item["coco_image_id"] = torch.tensor([42, 43])
            item["coco_target_text"] = ["sample caption one", "sample caption two"]
            item["coco_target_untruncated_length"] = (item["labels"] != -100).sum(dim=1)
            item["coco_target_truncated"] = torch.tensor([False, False])
        return TaskData(train=[item], validation=[item], ids={"test": [42, 43]})
    monkeypatch.setattr(training, "load_data", load)
    config["training"]["record_coco_presentations"] = False
    plain = training.train(config)
    assert not (plain / "coco_presentations.jsonl").exists()
    config["training"]["record_coco_presentations"] = True
    recorded = training.train(config)
    rows = [json.loads(line) for line in (recorded / "coco_presentations.jsonl").read_text().splitlines()]
    assert [row["phase"] for row in rows] == ["training", "training", "selection", "selection"]
    assert [row["microbatch"] for row in rows] == [0, 1, 0, 0]
    assert [row["snr_db"] for row in rows[2:]] == [None, 6.0]
    assert all(row["step"] == 1 and row["image_ids"] == [42, 43] for row in rows)
    for row in rows:
        assert row["target_token_ids"] == rows[0]["target_token_ids"]
    plain_state = torch.load(plain / "last.pt", weights_only=True)
    recorded_state = torch.load(recorded / "last.pt", weights_only=True)
    for name, value in plain_state["codec"].items():
        assert torch.equal(value, recorded_state["codec"][name])


def test_actual_truncation_is_recorded_without_replacing_sampled_target():
    class CharacterProcessor(Processor):
        def __call__(self, text=None, **kwargs):
            if "images" in kwargs:
                return super().__call__(text, **kwargs)
            assert text is not None
            ids = [ord(character) for character in text] + [1]
            if kwargs.get("truncation"):
                ids = ids[:kwargs["max_length"]]
            return {"input_ids": torch.tensor([ids]),
                    "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
    data = dataset(True, training=False)
    data.processor = CharacterProcessor()
    evidence = coco_input_evidence(default_collate([data[0]]))
    assert evidence["target_text"] == ["alpha"]
    assert evidence["target_token_ids"] == [[ord(c) for c in "alp"]]
    assert evidence["target_untruncated_lengths"] == [6]  # Characters plus EOS.
    assert evidence["target_truncated"] == [True]
