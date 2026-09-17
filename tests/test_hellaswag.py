"""HellaSwag collator and real HFLM likelihood path; no downloads."""
from typing import Any, cast

import torch

from jscc.data.hellaswag import Collator, selection_ids
from jscc.models.channel import AWGNChannel
from jscc.models.codec import Codec
from jscc.models.split_model import SplitModel
from model_helpers import tiny_backbone


def test_selection_is_disjoint_reproducible_and_independent_of_training_limit():
    full = selection_ids(100, 12, None, 0)
    small = selection_ids(100, 12, 8, 0)
    assert full == selection_ids(100, 12, None, 0)
    assert len(full["train_rows"]) == 88
    assert set(full["train_rows"]).isdisjoint(full["validation_rows"])
    assert set(full["train_rows"]) | set(full["validation_rows"]) == set(range(100))
    assert small["validation_rows"] == full["validation_rows"]
    assert small["train_rows"] == full["train_rows"][:8]
    assert full["validation_split"] == "train"


def test_loader_uses_train_holdout_and_preserves_historical_ids(monkeypatch):
    from datasets import Dataset, DatasetDict
    from jscc.data.hellaswag import load_data
    raw = DatasetDict({name: Dataset.from_list([
        {"ctx": str(offset + i), "endings": ["1", "2"], "label": 0} for i in range(8)])
        for name, offset in (("train", 10), ("validation", 100))})
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: raw)
    class Tokenizer:
        pad_token_id = 0

        def __call__(self, texts, **kwargs):
            return {"input_ids": [[int(text)] for text in texts],
                    "attention_mask": [[1] for _ in texts]}
    config = {"data": {"name": "fixture", "revision": "fixed", "num_train": None,
                       "num_validation": 2, "max_length": 8, "num_workers": 0},
              "model": {"device": "cpu"}, "training": {"batch_size": 2}}
    new = load_data(config, Tokenizer())
    assert all(int(value) < 100 for batch in new.validation for value in batch["input_ids"].flatten())
    old_ids = {"train_rows": [0, 1], "validation_rows": [0, 1]}
    old = load_data(config, Tokenizer(), old_ids)
    assert next(iter(old.validation))["input_ids"].tolist() == [[100], [101]]
    assert old.ids == old_ids


def test_hellaswag_collator_masks_only_padding():
    result = Collator(0)([{"input_ids": [1, 2], "attention_mask": [1, 1], "label_ids": [3]},
                          {"input_ids": [4], "attention_mask": [1], "label_ids": [5, 6]}])
    assert result["input_ids"].tolist() == [[1, 2], [4, 0]]
    assert result["labels"].tolist() == [[3, -100], [5, 6]]


def test_hellaswag_harness_scores_through_real_codec():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    from lm_eval.api.instance import Instance
    from lm_eval.api.registry import get_model
    token_impl = Tokenizer(WordLevel({"[PAD]": 0, "[EOS]": 1, "[BOS]": 2, "[UNK]": 3,
                                      "a": 4, "person": 5, "runs": 6, "sleeps": 7}, unk_token="[UNK]"))
    token_impl.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=token_impl, pad_token="[PAD]",
                                        eos_token="[EOS]", bos_token="[BOS]", unk_token="[UNK]")
    base = tiny_backbone()
    codec = Codec(16, {"hidden_dim": 16, "bottleneck_dim": 8, "n_res_blocks": 1,
                       "activation": "gelu", "layernorm": "both", "snr_film": False})
    model = SplitModel(base, codec, AWGNChannel(), {"stack": "enc", "where": "after_final_norm"},
                       {"normalize_power": True, "clean_film_snr": 18.0}).eval()
    harness_class = cast(type[Any], get_model("hf"))
    adapter = harness_class(pretrained=base, tokenizer=tokenizer, backend="seq2seq", batch_size=2, max_length=16)
    requests = [Instance(request_type="loglikelihood", doc={}, arguments=("a person", ending), idx=i)
                for i, ending in enumerate((" runs", " sleeps"))]
    with model.transmission(None):
        scores = adapter.loglikelihood(requests)
    assert len(scores) == 2 and all(torch.isfinite(torch.tensor(score[0])) for score in scores)
    assert model.reconstruction is not None
