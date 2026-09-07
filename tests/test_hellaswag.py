"""HellaSwag collator and real HFLM likelihood path; no downloads."""
from typing import Any, cast

import torch

from jscc.data.hellaswag import Collator
from jscc.models.channel import AWGNChannel
from jscc.models.codec import Codec
from jscc.models.split_model import SplitModel
from model_helpers import tiny_backbone


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
