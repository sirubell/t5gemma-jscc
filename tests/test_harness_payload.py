"""Payload validity must follow actual harness continuations, not padded values."""
from typing import Any, cast

import pytest
import torch

from jscc.models.channel import AWGNChannel
from jscc.models.codec import Codec
from jscc.models.split_model import SplitModel
from model_helpers import tiny_backbone


def fixture_model():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    impl = Tokenizer(WordLevel({"[PAD]": 0, "[EOS]": 1, "[BOS]": 2, "[UNK]": 3, "x": 4}, unk_token="[UNK]"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=impl, pad_token="[PAD]", eos_token="[EOS]", bos_token="[BOS]", unk_token="[UNK]")
    codec = Codec(16, {"hidden_dim": 16, "bottleneck_dim": 4, "n_res_blocks": 1,
                       "activation": "gelu", "layernorm": "none", "snr_film": False})
    model = SplitModel(tiny_backbone(), codec, AWGNChannel(),
                       {"stack": "dec", "where": "after_layer", "index": 0},
                       {"normalize_power": True, "clean_film_snr": 18.0}).eval()
    return tokenizer, model


def make_adapter(model, tokenizer):
    from jscc.harness_payload import payload_harness_class
    from lm_eval.api.registry import get_model
    cls = payload_harness_class(cast(type[Any], get_model("hf")))
    return cls(communication_model=model, pretrained=model.base, tokenizer=tokenizer,
               backend="seq2seq", batch_size=2, max_length=16)


@pytest.mark.parametrize("snr", [None, -6.0])
def test_real_harness_counts_and_scores(snr):
    tokenizer, model = fixture_model()
    adapter = make_adapter(model, tokenizer)
    # A real zero token INSIDE a continuation must count, as must its EOS.
    requests = [(("c1", "a"), [2, 4, 1], [4, 0, 1]),
                (("c2", "b"), [2, 4, 4, 1], [4, 1]),
                (("c3", "c"), [2, 1], [1])]
    from lm_eval.api.registry import get_model
    legacy = cast(Any, get_model("hf"))(pretrained=model.base, tokenizer=tokenizer,
                                       backend="seq2seq", batch_size=2, max_length=16)
    torch.manual_seed(12)
    with model.transmission(snr):
        before = legacy._loglikelihood_tokens(requests, disable_tqdm=True)
        allocated = dict(model.channel_uses_allocated)
        memory = model.valid_payload_counts['memory']
    torch.manual_seed(12)
    with model.transmission(snr):
        after = adapter._loglikelihood_tokens(requests, disable_tqdm=True)
        assert model.valid_payload_counts['hidden'] == 6 * 4
        assert model.channel_uses_allocated == allocated
        assert model.valid_payload_counts['memory'] == memory
    assert after == before
    assert model._decoder_valid_mask is None


def test_payload_scope_does_not_reset_counters_and_restores_after_exception():
    _, model = fixture_model()
    with model.transmission(None):
        model.channel_uses_valid['hidden'] = 9
        prior = torch.ones(1, 3, dtype=torch.bool)
        with model.decoder_payload_validity(prior):
            with pytest.raises(RuntimeError):
                with model.decoder_payload_validity(torch.ones(2, 2, dtype=torch.bool)):
                    raise RuntimeError('injected')
            assert model._decoder_valid_mask is prior
        assert model._decoder_valid_mask is None
        assert model.channel_uses_valid['hidden'] == 9


def test_ambiguous_actual_lengths_fail_instead_of_guessing_from_pad():
    from jscc.harness_payload import ContinuationLengths
    index = ContinuationLengths([(("x", "a"), [2, 4], [4]),
                                 (("x", "b"), [2, 4], [4, 0])], 16)
    with pytest.raises(ValueError, match='ambiguous'):
        index.mask(torch.tensor([[2, 4]]), torch.ones(1, 2), torch.tensor([[4, 0]]))
