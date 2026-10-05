"""CPU semantic checks for retained COCO scientific helpers."""

import hashlib
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from model_helpers import tiny_backbone
from jscc.coco_diagnostic import (
    MAX_REQUESTS, RequestBudget, diagnostic_inputs, forward_endpoint, prefix_inputs, prefix_variants,
    prepare_pair, score_logits, teacher_student_kl,
)
from jscc.models.channel import AWGNChannel
from jscc.models.codec import Codec
from jscc.models.split_model import SplitModel


class Tokenizer:
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        mapping = {"a": 5, "b": 6, ".": 7, "\n": 8}
        ids = [mapping.get(char, 9) for char in text]
        if add_special_tokens:
            ids += [self.eos_token_id]
        return {"input_ids": torch.tensor([ids]) if return_tensors == "pt" else ids}


class Processor:
    tokenizer = Tokenizer()

    def __call__(self, images, text, **_kwargs):
        # One or two crops according to pixel value, preserving image order.
        crops = []
        for image in images:
            number = image.getpixel((0, 0))[0]
            crops.extend([torch.full((3, 8, 8), number, dtype=torch.float32)] * (1 + number % 2))
        return {"input_ids": torch.tensor([[2, 3, 4]]),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
                "pixel_values": torch.stack(crops)}


def image(value):
    return Image.new("RGB", (8, 8), (value, 0, 0))


def test_query_only_pair_keeps_demo_pixels_and_text_with_variable_crops():
    processor = Processor()
    pair, segments, prompt = prepare_pair(processor, [image(n) for n in range(4)],
                                          ["a", "b", "a.", "b."], image(4), image(5))
    assert prompt.endswith("<start_of_image>")
    assert [item["crop_count"] for item in segments[0]] == [1, 2, 1, 2, 1]
    assert [item["crop_count"] for item in segments[1]] == [1, 2, 1, 2, 2]
    assert torch.equal(pair[0]["input_ids"], pair[1]["input_ids"])
    assert torch.equal(pair[0]["pixel_values"][:6], pair[1]["pixel_values"][:6])
    with pytest.raises(ValueError, match="identical query pixels"):
        prepare_pair(processor, [image(n) for n in range(4)], ["a"] * 4, image(4), image(4))


def test_prefix_native_shift_scores_position_after_complete_caption():
    tokenizer = Tokenizer()
    base = tiny_backbone()
    decoder, lexical = prefix_inputs(tokenizer, base, "ab.")
    assert decoder is not None
    assert isinstance(lexical, torch.Tensor)
    assert lexical.tolist() == [[5, 6, 7]]
    assert decoder.tolist() == [[2, 5, 6, 7]]  # T5Gemma-2 native BOS2.
    assert prefix_variants("ab.") == {"complete": ("ab.", False),
                                       "without_terminal_period": ("ab", False),
                                       "plus_newline": ("ab.\n", False)}
    assert prefix_variants("ab")["without_terminal_period"] == ("ab", True)
    assert prefix_inputs(tokenizer, base, "a" * 65)[0] is None
    class EOSTokenizer:
        eos_token_id = 1
        def __call__(self, *_args, **_kwargs):
            return {"input_ids": torch.tensor([[5, 1]])}
    with pytest.raises(ValueError, match="EOS"):
        prefix_inputs(EOSTokenizer(), base, "a")


def test_endpoint_logits_nonfinite_and_newline_semantics():
    logits = torch.tensor([0.0, 3.0, 1.0, 2.0])
    values = score_logits(logits, eos_id=1, newline_ids=[3, 2])
    assert values["eos_rank"] == 1
    assert values["newline_first_token_rank"] == 2
    assert values["newline_sequence_probability"] is None
    assert score_logits(logits, 1, [3])["newline_sequence_probability"] == values["newline_first_token_probability"]
    assert teacher_student_kl(logits, logits) == pytest.approx(0.0, abs=1e-7)
    with pytest.raises(FloatingPointError):
        score_logits(torch.tensor([0.0, float("nan")]), 0, [1])


def test_diagnostic_inputs_casts_only_pixels_and_preserves_original_inputs():
    model = SimpleNamespace(base=torch.nn.Linear(2, 2, dtype=torch.bfloat16))
    inputs = {"input_ids": torch.tensor([[2, 3]], dtype=torch.int64),
              "attention_mask": torch.tensor([[1, 0]], dtype=torch.int32),
              "pixel_values": torch.tensor([0.125, 0.5], dtype=torch.float32)}
    original = dict(inputs)
    snapshots = {key: value.clone() for key, value in inputs.items()}

    source = diagnostic_inputs(model, inputs)

    assert source is not inputs
    assert source.keys() == inputs.keys()
    assert all(value.device == model.base.weight.device for value in source.values())
    assert source["pixel_values"].dtype == torch.bfloat16
    assert torch.equal(source["pixel_values"], inputs["pixel_values"].bfloat16())
    for key in ("input_ids", "attention_mask"):
        assert source[key].dtype == inputs[key].dtype
        assert torch.equal(source[key], inputs[key])
    for key, value in inputs.items():
        assert value is original[key]
        assert value.dtype == snapshots[key].dtype
        assert torch.equal(value, snapshots[key])


def test_decoder_receiver_memory_is_used_in_real_tiny_full_forward():
    config = {"hidden_dim": 16, "bottleneck_dim": 8, "n_res_blocks": 1,
              "activation": "gelu", "layernorm": "none", "snr_film": False}
    model = SplitModel(tiny_backbone(3), Codec(16, config), AWGNChannel(),
                       {"stack": "dec", "where": "after_layer", "index": 0},
                       {"normalize_power": True, "clean_film_snr": 18.0}).eval()
    inputs = {"input_ids": torch.tensor([[2, 3, 4]]),
              "attention_mask": torch.ones(1, 3, dtype=torch.long),
              "pixel_values": torch.empty((0, 3, 8, 8))}
    # Text-only tiny input exercises the actual SplitModel full-forward path.
    inputs.pop("pixel_values")
    prefix = torch.tensor([[2, 5]])
    coded = forward_endpoint(model, inputs, prefix)
    assert coded.shape == (32,)
    assert model.channel_uses["memory"] > 0
    vanilla = forward_endpoint(model, inputs, prefix, vanilla=True)
    assert vanilla.shape == (32,)
    assert model.channel_uses["memory"] == 0


def test_request_attempts_count_failures_and_deadline_before_new_work():
    ticks = iter([0.0, 2.0, 8.1])
    budget = RequestBudget(10, 2, now=lambda: next(ticks))
    assert budget.begin() == 1
    assert budget.attempts == 1 and budget.completed == 0
    with pytest.raises(TimeoutError, match="Deadline reserve"):
        budget.begin()
    assert budget.attempts == 1
    budget.complete(decoder_calls=1, encoder_calls=1)
    assert (budget.completed, budget.decoder_calls, budget.encoder_calls) == (1, 1, 1)
    assert MAX_REQUESTS == 1024


def test_bfloat16_feature_signatures_hash_native_bits():
    """Job18383 reached feature logging, where NumPy rejects BF16 tensors."""
    from types import SimpleNamespace
    from jscc.coco_diagnostic import feature_signatures, tensor_sha256

    values = torch.tensor([[1.0, -2.0, 3.5]], dtype=torch.bfloat16)
    # BF16 words 0x3f80,0xc000,0x4060 in native little-endian order.
    import struct
    raw = struct.pack('=HHH', 0x3f80, 0xc000, 0x4060)
    expected = hashlib.sha256(b'(1, 3)torch.bfloat16' + raw).hexdigest()
    model = SimpleNamespace(activation=values, reconstruction=values.clone(),
                            memory_activation=None, memory_reconstruction=None)
    signatures = feature_signatures(model)
    assert signatures['boundary_input'] == expected
    assert signatures['boundary_reconstruction'] == expected
    assert signatures['receiver_memory_input'] is None
    assert tensor_sha256(values.reshape(3)) != expected
    assert tensor_sha256(values.float()) != expected


def test_tensor_hash_preserves_legacy_float32_and_noncontiguous_values():
    from jscc.coco_diagnostic import tensor_sha256

    values = torch.tensor([[1.0, -2.0], [3.5, 0.0]], dtype=torch.float32).T
    expected = hashlib.sha256(str(tuple(values.shape)).encode() +
                              str(values.dtype).encode() +
                              values.contiguous().numpy().tobytes()).hexdigest()
    assert tensor_sha256(values) == expected
    scalar = torch.tensor(1.0, dtype=torch.bfloat16)
    assert len(tensor_sha256(scalar)) == 64


def test_request_budget_keeps_fixed_attempt_ceiling_when_work_fails():
    budget = RequestBudget(10, 2, now=lambda: 0.0)
    for attempt in range(1, 1025):
        assert budget.begin() == attempt
    assert budget.attempts == MAX_REQUESTS and budget.completed == 0
    with pytest.raises(RuntimeError, match="Logical request ceiling"):
        budget.begin()
    assert budget.attempts == MAX_REQUESTS
