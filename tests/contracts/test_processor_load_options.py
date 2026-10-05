"""Model execution options must not become processor/tokenizer semantics."""

from copy import deepcopy

import pytest
import torch
import transformers

from contracts.test_model_build_config_identity import acquired_config
from shared.test_coco_loaded_tokenizer_identity import loaded as loaded

from jscc.models.split_model import build_model
from jscc.shared_inputs import _validate_coco_processor


@pytest.mark.parametrize("task", ["coco", "hellaswag"])
@pytest.mark.parametrize("attention", [None, "sdpa", "eager"])
@pytest.mark.parametrize("local_files_only", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_factory_routes_loader_options_without_mutating_config(
        monkeypatch, task, attention, local_files_only, dtype):
    config, expected_processor = acquired_config(monkeypatch, task, "enc", None, 32)
    config["model"].update(name="pinned-backbone", revision="immutable-revision", dtype=dtype)
    config["model"].pop("attn_implementation", None)
    if attention is not None:
        config["model"]["attn_implementation"] = attention
    authored = deepcopy(config)
    processor_cls = transformers.AutoProcessor if task == "coco" else transformers.AutoTokenizer
    model_loader = transformers.AutoModelForSeq2SeqLM.from_pretrained
    calls = []

    def load_processor(name, **kwargs):
        calls.append(("processor", name, kwargs))
        return expected_processor

    def load_model(name, **kwargs):
        calls.append(("model", name, kwargs))
        return model_loader(name, **kwargs)

    def wrong_processor(*args, **kwargs):
        pytest.fail("factory selected the wrong task processor")

    other_cls = transformers.AutoTokenizer if task == "coco" else transformers.AutoProcessor
    monkeypatch.setattr(processor_cls, "from_pretrained", load_processor)
    monkeypatch.setattr(other_cls, "from_pretrained", wrong_processor)
    monkeypatch.setattr(transformers.AutoModelForSeq2SeqLM, "from_pretrained", load_model)

    processor, wrapper = build_model(config, local_files_only=local_files_only)

    common: dict[str, str | bool] = {"revision": "immutable-revision"}
    if local_files_only:
        common["local_files_only"] = True
    model_options = {**common, "dtype": getattr(torch, dtype), "low_cpu_mem_usage": True}
    if attention is not None:
        model_options["attn_implementation"] = attention
    assert calls == [("processor", "pinned-backbone", common),
                     ("model", "pinned-backbone", model_options)]
    assert processor is expected_processor
    assert config == authored
    assert wrapper.base is not None
    assert wrapper.codec.encoder[0].in_features == 32
    assert all(parameter.dtype == torch.float32 for parameter in wrapper.codec.parameters())
    assert all(parameter.device.type == "cpu" for parameter in wrapper.parameters())


@pytest.mark.parametrize("attention", [None, "sdpa"])
def test_factory_real_local_processor_matches_asset_guard(monkeypatch, loaded, attention):
    assets, _, _ = loaded
    processor_loader = transformers.AutoProcessor.from_pretrained
    tokenizer_loader = transformers.AutoTokenizer.from_pretrained
    config, _ = acquired_config(monkeypatch, "coco", "enc", None, 32)
    config["model"].update(name=str(assets), revision="immutable-revision")
    config["model"].pop("attn_implementation", None)
    if attention is not None:
        config["model"]["attn_implementation"] = attention
    authored = deepcopy(config)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", processor_loader)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer_loader)

    processor, _ = build_model(config, local_files_only=True)

    _validate_coco_processor(processor, assets)
    assert "attn_implementation" not in processor.tokenizer.init_kwargs
    assert config == authored


def test_asset_guard_rejects_attention_option_in_real_processor_metadata(loaded):
    assets, _, _ = loaded
    processor = transformers.AutoProcessor.from_pretrained(
        assets, local_files_only=True, attn_implementation="sdpa")
    assert processor.tokenizer.init_kwargs["attn_implementation"] == "sdpa"

    with pytest.raises(ValueError, match="tokenizer differs"):
        _validate_coco_processor(processor, assets)
