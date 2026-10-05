"""Acquired decoder width must not rewrite the ordinary bound configuration."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from jscc.config import load_config
from jscc.models.split_model import build_model
from jscc.ordinary_runs import _checkpoint

REPO = Path(__file__).resolve().parents[2]


class AcquiredBackbone(nn.Module):
    """Only acquisition is mocked; construct the real Codec and SplitModel."""

    def __init__(self, hidden_size):
        super().__init__()
        self.config = SimpleNamespace(decoder=SimpleNamespace(hidden_size=hidden_size))
        self.weight = nn.Parameter(torch.ones(1))
        self.encoder = nn.Module()
        self.decoder = nn.Module()
        for stack in (self.encoder, self.decoder):
            layer = nn.Identity()
            layer.self_attn = nn.Identity()
            stack.layers = nn.ModuleList([layer])
            stack.norm = nn.Identity()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder


def acquired_config(monkeypatch, task, stack, input_dim, hidden_size):
    import transformers

    config = load_config(REPO / f"configs/experiments/ordinary/{task}.yaml")
    config["model"].update(device="cpu", dtype="float32", sdpa_backend_policy="auto",
                           numerical_policy="native")
    config["split"] = {"stack": stack, "where": "before_first_layer"}
    config["codec"].update(input_dim=input_dim, architecture="direct_affine",
                           layernorm="none", n_res_blocks=0, snr_film=False,
                           bottleneck_dim=8, dropout=0.0)
    config["codec"].pop("memory", None)
    processor = object()
    base = AcquiredBackbone(hidden_size)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *a, **k: processor)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: processor)
    monkeypatch.setattr(transformers.AutoModelForSeq2SeqLM, "from_pretrained", lambda *a, **k: base)
    return config, processor


@pytest.mark.parametrize("task", ["coco", "hellaswag"])
@pytest.mark.parametrize("stack", ["enc", "dec"])
@pytest.mark.parametrize("input_dim,hidden_size", [(None, 1152), (1152, 1152), (64, 1152), (None, 32)])
def test_build_preserves_bound_config_and_uses_acquired_width(
        monkeypatch, tmp_path, task, stack, input_dim, hidden_size):
    config, expected_processor = acquired_config(monkeypatch, task, stack, input_dim, hidden_size)
    bound = copy.deepcopy(config)

    processor, model = build_model(config, local_files_only=True)

    assert processor is expected_processor
    assert config == bound
    assert model.codec.encoder[0].in_features == hidden_size
    assert model.codec.decoder[-1].out_features == hidden_size
    if stack == "dec":
        assert model.memory_codec is not None
        assert model.memory_codec.encoder[0].in_features == hidden_size
        assert model.memory_codec.decoder[-1].out_features == hidden_size
    hidden = torch.ones(1, 2, hidden_size)
    assert model.codec.decode(model.codec.encode(hidden), None).shape == hidden.shape
    checkpoint = tmp_path / "best.pt"
    torch.save({"config": config, "step": 1}, checkpoint)
    assert _checkpoint(checkpoint, role="ordinary-best", expected_config=bound)["step"] == 1
    changed = copy.deepcopy(bound)
    changed["codec"]["input_dim"] = hidden_size + 1
    with pytest.raises(ValueError, match="configuration/step differs"):
        _checkpoint(checkpoint, role="ordinary-best", expected_config=changed)


@pytest.mark.parametrize("failure", ["codec", "split"])
def test_failed_setup_preserves_bound_config(monkeypatch, failure):
    config, _ = acquired_config(monkeypatch, "hellaswag", "enc", None, 1152)
    if failure == "codec":
        config["codec"]["architecture"] = "invalid"
        message = "unsupported codec.architecture"
    else:
        config["split"]["where"] = "invalid"
        message = "unknown split.where"
    bound = copy.deepcopy(config)

    with pytest.raises(ValueError, match=message):
        build_model(config, local_files_only=True)

    assert config == bound
