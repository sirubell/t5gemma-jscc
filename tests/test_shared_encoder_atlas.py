"""S0 gates use real tiny encoder layers, masks, channel and codec gradients."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import torch

from jscc.activation_replay import canonical_digest, identity
from jscc.models.channel import AWGNChannel
from jscc.models.codec import Codec
from jscc.models.split_model import SplitModel, stack_module
from model_helpers import tiny_backbone
from scripts import shared_encoder_atlas as atlas


@pytest.fixture
def config():
    return {"task": "hellaswag", "seed": 0,
            "model": {"name": "tiny", "revision": "fixed", "device": "cpu", "dtype": "float32"},
            "split": {"stack": "enc", "where": "after_layer", "index": 9},
            "codec": {"hidden_dim": 16, "bottleneck_dim": 8, "n_res_blocks": 1,
                      "activation": "gelu", "dropout": 0.0, "layernorm": "none", "snr_film": False},
            "channel": {"type": "awgn", "normalize_power": True, "clean_film_snr": 18.0},
            "training": {"deterministic_algorithms": True},
            "data": {"max_length": 512, "prompt_policy": {"mode": "five_shot", "source_max_length": 2048}}}


@pytest.fixture
def model(config):
    torch.manual_seed(9)
    return SplitModel(tiny_backbone(num_hidden_layers=20), Codec(16, config["codec"]),
                      AWGNChannel(), config["split"], config["channel"]).eval()


def frozen_views(config):
    rows = []
    for index in range(128):
        row = {"row_id": index, "group": "train" if index < 64 else "selection",
               "input_ids": [3, 4, 5] if index % 2 else [6, 7], "label_ids": [8, 9],
               "demo_ids": [200, 201, 202, 203, 204]}
        row.update(input_sha256=canonical_digest(row["input_ids"]),
                   target_sha256=canonical_digest(row["label_ids"]))
        row["view_sha256"] = canonical_digest(row)
        rows.append(row)
    return {"schema": atlas.VIEW_SCHEMA, "config_sha256": canonical_digest(config),
            "tokenizer": {"name": "tiny", "revision": "fixed"}, "pad_token_id": 0,
            "data_ids": {"train_rows": [*range(64), *range(200, 205)],
                         "validation_rows": list(range(64, 128)), "validation_split": "train"}, "rows": rows}


def batch(config, count=4):
    return atlas.batch_from_rows(frozen_views(config)["rows"][:count], 0)


def hook_counts(model):
    return [len(layer._forward_hooks) for layer in stack_module(model.base, "enc").layers]


def test_clean_atlas_stops_at19_restores_main_hook_and_excludes_codec(model, config):
    before = hook_counts(model)
    suffix = stack_module(model.base, "enc").norm.register_forward_pre_hook(lambda *_: pytest.fail("encoder suffix executed"))
    calls = []
    handle = model.codec.register_forward_hook(lambda *_: calls.append("codec"))
    # Fail if encode is reached: Codec.forward is not the communication entrypoint.
    original = model.codec.encode
    model.codec.encode = lambda *_: pytest.fail("clean atlas invoked codec")
    try:
        found = atlas.capture(model, batch(config), time.monotonic() + 30)
    finally:
        model.codec.encode = original
        handle.remove()
        suffix.remove()
    assert tuple(found) == atlas.SITES
    assert all(value.shape == (4, 3, 16) for value in found.values())
    assert hook_counts(model) == before
    assert not calls and model.channel_uses == {"hidden": 0, "memory": 0}
    assert not torch.equal(found[4], found[19])
    with pytest.raises(ValueError, match="only l4"):
        atlas.capture(model, batch(config), time.monotonic() + 30, (14,))


@pytest.mark.parametrize("site", atlas.SITES)
@pytest.mark.parametrize("snr", [None, 6.0])
def test_real_online_replay_gradients_noise_masks_counts_and_one_site(model, config, site, snr):
    before = hook_counts(model)
    sample = batch(config, 64)
    state = {key: tensor.clone() for key, tensor in model.codec.state_dict().items()}
    result = atlas.parity_pair(model, sample, site, snr, time.monotonic() + 30)
    assert result["passed"] and result["online_hook_calls"] == 1
    assert result["allocated_coordinates"] == 64 * 3 * 8
    assert result["valid_coordinates"] == 32 * 5 * 8
    assert len(result["awgn_draws"]) == (snr is not None)
    assert hook_counts(model) == before
    assert all(torch.equal(value, model.codec.state_dict()[key]) for key, value in state.items())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(row["max_abs_error"] == 0 for row in result["comparisons"])


def test_capture_and_parity_restore_hooks_on_deadline_and_codec_failure(model, config, monkeypatch):
    before = hook_counts(model)
    with pytest.raises(TimeoutError):
        atlas.capture(model, batch(config), time.monotonic() - 1)
    assert hook_counts(model) == before
    def fail(*_):
        raise RuntimeError("injected codec failure")
    monkeypatch.setattr(model.codec, "encode", fail)
    with pytest.raises(RuntimeError, match="injected"):
        atlas.parity_pair(model, batch(config), 4, None, time.monotonic() + 30)
    assert hook_counts(model) == before
    assert model.bypass is False


def test_nonfinite_activation_fails_and_restores_hooks(model, config):
    layers = stack_module(model.base, "enc").layers
    bad = layers[3].register_forward_hook(lambda _m, _a, output: output * float("nan"))
    before = hook_counts(model)
    try:
        with pytest.raises(FloatingPointError, match="activation"):
            atlas.capture(model, batch(config), time.monotonic() + 30)
        assert hook_counts(model) == before
    finally:
        bad.remove()


def test_views_require_disjoint_exact_rows_views_and_query_exclusion(config):
    views = frozen_views(config)
    assert len(atlas.validate_views(views, config)) == 128
    for mutate, message in [
        (lambda x: x["rows"].pop(), "128"),
        (lambda x: x["rows"][0]["input_ids"].append(1), "token hash"),
        (lambda x: x["rows"][0]["demo_ids"].__setitem__(0, 0), "excluding query"),
        (lambda x: x["data_ids"]["validation_rows"].append(0), "disjoint"),
        (lambda x: x["rows"][0].__setitem__("view_sha256", "wrong"), "view hash"),
    ]:
        changed = copy.deepcopy(views)
        mutate(changed)
        with pytest.raises(ValueError, match=message):
            atlas.validate_views(changed, config)


def test_completed_artifact_counts_corruption_and_unchanged_v1_guard(tmp_path, model, config, monkeypatch):
    monkeypatch.setattr(atlas, "validate_config", lambda _: None)  # Tiny architecture only in this CPU fixture.
    output = tmp_path / "s0"
    result = atlas.run(config, frozen_views(config), output,
                       loader=lambda _: (SimpleNamespace(pad_token_id=0), model))
    assert result["site_presentations"] == 384 and result["parity_pairs"] == 6
    assert result["optimizer_updates"] == 0
    assert len(atlas.read_complete(output)["shards"]) == 8
    parity = json.loads((output / "parity.json").read_text())
    assert all(row["passed"] for row in parity)
    shard = output / result["shards"][0]["file"]
    shard.write_bytes(shard.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="shard hash"):
        atlas.read_complete(output)
    changed = copy.deepcopy(config)
    changed["split"]["index"] = 4
    with pytest.raises(ValueError, match="restricted to HellaSwag enc_l9"):
        identity(changed, frozen_views(config)["data_ids"])


def test_deadline_includes_model_load_and_partial_receipt(tmp_path, config, monkeypatch):
    monkeypatch.setattr(atlas, "validate_config", lambda _: None)
    def load(_):
        time.sleep(0.02)
        return None, None
    output = tmp_path / "timeout"
    with pytest.raises(TimeoutError):
        atlas.run(config, frozen_views(config), output, max_seconds=0.01, loader=load)
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "FAILED_OR_PARTIAL" and status["optimizer_updates"] == 0
    assert not (output / "manifest.json").exists()


def test_cache_cap_has_no_complete_manifest_and_rejects_existing_directory(tmp_path, model, config, monkeypatch):
    monkeypatch.setattr(atlas, "validate_config", lambda _: None)
    output = tmp_path / "cap"
    with pytest.raises(RuntimeError, match="byte cap"):
        atlas.run(config, frozen_views(config), output, max_cache_bytes=1024,
                  loader=lambda _: (SimpleNamespace(pad_token_id=0), model))
    assert json.loads((output / "status.json").read_text())["status"] == "FAILED_OR_PARTIAL"
    assert not (output / "manifest.json").exists()
    with pytest.raises(FileExistsError):
        atlas.run(config, frozen_views(config), output)


def test_config_rejects_other_architecture(config):
    with pytest.raises(ValueError, match="adopted residual"):
        atlas.validate_config(config)


def test_prepare_views_uses_native_builder_fixed_ids_and_train_only_demos(tmp_path, config, monkeypatch):
    import datasets
    import transformers
    monkeypatch.setattr(atlas, "validate_config", lambda _: None)
    config["data"].update(name="fixture", revision="fixed")
    config["data"]["prompt_policy"].update(version="prompt-alignment-v1", seed=3)
    class Rows:
        _fingerprint = "fixture-fingerprint"
        def __init__(self, docs):
            self.docs = docs
        def select(self, ids):
            return Rows([self.docs[index] for index in ids])
        def __iter__(self):
            return iter(self.docs)
        def to_dict(self):
            return {key: [doc[key] for doc in self.docs] for key in self.docs[0]}
    raw = Rows([{"source_id": f"source-{index}", "activity_label": "test", "ctx_a": "a",
                 "ctx_b": "b", "ctx": "a b", "endings": ["one", "two", "three", "four"], "label": "0"}
                for index in range(205)])
    class Tokenizer:
        pad_token_id = 0
        def __call__(self, texts, **_):
            return {"input_ids": [[3, 4, 5] for _ in texts], "attention_mask": [[1, 1, 1] for _ in texts]}
    monkeypatch.setattr(datasets, "load_dataset", lambda *_args, **_kwargs: {"train": raw})
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: Tokenizer())
    views = atlas.prepare_views(config, frozen_views(config)["data_ids"], tmp_path / "prepared")
    assert [row["row_id"] for row in views["rows"]] == list(range(128))
    assert len(atlas.validate_views(views, config)) == 128
    assert all(row["row_id"] not in row["demo_ids"] for row in views["rows"])
    assert all(row["source_id"] not in row["demo_source_ids"] for row in views["rows"])
    assert json.loads((tmp_path / "prepared" / "status.json").read_text())["status"] == "COMPLETE"


def test_cli_hard_ceiling_kills_child_and_records_partial(tmp_path):
    output = tmp_path / "hard-timeout"
    result = subprocess.run([sys.executable, "-m", "scripts.shared_encoder_atlas",
                             "--config", "unused.yaml", "--views", "unused.json",
                             "--output", str(output), "--max-seconds", "0.01"],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    status = json.loads((output / "status.json").read_text())
    assert status["error"] == "HardWallTimeout" and status["status"] == "FAILED_OR_PARTIAL"
    assert status["optimizer_updates"] == 0 and not (output / "manifest.json").exists()
