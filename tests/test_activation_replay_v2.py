"""Complete sequence v2 contracts on tiny real encoder modules."""
import copy
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from jscc.activation_replay import (
    canonical_digest, capture_clean_sites, capture_sequences, capture_source_inventory,
    open_replay, producer_spec_v2, sequence_view_v2,
)
from jscc.models.channel import AWGNChannel
from jscc.models.codec import Codec
from jscc.models.split_model import SplitModel, resolve_encoder_site
from jscc.losses import reconstruction_loss
from model_helpers import tiny_backbone


@pytest.fixture
def model():
    torch.manual_seed(19)
    codec = {"hidden_dim": 16, "bottleneck_dim": 8, "n_res_blocks": 1,
             "activation": "gelu", "dropout": 0.0, "layernorm": "none", "snr_film": False}
    return SplitModel(tiny_backbone(), Codec(16, codec), AWGNChannel(),
                      {"stack": "enc", "where": "after_final_norm"},
                      {"type": "awgn", "normalize_power": True, "clean_film_snr": 18.0}).eval()


@pytest.fixture
def batch(model):
    labels = torch.tensor([[8, 9, -100], [10, 11, 12]])
    return {"input_ids": torch.tensor([[3, 4, 0, 0], [5, 6, 7, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]]),
            "labels": labels, "decoder_input_ids": model.base.prepare_decoder_input_ids_from_labels(labels=labels),
            "decoder_attention_mask": labels.ne(-100), "row_ids": torch.tensor([1, 2]),
            "batch_view_id": "batch0", "view_ids": ["v1", "v2"],
            "source_family_ids": ["f1", "f2"], "demo_ids": [[3], [3]],
            "position_roles": [["demo", "query", "padding", "padding"],
                               ["demo", "query", "query", "padding"]],
            "source_views": [{"row_id": i, "native_prefix": "reference", "demo_source_family_ids": ["f3"]}
                             for i in [1, 2]]}


@pytest.fixture
def spec(model, batch):
    site = resolve_encoder_site(model.base, model.split, "tiny-fixed")
    return producer_spec_v2(
        source_root=Path(__file__).resolve().parents[1],
        backbone={"model_revision": "tiny-fixed", "tokenizer_revision": "tiny-fixed",
                  "weights_sha256": "fixture-weights", "config_sha256": "fixture-config"},
        environment={"libraries": {"torch": torch.__version__}, "backend": "eager", "activation_dtype": "torch.float32"},
        data={"task": "hellaswag", "fingerprints": {"train": "fixture"}, "source_family_ids": ["f1", "f2", "f3"]},
        policy={"native_prefix": "reference", "site_authorization": {"sites": {"enc_fn": "trained"}}},
        sites=[{**asdict(site), "site_id": site.site_id, "split": site.split}],
        views=[sequence_view_v2(batch)], data_role="optimization",
        disjointness={"excluded_source_family_ids": ["selection-family"]})


def requirement(manifest):
    spec = manifest["producer"]
    return {"manifest_sha256": manifest["manifest_sha256"], "producer_sha256": manifest["producer_sha256"],
            "capability": "local_reconstruction", "data_role": "optimization", "sites": ["enc_fn"],
            "views": ["batch0"], "activation_dtype": spec["environment"]["activation_dtype"],
            "learner_source_sha256": "fixture-learner",
            "parity_receipt": {"status": "passed", "producer_sha256": manifest["producer_sha256"],
                               "capability": "local_reconstruction", "learner_source_sha256": "fixture-learner",
                               "evidence_sha256": "fixture-parity"}}


def test_complete_shapes_masks_native_targets_views_and_immutable_read(tmp_path, model, batch, spec):
    directory = tmp_path / "bank"
    manifest = capture_sequences(directory, spec, [batch], model)
    replay = open_replay(directory, requirement(manifest))
    restored = replay.read("batch0", "enc_fn")
    assert restored["activation"].shape == (2, 4, 16)
    for key, value in batch.items():
        if torch.is_tensor(value):
            assert torch.equal(restored[key], value)
        else:
            assert restored[key] == value
    assert manifest["shards"][0]["valid_vectors"] == 5
    assert manifest["shards"][0]["allocated_vectors"] == 8
    restored["input_ids"].zero_()
    assert torch.equal(replay.read("batch0", "enc_fn")["input_ids"], batch["input_ids"])
    public = replay.manifest
    public["producer"]["data_role"] = "heldout_after_freeze"
    assert replay.manifest["producer"]["data_role"] == "optimization"
    with pytest.raises(FileExistsError):
        capture_sequences(directory, spec, [batch], model)


@pytest.mark.parametrize("architecture,layernorm", [
    ("direct_affine", "none"), ("direct_outer_ln", "both"),
    ("residual_mlp", "none"), ("residual_mlp", "both"),
])
@pytest.mark.parametrize("snr", [None, 6.0])
def test_online_local_activation_reconstruction_codec_gradient_parity(
    tmp_path, model, batch, spec, snr, architecture, layernorm,
):
    # Capture identity is clean-backbone-only; all accepted learners reuse it.
    from jscc.models.codec import Codec
    model.codec = Codec(16, {**model.codec.config, "architecture": architecture,
                            "layernorm": layernorm,
                            "n_res_blocks": 2 if architecture == "residual_mlp" else 0})
    manifest = capture_sequences(tmp_path / "bank", spec, [batch], model)
    activation = open_replay(tmp_path / "bank", requirement(manifest)).read("batch0", "enc_fn")["activation"]
    mask = batch["attention_mask"]
    torch.manual_seed(103)
    with model.transmission(snr, encoder_mask=mask):
        model.base.get_encoder()(input_ids=batch["input_ids"], attention_mask=mask, return_dict=True)
    online_activation = model.activation.clone()
    online_reconstruction = model.reconstruction.clone()
    online = reconstruction_loss(model.reconstruction, model.activation, mask)
    online.backward()
    gradients = []
    for parameter in model.codec.parameters():
        assert parameter.grad is not None
        gradients.append(parameter.grad.clone())
    model.zero_grad(set_to_none=True)
    torch.manual_seed(103)
    with model.transmission(snr, encoder_mask=mask):
        reconstruction = model._transmit(activation, model.codec, "hidden", mask)
    local = reconstruction_loss(reconstruction, activation, mask)
    local.backward()
    torch.testing.assert_close(activation, online_activation, rtol=0, atol=0)
    torch.testing.assert_close(reconstruction, online_reconstruction, rtol=0, atol=0)
    torch.testing.assert_close(online, local, rtol=0, atol=0)
    for parameter, gradient in zip(model.codec.parameters(), gradients, strict=True):
        torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in model.base.parameters())


@pytest.mark.parametrize("field,value", [("data_role", "heldout_after_freeze"), ("capability", "functional_suffix"),
                                         ("sites", ["enc_l1"]), ("views", ["repacked"]),
                                         ("activation_dtype", "torch.bfloat16")])
def test_consumer_contract_cannot_weaken_or_relabel(tmp_path, model, batch, spec, field, value):
    manifest = capture_sequences(tmp_path / "bank", spec, [batch], model)
    needs = requirement(manifest)
    needs[field] = value
    with pytest.raises(ValueError):
        open_replay(tmp_path / "bank", needs)


def test_corrupt_shard_rejected_at_open_and_each_read(tmp_path, model, batch, spec):
    manifest = capture_sequences(tmp_path / "bank", spec, [batch], model)
    replay = open_replay(tmp_path / "bank", requirement(manifest))
    shard = tmp_path / "bank" / manifest["shards"][0]["file"]
    shard.write_bytes(shard.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        replay.read("batch0", "enc_fn")
    with pytest.raises(ValueError, match="checksum"):
        open_replay(tmp_path / "bank", requirement(manifest))


def test_incomplete_and_changed_native_targets_rejected(tmp_path, model, batch, spec):
    bad = copy.deepcopy(batch)
    bad["labels"][0, 0] = 14
    with pytest.raises(ValueError, match="ordered batch view"):
        capture_sequences(tmp_path / "bad", spec, [bad], model)
    assert (tmp_path / "bad" / "incomplete.json").exists()
    assert not (tmp_path / "bad" / "manifest.json").exists()
    with pytest.raises(FileNotFoundError):
        open_replay(tmp_path / "bad", {})
    with pytest.raises(StopIteration):
        capture_sequences(tmp_path / "short", spec, [], model)
    with pytest.raises(RuntimeError, match="byte cap"):
        capture_sequences(tmp_path / "cap", spec, [batch], model, max_cache_bytes=1)


def test_capture_dependency_closure_retains_bytes_and_excludes_only_unimported_learner(tmp_path):
    files = {"jscc/activation_replay.py": "from .models.wrapper import capture\n",
             "jscc/models/wrapper.py": "from ..data.prompt import tokenize\n",
             "jscc/data/prompt.py": "from ..preprocess import transform\n",
             "jscc/preprocess.py": "transform = 1\n", "jscc/runtime.py": "",
             "jscc/config.py": "", "jscc/presentation.py": "", "jscc/__init__.py": "",
             "jscc/training.py": "learner = 1\n", "uv.lock": "lock", "pyproject.toml": "",
             "configs/model.yaml": "capture: true\n"}
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    before = capture_source_inventory(tmp_path)
    assert "jscc/preprocess.py" in before
    assert "jscc/training.py" not in before
    (tmp_path / "jscc/training.py").write_text("learner = 2\n")
    assert capture_source_inventory(tmp_path) == before
    for name in ("jscc/preprocess.py", "jscc/data/prompt.py", "configs/model.yaml", "uv.lock"):
        path = tmp_path / name
        original = path.read_text()
        path.write_text(original + "\n")
        assert canonical_digest(capture_source_inventory(tmp_path)) != canonical_digest(before)
        path.write_text(original)
    (tmp_path / "jscc/models/wrapper.py").write_text("from ..training import learner\n")
    assert "jscc/training.py" in capture_source_inventory(tmp_path)
    for name, entry in before.items():
        assert bytes.fromhex(entry["content_hex"]) == (tmp_path / name).read_bytes() or name == "jscc/models/wrapper.py"


def test_source_archive_corruption_and_family_leaks_rejected(tmp_path, model, batch, spec):
    broken = copy.deepcopy(spec)
    broken["source_files"]["uv.lock"]["content_hex"] = "00"
    with pytest.raises(ValueError, match="source archive"):
        capture_sequences(tmp_path / "broken", broken, [batch], model)
    broken = copy.deepcopy(spec)
    broken["disjointness"]["excluded_source_family_ids"] = ["f3"]
    with pytest.raises(ValueError, match="demonstration"):
        capture_sequences(tmp_path / "leak", broken, [batch], model)


def test_cross_learner_reuse_requires_new_applicable_parity_receipt(tmp_path, model, batch, spec):
    manifest = capture_sequences(tmp_path / "bank", spec, [batch], model)
    needs = requirement(manifest)
    needs["learner_source_sha256"] = "new-learner-source"
    with pytest.raises(ValueError, match="parity receipt"):
        open_replay(tmp_path / "bank", needs)
    needs["parity_receipt"]["learner_source_sha256"] = "new-learner-source"
    assert torch.equal(open_replay(tmp_path / "bank", needs).read("batch0", "enc_fn")["input_ids"], batch["input_ids"])


def test_raw_last_block_and_final_norm_are_distinct(model, batch, spec):
    raw = resolve_encoder_site(model.base, {"stack": "enc", "where": "after_layer", "index": 1}, "tiny-fixed")
    sites = [{**asdict(raw), "site_id": raw.site_id, "split": raw.split}, *spec["sites"]]
    values = capture_clean_sites(model, batch, sites, {"sites": {"enc_l1": "diagnostic", "enc_fn": "trained"}})
    assert not torch.equal(values["enc_l1"], values["enc_fn"])
    assert model.split == {"stack": "enc", "where": "after_final_norm"}
    with pytest.raises(ValueError, match="freeze receipt"):
        capture_clean_sites(model, batch, spec["sites"], {"sites": {"enc_fn": "heldout"}})


def test_reference_prefix_must_match_native_preparation(tmp_path, model, batch, spec):
    changed = copy.deepcopy(batch)
    changed["decoder_input_ids"][:, 0] = 0
    spec["views"] = [sequence_view_v2(changed)]
    with pytest.raises(ValueError, match="native decoder"):
        capture_sequences(tmp_path / "prefix", spec, [changed], model)


def test_v1_source_guard_stays_exact_and_v2_loader_never_relabels(tmp_path, monkeypatch):
    from jscc import activation_replay
    from test_local_reconstruction import recipe, model_for, batch as legacy_batch

    config = recipe(tmp_path)
    directory = tmp_path / "legacy"
    manifest = activation_replay.write_cache(directory, config,
        {"train_rows": [1, 2], "validation_rows": [3], "validation_split": "train"},
        [legacy_batch()], model_for(config))
    original = (directory / "manifest.json").read_bytes()
    monkeypatch.setattr(activation_replay, "training_source_digest", lambda: "different-learner-source")
    with pytest.raises(ValueError, match="stale cache identity"):
        activation_replay.load_manifest(directory, config)
    with pytest.raises(ValueError, match="manifest identity"):
        open_replay(directory, {"producer_sha256": manifest["identity_sha256"]})
    assert (directory / "manifest.json").read_bytes() == original


def test_objective_validation_replay_cannot_be_used_for_optimization(tmp_path, model, batch, spec):
    spec["data_role"] = "objective_validation"
    manifest = capture_sequences(tmp_path / "validation", spec, [batch], model)
    needs = requirement(manifest)
    needs.update(capability="objective_validation", data_role="objective_validation")
    needs["parity_receipt"]["capability"] = "objective_validation"
    assert torch.equal(open_replay(tmp_path / "validation", needs).read("batch0", "enc_fn")["labels"], batch["labels"])
    needs["capability"] = "local_reconstruction"
    needs["parity_receipt"]["capability"] = "local_reconstruction"
    with pytest.raises(ValueError, match="optimization data"):
        open_replay(tmp_path / "validation", needs)


def test_objective_validation_capability_rejects_optimization_data(tmp_path, model, batch, spec):
    manifest = capture_sequences(tmp_path / "optimization", spec, [batch], model)
    needs = requirement(manifest)
    needs["capability"] = "objective_validation"
    needs["parity_receipt"]["capability"] = "objective_validation"
    with pytest.raises(ValueError, match="objective_validation data"):
        open_replay(tmp_path / "optimization", needs)


def test_runner_binds_replay_to_current_learner_and_exact_prepared_views(tmp_path, model, batch, spec):
    from jscc.baseline_protocol import open_baseline_replay, read_replay_batch, batch_identity
    site = resolve_encoder_site(model.base, model.split, "tiny-fixed")
    manifest = capture_sequences(tmp_path / "bank", spec, [batch], model)
    needs = requirement(manifest)
    options = dict(source_identity="fixture-learner", role="optimization", site=site,
                   backbone=spec["backbone"], dtype=torch.float32)
    replay = open_baseline_replay(tmp_path / "bank", needs, **options)
    ref = {"batch_view_id": "batch0", "view_sha256": batch_identity(batch)}
    assert torch.equal(read_replay_batch(replay, ref)["input_ids"], batch["input_ids"])
    with pytest.raises(ValueError, match="ordered stream"):
        read_replay_batch(replay, {**ref, "view_sha256": "other-online-view"})
    with pytest.raises(ValueError, match="consumer source/role"):
        open_baseline_replay(tmp_path / "bank", needs, **{**options, "source_identity": "new-learner"})
    with pytest.raises(ValueError, match="consumer source/role"):
        open_baseline_replay(tmp_path / "bank", needs, **{**options, "role": "objective_validation"})
    with pytest.raises(ValueError, match="backbone/site/precision"):
        open_baseline_replay(tmp_path / "bank", needs, **{**options, "backbone": {**spec["backbone"], "model_revision": "other"}})
