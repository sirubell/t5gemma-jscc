import copy
from dataclasses import asdict

import pytest
import torch

from jscc.models.channel import AWGNChannel
from jscc.models.codec import Codec
from jscc.models.split_model import SplitModel, stack_module, resolve_encoder_site
from jscc.sharing_geometry_capture import capture_geometry_atlas, capture_geometry_batch
from model_helpers import tiny_backbone


@pytest.fixture
def model():
    torch.manual_seed(4)
    codec = {
        "hidden_dim": 16,
        "bottleneck_dim": 8,
        "n_res_blocks": 0,
        "activation": "gelu",
        "dropout": 0.0,
        "layernorm": "none",
        "snr_film": False,
    }
    return SplitModel(
        tiny_backbone(num_hidden_layers=26),
        Codec(16, codec),
        AWGNChannel(),
        {"stack": "enc", "where": "after_layer", "index": 9},
        {"normalize_power": True, "clean_film_snr": 18.0},
    ).eval()


def batch(prefix="train"):
    return {
        "input_ids": torch.tensor([[3, 4, 5, 6], [7, 8, 9, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
        "query_ids": [prefix + "-0", prefix + "-1"],
        "view_ids": ["native-a", "native-b"],
        "token_positions": torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
        "token_roles": [
            ["demo", "demo", "query", "query"],
            ["demo", "query", "query", "padding"],
        ],
        "provenance": {
            "synthetic": True,
            "fixture": "complete-four-position-native-input",
        },
    }


def hook_counts(model):
    stack = stack_module(model.base, "enc")
    return [len(m._forward_hooks) for m in [*stack.layers, stack.norm]]


def test_real_26_layer_capture_has_raw_final_and_unchanged_native_layout(
    model, monkeypatch
):
    inputs = batch()
    before = copy.deepcopy(inputs)
    counts = hook_counts(model)
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    previous_activation = torch.ones(1)
    model.activation = previous_activation
    model.channel_uses = {"hidden": 123, "memory": 0}
    monkeypatch.setattr(model.codec, "encode", lambda *_: pytest.fail("codec reached"))
    monkeypatch.setattr(
        model.channel, "forward", lambda *_: pytest.fail("channel reached")
    )
    result = capture_geometry_batch(model, inputs, model_revision="synthetic-fixed")
    assert set(result["records"]) == {"enc_l9", "enc_l19", "enc_l25", "enc_fn"}
    for records in result["records"].values():
        for i, record in enumerate(records):
            assert record["activations"].shape == (4, 16)
            assert not record["activations"].requires_grad
            assert torch.equal(record["mask"], inputs["attention_mask"][i])
            assert torch.equal(record["token_positions"], inputs["token_positions"][i])
            assert record["token_roles"] == inputs["token_roles"][i]
    assert result["receipt"]["raw_final_site"] == "enc_l25"
    assert result["receipt"]["resolved_sites"]["enc_l25"]["layer_count"] == 26
    assert result["receipt"]["provenance"] == before["provenance"]
    assert not torch.equal(
        result["records"]["enc_l25"][0]["activations"],
        result["records"]["enc_fn"][0]["activations"],
    )
    with torch.no_grad():
        expected_final = stack_module(model.base, "enc").norm(
            result["records"]["enc_l25"][0]["activations"]
        )
    torch.testing.assert_close(
        expected_final, result["records"]["enc_fn"][0]["activations"]
    )
    assert hook_counts(model) == counts
    assert model.activation is previous_activation
    assert model.channel_uses == {"hidden": 123, "memory": 0}
    assert model.bypass is False
    assert all(not m.training for m in model.base.modules())
    assert all(
        torch.equal(value, model.state_dict()[key]) for key, value in weights.items()
    )
    assert all(p.grad is None for p in model.parameters())
    assert torch.equal(inputs["input_ids"], before["input_ids"])
    assert torch.equal(inputs["attention_mask"], before["attention_mask"])


def test_heldout_gate_precedes_any_encoder_work(model, monkeypatch):
    encoder = model.base.get_encoder()
    monkeypatch.setattr(
        encoder, "forward", lambda **_: pytest.fail("encoder reached before freeze")
    )
    with pytest.raises(ValueError, match="file-backed freeze"):
        capture_geometry_batch(
            model, batch(), model_revision="synthetic", include_heldout=True
        )
    with pytest.raises((ValueError, KeyError)):
        capture_geometry_batch(
            model,
            batch(),
            model_revision="synthetic",
            include_heldout=True,
            freeze_reference={"complete": True},
            obligation="geometry:atlas",
        )


def test_postfreeze_atlas_verifies_before_every_capture_and_analysis(
    model, monkeypatch
):
    import jscc.sharing_geometry_capture as module

    events = []
    reference = {"fixture": "verified-by-test-double"}

    def verify(value, *, obligation):
        assert value == reference and obligation == "geometry:atlas"
        events.append("verify")
        return {
            "obligation_plan": {"geometry": [obligation]},
            "synthetic": True,
            "site_policy": capture_policy(model),
        }

    monkeypatch.setattr(module, "verify_study_freeze", verify)
    handle = model.base.get_encoder().register_forward_pre_hook(
        lambda *_: events.append("encoder")
    )
    try:
        result = capture_geometry_atlas(
            model,
            {"train": [batch()], "selection": [batch("selection")]},
            model_revision="synthetic-fixed",
            include_heldout=True,
            freeze_reference=reference,
            obligation="geometry:atlas",
            sample_cap=5,
        )
    finally:
        handle.remove()
    assert events == ["verify", "encoder", "verify", "encoder", "verify"]
    for partition in ("train", "selection"):
        analysis = result["analysis"]["partitions"][partition]
        assert analysis["actual_query_count"] == 2
        assert analysis["full_valid_rows"] == 7
        assert analysis["sampled_rows"] == 5
        assert set(analysis["sites"]) == {
            "enc_l9",
            "enc_l14",
            "enc_l19",
            "enc_l25",
            "enc_fn",
        }
        assert result["capture_receipts"][partition][0]["freeze_reference"] == reference
        assert result["capture_receipts"][partition][0]["phase"] == "post_freeze"


def test_exception_restores_hooks_masks_condition_and_counts(model):
    stack = stack_module(model.base, "enc")
    counts = hook_counts(model)
    sentinel = torch.ones(2, 4)
    model._encoder_valid_mask = sentinel
    model.snr_db = 6
    model.channel_uses_valid = {"hidden": 17, "memory": 0}

    def fail(*_):
        raise RuntimeError("fixture encoder failure")

    handle = stack.layers[20].register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="fixture encoder failure"):
            capture_geometry_batch(model, batch(), model_revision="synthetic")
    finally:
        handle.remove()
    assert hook_counts(model) == counts
    assert model._encoder_valid_mask is sentinel
    assert model.snr_db == 6 and not model.bypass
    assert model.channel_uses_valid == {"hidden": 17, "memory": 0}
    assert model._forward_active == 0


def test_bad_metadata_and_leakage_fail_before_capture(model, monkeypatch):
    monkeypatch.setattr(
        model.base.get_encoder(), "forward", lambda **_: pytest.fail("encoder reached")
    )
    inputs = batch()
    inputs["token_roles"][0][0] = "unknown"
    with pytest.raises(ValueError, match="labeled"):
        capture_geometry_batch(model, inputs, model_revision="synthetic")
    with pytest.raises(ValueError, match="disjoint"):
        capture_geometry_atlas(
            model,
            {"train": [batch()], "selection": [batch()]},
            model_revision="synthetic",
        )


def capture_policy(model):
    result = {}
    for name in ("enc_l9", "enc_l19", "enc_fn", "enc_l14"):
        spec = (
            {"stack": "enc", "where": "after_final_norm"}
            if name == "enc_fn"
            else {"stack": "enc", "where": "after_layer", "index": int(name[5:])}
        )
        result[name] = {
            "site": asdict(resolve_encoder_site(model.base, spec, "synthetic-fixed")),
            "role": "heldout_after_freeze" if name == "enc_l14" else "trained",
        }
    return result


@pytest.mark.parametrize(
    "field,value", [("model_revision", "other"), ("layer_count", 27), ("hidden_dim", 8)]
)
def test_heldout_capture_requires_matching_actual_backbone(
    model, monkeypatch, field, value
):
    import jscc.sharing_geometry_capture as module

    policy = capture_policy(model)
    policy["enc_l14"]["site"][field] = value
    monkeypatch.setattr(
        module,
        "verify_study_freeze",
        lambda *a, **kw: {
            "obligation_plan": {"geometry": ["geometry:atlas"]},
            "site_policy": policy,
        },
    )
    monkeypatch.setattr(
        model.base.get_encoder(), "forward", lambda **_: pytest.fail("encoder reached")
    )
    with pytest.raises(ValueError, match="actual capture backbone"):
        capture_geometry_batch(
            model,
            batch(),
            model_revision="synthetic-fixed",
            include_heldout=True,
            freeze_reference={"fixture": True},
            obligation="geometry:atlas",
        )
