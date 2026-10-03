"""Real tiny BF16 backbone checks for the explicit mixed numerical runtime."""
import copy
from typing import Any, cast

import pytest
import torch

from jscc.config import load_config, validate_numerical_policy
from jscc.models.channel import AWGNChannel, normalize_power
from jscc.models.codec import Codec
from jscc.models.split_model import SplitModel
from jscc.runtime import autocast_for, precision_telemetry
from tests.model_helpers import tiny_backbone


def make_model(architecture="direct_affine", norm="none", policy="codec_receiver_fp32") -> Any:
    torch.manual_seed(7)
    base = tiny_backbone()
    cast(Any, base).to(dtype=torch.bfloat16)
    codec = Codec(16, {"architecture": architecture, "hidden_dim": 16, "bottleneck_dim": 8,
                       "n_res_blocks": 2 if architecture == "residual_mlp" else 0,
                       "activation": "gelu", "dropout": 0.0, "layernorm": norm,
                       "snr_film": False})
    return SplitModel(base, codec, AWGNChannel(), {"stack": "enc", "where": "after_final_norm"},
                      {"normalize_power": True, "clean_film_snr": 30}, numerical_policy=policy)


def inputs():
    return {"input_ids": torch.tensor([[3, 4, 5], [6, 7, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
            "decoder_input_ids": torch.tensor([[0, 8], [0, 9]]), "use_cache": False}


def objective(model, batch):
    with autocast_for(model), model.transmission(bypass=True), torch.no_grad():
        teacher = model(**batch).logits.detach()
    with autocast_for(model), model.transmission():
        logits = model(**batch).logits
        # This deliberately includes R's BF16 cast in the gradient path.
        loss = (logits - teacher).square().mean() + .1 * (
            model.reconstruction.float() - model.activation.float()).square().mean()
    return loss


@pytest.mark.parametrize("architecture,norm", [("direct_affine", "none"), ("direct_outer_ln", "both"),
                                              ("residual_mlp", "none"), ("residual_mlp", "both")])
def test_recipe_storage_gradients_optimizer_reload_and_split_reduction(architecture, norm):
    model = make_model(architecture, norm)
    batch = inputs()
    storage = [(name, id(p), p.data_ptr(), p.dtype) for name, p in model.base.named_parameters()]
    assert model.base.lm_head.out_proj.weight is model.base.get_encoder().text_model.embed_tokens.weight
    assert all(not p.requires_grad for p in model.base.parameters())
    optimizer = torch.optim.AdamW(model.codec.parameters(), lr=1e-3)
    loss = objective(model, batch)
    loss.backward()
    native = [p.grad.clone() for p in model.codec.parameters()]
    optimizer.zero_grad()
    for row in range(2):
        objective(model, {k: v[row:row + 1] if torch.is_tensor(v) else v for k, v in batch.items()}).div(2).backward()
    for actual, expected in zip(model.codec.parameters(), native):
        torch.testing.assert_close(actual.grad, expected, atol=2e-5, rtol=3e-3)
    assert all(p.grad is None for p in model.base.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.codec.parameters())
    optimizer.step()
    saved_codec, saved_optimizer = copy.deepcopy(model.codec.state_dict()), copy.deepcopy(optimizer.state_dict())
    restored = make_model(architecture, norm)
    restored.load_communication_state({"codec": saved_codec, "channel": model.channel.state_dict()})
    resumed = torch.optim.AdamW(restored.codec.parameters(), lr=1e-3)
    resumed.load_state_dict(saved_optimizer)
    for candidate, opt in ((model, optimizer), (restored, resumed)):
        opt.zero_grad()
        objective(candidate, batch).backward()
        opt.step()
    for a, b in zip(model.codec.parameters(), restored.codec.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert storage == [(name, id(p), p.data_ptr(), p.dtype) for name, p in model.base.named_parameters()]
    assert not any("cache" in key or "original" in key for key in model.base.state_dict())
    assert precision_telemetry(model)["numerical_runtime"]["compute_cache_bytes"] > 0


def test_boundary_matches_diagnostic_r_semantics_and_unrounded_k():
    model = make_model()
    batch = inputs()
    with model.transmission(bypass=True):
        model(**batch)
    hidden = model.activation
    # Frozen diagnostic boundary_fp32 fixture: FP32 encode/power/decode;
    # original BF16 target and quantized BF16 prediction retained for R.
    with torch.autocast("cpu", enabled=False):
        expected = model.codec.decode(normalize_power(model.codec.encode(hidden.float()),
                                                      batch["attention_mask"], mask_representation="binary"), 30)
    with model.transmission(encoder_mask=batch["attention_mask"]):
        actual = model._roundtrip(hidden)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(model.reconstruction, expected.bfloat16(), atol=0, rtol=0)
    assert model.activation.dtype == model.reconstruction.dtype == torch.bfloat16
    assert actual.dtype == torch.float32
    assert (actual != model.reconstruction.float()).any()


def test_direct_base_and_cached_generation_match_wrapper():
    model = make_model()
    batch = inputs()
    with model.transmission():
        wrapped = model(**batch).logits
    with model.transmission():
        direct = model.base(**batch).logits
    torch.testing.assert_close(wrapped, direct, atol=0, rtol=0)
    assert direct.dtype == torch.float32
    generation = {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask")}
    with model.transmission():
        cached = model.generate(**generation, max_new_tokens=3, do_sample=False, use_cache=True)
    with model.transmission():
        full = model.base.generate(**generation, max_new_tokens=3, do_sample=False, use_cache=False)
    assert torch.equal(cached, full)


def test_native_unchanged_and_config_rejections():
    model = make_model(policy="native")
    with autocast_for(model):
        assert model(**inputs()).logits.dtype == torch.bfloat16
    assert "numerical_runtime" not in precision_telemetry(model)
    config = load_config("configs/tasks/hellaswag.yaml")
    config["model"]["numerical_policy"] = "codec_receiver_fp32"
    for change in ({"dtype": "float32"}, {"sdpa_backend_policy": "flash_math"}, {"numerical_policy": "unknown"}):
        with pytest.raises(ValueError):
            validate_numerical_policy({**config["model"], **change}, {"stack": "enc", "where": "after_final_norm"})
    with pytest.raises(ValueError, match="after_final_norm"):
        validate_numerical_policy(config["model"], config["split"])
    with pytest.raises(ValueError, match="bfloat16"):
        SplitModel(tiny_backbone(), model.codec, AWGNChannel(), {"stack": "enc", "where": "after_final_norm"},
                   model.channel_config, numerical_policy="codec_receiver_fp32")


def test_cache_refresh_and_original_hierarchy_survive_state_load_and_device_move():
    model = make_model()
    native = make_model(policy="native")
    assert set(model.base.state_dict()) == set(native.base.state_dict())
    model(**inputs())
    old_cache = model.base.lm_head._copies[0]["out_proj.weight"]
    state = copy.deepcopy(model.base.state_dict())
    model.base.load_state_dict(state)
    model.to("cpu")
    model(**inputs())
    assert model.base.lm_head._copies[0]["out_proj.weight"] is not old_cache
    assert all(not p.requires_grad for group in model.base.lm_head._copies for p in group.values())
    assert model.base.lm_head.out_proj.weight is model.base.get_encoder().text_model.embed_tokens.weight


def test_local_replay_r_uses_bf16_prediction():
    from jscc.local_reconstruction import local_batch_values
    from jscc.losses import reconstruction_loss_stats
    from jscc.baseline_protocol import objective_settings

    model = make_model()
    with model.transmission(bypass=True):
        model(**inputs())
    shard = {"activation": model.activation, "attention_mask": inputs()["attention_mask"]}
    actual = local_batch_values(model, shard, None, objective_settings("local"))
    expected, count = reconstruction_loss_stats(model.reconstruction, shard["activation"], shard["attention_mask"])
    assert model.reconstruction.dtype == torch.bfloat16
    assert "hidden_numerator" in actual and "hidden_denominator" in actual
    torch.testing.assert_close(actual["hidden_numerator"], expected, atol=0, rtol=0)
    torch.testing.assert_close(actual["hidden_denominator"], count, atol=0, rtol=0)


def test_cached_decoder_keeps_fp32_kv_and_matches_full_logits():
    model = make_model()
    batch = inputs()
    with model.transmission():
        first = model.base(**{**batch, "decoder_input_ids": batch["decoder_input_ids"][:, :1], "use_cache": True})
        cache = first.past_key_values
        assert cache is not None
        # Reuse the exact reconstructed memory; no second channel draw.
        memory = first.encoder_last_hidden_state
        from transformers.modeling_outputs import BaseModelOutput
        second = model.base(encoder_outputs=BaseModelOutput(last_hidden_state=memory),
                            attention_mask=batch["attention_mask"],
                            decoder_input_ids=batch["decoder_input_ids"][:, 1:],
                            past_key_values=cache, use_cache=True)
        full = model.base(encoder_outputs=BaseModelOutput(last_hidden_state=memory),
                          attention_mask=batch["attention_mask"],
                          decoder_input_ids=batch["decoder_input_ids"], use_cache=False)
    assert second.past_key_values is cache
    torch.testing.assert_close(second.logits[:, -1], full.logits[:, -1], atol=1e-6, rtol=1e-5)
    assert cache.self_attention_cache.layers[0].keys.dtype == torch.float32


def test_runtime_yaml_policy_override_is_validated(tmp_path):
    import yaml
    config = load_config("configs/tasks/hellaswag.yaml")
    config["split"] = {"stack": "enc", "where": "after_final_norm"}
    config["runtime"] = {"numerical_policy": "codec_receiver_fp32"}
    path = tmp_path / "runtime.yaml"
    path.write_text(yaml.safe_dump(config))
    assert load_config(path)["model"]["numerical_policy"] == "codec_receiver_fp32"
    config["runtime"]["dtype"] = "float32"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="bfloat16"):
        load_config(path)


def test_inference_warmed_cache_can_train():
    model = make_model()
    with torch.inference_mode():
        model(**inputs())
    objective(model, inputs()).backward()
    assert all(p.grad is not None for p in model.codec.parameters())


def mixed_learner():
    from jscc.baseline_protocol import BaselineLearner
    return BaselineLearner(make_model(), run_id="mixed-cpu", task="hellaswag",
        identity={"source": "source", "config": "config", "data": "data", "parent": None},
        pairing_id="mixed-pair", synthetic=True, effective_batch=64, final_step=4)


def mixed_batch():
    return {"input_ids": inputs()["input_ids"].repeat(32, 1),
            "attention_mask": inputs()["attention_mask"].repeat(32, 1),
            "labels": torch.tensor([[8, 9], [10, -100]]).repeat(32, 1),
            "row_ids": torch.arange(64), "source_family_ids": [f"family-{i}" for i in range(64)]}


def mixed_saved_state(tmp_path, learner):
    from dataclasses import asdict
    from jscc.baseline_protocol import comparison_refs
    from jscc.experiment_state import save_state, open_state
    from jscc.models.split_model import resolve_encoder_site
    metadata = dict(source_identity="source", config_identity="config", data_identity="data",
        parent_identity=None, initialization_identity="initial", stream_identity="stream",
        protocol_identity="tiny-mixed", lineage=[], completed_updates=learner.completed,
        phase="both", snapshot_role="trained", numerical_policy="codec_receiver_fp32",
        model_state_contract="codec-only-stateless-channel-v1",
        site=asdict(resolve_encoder_site(learner.model.base, learner.model.split, "tiny")))
    metadata["comparison_controls"] = comparison_refs(learner, metadata, "combined")
    reference = save_state(tmp_path / "mixed.pt", model=learner.model.codec,
        optimizer=learner.optimizer, scheduler=learner.scheduler, scaler=learner.scaler,
        metadata=metadata, stream_state={"completed_updates": learner.completed,
                                        "offset": learner.offset, "source_valid_tokens": learner.valid_tokens})
    return open_state(reference, expected=metadata)


def test_mixed_real_objective_records_and_state_carry(tmp_path):
    from jscc.baseline_protocol import objective_record_request, objective_event_payload
    from jscc.experiment_records import observation_identity
    from jscc.experiment_state import restore_state
    learner = mixed_learner()
    batch = mixed_batch()
    values, _, _ = learner._values(batch, "combined", 1, 0, "train", "no_noise")
    torch.testing.assert_close(values["loss"], values["kl"] + .1 * values["nmse"])
    assert float(values["kl"].detach()) > 0
    assert learner.model.reconstruction.dtype == torch.bfloat16
    before = copy.deepcopy(learner.model.codec.state_dict())
    learner.update([batch])  # Actual zero-LR warmup step still initializes AdamW.
    learner.update([batch])
    assert any(not torch.equal(before[k], v) for k, v in learner.model.codec.state_dict().items())
    state = mixed_saved_state(tmp_path, learner)
    request = objective_record_request(learner, state, [batch], "combined")
    identity = observation_identity(request)  # Actual producer crosses records schema boundary.
    assert request["precision"] == "torch.bfloat16:codec_receiver_fp32"
    assert isinstance(request["details"]["outputs"], str)
    records = learner.validate([batch])
    payload = objective_event_payload(learner, state, records, [batch], "observations.json")
    assert payload["identity"] == identity
    assert len(payload["conditions"]) == 6
    resumed = mixed_learner()
    stream = restore_state(state, model=resumed.model.codec, optimizer=resumed.optimizer,
                           scheduler=resumed.scheduler, scaler=resumed.scaler)
    resumed.completed = resumed.attempted = stream["completed_updates"]
    resumed.offset, resumed.valid_tokens = stream["offset"], stream["source_valid_tokens"]
    original_event, resumed_event = learner.update([batch]), resumed.update([batch])
    assert original_event["payload"]["objective"] == resumed_event["payload"]["objective"]
    for a, b in zip(learner.model.codec.parameters(), resumed.model.codec.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert learner.scheduler.state_dict() == resumed.scheduler.state_dict()
    for key, original in learner.optimizer.state_dict()["state"].items():
        for name, tensor in original.items():
            torch.testing.assert_close(tensor, resumed.optimizer.state_dict()["state"][key][name], atol=0, rtol=0)


def test_mixed_evaluation_requires_matching_request_and_checkpoint_policy(tmp_path):
    from jscc.evaluation import validate_evaluation_target
    from jscc.experiment_state import ValidatedState
    learner = mixed_learner()
    learner.update([mixed_batch()])
    state = mixed_saved_state(tmp_path, learner)
    meta = state.payload["metadata"]
    request = {"checkpoint_sha256": state.reference.sha256, "learner_kind": state.payload["kind"],
        "data": meta["data_identity"], "comparison": meta["comparison_controls"],
        "step": meta["completed_updates"], "target_site": meta["site"], "target_role": "trained",
        "settings": {"numerical_policy": "codec_receiver_fp32"}, "precision": "bf16",
        "backend": learner.model.base.config.decoder._attn_implementation,
        **{key: meta[key + "_identity"] for key in ("config", "source", "protocol", "parent")}}
    validate_evaluation_target(state, request, learner.model)
    with pytest.raises(ValueError, match="Observation numerical policy"):
        validate_evaluation_target(state, {**request, "settings": {}}, learner.model)
    changed = copy.deepcopy(state.payload)
    changed["metadata"].pop("numerical_policy")
    with pytest.raises(ValueError, match="Checkpoint numerical policy"):
        validate_evaluation_target(ValidatedState(changed, state.reference), request, learner.model)


def test_mixed_evaluation_identity_is_stable_across_cache_warmup():
    from types import SimpleNamespace
    from jscc.evaluation_policy import evaluation_identity, digest
    model = make_model()
    tokenizer = SimpleNamespace(special_tokens_map={})
    adapter = SimpleNamespace(forward_logits_dtypes={"torch.float32"}, max_length=32)
    def identity():
        return evaluation_identity(model, tokenizer, {"num_samples": 2, "num_fewshot": 0,
            "batch_size": 2}, {"revision": "fixed"}, {}, adapter)
    before = identity()
    with torch.inference_mode():
        model(**inputs())
    assert digest(before) == digest(identity())
    assert before["numerical_runtime"]["receiver_compute"] == "float32"
    assert "compute_cache_bytes" not in before["numerical_runtime"]
