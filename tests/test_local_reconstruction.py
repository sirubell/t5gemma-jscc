"""Offline CPU gates for encoder-only capture, replay and phase transfer."""

import copy
from pathlib import Path

import pytest
import torch

from jscc.activation_replay import (capture_encoder_microbatch, identity, load_manifest,
                                    load_shard, write_cache)
from jscc.local_reconstruction import local_batch_values, validate_phase_transfer
from jscc.losses import reconstruction_loss_stats
from jscc.models.channel import AWGNChannel
from jscc.models.codec import Codec
from jscc.models.split_model import SplitModel
from jscc.presentation import PresentationSampler
from jscc.runtime import model_inputs
from model_helpers import tiny_backbone


def recipe(tmp_path):
    return {"task": "hellaswag", "seed": 0, "protocol": "corrected-baseline-v2-native-decoder-inputs",
            "run": {"name": "cpu", "output_dir": str(tmp_path)},
            "model": {"name": "tiny", "revision": "fixed", "device": "cpu", "dtype": "float32"},
            "split": {"stack": "enc", "where": "after_layer", "index": 9},
            "codec": {"hidden_dim": 16, "bottleneck_dim": 8, "n_res_blocks": 1,
                      "activation": "gelu", "layernorm": "none", "snr_film": False},
            "channel": {"type": "awgn", "kwargs": {}, "normalize_power": True,
                        "train_noise": True, "train_snr_range": [-6.0, 18.0], "clean_film_snr": 18.0},
            "data": {"name": "fixture", "revision": "fixed", "max_length": 8,
                     "num_validation": 1, "num_train": None},
            "evaluation": {"mode": "fixture"},
            "training": {"max_steps": 1, "batch_size": 2, "gradient_accumulation": 1,
                         "presentation_stream": {"policy": "epoch-permutations-v1",
                                                 "seed": 0, "total_presentations": 2},
                         "paired_randomness": {"policy": "step-microbatch-stream-v1", "seed": 0},
                         "kl_weight": 0.0, "mse_weight": 0.1}}


def model_for(config):
    base = tiny_backbone(num_hidden_layers=10)
    codec = Codec(16, config["codec"])
    return SplitModel(base, codec, AWGNChannel(), config["split"], config["channel"]).eval()


def batch():
    return {"input_ids": torch.tensor([[3, 4, 5, 0], [6, 7, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
            "labels": torch.tensor([[8, 9, -100], [10, -100, -100]]),
            "row_ids": torch.tensor([1, 2])}


@torch.enable_grad()
def test_encoder_capture_matches_online_hook_and_local_gradient_without_suffix(tmp_path, monkeypatch):
    torch.manual_seed(7)
    config = recipe(tmp_path)
    model = model_for(config)
    sample = batch()
    activation = capture_encoder_microbatch(model, sample)
    assert model.activation is None  # The split codec hook was never entered.
    kwargs, _ = model_inputs(sample, model)
    snr = torch.tensor([[[0.0]], [[12.0]]])
    with torch.enable_grad(), model.channel.replay(0, "train:1:0"), model.transmission(
            snr, encoder_mask=sample["attention_mask"]):
        model(**kwargs)
    online_activation = model.activation
    assert online_activation is not None
    assert torch.equal(activation, online_activation)
    online_reconstruction = model.reconstruction
    assert online_reconstruction is not None
    online_loss = reconstruction_loss_stats(online_reconstruction, online_activation,
                                            sample["attention_mask"])
    expected_grad = torch.autograd.grad(online_loss[0] / online_loss[1],
                                        tuple(model.codec.parameters()))
    original_forward = model.base.forward
    def forbidden(*_args, **_kwargs):
        raise AssertionError("local replay executed frozen suffix or decoder")
    monkeypatch.setattr(model.base, "forward", forbidden)
    shard = {**sample, "activation": activation}
    with torch.enable_grad(), model.channel.replay(0, "train:1:0"):
        values = local_batch_values(model, shard, snr, config["training"])
    actual_grad = torch.autograd.grad(values["nmse"], tuple(model.codec.parameters()))
    assert torch.allclose(values["nmse"], online_loss[0] / online_loss[1], atol=1e-6)
    for online, local in zip(expected_grad, actual_grad):
        torch.testing.assert_close(local, online, atol=1e-6, rtol=1e-5)
    assert all(parameter.grad is None for parameter in model.base.parameters())
    monkeypatch.setattr(model.base, "forward", original_forward)


def test_replay_keeps_padding_power_and_paired_noise(tmp_path):
    config = recipe(tmp_path)
    model = model_for(config)
    shard = {**batch(), "activation": capture_encoder_microbatch(model, batch())}
    snr = torch.tensor([[[0.0]], [[12.0]]])
    with model.channel.replay(0, "train:1:0", capture=True):
        first = local_batch_values(model, shard, snr, config["training"])
        first_draw = copy.deepcopy(model.channel.draw_summaries)
    with model.channel.replay(0, "train:1:0", capture=True):
        second = local_batch_values(model, shard, snr, config["training"])
        second_draw = copy.deepcopy(model.channel.draw_summaries)
    assert first_draw == second_draw and first_draw[0]["shape"] == [2, 4, 8]
    torch.testing.assert_close(first["nmse"], second["nmse"])
    assert first.get("hidden_denominator") == 2
    padded = copy.deepcopy(shard)
    padded["activation"] = torch.cat([shard["activation"], torch.full((2, 1, 16), 100.0)], dim=1)
    padded["attention_mask"] = torch.cat([shard["attention_mask"], torch.zeros((2, 1), dtype=torch.long)], dim=1)
    with model.channel.replay(0, "train:1:0"):
        changed = local_batch_values(model, padded, None, config["training"])
    torch.testing.assert_close(changed["nmse"], local_batch_values(model, shard, None, config["training"])["nmse"])


def test_cache_roundtrip_rejects_stale_source_tampering_and_selection_leak(tmp_path):
    config = recipe(tmp_path)
    data_ids = {"train_rows": [1, 2], "validation_rows": [3], "validation_split": "train"}
    model = model_for(config)
    cache = tmp_path / "cache"
    write_cache(cache, config, data_ids, [batch()], model)
    manifest = load_manifest(cache, config)
    saved = load_shard(cache, manifest, 0)
    assert torch.equal(saved["activation"], capture_encoder_microbatch(model, batch()))
    stale = copy.deepcopy(config)
    stale["data"]["revision"] = "other"
    with pytest.raises(ValueError, match="stale cache"):
        load_manifest(cache, stale)
    with pytest.raises(ValueError, match="overlap"):
        identity(config, {"train_rows": [1, 3], "validation_rows": [3], "validation_split": "train"})
    changed = copy.deepcopy(manifest)
    changed["shards"][0]["row_ids"] = [1, 3]
    import json
    (cache / "manifest.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="selection"):
        load_manifest(cache, config)


def test_presentation_suffix_and_weight_only_phase_policy(tmp_path):
    all_ids = PresentationSampler([11, 12, 13], 8, seed=5).actual_ids
    b = PresentationSampler([11, 12, 13], 4, seed=5).actual_ids
    c = PresentationSampler([11, 12, 13], 4, seed=5, start=4).actual_ids
    assert torch.equal(torch.cat([b, c]), all_ids)
    config = recipe(tmp_path)
    config["training"]["presentation_stream"] = {"policy": "epoch-permutations-v1", "seed": 0,
                                                    "total_presentations": 2, "start_presentation": 2}
    state = {"config": recipe(tmp_path), "data_ids": {"train_rows": [1, 2],
             "validation_rows": [3], "validation_split": "train"}, "step": 1,
             "codec": {"weight": torch.tensor([1.0])}, "channel": {}, "memory_codec": None,
             "optimizer": {"sentinel": "must not load"}, "scheduler": {"sentinel": "must not load"},
             "scaler": {"sentinel": "must not load"}}
    from jscc.activation_replay import canonical_digest, file_digest
    from jscc.local_reconstruction import parent_recipe_sha256
    from jscc.presentation import training_source_digest
    state["phase_record"] = {"phase": "B-local-reconstruction", "terminal_step": 1,
                             "source_sha256": training_source_digest(),
                             "data_ids_sha256": canonical_digest(state["data_ids"]),
                             "cache_identity_sha256": "fixed-cache", "presentations": 2}
    path = Path(tmp_path) / "b.pt"
    torch.save(state, path)
    (tmp_path / "completion.json").write_text(__import__("json").dumps({
        "status": "complete", "checkpoint": "b.pt", "checkpoint_sha256": file_digest(path)}))
    with pytest.raises(ValueError, match="requires a declared B parent"):
        validate_phase_transfer(path, config)
    config["training"]["phase_transfer"] = {"parent_recipe_sha256": parent_recipe_sha256(state["config"]),
                                             "parent_step": 1}
    result = validate_phase_transfer(path, config)
    assert result["metadata"]["optimizer_state_loaded"] is False
    assert result["metadata"]["scheduler_state_loaded"] is False
    assert result["metadata"]["scaler_state_loaded"] is False
    bad = copy.deepcopy(config)
    bad["data"]["revision"] = "changed"
    with pytest.raises(ValueError, match="data differs"):
        validate_phase_transfer(path, bad)
    unplanned = copy.deepcopy(state)
    unplanned["config"]["seed"] = 1
    unplanned["config"]["training"]["lr"] = 0.0004
    unplanned["config"]["training"]["paired_randomness"]["seed"] = 1
    torch.save(unplanned, path)
    (tmp_path / "completion.json").write_text(__import__("json").dumps({
        "status": "complete", "checkpoint": "b.pt", "checkpoint_sha256": file_digest(path)}))
    with pytest.raises(ValueError, match="declared B parent recipe"):
        validate_phase_transfer(path, config)


@torch.enable_grad()
def test_b_terminal_transfers_to_actual_c_update_and_evaluator_state(tmp_path, monkeypatch):
    from jscc.config import load_config
    from jscc.data import TaskData
    from jscc.local_reconstruction import run_local
    from jscc.training import train
    import jscc.training as training_module
    import jscc.models.split_model as split_module

    config = load_config(Path(__file__).parents[1] / "configs/tasks/hellaswag.yaml")
    config.update(seed=0)
    config["model"].update(name="tiny", revision="fixed", device="cpu", dtype="float32")
    config["codec"].update(hidden_dim=16, bottleneck_dim=8, n_res_blocks=1)
    config["data"].update(name="fixture", revision="fixed", max_length=8,
                          num_validation=1, num_train=2)
    config["data"].pop("prompt_policy", None)  # Synthetic tokens have no archived prompt rows.
    config["run"].update(name="b-cpu", output_dir=str(tmp_path))
    config["training"].update(max_steps=1, schedule_steps=1, batch_size=2,
                              gradient_accumulation=1, eval_every=1, min_steps=1,
                              save_steps=[1], selection_steps=[1], patience=None,
                              feature_summary=None, log_every=1)
    config["training"]["presentation_stream"].update(total_presentations=2,
                                                       start_presentation=0)
    data_ids = {"train_rows": [1, 2], "validation_rows": [3], "validation_split": "train"}
    capture_model = model_for(config)
    cache = tmp_path / "cache"
    write_cache(cache, config, data_ids, [batch()], capture_model)
    monkeypatch.setattr(split_module, "build_model", lambda cfg: (None, model_for(cfg)))
    b_run = run_local(config, cache)
    b_checkpoint = b_run / "last.pt"
    b_state = torch.load(b_checkpoint, map_location="cpu", weights_only=True)
    assert b_state["step"] == 1 and b_state["phase_record"]["terminal_step"] == 1
    evaluator_model = model_for(config)
    evaluator_model.load_communication_state(b_state)

    c_config = copy.deepcopy(config)
    c_config["run"]["name"] = "c-cpu"
    c_config["training"]["presentation_stream"]["start_presentation"] = 2
    from jscc.local_reconstruction import parent_recipe_sha256
    c_config["training"]["phase_transfer"] = {"cache_identity_sha256":
        b_state["phase_record"]["cache_identity_sha256"], "parent_step": 1,
        "parent_recipe_sha256": parent_recipe_sha256(b_state["config"])}
    def build(cfg):
        return None, model_for(cfg)
    class TinyTrain:
        def __init__(self):
            self.sampler = PresentationSampler(data_ids["train_rows"], 2, 0, start=2)
        def __iter__(self):
            selected = batch()
            selected["row_ids"] = self.sampler.actual_ids.clone()
            yield selected
    seen_ids = []
    def load_data(_config, _processor, saved_ids=None):
        assert saved_ids == data_ids
        seen_ids.append(saved_ids)
        return TaskData(train=TinyTrain(), validation=[batch()], ids=data_ids)
    monkeypatch.setattr(training_module, "build_model", build)
    monkeypatch.setattr(training_module, "load_data", load_data)
    c_run = train(c_config, initial_checkpoint=b_checkpoint)
    c_state = torch.load(c_run / "last.pt", map_location="cpu", weights_only=True)
    assert seen_ids == [data_ids]
    assert c_state["step"] == 1
    assert c_state["phase_transfer"]["optimizer_state_loaded"] is False
    assert c_state["phase_transfer"]["parent_step"] == 1
    assert c_state["optimizer"]["state"]  # Fresh optimizer received one update.
    assert c_state["scheduler"]["last_epoch"] == 1
    evaluator_model.load_communication_state(c_state)
    # Final checkpoint serialization may exhaust the wall cap after the last
    # optimizer update; that run must not claim FULL_BUDGET_COMPLETED.
    import json
    from types import SimpleNamespace
    clock = [0.0]
    def tick():
        clock[0] += 0.001
        return clock[0]
    monkeypatch.setattr(training_module, "time", SimpleNamespace(monotonic=tick))
    original_save = training_module.save_checkpoint
    def crossed(path, *args, **kwargs):
        original_save(path, *args, **kwargs)
        if Path(path).name == "step_000001.pt":
            clock[0] = 11.0
    monkeypatch.setattr(training_module, "save_checkpoint", crossed)
    overdue = copy.deepcopy(c_config)
    overdue["run"]["name"] = "c-overdue"
    with pytest.raises(TimeoutError, match="partial evidence"):
        train(overdue, initial_checkpoint=b_checkpoint, deadline=10.0)
    overdue_runs = list(tmp_path.glob("*-c-overdue-*"))
    assert len(overdue_runs) == 1
    overdue_run = overdue_runs[0]
    completion = json.loads((overdue_run / "completion.json").read_text())
    assert completion["status"] == "PARTIAL"
    assert completion["reason"] == "wall_deadline_overrun"


def test_prepare_plan_explicit_budgets_and_c_d_policy_guards(tmp_path):
    import yaml
    from scripts.hellaswag_two_stage import check_plan, prepare_plan
    base = Path(__file__).parents[1] / "configs/tasks/hellaswag.yaml"
    plan_dir = tmp_path / "plan"
    result = prepare_plan(str(base), plan_dir, total_updates=3,
                          local_updates=1, functional_updates=2, functional_lr=0.0001)
    assert result["arms"]["A"]["presentations"] == 192
    assert result["arms"]["B"]["presentations"] == 64
    assert result["arms"]["C"]["presentation_start"] == 64
    assert result["arms"]["D"]["presentation_start"] == 64
    assert yaml.safe_load((plan_dir / "c.yaml").read_text())["training"]["lr"] == 0.0001
    with pytest.raises(FileExistsError, match="fresh and empty"):
        prepare_plan(str(base), plan_dir, total_updates=3, local_updates=1, functional_updates=2)
    with pytest.raises(ValueError, match="U = L"):
        prepare_plan(str(base), tmp_path / "bad", total_updates=3,
                     local_updates=1, functional_updates=1)
    paths = {arm: str(plan_dir / f"{arm.lower()}.yaml") for arm in "ABCD"}
    d = yaml.safe_load((plan_dir / "d.yaml").read_text())
    d["training"]["lr"] = 0.0002
    (plan_dir / "d.yaml").write_text(yaml.safe_dump(d))
    with pytest.raises(ValueError, match="functional training"):
        check_plan(paths)
    d["training"]["lr"] = 0.0001
    d["evaluation"]["batch_size"] = 1
    (plan_dir / "d.yaml").write_text(yaml.safe_dump(d))
    with pytest.raises(ValueError, match="evaluation"):
        check_plan(paths)
    d["evaluation"]["batch_size"] = 64
    (plan_dir / "d.yaml").write_text(yaml.safe_dump(d))
    c = yaml.safe_load((plan_dir / "c.yaml").read_text())
    c["training"]["phase_transfer"]["parent_recipe_sha256"] = "0" * 64
    (plan_dir / "c.yaml").write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError, match="planned B recipe"):
        check_plan(paths)


def test_expired_capture_leaves_partial_receipt_without_manifest(tmp_path):
    import time
    config = recipe(tmp_path)
    data_ids = {"train_rows": [1, 2], "validation_rows": [3], "validation_split": "train"}
    cache = tmp_path / "expired"
    with pytest.raises(TimeoutError, match="deadline"):
        write_cache(cache, config, data_ids, [batch()], model_for(config),
                    deadline=time.monotonic() - 1)
    assert (cache / "partial.json").exists()
    assert not (cache / "manifest.json").exists()


def test_capture_crossing_deadline_during_final_manifest_is_partial(tmp_path, monkeypatch):
    import itertools
    from types import SimpleNamespace
    import jscc.activation_replay as replay_module
    clock = itertools.chain([0.0] * 4, itertools.repeat(11.0))
    monkeypatch.setattr(replay_module, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    config = recipe(tmp_path)
    ids = {"train_rows": [1, 2], "validation_rows": [3], "validation_split": "train"}
    cache = tmp_path / "crossed"
    with pytest.raises(TimeoutError, match="manifest publication"):
        write_cache(cache, config, ids, [batch()], model_for(config), deadline=10.0)
    assert (cache / "partial.json").exists()
    assert not (cache / "manifest.json").exists()


def test_cache_byte_cap_stops_before_accepting_shard(tmp_path):
    import json
    config = recipe(tmp_path)
    data_ids = {"train_rows": [1, 2], "validation_rows": [3], "validation_split": "train"}
    cache = tmp_path / "capped"
    with pytest.raises(RuntimeError, match="byte cap"):
        write_cache(cache, config, data_ids, [batch()], model_for(config),
                    max_cache_bytes=1)
    partial = json.loads((cache / "partial.json").read_text())
    assert partial["cache_bytes"] == 0
    assert partial["captured_presentations"] == 0
    assert not list(cache.glob("*.pt"))
    assert not (cache / "manifest.json").exists()


def test_prompt_view_identity_checks_tokens_and_demo_selection(tmp_path):
    from jscc.activation_replay import canonical_digest
    config = recipe(tmp_path)
    config["data"]["prompt_policy"] = {"version": "prompt-alignment-v1", "mode": "five_shot"}
    ids = {"train_rows": [1, 2], "validation_rows": [3], "validation_split": "train"}
    sample = batch()
    rows = []
    for position, row_id in enumerate([1, 2]):
        source = sample["input_ids"][position][sample["attention_mask"][position].bool()].tolist()
        target = sample["labels"][position][sample["labels"][position] != -100].tolist()
        rows.append({"prompt_row_id": row_id, "demo_ids": [2 if row_id == 1 else 1],
                     "input_hash": canonical_digest(source), "target_hash": canonical_digest(target),
                     "source_hash": "source", "text_hash": "prompt"})
    evidence = {"train": {"rows": rows}, "policy": config["data"]["prompt_policy"]}
    cache = tmp_path / "prompt-cache"
    manifest = write_cache(cache, config, ids, [sample], model_for(config), prompt_evidence=evidence)
    assert len(manifest["shards"][0]["view_ids"]) == 2
    stale = copy.deepcopy(evidence)
    stale["train"]["rows"][0]["input_hash"] = "wrong"
    with pytest.raises(ValueError, match="prompt view tokens"):
        write_cache(tmp_path / "wrong-view", config, ids, [sample], model_for(config),
                    prompt_evidence=stale)
    leaking = copy.deepcopy(evidence)
    leaking["train"]["rows"][0]["demo_ids"] = [3]
    with pytest.raises(ValueError, match="demonstration leaks"):
        write_cache(tmp_path / "leak", config, ids, [sample], model_for(config),
                    prompt_evidence=leaking)


def test_interrupted_local_phase_has_no_terminal_checkpoint(tmp_path, monkeypatch):
    import jscc.local_reconstruction as local_module
    import jscc.models.split_model as split_module
    config = recipe(tmp_path)
    config["training"].update(lr=0.0002, weight_decay=0.01, warmup_ratio=0.0,
                              min_lr_ratio=0.0, grad_clip=1.0, streamed_backward=True)
    ids = {"train_rows": [1, 2], "validation_rows": [3], "validation_split": "train"}
    cache = tmp_path / "cache"
    write_cache(cache, config, ids, [batch()], model_for(config))
    monkeypatch.setattr(split_module, "build_model", lambda cfg: (None, model_for(cfg)))
    def timed_out(*_args, **_kwargs):
        raise TimeoutError("test deadline")
    monkeypatch.setattr(local_module, "load_shard", timed_out)
    with pytest.raises(TimeoutError, match="deadline"):
        local_module.run_local(config, cache)
    run_dirs = [path for path in tmp_path.iterdir() if path.is_dir() and path.name.startswith("20")]
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "completion.json").exists()
    assert not (run_dirs[0] / "last.pt").exists()


@torch.enable_grad()
def test_b_crossing_deadline_during_checkpoint_cannot_transfer(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import jscc.local_reconstruction as local_module
    import jscc.models.split_model as split_module
    config = recipe(tmp_path)
    config["training"].update(lr=0.0002, weight_decay=0.01, warmup_ratio=0.0,
                              min_lr_ratio=0.0, grad_clip=1.0, streamed_backward=True)
    ids = {"train_rows": [1, 2], "validation_rows": [3], "validation_split": "train"}
    cache = tmp_path / "cache"
    write_cache(cache, config, ids, [batch()], model_for(config))
    monkeypatch.setattr(split_module, "build_model", lambda cfg: (None, model_for(cfg)))
    clock = [0.0]
    monkeypatch.setattr(local_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    original_save = local_module.save_checkpoint
    def crossed(*args, **kwargs):
        original_save(*args, **kwargs)
        clock[0] = 11.0
    monkeypatch.setattr(local_module, "save_checkpoint", crossed)
    with pytest.raises(TimeoutError, match="terminal serialization"):
        local_module.run_local(config, cache, deadline=10.0)
    run_dirs = [path for path in tmp_path.iterdir() if path.is_dir() and path.name.startswith("20")]
    assert len(run_dirs) == 1
    run = run_dirs[0]
    assert not (run / "last.pt").exists()
    assert (run / "last.partial.pt").exists()
    assert json.loads((run / "completion.json").read_text())["status"] == "FAILED_OR_PARTIAL"
    from jscc.local_reconstruction import parent_recipe_sha256
    c_config = copy.deepcopy(config)
    c_config["training"]["presentation_stream"]["start_presentation"] = 2
    c_config["training"]["phase_transfer"] = {"parent_recipe_sha256": parent_recipe_sha256(config),
                                                 "parent_step": 1}
    with pytest.raises(ValueError, match="partial or completion receipt"):
        validate_phase_transfer(run / "last.partial.pt", c_config)
