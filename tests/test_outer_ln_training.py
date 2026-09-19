"""Offline gates for the opt-in formal outer-LN input/randomness contract."""
import json
import gzip

import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from typing import cast

from jscc.data.hellaswag import Collator
from jscc.models.channel import AWGNChannel
from jscc.presentation import PresentationSampler, feature_statistics, initialization_evidence
from jscc import training
from jscc.config import load_config
from jscc.data import TaskData
from test_core import toy_model
from pathlib import Path


def test_exact_640k_stream_crosses_epoch_tails_and_replays(tmp_path):
    pool = [11, 23, 37, 48, 52, 69, 74]
    sampler = PresentationSampler(pool, 640000, 0)
    torch.manual_seed(987)
    replay = PresentationSampler(pool, 640000, 0)
    assert torch.equal(sampler.actual_ids, replay.actual_ids)
    assert len(sampler) == 5000 * 128
    assert sorted(sampler.actual_ids[:7].tolist()) == pool
    assert sorted(sampler.actual_ids[7:14].tolist()) == pool
    manifest = sampler.save(tmp_path, 128)
    assert len(manifest["segment_sha256"]) == 5000
    with gzip.open(tmp_path / "presentation_ids.json.gz", "rt") as source:
        assert len(json.load(source)) == 640000


def test_noise_replay_independent_of_global_rng_and_stream_interleaving():
    channel = AWGNChannel()
    z = torch.zeros(2, 3, 4)
    before = torch.get_rng_state()
    with channel.replay(0, "train:12:0", capture=True):
        first = channel.transmit(z, 0, stream="hidden")
        memory = channel.transmit(z, 0, stream="memory")
        second = channel.transmit(z, 0, stream="hidden")
    assert torch.equal(before, torch.get_rng_state())
    assert not torch.equal(first, second)
    torch.randn(100)
    with channel.replay(0, "train:12:0", capture=True):
        paired_memory = channel.transmit(z, 0, stream="memory")
        paired = channel.transmit(z, 6, stream="hidden")
    assert torch.equal(memory, paired_memory)
    torch.testing.assert_close(paired, first * 10**(-6/20), rtol=0, atol=0)
    with channel.replay(0, "train:13:0"):
        assert not torch.equal(first, channel(z, 0))


def test_feature_summary_excludes_padding():
    value = torch.tensor([[[1., 3.], [1000., -1000.]]])
    result = feature_statistics(value, torch.tensor([[True, False]]))
    assert result["valid_tokens"] == 1
    assert result["mean"]["mean"] == 2
    assert result["std"]["mean"] == 1


def test_outer_norm_does_not_change_core_initialization(tmp_path):
    from jscc.models.codec import Codec
    from types import SimpleNamespace
    config = dict(hidden_dim=8, bottleneck_dim=4, n_res_blocks=2, activation="gelu", dropout=0., snr_film=False)
    evidence = []
    for norm in ("none", "both"):
        torch.manual_seed(0)
        codec = Codec(8, dict(config, layernorm=norm))
        path = tmp_path / norm
        path.mkdir()
        evidence.append(initialization_evidence(SimpleNamespace(codec=codec, memory_codec=None), path)["hidden"])
    assert evidence[0]["core_sha256"] == evidence[1]["core_sha256"]
    assert evidence[0]["internal_layernorm_count"] == 4
    assert evidence[0]["parameter_count"] < evidence[1]["parameter_count"]


def test_fixed_stream_train_records_consumed_ids_noise_features_and_counts(tmp_path, monkeypatch):
    config = load_config(Path(__file__).parents[1] / "configs/tasks/hellaswag.yaml")
    config["run"].update(output_dir=str(tmp_path), name="stream")
    config["model"].update(device="cpu", dtype="float32")
    config["training"].update(max_steps=2, schedule_steps=2, batch_size=2, eval_every=2,
        gradient_accumulation=1, validation_batches=1, patience=None, save_steps=[2], log_every=1,
        streamed_backward=True, valid_only_kl=True,
        presentation_stream={"seed":0, "total_presentations":4, "policy":"epoch-permutations-v1"},
        paired_randomness={"seed":0, "policy":"step-microbatch-stream-v1", "audit_steps":[1,2]},
        feature_summary={"selection_count":2, "steps":[0,2]})
    rows = [{"input_ids":[1,2], "attention_mask":[1,1], "label_ids":[3,4], "row_id":i} for i in range(3)]
    sampler = PresentationSampler([0,1,2],4,0)
    loader = DataLoader(cast(Dataset, rows),batch_size=2,sampler=sampler,collate_fn=Collator(0))
    validation = DataLoader(cast(Dataset, rows),batch_size=2,collate_fn=Collator(0))
    def build(config):
        model = toy_model("dec")
        model.codec.film = None
        assert model.memory_codec is not None
        model.memory_codec.film = None
        return None, model
    monkeypatch.setattr(training,"build_model",build)
    monkeypatch.setattr(training,"load_data",lambda *args:TaskData(train=loader,validation=validation,ids={"validation_rows":[10,11,12]}))
    run = training.train(config)
    completion = json.loads((run / "completion.json").read_text())
    assert completion["presentations"] == 4 and completion["presentation_budget_verified"]
    manifest = json.loads((run / "presentations.json").read_text())
    assert completion["consumed_ids_bytes_sha256"] == manifest["actual_ids_bytes_sha256"]
    assert completion["status"] == "FULL_BUDGET_COMPLETED"
    assert completion["hidden_latent_valid"] > 0 and completion["memory_latent_valid"] > 0
    features = [json.loads(line) for line in (run / "feature_summary.jsonl").read_text().splitlines()]
    assert [row["step"] for row in features] == [0,2]
    assert set(features[0]["streams"]) == {"hidden","memory"}
    audits = [json.loads(line) for line in (run / "randomness_audit.jsonl").read_text().splitlines()]
    assert len(audits) == 2
    assert {draw["stream"] for draw in audits[0]["draws"]} == {"hidden","memory"}
    with pytest.raises(ValueError,match="fresh-only"):
        training.train(config,resume=run / "last.pt")
    original_aggregate = training.aggregate_batch_losses
    def nonfinite(values, settings):
        result = original_aggregate(values, settings)
        result["loss"] = result["loss"] * float("nan")
        return result
    monkeypatch.setattr(training, "aggregate_batch_losses", nonfinite)
    with pytest.raises(FloatingPointError, match="nonfinite loss"):
        training.train(config)
    failures = [json.loads(path.read_text()) for path in tmp_path.glob("*/completion.json")]
    failed = next(item for item in failures if item["reason"] == "failure")
    assert failed["step"] == 0 and failed["attempted_step"] == 1
    assert failed["presentations"] == 2 and failed["status"] == "FAILED_OR_PARTIAL"

    def broken_features(*args):
        raise RuntimeError("injected step-zero feature failure")
    monkeypatch.setattr(training, "record_feature_summary", broken_features)
    with pytest.raises(RuntimeError, match="step-zero feature failure"):
        training.train(config)
    failures = [(path, json.loads(path.read_text())) for path in tmp_path.glob("*/completion.json")]
    path, failed = next((path, item) for path, item in failures if item.get("message") == "injected step-zero feature failure")
    assert failed["step"] == 0 and failed["attempted_step"] == 0
    assert failed["presentations"] == 0 and failed["status"] == "FAILED_OR_PARTIAL"
    assert (path.parent / "failure.pt").is_file()


def test_replay_preserves_legacy_two_argument_awgn_subclass():
    class RecordingAWGN(AWGNChannel):
        def forward(self, z, snr_db):
            self.calls = getattr(self, "calls", 0) + 1
            return super().forward(z, snr_db)
    channel = RecordingAWGN()
    z = torch.zeros(2, 3, 4)
    with channel.replay(0, "legacy-subclass"):
        hidden = channel.transmit(z, 0, stream="hidden")
        memory = channel.transmit(z, 0, stream="memory")
    assert channel.calls == 2
    assert not torch.equal(hidden, memory)
    assert channel._stream == "hidden"
