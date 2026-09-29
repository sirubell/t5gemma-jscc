"""CPU protocol/safety checks for the frozen COCO diagnostic."""

import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

from PIL import Image
import pytest
import torch
import yaml

from model_helpers import tiny_backbone
from jscc.coco_diagnostic import (
    MAX_REQUESTS, RequestBudget, diagnostic_inputs, forward_endpoint, prefix_inputs, prefix_variants,
    prepare_manifest, prepare_pair, runtime_packages, score_logits, source_digest, teacher_student_kl,
    validate_selection, verify_historical_archive, verify_plan,
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


def test_setup_reserve_stops_before_rows_or_model_and_preserves_source_receipt(tmp_path, monkeypatch):
    from jscc import coco_diagnostic as diagnostic
    original_budget = diagnostic.RequestBudget
    class ExpiredSetupBudget(original_budget):
        def __init__(self, max_seconds, reserve_seconds):
            super().__init__(max_seconds, reserve_seconds, now=lambda: 0.0)
            self.now = lambda: self.stop_new
    monkeypatch.setattr(diagnostic, "RequestBudget", ExpiredSetupBudget)
    monkeypatch.setattr(diagnostic, "load_rows_offline", lambda *_: pytest.fail("rows loaded in reserve"))
    monkeypatch.setattr(diagnostic, "load_mode", lambda *_: pytest.fail("model loaded in reserve"))
    historical = {"archive_sha256": "frozen"}
    plan = {"modes": {"enc_l9": {"config": {}}}, "ids": {}, "selection": {},
            "manifest_sha256": "a" * 64, "selection_sha256": "b" * 64,
            "active_source_files": {}, "historical_source_manifest": historical}
    result = diagnostic.run(plan, tmp_path / "reserve", max_seconds=10, reserve_seconds=2)
    assert result["status"] == "stopped_deadline"
    assert result["attempts"] == 0
    assert json.loads((tmp_path / "reserve" / "status.json").read_text())["historical_source"] == historical


def test_frozen_selection_detects_split_leakage():
    images = list(range(32))
    ids = {"train_ids": [100], "demo_ids": [200, 201, 202, 203],
           "selection_ids": images, "report_ids": [300]}
    selection = {"image_ids": images, "permuted_query_image_ids": images[1:] + images[:1],
                 "demo_ids": ids["demo_ids"], "no_report_overlap": True}
    validate_selection(selection, ids)
    bad = copy.deepcopy(ids)
    bad["report_ids"] = [0]
    with pytest.raises(ValueError, match="overlap"):
        validate_selection(selection, bad)
    bad = copy.deepcopy(selection)
    bad["permuted_query_image_ids"][0] = 0
    with pytest.raises(ValueError, match="permutation"):
        validate_selection(bad, ids)


def test_check_only_requires_real_hashes_and_independent_historical_binding(tmp_path, monkeypatch):
    from jscc import coco_diagnostic as diagnostic
    repo = Path(__file__).resolve().parents[1]
    images = list(range(32))
    ids = {"train_ids": [100], "demo_ids": [200, 201, 202, 203],
           "selection_ids": images, "report_ids": [300]}
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps({"image_ids": images,
        "permuted_query_image_ids": images[1:] + images[:1],
        "demo_ids": ids["demo_ids"], "no_report_overlap": True}))
    historical = tmp_path / "historical-source.json"
    archive_path = tmp_path / "execution.tar.gz"
    archive_members = {"jscc/training.py": b"example training source", "train.py": b"entry",
                       "uv.lock": b"lock", "pyproject.toml": b"project"}
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, content in archive_members.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    historical.write_text(json.dumps({name: hashlib.sha256(content).hexdigest()
                                      for name, content in archive_members.items()}))
    historical_digest = verify_historical_archive(archive_path, historical)
    index = {}
    bindings = {}
    for mode in ("vanilla", "enc_l9", "enc_fn", "dec_l8"):
        checkpoint = tmp_path / f"{mode}-step_000500.pt"
        checkpoint.write_bytes(b"offline fixture")
        split = ({"stack": "enc", "where": "after_layer", "index": 9} if mode == "enc_l9" else
                 {"stack": "dec", "where": "after_layer", "index": 8} if mode == "dec_l8" else
                 {"stack": "enc", "where": "after_final_norm"})
        config = {"task": "coco", "model": {"name": "example", "revision": "model-rev",
                  "dtype": "bfloat16", "sdpa_backend_policy": "flash_math"},
                  "data": {"name": "coco", "revision": "data-rev", "karpathy_name": "split",
                           "karpathy_revision": "split-rev", "num_demos": 4},
                  "evaluation": {"max_new_tokens": 64}, "training": {"max_steps": 500}, "split": split}
        config_path, run_path, ids_path = (tmp_path / f"{mode}-{suffix}" for suffix in ("config.yaml", "run.json", "ids.json"))
        config_path.write_text(yaml.safe_dump(config))
        ids_path.write_text(json.dumps(ids))
        training_source = historical_digest
        run_path.write_text(json.dumps({"resolved_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                                        "training_source_sha256": training_source,
                                        "source": {"revision": None}}))
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        index[mode] = {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
                       "source": {"revision": None}}
        bindings[mode] = {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
            "config": str(config_path), "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "run": str(run_path), "run_sha256": hashlib.sha256(run_path.read_bytes()).hexdigest(),
            "data_ids": str(ids_path), "data_ids_sha256": hashlib.sha256(ids_path.read_bytes()).hexdigest(),
            "training_source_sha256": training_source}
    index_path = tmp_path / "checkpoints.json"
    index_path.write_text(json.dumps(index))
    monkeypatch.setattr(diagnostic, "FROZEN_SELECTION_SHA256", hashlib.sha256(selection_path.read_bytes()).hexdigest())
    monkeypatch.setattr(diagnostic, "FROZEN_CHECKPOINT_INDEX_SHA256", hashlib.sha256(index_path.read_bytes()).hexdigest())
    manifest = {"protocol": "coco-query-endpoint-v1", "batch_size": 1, "max_new_tokens": 64,
                "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                "active_source_files": source_digest(repo), "runtime_packages": runtime_packages(),
                "historical_source": {
                    "archive": {"path": str(archive_path),
                                "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest()},
                    "manifest": {"path": str(historical),
                                 "sha256": hashlib.sha256(historical.read_bytes()).hexdigest()}},
                "modes": bindings}
    manifest_path = tmp_path / "manifest.json"
    bindings_path = tmp_path / "bindings.json"
    bindings_path.write_text(json.dumps({"historical_source": {
        "archive": str(archive_path), "manifest": str(historical)},
        "modes": {mode: {key: value for key, value in mode_binding.items()
                         if key in ("checkpoint", "config", "run", "data_ids")}
                  for mode, mode_binding in bindings.items()}}))
    plan = prepare_manifest(repo=repo, selection_path=selection_path,
                            checkpoint_index_path=index_path, bindings_path=bindings_path,
                            manifest_path=manifest_path)
    assert plan["ids"] == ids
    assert json.loads(manifest_path.read_text()) == manifest
    plan = verify_plan(repo=repo, selection_path=selection_path,
                       checkpoint_index_path=index_path, manifest_path=manifest_path)
    assert plan["ids"] == ids
    manifest["modes"]["dec_l8"]["training_source_sha256"] = None
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="historical training source binding"):
        verify_plan(repo=repo, selection_path=selection_path,
                    checkpoint_index_path=index_path, manifest_path=manifest_path)


@pytest.mark.parametrize("cross_reserve_in_preparation", [False, True])
def test_run_orchestrates_exact_matrix_and_durable_items_offline(tmp_path, monkeypatch,
                                                               cross_reserve_in_preparation):
    from jscc import coco_diagnostic as diagnostic

    class DummyBase(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class DummyModel:
        def __init__(self):
            self.base = DummyBase()
            self.activation = self.reconstruction = None
            self.memory_activation = self.memory_reconstruction = None

    class DummyTokenizer:
        eos_token_id = 1
        def __call__(self, text, **_kwargs):
            return {"input_ids": [8] if text == "\n" else [5]}

    class DummyProcessor:
        tokenizer = DummyTokenizer()

    selection = {"image_ids": list(range(32)),
                 "permuted_query_image_ids": list(range(1, 32)) + [0],
                 "demo_ids": [100, 101, 102, 103]}
    rows = {i: {"image": image(i), "answer": ["a."], "file_name": f"COCO_{i}.jpg"}
            for i in (*selection["image_ids"], *selection["demo_ids"])}
    model_info = {"config": {"model": {"revision": "r", "sdpa_backend_policy": "flash_math",
                                         "dtype": "bfloat16"}}, "checkpoint_sha256": "a" * 64,
                  "training_source_sha256": "b" * 64, "config_sha256": "c" * 64}
    plan = {"modes": {mode: model_info for mode in diagnostic.MODES},
            "selection": selection, "ids": {}, "manifest_sha256": "d" * 64,
            "selection_sha256": "e" * 64, "active_source_files": {}}
    monkeypatch.setattr(diagnostic, "load_rows_offline", lambda *_args: rows)
    monkeypatch.setattr(diagnostic, "load_mode", lambda *_args: (DummyProcessor(), DummyModel()))
    monkeypatch.setattr(diagnostic, "prepare_pair", lambda *_args: (
        [{"input_ids": torch.tensor([[2]]), "attention_mask": torch.tensor([[1]]),
          "pixel_values": torch.zeros(1, 3, 8, 8)}] * 2,
        [[{"rgb_sha256": "x", "crop_count": 1}]*5]*2, "prompt"))
    monkeypatch.setattr(diagnostic, "prefix_inputs", lambda *_args: (torch.tensor([[2, 5]]), torch.tensor([[5]])))
    monkeypatch.setattr(diagnostic, "forward_endpoint", lambda *_args, **_kwargs: torch.tensor([0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    monkeypatch.setattr(diagnostic, "generate_caption", lambda *_args, **_kwargs: {
        "generated_token_ids": [2, 5, 1], "raw_caption": "a.", "cleaned_caption": "a.",
        "decoder_forward_calls_observed": 2, "eos_emitted": True, "truncated": False,
        "repeated_lines": False, "trigram_repetition": False})
    if cross_reserve_in_preparation:
        clock = [0.0]
        class PreparationBudget(diagnostic.RequestBudget):
            def __init__(self, max_seconds, reserve_seconds):
                super().__init__(max_seconds, reserve_seconds, now=lambda: clock[0])
        original_prepare = diagnostic.prepare_pair
        def cross(*args):
            prepared = original_prepare(*args)
            clock[0] = 120.0
            return prepared
        monkeypatch.setattr(diagnostic, "RequestBudget", PreparationBudget)
        monkeypatch.setattr(diagnostic, "prepare_pair", cross)
        monkeypatch.setattr(diagnostic, "tensor_sha256", lambda *_: pytest.fail("hashed tensors after reserve"))
    result = diagnostic.run(plan, tmp_path / "out", max_seconds=120, reserve_seconds=0)
    if cross_reserve_in_preparation:
        assert result["status"] == "stopped_deadline"
        assert result["attempts"] == result["completed"] == 0
        assert json.loads((tmp_path / "out" / "status.json").read_text())["status"] == "stopped_deadline"
        return
    assert result["status"] == "complete"
    assert result["attempts"] == result["completed"] == 1024
    assert result["timing_stage"]["requests"] == 128
    assert result["decoder_calls_observed"] == 768 + 256 * 2
    lines = [json.loads(line) for line in (tmp_path / "out" / "items.jsonl").read_text().splitlines()]
    finals = [line for line in lines if line.get("status") == "complete"]
    partials = [line for line in lines if line.get("status") == "partial"]
    assert len(finals) == len(partials) == 1024
    assert len({line["request_id"] for line in finals}) == 1024
    assert sum(line["phase"] == "timing" for line in finals) == 128


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
