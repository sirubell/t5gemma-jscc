"""Bounded encoder l4/l9/l19 atlas and local replay parity; zero updates.

Run as ``uv run --locked python -m scripts.shared_encoder_atlas --help``.
The CLI supervisor enforces the wall ceiling including child startup/model load.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import json
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch

from jscc.activation_replay import canonical_digest, file_digest
from jscc.config import load_config
from jscc.data.hellaswag import Collator
from jscc.losses import reconstruction_loss_stats
from jscc.models.channel import AWGNChannel
from jscc.models.split_model import build_model, stack_module
from jscc.presentation import tensor_digest, training_source_digest
from jscc.runtime import configure_training_determinism, seed_everything, source_state

SITES = (4, 9, 19)
SCHEMA = "shared-encoder-s0-atlas-v1"
VIEW_SCHEMA = "shared-encoder-s0-views-v1"
MAX_BYTES = 1024 ** 3
MAX_SECONDS = 600


class BoundaryReached(Exception):
    """Stop before any encoder suffix or decoder executes."""


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("S0 wall deadline exceeded")


def write_json(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def budget_json(path, value, limit):
    """Leave 1 KiB for terminal failure status; never publish an oversized payload."""
    size = len((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())
    used = sum(item.stat().st_size for item in path.parent.iterdir() if item != path)
    if used + size + 1024 > limit:
        raise RuntimeError("serialized byte cap would be exceeded by JSON artifact")
    write_json(path, value)


def finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"nonfinite {name}")


def validate_views(manifest, config):
    """Require supplied, fully tokenized views; never select or truncate rows."""
    if manifest.get("schema") != VIEW_SCHEMA:
        raise ValueError("unsupported frozen-view schema")
    if manifest.get("config_sha256") != canonical_digest(config):
        raise ValueError("frozen views do not match resolved configuration")
    if manifest.get("tokenizer") != {key: config["model"][key] for key in ("name", "revision")}:
        raise ValueError("tokenizer identity differs from pinned model")
    ids = manifest["data_ids"]
    if any(type(row_id) is not int or row_id < 0 for row_id in [*ids["train_rows"], *ids["validation_rows"]]):
        raise ValueError("data IDs must be nonnegative integers")
    train, selection = set(ids["train_rows"]), set(ids["validation_rows"])
    if ids.get("validation_split") != "train" or train & selection:
        raise ValueError("selection must be disjoint and train-derived")
    if not train or not selection or len(train) != len(ids["train_rows"]) or len(selection) != len(ids["validation_rows"]):
        raise ValueError("data IDs must be nonempty and unique")
    rows = manifest["rows"]
    if len(rows) != 128:
        raise ValueError("S0 requires exactly 128 frozen views")
    if len({row["row_id"] for row in rows}) != 128:
        raise ValueError("atlas rows must be unique")
    for index, row in enumerate(rows):
        group = "train" if index < 64 else "selection"
        if row.get("group") != group or row["row_id"] not in (train if group == "train" else selection):
            raise ValueError("rows must be 64 train then 64 disjoint selection views")
        demos = row["demo_ids"]
        if len(demos) != 5 or len(set(demos)) != 5 or not set(demos).issubset(train) or row["row_id"] in demos:
            raise ValueError("five distinct train-only demos required, excluding query")
        if "source_id" in row and row["source_id"] is not None:
            if len(row.get("demo_source_ids", [])) != 5 or row["source_id"] in row["demo_source_ids"]:
                raise ValueError("demo source overlaps query source")
        for key in ("input_ids", "label_ids"):
            tokens = row[key]
            if not tokens or any(type(token) is not int or token < 0 for token in tokens):
                raise ValueError("frozen tokens must be nonempty nonnegative integers")
        if len(row["input_ids"]) > config["data"]["prompt_policy"]["source_max_length"]:
            raise ValueError("frozen input exceeds declared source cap; no truncation allowed")
        if row["input_sha256"] != canonical_digest(row["input_ids"]) or row["target_sha256"] != canonical_digest(row["label_ids"]):
            raise ValueError("frozen token hash mismatch")
        basis = {key: value for key, value in row.items() if key != "view_sha256"}
        if row["view_sha256"] != canonical_digest(basis):
            raise ValueError("frozen view hash mismatch")
    if type(manifest.get("pad_token_id")) is not int or manifest["pad_token_id"] < 0:
        raise ValueError("explicit pad token ID required")
    if config["data"].get("prompt_policy", {}).get("mode") != "five_shot":
        raise ValueError("explicit five-shot prompt policy required")
    return rows


def validate_config(config):
    if config["task"] != "hellaswag" or config["split"] != {"stack": "enc", "where": "after_layer", "index": 9}:
        raise ValueError("S0 requires HellaSwag baseline enc_l9 configuration")
    codec = config["codec"]
    expected = {"hidden_dim": 1152, "bottleneck_dim": 512, "n_res_blocks": 2,
                "layernorm": "none", "snr_film": False}
    if any(codec.get(key) != value for key, value in expected.items()) or codec.get("architecture", "residual_mlp") != "residual_mlp":
        raise ValueError("S0 requires adopted residual H1152/B512 baseline")
    if codec.get("activation") != "gelu" or codec.get("dropout", 0.0) != 0.0:
        raise ValueError("S0 requires GELU and zero dropout")
    if config["channel"]["type"] != "awgn" or not config["channel"]["normalize_power"]:
        raise ValueError("S0 requires normalized AWGN channel")


def batch_from_rows(rows, pad_id):
    return Collator(pad_id)([{"input_ids": row["input_ids"], "label_ids": row["label_ids"],
                             "attention_mask": [1] * len(row["input_ids"]),
                             "row_id": row["row_id"]} for row in rows])


@contextmanager
def without_main_hook(model):
    """Restore the existing main hook even when a diagnostic fails."""
    if model.split != {"stack": "enc", "where": "after_layer", "index": 9} or model.memory_codec is not None:
        raise ValueError("diagnostic routing requires the encoder l9 wrapper")
    layers = stack_module(model.base, "enc").layers
    if len(layers) <= max(SITES):
        raise ValueError("backbone lacks required encoder sites")
    model.handle.remove()
    try:
        yield layers
    finally:
        model.handle = layers[9].register_forward_hook(model._hook)


def encoder_forward(model, batch):
    device = next(model.base.parameters()).device
    with model.attention_context():
        model.base.get_encoder()(input_ids=batch["input_ids"].to(device),
                                 attention_mask=batch["attention_mask"].to(device), return_dict=True)


def capture(model, batch, deadline, sites=SITES):
    """Unmodified full sequences, stop immediately after the deepest site."""
    if tuple(sites) != SITES and (len(sites) != 1 or sites[0] not in SITES):
        raise ValueError("only l4/l9/l19 may be captured")
    check_deadline(deadline)
    found, handles = {}, []
    with without_main_hook(model) as layers:
        try:
            for site in sites:
                def observe(_module, _args, output, site=site):
                    check_deadline(deadline)
                    value = output[0] if isinstance(output, tuple) else output
                    finite(value, "activation")
                    if site in found:
                        raise RuntimeError("site captured more than once")
                    if value.ndim != 3 or value.shape[:2] != batch["attention_mask"].shape or value.shape[-1] != model.codec.encoder[0].in_features:
                        raise ValueError("site dimension or full-mask shape mismatch")
                    found[site] = value.detach().clone()
                    if site == max(sites):
                        raise BoundaryReached
                handles.append(layers[site].register_forward_hook(observe))
            with torch.no_grad(), model.transmission(None, bypass=True, encoder_mask=batch["attention_mask"]):
                try:
                    encoder_forward(model, batch)
                except BoundaryReached:
                    pass
            if set(found) != set(sites) or sum(model.channel_uses.values()) != 0:
                raise RuntimeError("clean atlas routing/count mismatch")
        finally:
            for handle in handles:
                handle.remove()
    check_deadline(deadline)
    return found


def local_loss(reconstructed, activation, mask):
    numerator, denominator = reconstruction_loss_stats(reconstructed, activation, mask)
    return numerator / denominator


def parity_pair(model, batch, site, snr, deadline, seed=0):
    """One online insertion versus replay; gradients only through local codec."""
    if site not in SITES or snr not in (None, 6.0):
        raise ValueError("unsupported S0 parity condition")
    clean = capture(model, batch, deadline, (site,))[site]
    mask = batch["attention_mask"].to(clean.device)
    namespace = f"shared-encoder-s0:l{site}"
    calls = 0
    online: dict[str, Any] = {}
    with without_main_hook(model) as layers:
        def transmit(_module, _args, output):
            nonlocal calls
            check_deadline(deadline)
            calls += 1
            activation = output[0] if isinstance(output, tuple) else output
            online["activation"] = activation.detach().clone()
            online["reconstruction"] = model._roundtrip(activation)
            raise BoundaryReached
        handle = layers[site].register_forward_hook(transmit)
        try:
            with torch.enable_grad(), model.channel.replay(seed, namespace, capture=True), model.transmission(snr, encoder_mask=mask):
                try:
                    encoder_forward(model, batch)
                except BoundaryReached:
                    pass
                if calls != 1:
                    raise RuntimeError("online parity must transmit exactly once")
                online["counts"] = (dict(model.channel_uses), dict(model.channel_uses_valid))
                online["draws"] = copy.deepcopy(model.channel.draw_summaries)
                online["loss"] = local_loss(online["reconstruction"], online["activation"], mask)
                parameters = tuple(model.codec.parameters())
                online["gradients"] = torch.autograd.grad(online["loss"], parameters)
        finally:
            handle.remove()
    check_deadline(deadline)
    with torch.enable_grad(), model.channel.replay(seed, namespace, capture=True), model.transmission(snr, encoder_mask=mask):
        reconstructed = model._roundtrip(clean)
        loss = local_loss(reconstructed, clean, mask)
        gradients = torch.autograd.grad(loss, tuple(model.codec.parameters()))
        counts = (dict(model.channel_uses), dict(model.channel_uses_valid))
        draws = copy.deepcopy(model.channel.draw_summaries)
    expected_allocated = batch["attention_mask"].numel() * model.codec.config["bottleneck_dim"]
    expected_valid = int(mask.sum()) * model.codec.config["bottleneck_dim"]
    expected_counts = ({"hidden": expected_allocated, "memory": 0}, {"hidden": expected_valid, "memory": 0})
    if counts != expected_counts or online["counts"] != expected_counts or draws != online["draws"]:
        raise RuntimeError("online/replay payload or actual AWGN draw mismatch")
    if len(draws) != (0 if snr is None else 1):
        raise RuntimeError("unexpected noise draw count")
    pairs = [("activation", clean, online["activation"]),
             ("reconstruction", reconstructed, online["reconstruction"]), ("nmse", loss, online["loss"])]
    pairs += [(f"gradient_{index}", actual, expected) for index, (actual, expected) in enumerate(zip(gradients, online["gradients"]))]
    comparisons = []
    for name, actual, expected in pairs:
        finite(actual, name)
        finite(expected, name)
        delta = float((actual.detach().float() - expected.detach().float()).abs().max())
        comparisons.append({"name": name, "equal": torch.equal(actual, expected), "max_abs_error": delta,
                            "online_sha256": tensor_digest(expected.reshape(1) if expected.ndim == 0 else expected),
                            "replay_sha256": tensor_digest(actual.reshape(1) if actual.ndim == 0 else actual)})
    result = {"site": site, "snr_db": snr, "online_hook_calls": calls, "optimizer_updates": 0,
              "allocated_coordinates": expected_allocated, "valid_coordinates": expected_valid,
              "awgn_draws": draws, "comparisons": comparisons,
              "passed": all(row["equal"] for row in comparisons), "nmse": float(loss.detach())}
    check_deadline(deadline)
    return result


def activation_summary(value, mask):
    valid = value.float()[mask.bool()]
    finite(valid, "valid activation")
    counts = mask.sum(1).float() * value.shape[-1]
    energy = (value.float().square() * mask[..., None]).sum((1, 2)) / counts
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "valid_vectors": int(mask.sum()), "allocated_vectors": mask.numel(),
            "sample_rms": energy.sqrt().tolist(), "near_zero_energy_samples": int((energy < 1e-8).sum()),
            "channel_mean": valid.mean(0).tolist(), "channel_variance": valid.var(0, unbiased=False).tolist()}


def run(config, views, output, *, max_seconds: float = MAX_SECONDS, max_cache_bytes=MAX_BYTES,
        loader=build_model, started=None):
    """CPU-testable runner. CLI adds a hard subprocess timeout around this."""
    started = time.monotonic() if started is None else started
    deadline = started + max_seconds
    if not 0 < max_seconds <= MAX_SECONDS or not 1024 <= max_cache_bytes <= MAX_BYTES:
        raise ValueError("S0 caps must be positive and require <=600 seconds and 1 KiB..1 GiB")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    progress: dict[str, Any] = {"schema": SCHEMA, "status": "RUNNING", "optimizer_updates": 0,
                                "captured_microbatches": 0, "site_presentations": 0, "parity_pairs": 0}
    def receipt(status, **extra):
        progress.update(status=status, elapsed_seconds=time.monotonic() - started, **extra)
        write_json(output / "status.json", progress)
    receipt("RUNNING", stage="validation")
    try:
        check_deadline(deadline)
        validate_config(config)
        rows = validate_views(views, config)
        config = copy.deepcopy(config)
        identity = {"config_sha256": canonical_digest(config), "views_sha256": canonical_digest(views),
                    "source": source_state(), "training_source_sha256": training_source_digest(),
                    "script_sha256": file_digest(Path(__file__)), "sites": list(SITES),
                    "max_seconds": max_seconds, "max_serialized_bytes": max_cache_bytes,
                    "parity_policy": "exact-equality; no-noise and replay-audited AWGN +6dB"}
        budget_json(output / "identity.json", identity, max_cache_bytes)
        budget_json(output / "config.json", config, max_cache_bytes)
        budget_json(output / "views.json", views, max_cache_bytes)
        if sum(path.stat().st_size for path in output.iterdir()) > max_cache_bytes:
            raise RuntimeError("serialized byte cap exceeded by input receipts")
        configure_training_determinism(config["training"])
        seed_everything(config["seed"])
        receipt("RUNNING", stage="model_load")
        load_started = time.monotonic()
        if config["model"]["device"].startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(torch.device(config["model"]["device"]))
        processor, model = loader(config)
        check_deadline(deadline)
        if processor.pad_token_id != views["pad_token_id"]:
            raise ValueError("frozen padding ID differs from loaded tokenizer")
        if not isinstance(model.channel, AWGNChannel):
            raise ValueError("S0 requires the replay-audited AWGN implementation")
        model.eval()
        if any(parameter.requires_grad for parameter in model.base.parameters()):
            raise ValueError("backbone must be frozen")
        initial = {name: tensor_digest(value) for name, value in model.codec.state_dict().items()}
        model_load_seconds = time.monotonic() - load_started
        shards = []
        capture_started = time.monotonic()
        for offset in range(0, 128, 16):
            receipt("RUNNING", stage="atlas")
            batch = batch_from_rows(rows[offset:offset + 16], views["pad_token_id"])
            found = capture(model, batch, deadline)
            saved = {**batch, "activations": {site: value.cpu() for site, value in found.items()},
                     "view_ids": [row["view_sha256"] for row in rows[offset:offset + 16]]}
            path = output / f"atlas-{offset // 16:02d}.pt"
            temporary = path.with_suffix(".partial")
            # Serialization overhead is checked before publication as well.
            estimate = sum(t.numel() * t.element_size() for t in batch.values()) + sum(t.numel() * t.element_size() for t in found.values())
            used = sum(p.stat().st_size for p in output.iterdir())
            if used + estimate + 1024 > max_cache_bytes:
                raise RuntimeError("serialized byte cap would be exceeded")
            torch.save(saved, temporary)
            if sum(p.stat().st_size for p in output.iterdir()) + 1024 > max_cache_bytes:
                temporary.unlink()
                raise RuntimeError("serialized byte cap exceeded")
            check_deadline(deadline)
            temporary.replace(path)
            shards.append({"file": path.name, "sha256": file_digest(path), "bytes": path.stat().st_size,
                           "row_ids": batch["row_ids"].tolist(), "view_ids": saved["view_ids"],
                           "input_ids_sha256": tensor_digest(batch["input_ids"]),
                           "mask_sha256": tensor_digest(batch["attention_mask"]),
                           "sites": {str(site): {"activation_sha256": tensor_digest(value),
                                                 **activation_summary(value, batch["attention_mask"])}
                                     for site, value in saved["activations"].items()}})
            progress.update(captured_microbatches=len(shards), site_presentations=len(shards) * 16 * len(SITES))
        capture_seconds = time.monotonic() - capture_started
        parity_started = time.monotonic()
        # Retokenization is unnecessary: use the exact frozen tokens, newly padded as batch64.
        batch = batch_from_rows(rows[:64], views["pad_token_id"])
        parity = []
        for site in SITES:
            for snr in (None, 6.0):
                receipt("RUNNING", stage="parity")
                result = parity_pair(model, batch, site, snr, deadline, config["seed"])
                parity.append(result)
                budget_json(output / "parity.json", parity, max_cache_bytes)
                progress["parity_pairs"] = len(parity)
                if not result["passed"]:
                    raise RuntimeError("online/replay exact parity failed; see parity.json")
        if initial != {name: tensor_digest(value) for name, value in model.codec.state_dict().items()}:
            raise RuntimeError("codec state changed despite zero-update contract")
        device = next(model.base.parameters()).device
        memory = ({"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)} if device.type == "cuda" else None)
        manifest = {"schema": SCHEMA, "status": "complete", "identity": identity,
                    "identity_sha256": canonical_digest(identity), "optimizer_updates": 0,
                    "atlas_rows": 128, "site_presentations": 384, "microbatches": 8,
                    "parity_pairs": 6, "parity_rows_per_pair": 64, "sites": list(SITES),
                    "codec_initial_sha256": canonical_digest(initial), "shards": shards,
                    "parity_sha256": file_digest(output / "parity.json"),
                    "model_load_seconds": model_load_seconds, "capture_seconds": capture_seconds,
                    "parity_seconds": time.monotonic() - parity_started, "cuda_memory": memory,
                    "runtime": {"torch": str(torch.__version__), "transformers": version("transformers"),
                                "cuda": torch.version.cuda, "device": str(device),
                                "backbone_dtype": str(next(model.base.parameters()).dtype),
                                "codec_dtype": str(next(model.codec.parameters()).dtype),
                                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None}}
        check_deadline(deadline)
        budget_json(output / "manifest.json", manifest, max_cache_bytes)
        receipt("COMPLETE", stage="complete")
        if sum(p.stat().st_size for p in output.iterdir()) > max_cache_bytes:
            raise RuntimeError("serialized byte cap exceeded by final receipts")
        check_deadline(deadline)
        return manifest
    except BaseException as exc:
        (output / "manifest.json").unlink(missing_ok=True)
        receipt("FAILED_OR_PARTIAL", error=type(exc).__name__, message=str(exc))
        raise


def read_complete(output):
    """Verify the diagnostic artifact; never feed this schema to production replay."""
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    status = json.loads((output / "status.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete" or status.get("status") != "COMPLETE":
        raise ValueError("incomplete or unsupported S0 artifact")
    if (manifest.get("sites") != list(SITES) or manifest.get("atlas_rows") != 128
            or manifest.get("site_presentations") != 384 or manifest.get("microbatches") != 8
            or manifest.get("parity_pairs") != 6 or manifest.get("optimizer_updates") != 0
            or len(manifest.get("shards", [])) != 8):
        raise ValueError("S0 count/site contract mismatch")
    if canonical_digest(manifest["identity"]) != manifest["identity_sha256"]:
        raise ValueError("identity hash mismatch")
    if file_digest(output / "parity.json") != manifest["parity_sha256"]:
        raise ValueError("parity hash mismatch")
    if (canonical_digest(json.loads((output / "views.json").read_text())) != manifest["identity"]["views_sha256"]
            or canonical_digest(json.loads((output / "config.json").read_text())) != manifest["identity"]["config_sha256"]):
        raise ValueError("frozen input identity mismatch")
    for shard in manifest["shards"]:
        path = output / shard["file"]
        if path.parent != output or file_digest(path) != shard["sha256"] or path.stat().st_size != shard["bytes"]:
            raise ValueError("shard hash/size mismatch")
    return manifest



def prepare_views(config, ids, output, *, max_seconds: float = MAX_SECONDS, max_cache_bytes=MAX_BYTES, started=None):
    """Use the existing native five-shot builder; materialize only 128 query views."""
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from jscc.data.hellaswag_prompts import PromptBuilder, validate_policy

    started = time.monotonic() if started is None else started
    deadline = started + max_seconds
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    status = {"schema": VIEW_SCHEMA, "optimizer_updates": 0, "status": "RUNNING"}
    write_json(output / "status.json", status)
    try:
        validate_config(config)
        policy = config["data"].get("prompt_policy", {})
        validate_policy(policy, config["data"]["max_length"])
        if policy["mode"] != "five_shot":
            raise ValueError("S0 needs native five-shot views")
        if ids.get("validation_split") != "train" or set(ids["train_rows"]) & set(ids["validation_rows"]):
            raise ValueError("supplied data IDs must have a disjoint train selection holdout")
        if len(ids["train_rows"]) < 64 or len(ids["validation_rows"]) < 64:
            raise ValueError("at least 64 IDs per group required")
        selected = list(ids["train_rows"][:64]) + list(ids["validation_rows"][:64])
        budget_json(output / "selected_rows.json", {"policy": "first64-in-supplied-ID-order-per-group",
                                                    "train": selected[:64], "selection": selected[64:],
                                                    "data_ids_sha256": canonical_digest(ids)}, max_cache_bytes)
        check_deadline(deadline)
        raw = load_dataset(config["data"]["name"], revision=config["data"]["revision"])["train"]
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"], revision=config["model"]["revision"])
        check_deadline(deadline)
        builder = PromptBuilder(raw, ids, policy)
        rows = []
        for offset in range(0, 128, 16):
            check_deadline(deadline)
            row_ids = selected[offset:offset + 16]
            tokenized = builder.tokenize(raw.select(row_ids).to_dict(), row_ids, tokenizer)
            for position, row_id in enumerate(row_ids):
                row = {"row_id": row_id, "group": "train" if offset < 64 else "selection",
                       "input_ids": tokenized["input_ids"][position], "label_ids": tokenized["label_ids"][position],
                       "demo_ids": tokenized["demo_ids"][position], "source_id": tokenized["source_id"][position],
                       "demo_source_ids": tokenized["demo_source_ids"][position],
                       "source_length": tokenized["source_length"][position],
                       "source_sha256": tokenized["source_hash"][position],
                       "prompt_text_sha256": tokenized["text_hash"][position]}
                row["input_sha256"] = canonical_digest(row["input_ids"])
                row["target_sha256"] = canonical_digest(row["label_ids"])
                row["view_sha256"] = canonical_digest(row)
                rows.append(row)
        views = {"schema": VIEW_SCHEMA, "config_sha256": canonical_digest(config), "data_ids": ids,
                 "tokenizer": {key: config["model"][key] for key in ("name", "revision")},
                 "pad_token_id": tokenizer.pad_token_id, "rows": rows,
                 "source": source_state(), "training_source_sha256": training_source_digest(),
                 "script_sha256": file_digest(Path(__file__)), "preprocess_sha256": builder.helper_digest,
                 "dataset_fingerprint": raw._fingerprint,
                 "row_selection_policy": "first64-in-supplied-ID-order-per-group",
                 "source_truncation_policy": "existing prompt-alignment-v1 left2048; target512"}
        validate_views(views, config)
        check_deadline(deadline)
        budget_json(output / "views.json", views, max_cache_bytes)
        write_json(output / "status.json", {**status, "status": "COMPLETE", "views": 128,
                                             "views_file_sha256": file_digest(output / "views.json"),
                                             "elapsed_seconds": time.monotonic() - started})
        check_deadline(deadline)
        return views
    except BaseException as exc:
        (output / "views.json").unlink(missing_ok=True)
        write_json(output / "status.json", {**status, "status": "FAILED_OR_PARTIAL", "error": type(exc).__name__,
                                             "message": str(exc), "elapsed_seconds": time.monotonic() - started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--views", help="explicit frozen tokenized row/view JSON")
    parser.add_argument("--prepare-views", action="store_true", help="prepare128 native five-shot views; no model load")
    parser.add_argument("--data-ids", help="existing pinned data_ids.json for --prepare-views")
    parser.add_argument("--output", required=True, help="new directory; existing paths are rejected")
    parser.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    parser.add_argument("--max-cache-bytes", type=int, default=MAX_BYTES)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= MAX_SECONDS or not 1024 <= args.max_cache_bytes <= MAX_BYTES:
        parser.error("caps require <=600 seconds and 1 KiB..1 GiB")
    if (args.prepare_views and (not args.data_ids or args.views)) or (not args.prepare_views and (not args.views or args.data_ids)):
        parser.error("use --prepare-views --data-ids OR --views")
    output = Path(args.output)
    if output.exists():
        parser.error("output must be a new directory")
    started = time.monotonic()
    if args.worker:
        if args.prepare_views:
            prepare_views(load_config(args.config), json.loads(Path(args.data_ids).read_text()), output,
                          max_seconds=args.max_seconds, max_cache_bytes=args.max_cache_bytes, started=started)
            return
        run(load_config(args.config), json.loads(Path(args.views).read_text()), output,
            max_seconds=args.max_seconds, max_cache_bytes=args.max_cache_bytes, started=started)
        return
    command = [sys.executable, "-m", "scripts.shared_encoder_atlas", *sys.argv[1:], "--worker"]
    process = subprocess.Popen(command)
    try:
        code = process.wait(timeout=max(0.001, args.max_seconds - (time.monotonic() - started)))
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        output.mkdir(parents=True, exist_ok=True)
        (output / "manifest.json").unlink(missing_ok=True)
        if args.prepare_views:
            (output / "views.json").unlink(missing_ok=True)
        status_path = output / "status.json"
        try:
            previous = json.loads(status_path.read_text())
        except (OSError, ValueError):
            previous = {}
        write_json(status_path, {**previous, "schema": SCHEMA, "status": "FAILED_OR_PARTIAL",
                                 "error": "HardWallTimeout", "optimizer_updates": 0,
                                 "elapsed_seconds": time.monotonic() - started})
        raise SystemExit("S0 hard wall ceiling reached; child killed, partial artifacts retained")
    if code:
        output.mkdir(parents=True, exist_ok=True)
        (output / "manifest.json").unlink(missing_ok=True)
        if args.prepare_views:
            (output / "views.json").unlink(missing_ok=True)
        status_path = output / "status.json"
        try:
            previous = json.loads(status_path.read_text())
        except (OSError, ValueError):
            previous = {}
        if previous.get("status") != "FAILED_OR_PARTIAL":
            write_json(status_path, {**previous, "schema": SCHEMA, "status": "FAILED_OR_PARTIAL",
                                     "error": "WorkerExit", "returncode": code, "optimizer_updates": 0,
                                     "elapsed_seconds": time.monotonic() - started})
        raise SystemExit(code)
    if args.prepare_views:
        status = json.loads((output / "status.json").read_text())
        if status.get("status") != "COMPLETE" or file_digest(output / "views.json") != status["views_file_sha256"]:
            raise ValueError("view preparation incomplete or hash mismatch")
    else:
        read_complete(output)


if __name__ == "__main__":
    main()
