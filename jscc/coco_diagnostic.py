"""Frozen COCO query/endpoint diagnostic. No training or remote access."""

from __future__ import annotations

from contextlib import contextmanager
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import signal
import tarfile
import time
from typing import Any, cast

import torch
import yaml

from .data.coco import caption_prompt
from .evaluation import clean_caption
from .runtime import autocast_for, prepare_trainable_parameters


MODES = ("vanilla", "enc_l9", "enc_fn", "dec_l8")
VARIANTS = ("complete", "without_terminal_period", "plus_newline")
CONTROLS = ("matched", "permuted")
MAX_REQUESTS = 1024
TIMING_REQUESTS = 128
FROZEN_SELECTION_SHA256 = "bf568250961cb71ac4b00aa65811b2854736e4b9e3734a48d05fa5bbce3ba6b8"
FROZEN_CHECKPOINT_INDEX_SHA256 = "309a071f95405d58007bfaed497ffd5a74dc4ac6baa8c8bb846d11758b7a7b1a"
EXTRA_SOURCE = ("scripts/coco_query_endpoint_diagnostic.py", "train.py", "pyproject.toml", "uv.lock")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_sha256(value: Any) -> str:
    return bytes_sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def require_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"Missing or invalid SHA-256 for {label}")
    return value


def check_hash(path: Path, expected: Any, label: str) -> str:
    expected = require_digest(expected, label)
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {path}")
    return actual


def validate_selection(selection: dict[str, Any], ids: dict[str, Any]) -> None:
    image_ids = selection["image_ids"]
    donors = selection["permuted_query_image_ids"]
    demos = selection["demo_ids"]
    if len(image_ids) != 32 or len(set(image_ids)) != 32 or len(demos) != 4 or len(set(demos)) != 4:
        raise ValueError("Selection requires 32 unique images and four unique demos")
    if donors != image_ids[1:] + image_ids[:1]:
        raise ValueError("Query donor permutation must be the frozen one-step cycle")
    if image_ids != ids["selection_ids"][:32] or demos != ids["demo_ids"]:
        raise ValueError("Frozen selection/demo IDs differ from saved run IDs")
    groups = [ids[key] for key in ("train_ids", "demo_ids", "selection_ids", "report_ids")]
    flat = [number for group in groups for number in group]
    if len(flat) != len(set(flat)):
        raise ValueError("Saved train/demo/selection/report IDs overlap or contain duplicates")
    if not selection.get("no_report_overlap") or set(image_ids) & set(ids["report_ids"]):
        raise ValueError("Selection/report overlap")


def source_digest(repo: Path) -> dict[str, str]:
    files = sorted(repo.joinpath("jscc").rglob("*.py"))
    files += [repo / name for name in EXTRA_SOURCE]
    return {str(path.relative_to(repo)): sha256(path) for path in files}


def runtime_packages() -> dict[str, str]:
    return {name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "datasets", "pillow")}


def prepare_manifest(*, repo: Path, selection_path: Path, checkpoint_index_path: Path,
                     bindings_path: Path, manifest_path: Path) -> dict[str, Any]:
    """Hash explicitly named local artifacts; no model/data loading."""
    bindings = read_json(bindings_path)
    index = read_json(checkpoint_index_path)
    historical = bindings["historical_source"]
    for key in ("archive", "manifest"):
        historical[key] = {"path": str(Path(historical[key]).resolve()),
                           "sha256": sha256(Path(historical[key]))}
    archive_digest = verify_historical_archive(Path(historical["archive"]["path"]),
                                               Path(historical["manifest"]["path"]))
    modes = {}
    for mode in MODES:
        paths = {key: Path(bindings["modes"][mode][key]).resolve()
                 for key in ("checkpoint", "config", "run", "data_ids")}
        if sha256(paths["checkpoint"]) != index[mode]["checkpoint_sha256"]:
            raise ValueError(f"{mode} checkpoint hash differs from frozen index")
        run = read_json(paths["run"])
        if run.get("training_source_sha256") != archive_digest:
            raise ValueError(f"{mode} training source differs from historical archive")
        modes[mode] = {key: str(path) for key, path in paths.items()}
        modes[mode].update(checkpoint_sha256=sha256(paths["checkpoint"]),
                           config_sha256=sha256(paths["config"]),
                           run_sha256=sha256(paths["run"]),
                           data_ids_sha256=sha256(paths["data_ids"]),
                           training_source_sha256=archive_digest)
    manifest = {"protocol": "coco-query-endpoint-v1", "batch_size": 1,
                "max_new_tokens": 64, "selection_sha256": sha256(selection_path),
                "active_source_files": source_digest(repo), "runtime_packages": runtime_packages(),
                "historical_source": historical, "modes": modes}
    with manifest_path.open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return verify_plan(repo=repo, selection_path=selection_path,
                       checkpoint_index_path=checkpoint_index_path, manifest_path=manifest_path)


def verify_historical_archive(archive_path: Path, manifest_path: Path) -> str:
    """Recompute the historical training_source_digest from actual archive bytes."""
    manifest = read_json(manifest_path)
    digest = hashlib.sha256()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = {member.name.removeprefix("./"): member for member in archive.getmembers() if member.isfile()}
        if set(members) != set(manifest):
            raise ValueError("Historical source manifest and archive file lists differ")
        for name, member in members.items():
            stream = archive.extractfile(member)
            if stream is None or bytes_sha256(stream.read()) != manifest[name]:
                raise ValueError(f"Historical source member mismatch: {name}")
        python = sorted(name for name in members if name.startswith("jscc/") and name.endswith(".py"))
        for name in python:
            digest.update(name.removeprefix("jscc/").encode())
            stream = archive.extractfile(members[name])
            assert stream is not None
            digest.update(stream.read())
        for name in ("train.py", "uv.lock", "pyproject.toml"):
            digest.update(name.encode())
            stream = archive.extractfile(members[name])
            assert stream is not None
            digest.update(stream.read())
    return digest.hexdigest()


def verify_plan(*, repo: Path, selection_path: Path, checkpoint_index_path: Path,
                manifest_path: Path) -> dict[str, Any]:
    """Read only files and JSON/YAML. This path never imports a model loader."""
    selection = read_json(selection_path)
    index = read_json(checkpoint_index_path)
    manifest = read_json(manifest_path)
    if sha256(selection_path) != FROZEN_SELECTION_SHA256 or sha256(checkpoint_index_path) != FROZEN_CHECKPOINT_INDEX_SHA256:
        raise ValueError("Selection or checkpoint index differs from frozen archive")
    if set(manifest["modes"]) != set(MODES) or set(index) & set(MODES) != set(MODES):
        raise ValueError("Manifest/index must bind vanilla and exactly three codecs")
    if manifest.get("protocol") != "coco-query-endpoint-v1":
        raise ValueError("Manifest protocol mismatch")
    if manifest.get("batch_size") != 1 or manifest.get("max_new_tokens") != 64:
        raise ValueError("Frozen batch/generation settings mismatch")
    if manifest.get("selection_sha256") != sha256(selection_path):
        raise ValueError("Frozen selection hash mismatch")
    actual_source = source_digest(repo)
    if manifest.get("active_source_files") != actual_source:
        raise ValueError("Active diagnostic source file hashes differ from verified manifest")
    if manifest.get("runtime_packages") != runtime_packages():
        raise ValueError("Installed runtime packages differ from verified manifest")
    historical = manifest.get("historical_source")
    if not isinstance(historical, dict):
        raise ValueError("Verified historical source archive and manifest are required")
    archive_binding, files_binding = historical["archive"], historical["manifest"]
    check_hash(Path(archive_binding["path"]), archive_binding.get("sha256"), "historical source archive")
    check_hash(Path(files_binding["path"]), files_binding.get("sha256"), "historical source manifest")
    archive_source_digest = verify_historical_archive(Path(archive_binding["path"]), Path(files_binding["path"]))
    modes = {}
    shared_ids = None
    shared_recipe = None
    for mode in MODES:
        binding = manifest["modes"][mode]
        archived = index[mode]
        checkpoint = Path(binding["checkpoint"])
        if Path(archived["checkpoint"]).name != checkpoint.name:
            raise ValueError(f"{mode} checkpoint filename differs from archive")
        check_hash(checkpoint, archived.get("checkpoint_sha256"), f"{mode} checkpoint")
        if binding.get("checkpoint_sha256") != archived["checkpoint_sha256"]:
            raise ValueError(f"{mode} checkpoint binding differs from archive")
        config_path, run_path, ids_path = (Path(binding[key]) for key in ("config", "run", "data_ids"))
        run = read_json(run_path)
        ids = read_json(ids_path)
        config_hash = check_hash(config_path, run.get("resolved_config_sha256"), f"{mode} resolved config")
        if binding.get("config_sha256") != config_hash or binding.get("run_sha256") != sha256(run_path) or binding.get("data_ids_sha256") != sha256(ids_path):
            raise ValueError(f"{mode} config/run/ID manifest binding mismatch")
        historical_digest = require_digest(run.get("training_source_sha256"), f"{mode} historical training source")
        if binding.get("training_source_sha256") != historical_digest or historical_digest != archive_source_digest:
            raise ValueError(f"{mode} historical training source binding mismatch")
        config = yaml.safe_load(config_path.read_text())
        if config["task"] != "coco" or config["model"]["dtype"] != "bfloat16" or config["model"]["sdpa_backend_policy"] != "flash_math":
            raise ValueError(f"{mode} frozen task/precision/attention policy mismatch")
        if config["evaluation"]["max_new_tokens"] != 64 or config["data"]["num_demos"] != 4:
            raise ValueError(f"{mode} generation/demo recipe mismatch")
        if config["split"] != ({"stack": "enc", "where": "after_layer", "index": 9} if mode == "enc_l9" else
                               {"stack": "enc", "where": "after_final_norm"} if mode in ("enc_fn", "vanilla") else
                               {"stack": "dec", "where": "after_layer", "index": 8}):
            raise ValueError(f"{mode} split differs from frozen route")
        if mode != "vanilla" and config["training"]["max_steps"] != 500:
            raise ValueError(f"{mode} training step recipe mismatch")
        if shared_ids is None:
            shared_ids = ids
        elif ids != shared_ids:
            raise ValueError("Route data IDs differ")
        recipe = (config["model"]["name"], config["model"]["revision"],
                  config["data"]["name"], config["data"]["revision"],
                  config["data"]["karpathy_name"], config["data"]["karpathy_revision"])
        if shared_recipe is None:
            shared_recipe = recipe
        elif recipe != shared_recipe:
            raise ValueError("Route model/data revisions differ")
        modes[mode] = {"checkpoint": str(checkpoint), "config": config, "data_ids": ids,
                       "checkpoint_sha256": archived["checkpoint_sha256"],
                       "config_sha256": config_hash, "data_ids_sha256": sha256(ids_path),
                       "training_source_sha256": historical_digest}
    assert shared_ids is not None
    validate_selection(selection, shared_ids)
    return {"selection": selection, "ids": shared_ids, "modes": modes,
            "manifest_sha256": sha256(manifest_path), "selection_sha256": sha256(selection_path),
            "checkpoint_index_sha256": sha256(checkpoint_index_path),
            "active_source_files": actual_source, "historical_source_manifest": historical}


def image_id(row: dict[str, Any]) -> int:
    return int(row["file_name"].rsplit("_", 1)[1].split(".")[0])


def load_rows_offline(config: dict[str, Any], ids: dict[str, Any], selection: dict[str, Any]):
    from datasets import load_dataset
    data = config["data"]
    raw = load_dataset(data["name"], split="val", revision=data["revision"])
    val = set(load_dataset(data["karpathy_name"], split="validation", revision=data["karpathy_revision"])["cocoid"])
    test = set(load_dataset(data["karpathy_name"], split="test", revision=data["karpathy_revision"])["cocoid"])
    if not set(selection["image_ids"]) <= val or set(selection["image_ids"]) & test:
        raise ValueError("Selection is outside Karpathy validation or overlaps test")
    if (set(ids["train_ids"]) | set(ids["demo_ids"])) & (val | test):
        raise ValueError("Train/demo IDs leak into Karpathy validation/test")
    wanted = set(selection["image_ids"]) | set(selection["demo_ids"])
    positions = {}
    for position, filename in enumerate(raw["file_name"]):
        number = image_id({"file_name": filename})
        if number in wanted:
            if number in positions:
                raise ValueError(f"Duplicate COCO image {number}")
            positions[number] = position
    if set(positions) != wanted:
        raise ValueError(f"Missing cached COCO images: {sorted(wanted - set(positions))}")
    return {number: cast(dict[str, Any], raw[position]) for number, position in positions.items()}


def prefix_variants(reference: str) -> dict[str, tuple[str, bool]]:
    removed = reference[:-1] if reference.endswith(".") else reference
    return {"complete": (reference, False),
            "without_terminal_period": (removed, removed == reference),
            "plus_newline": (reference + "\n", False)}


def prefix_inputs(tokenizer, base, text: str, *, max_positions: int = 65):
    """Native shift: BOS then every lexical prefix token, score at last position."""
    lexical = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    eos = tokenizer.eos_token_id
    if eos is None or eos in lexical[0].tolist():
        raise ValueError("Prefix contains EOS or tokenizer has no EOS")
    if lexical.shape[1] + 1 > max_positions:
        return None, lexical
    labels = torch.cat((lexical, torch.tensor([[eos]], dtype=lexical.dtype)), dim=1)
    decoder = base.prepare_decoder_input_ids_from_labels(labels=labels)
    if decoder.shape[1] != labels.shape[1] or not torch.equal(decoder[:, 1:], lexical):
        raise ValueError("Native decoder shift does not align with reference prefix")
    return decoder, lexical


def rank_of(logits: torch.Tensor, token_id: int) -> int:
    return int((logits > logits[token_id]).sum().item()) + 1


def score_logits(logits: torch.Tensor, eos_id: int, newline_ids: list[int]) -> dict[str, Any]:
    values = logits.detach().float().cpu().flatten()
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("Nonfinite raw endpoint logits")
    probabilities = torch.softmax(values, dim=-1)
    first = newline_ids[0] if newline_ids else None
    return {"next_token_top1": int(values.argmax()), "eos_probability": float(probabilities[eos_id]),
            "eos_rank": rank_of(values, eos_id),
            "newline_first_token_probability": float(probabilities[first]) if first is not None else None,
            "newline_first_token_rank": rank_of(values, first) if first is not None else None,
            "newline_sequence_probability": float(probabilities[first]) if len(newline_ids) == 1 else None}


def teacher_student_kl(teacher: torch.Tensor, student: torch.Tensor) -> float:
    left, right = teacher.detach().float().cpu(), student.detach().float().cpu()
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        raise FloatingPointError("Nonfinite teacher/student logits")
    log_p, log_q = torch.log_softmax(left, -1), torch.log_softmax(right, -1)
    return float(torch.sum(log_p.exp() * (log_p - log_q)))


def tensor_sha256(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().contiguous().cpu()
    return bytes_sha256(str(tuple(tensor.shape)).encode() + str(tensor.dtype).encode() + tensor.numpy().tobytes())


def reference_alignment(caption: str, references: list[str]) -> dict[str, float | None]:
    """Descriptive lexical F1 only; not CIDEr or a caption-quality estimate."""
    def tokens(value):
        return re.findall(r"\w+", value.lower())
    predicted = Counter(tokens(caption))
    scores = []
    for reference in references:
        expected = Counter(tokens(reference))
        overlap = sum((predicted & expected).values())
        denominator = sum(predicted.values()) + sum(expected.values())
        if denominator:
            scores.append(2 * overlap / denominator)
    return {"max_token_f1": max(scores) if scores else None}


def feature_signatures(model) -> dict[str, str | None]:
    return {name: tensor_sha256(value) if torch.is_tensor(value) else None
            for name, value in (("boundary_input", model.activation),
                                ("boundary_reconstruction", model.reconstruction),
                                ("receiver_memory_input", model.memory_activation),
                                ("receiver_memory_reconstruction", model.memory_reconstruction))}


def prepare_pair(processor, demos: list[Any], captions: list[str], matched: Any, donor: Any):
    prompt = caption_prompt(captions)
    images = ([image.convert("RGB") for image in demos] + [matched.convert("RGB")],
              [image.convert("RGB") for image in demos] + [donor.convert("RGB")])
    source = []
    details = []
    for group in images:
        processed = processor(images=group, text=prompt, return_tensors="pt", padding=True, truncation=False)
        item = {key: processed[key] for key in ("input_ids", "attention_mask", "pixel_values")}
        source.append(item)
        per_image = []
        for image in group:
            single = processor(images=[image], text="<start_of_image>", return_tensors="pt", padding=True, truncation=False)
            per_image.append({"rgb_sha256": bytes_sha256(image.tobytes()),
                              "pixel_sha256": tensor_sha256(single["pixel_values"]),
                              "crop_count": int(single["pixel_values"].reshape(-1, *single["pixel_values"].shape[-3:]).shape[0])})
        details.append(per_image)
    for key in ("input_ids", "attention_mask"):
        if not torch.equal(source[0][key], source[1][key]):
            raise ValueError(f"Query swap changed {key}")
    if details[0][:4] != details[1][:4] or details[0][4]["rgb_sha256"] == details[1][4]["rgb_sha256"]:
        raise ValueError("Query swap changed demos or reused identical query pixels")
    # Processor expansions must account for the five image segments; compare
    # separately processed crop counts and full multi-image crop count.
    for item, per_image in zip(source, details, strict=True):
        full_count = int(item["pixel_values"].reshape(-1, *item["pixel_values"].shape[-3:]).shape[0])
        if full_count != sum(entry["crop_count"] for entry in per_image):
            raise ValueError("Full processor crop count differs from image segments")
    left = source[0]["pixel_values"].reshape(-1, *source[0]["pixel_values"].shape[-3:])
    right = source[1]["pixel_values"].reshape(-1, *source[1]["pixel_values"].shape[-3:])
    demo_crops = sum(entry["crop_count"] for entry in details[0][:4])
    if not torch.equal(left[:demo_crops], right[:demo_crops]):
        raise ValueError("Processed demonstration image crops changed during query swap")
    if torch.equal(left[demo_crops:], right[demo_crops:]):
        raise ValueError("Processed query crops did not change")
    return source, details, prompt


class RequestBudget:
    def __init__(self, max_seconds: float, reserve_seconds: float, *, now=time.monotonic):
        if not math.isfinite(max_seconds) or max_seconds <= 0 or not math.isfinite(reserve_seconds) or not 0 <= reserve_seconds < max_seconds:
            raise ValueError("Require finite positive deadline and smaller nonnegative reserve")
        self.now = now
        self.start = now()
        self.deadline = self.start + max_seconds
        self.stop_new = self.deadline - reserve_seconds
        self.attempts = 0
        self.completed = 0
        self.decoder_calls = 0
        self.encoder_calls = 0

    def check_time(self):
        """Guard setup and pair preparation without consuming a request slot."""
        if self.now() >= self.stop_new:
            raise TimeoutError("Deadline reserve reached before setup or input preparation")

    def begin(self) -> int:
        if self.attempts >= MAX_REQUESTS:
            raise RuntimeError("Logical request ceiling reached")
        self.check_time()
        self.attempts += 1
        return self.attempts

    def complete(self, *, decoder_calls: int = 0, encoder_calls: int = 0):
        self.completed += 1
        self.decoder_calls += decoder_calls
        self.encoder_calls += encoder_calls


@contextmanager
def hard_alarm(seconds: float):
    if not hasattr(signal, "setitimer"):
        raise RuntimeError("POSIX wall-clock alarm required for run mode")
    previous = signal.getsignal(signal.SIGALRM)
    def expired(*_):
        raise TimeoutError("Hard diagnostic deadline reached")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class Journal:
    def __init__(self, output: Path):
        output.mkdir(parents=True, exist_ok=False)
        self.output = output
        self.stream = (output / "items.jsonl").open("x", buffering=1)

    def append(self, value: dict[str, Any]) -> None:
        self.stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self) -> None:
        self.stream.close()


def measure_memory(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda":
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def load_mode(info: dict[str, Any], mode: str):
    from .models.split_model import build_model
    config = info["config"]
    processor, model = build_model(config)
    prepare_trainable_parameters(model)
    if mode != "vanilla":
        state = torch.load(info["checkpoint"], map_location="cpu", weights_only=True)
        if int(state["step"]) != 500 or state["config"] != config or state["data_ids"] != info["data_ids"]:
            raise ValueError(f"{mode} checkpoint step/config/data IDs differ from archived receipt")
        model.load_communication_state(state)
    model.eval()
    return processor, model


def forward_endpoint(model, inputs, decoder, *, vanilla: bool = False):
    parameter = next(model.base.parameters())
    source = {key: value.to(device=parameter.device, dtype=parameter.dtype if key == "pixel_values" else None)
              for key, value in inputs.items()}
    decoder = decoder.to(parameter.device)
    with torch.no_grad(), autocast_for(model), model.observe_payload() as events, model.transmission(None, bypass=vanilla,
            encoder_mask=source["attention_mask"], decoder_mask=torch.ones_like(decoder, dtype=torch.bool)):
        with model.attention_context():
            result = model.base(**source, decoder_input_ids=decoder, use_cache=False, return_dict=True)
    if not vanilla and model.memory_codec is not None and not any(event["stream"] == "memory" for event in events):
        raise RuntimeError("Decoder receiver-memory codec was bypassed")
    return result.logits[0, -1].detach().float().cpu()


def generate_caption(model, processor, inputs, *, vanilla: bool = False):
    parameter = next(model.base.parameters())
    source = {key: value.to(device=parameter.device, dtype=parameter.dtype if key == "pixel_values" else None)
              for key, value in inputs.items()}
    decoder_calls = 0
    def counted(*_):
        nonlocal decoder_calls
        decoder_calls += 1
    handle = model.base.get_decoder().register_forward_hook(counted)
    try:
        with torch.no_grad(), autocast_for(model), model.observe_payload() as events, model.transmission(None, bypass=vanilla,
                encoder_mask=source["attention_mask"]):
            generated = model.generate(**source, max_new_tokens=64, do_sample=False,
                                       num_beams=1, use_cache=True)
    finally:
        handle.remove()
    if not vanilla and model.memory_codec is not None and not any(event["stream"] == "memory" for event in events):
        raise RuntimeError("Decoder receiver-memory codec was bypassed")
    tokens = generated[0].detach().cpu().tolist()
    raw = processor.tokenizer.decode(tokens, skip_special_tokens=True)
    eos = processor.tokenizer.eos_token_id
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    return {"generated_token_ids": tokens, "raw_caption": raw, "cleaned_caption": clean_caption(raw),
            "decoder_forward_calls_observed": decoder_calls,
            "eos_emitted": eos in tokens, "truncated": eos not in tokens,
            "repeated_lines": len(lines) != len(set(lines)),
            "trigram_repetition": len([tuple(raw.split()[i:i+3]) for i in range(max(0, len(raw.split())-2))]) !=
                                  len(set(tuple(raw.split()[i:i+3]) for i in range(max(0, len(raw.split())-2))))}


def run(plan: dict[str, Any], output: Path, max_seconds: float, reserve_seconds: float) -> dict[str, Any]:
    """One explicit, bounded local run; partial records are durable after every request."""
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    budget = RequestBudget(max_seconds, reserve_seconds)
    journal = Journal(output)
    status: dict[str, Any] = {"protocol": "coco-query-endpoint-v1", "status": "running",
        "expected_requests": MAX_REQUESTS, "timing_stage_requests": TIMING_REQUESTS,
        "manifest_sha256": plan["manifest_sha256"], "selection_sha256": plan["selection_sha256"],
        "active_source_files": plan["active_source_files"], "runtime_packages": runtime_packages(),
        "historical_source": plan.get("historical_source_manifest"),
        "training_updates": 0,
        "unsupported_requests": 0}
    def save_status():
        status.update(attempts=budget.attempts, completed=budget.completed,
                      decoder_calls_observed=budget.decoder_calls, encoder_calls_observed=budget.encoder_calls,
                      elapsed_seconds=budget.now() - budget.start)
        temporary = output / "status.tmp"
        temporary.write_text(json.dumps(status, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(output / "status.json")
    save_status()
    try:
        with hard_alarm(max_seconds):
            budget.check_time()
            rows = load_rows_offline(plan["modes"]["enc_l9"]["config"], plan["ids"], plan["selection"])
            selection = plan["selection"]
            # First four IDs are a complete included stage across all modes.
            for phase, indices in (("timing", range(4)), ("remainder", range(4, 32))):
                for mode in MODES:
                    budget.check_time()
                    loaded = budget.now()
                    processor, model = load_mode(plan["modes"][mode], mode)
                    model_load_seconds = budget.now() - loaded
                    tokenizer = processor.tokenizer
                    newline_ids = tokenizer("\n", add_special_tokens=False)["input_ids"]
                    if not newline_ids:
                        raise ValueError("Newline tokenizer encoding is empty")
                    demos = [rows[number]["image"] for number in selection["demo_ids"]]
                    captions = [rows[number]["answer"][0] for number in selection["demo_ids"]]
                    for index in indices:
                        budget.check_time()
                        pair_outputs: dict[str, dict[str, Any]] = {}
                        original = selection["image_ids"][index]
                        donor = selection["permuted_query_image_ids"][index]
                        original_row, donor_row = rows[original], rows[donor]
                        prep_start = budget.now()
                        budget.check_time()
                        pair, segments, prompt = prepare_pair(processor, demos, captions,
                            original_row["image"], donor_row["image"])
                        budget.check_time()
                        processing_seconds = budget.now() - prep_start
                        reference = original_row["answer"][0] if original_row["answer"] else None
                        if not isinstance(reference, str) or not reference:
                            raise ValueError(f"Image {original} has no recorded first selection target")
                        variants = prefix_variants(reference)
                        for control_index, control in enumerate(CONTROLS):
                            budget.check_time()
                            inputs = pair[control_index]
                            source_hashes = {key: tensor_sha256(value) for key, value in inputs.items()}
                            common = {"image_id": original, "query_image_id": original if control == "matched" else donor,
                                "demo_ids": selection["demo_ids"], "query_control": control, "model_mode": mode,
                                "condition": "vanilla" if mode == "vanilla" else "no_noise",
                                "checkpoint_sha256": plan["modes"][mode]["checkpoint_sha256"],
                                "training_source_sha256": plan["modes"][mode]["training_source_sha256"],
                                "source_files": plan["active_source_files"], "runtime_packages": status["runtime_packages"],
                                "source_sha256": json_sha256(plan["active_source_files"]),
                                "config_sha256": plan["modes"][mode]["config_sha256"],
                                "data_ids_sha256": plan["modes"][mode].get("data_ids_sha256"),
                                "model_revision": plan["modes"][mode]["config"]["model"]["revision"],
                                "processor_revision": plan["modes"][mode]["config"]["model"]["revision"],
                                "tokenizer_revision": plan["modes"][mode]["config"]["model"]["revision"],
                                "prompt_sha256": bytes_sha256(prompt.encode()), "input_tensor_sha256": source_hashes,
                                "image_segments": segments[control_index], "input_image_sha256": [x["rgb_sha256"] for x in segments[control_index]],
                                "references_original": original_row["answer"], "references_permuted": donor_row["answer"],
                                "reference_prefix_source": reference, "input_processing_seconds": processing_seconds,
                                "model_load_seconds": model_load_seconds, "phase": phase,
                                "encoder_mask": inputs["attention_mask"].tolist(),
                                "attention_backend": plan["modes"][mode]["config"]["model"]["sdpa_backend_policy"],
                                "dtype": plan["modes"][mode]["config"]["model"]["dtype"],
                                "cache_policy": "no endpoint replay; generation native cache",
                                "seed": plan["modes"][mode]["config"].get("seed")}
                            for kind, variant in [("generation", None), *[("endpoint", name) for name in VARIANTS]]:
                                request_id = f"{index:02d}-{mode}-{control}-{kind}-{variant or 'free'}"
                                attempt = budget.begin()
                                record: dict[str, Any] = dict(common, request_id=request_id, request_kind=kind,
                                    prefix_variant=variant, attempt=attempt, status="partial",
                                    requests_consumed=attempt, decoder_calls_consumed=None,
                                    encoder_calls_consumed=None, error=None,
                                    path="cached_generation" if kind == "generation" else "full_forward",
                                    prefix_text=None, prefix_token_ids=None, prefix_length=None,
                                    decoder_input_ids=None, decoder_mask=None, cache_position=None,
                                    eos_token_id=None, newline_token_ids=None, next_token_top1=None,
                                    eos_probability=None, eos_rank=None, newline_first_token_probability=None,
                                    newline_first_token_rank=None, newline_sequence_probability=None,
                                    teacher_student_kl=None, full_cached_max_abs_logit_delta=None,
                                    raw_logits_artifact=None, processed_scores_artifact=None,
                                    generated_token_ids=None, raw_caption=None, cleaned_caption=None,
                                    truncated=None, repeated_lines=None, trigram_regex=None,
                                    logits_artifact_sha256=None, early_stop_reason=None,
                                    peak_allocated_bytes=None, peak_reserved_bytes=None)
                                journal.append(record)
                                started = budget.now()
                                try:
                                    if kind == "generation":
                                        record.update(generate_caption(model, processor, inputs, vanilla=mode == "vanilla"),
                                            generation_kwargs={"max_new_tokens": 64, "do_sample": False,
                                                               "num_beams": 1, "use_cache": True})
                                        record["trigram_regex"] = record["trigram_repetition"]
                                        record["reference_alignment"] = {
                                            "original": reference_alignment(record["cleaned_caption"], original_row["answer"]),
                                            "donor": reference_alignment(record["cleaned_caption"], donor_row["answer"])}
                                        calls = record["decoder_forward_calls_observed"]
                                        record.update(decoder_calls_consumed=calls, encoder_calls_consumed=1)
                                    else:
                                        text, noop = variants[variant]
                                        decoder, lexical = prefix_inputs(tokenizer, model.base, text)
                                        record.update(prefix_text=text, prefix_noop=noop,
                                            prefix_token_ids=lexical[0].tolist(), prefix_length=int(lexical.shape[1]),
                                            newline_token_ids=newline_ids, eos_token_id=tokenizer.eos_token_id,
                                            decoder_input_ids=decoder[0].tolist() if decoder is not None else None,
                                            decoder_mask=[1] * decoder.shape[1] if decoder is not None else None)
                                        if decoder is None:
                                            record.update(status="unsupported", error="Native shifted prefix exceeds 65 decoder positions",
                                                decoder_calls_consumed=0, encoder_calls_consumed=0)
                                            budget.complete(decoder_calls=0, encoder_calls=0)
                                            status["unsupported_requests"] += 1
                                            continue
                                        logits = forward_endpoint(model, inputs, decoder, vanilla=mode == "vanilla")
                                        values = score_logits(logits, tokenizer.eos_token_id, newline_ids)
                                        record["feature_signatures"] = feature_signatures(model)
                                        if mode != "vanilla":
                                            teacher_path = output / f"{index:02d}-vanilla-{control}-endpoint-{variant}.pt"
                                            if not teacher_path.is_file():
                                                raise ValueError("Matched teacher endpoint artifact is missing")
                                            teacher = torch.load(teacher_path, map_location="cpu", weights_only=True)
                                            if not torch.equal(teacher["decoder_input_ids"], decoder.cpu()):
                                                raise ValueError("Teacher/student decoder prefixes differ")
                                            if teacher["input_tensor_sha256"] != source_hashes:
                                                raise ValueError("Teacher/student input tensors differ")
                                            record["teacher_student_kl"] = teacher_student_kl(teacher["raw_logits"], logits)
                                            record["teacher_student_max_abs_logit_delta"] = float(
                                                (teacher["raw_logits"].float() - logits.float()).abs().max())
                                        artifact = output / f"{request_id}.pt"
                                        torch.save({"raw_logits": logits, "decoder_input_ids": decoder.cpu(),
                                                    "prefix_token_ids": lexical.cpu(),
                                                    "input_tensor_sha256": source_hashes}, artifact)
                                        record.update(values, raw_logits_artifact=artifact.name,
                                            logits_artifact_sha256=sha256(artifact),
                                            decoder_calls_consumed=1, encoder_calls_consumed=1)
                                        calls = 1
                                    budget.complete(decoder_calls=calls, encoder_calls=1)
                                    record["status"] = "complete"
                                    if kind == "generation":
                                        pair_outputs[control] = {"generated_token_ids": record["generated_token_ids"],
                                                                 "raw_caption": record["raw_caption"],
                                                                 "cleaned_caption": record["cleaned_caption"]}
                                except Exception as exc:
                                    record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                                    raise
                                finally:
                                    record["elapsed_seconds"] = budget.now() - started
                                    record.update(measure_memory(next(model.base.parameters()).device))
                                    journal.append(record)
                                    save_status()
                        if len(pair_outputs) == 2:
                            journal.append({"record_kind": "paired_generation_summary", "image_id": original,
                                "donor_image_id": donor, "model_mode": mode,
                                "token_ids_changed": pair_outputs["matched"]["generated_token_ids"] != pair_outputs["permuted"]["generated_token_ids"],
                                "raw_caption_changed": pair_outputs["matched"]["raw_caption"] != pair_outputs["permuted"]["raw_caption"],
                                "cleaned_caption_changed": pair_outputs["matched"]["cleaned_caption"] != pair_outputs["permuted"]["cleaned_caption"]})
                    del model, processor
                if phase == "timing":
                    if budget.attempts != TIMING_REQUESTS:
                        raise RuntimeError("Timing stage request count differs from 128")
                    stage_seconds = budget.now() - budget.start
                    estimated_remaining = stage_seconds * 7
                    status["timing_stage"] = {"requests": TIMING_REQUESTS, "elapsed_seconds": stage_seconds,
                                              "conservative_remaining_estimate_seconds": estimated_remaining}
                    save_status()
                    if budget.now() + estimated_remaining >= budget.stop_new:
                        status["status"] = "stopped_after_timing"
                        status["early_stop_reason"] = "Timing stage does not fit remaining deadline reserve"
                        save_status()
                        return status
        if budget.attempts != MAX_REQUESTS or budget.completed != MAX_REQUESTS:
            raise RuntimeError("Completed matrix has incorrect logical request count")
        status["status"] = "complete_with_unsupported" if status["unsupported_requests"] else "complete"
    except TimeoutError as exc:
        status.update(status="stopped_deadline", early_stop_reason=str(exc))
    except Exception as exc:
        status.update(status="failed", early_stop_reason=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_status()
        journal.close()
    return status
