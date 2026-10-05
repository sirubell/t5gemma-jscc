"""Reusable COCO query/endpoint scientific helpers and saved-receipt hashing."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any

import torch

from .data.coco import caption_prompt
from .evaluation import clean_caption
from .runtime import autocast_for


MODES = ("vanilla", "enc_l9", "enc_fn", "dec_l8")
MAX_REQUESTS = 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


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
    # Hash native storage bytes: NumPy cannot represent BF16 directly.
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    return bytes_sha256(str(tuple(tensor.shape)).encode() + str(tensor.dtype).encode() + raw)


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


def diagnostic_inputs(model, inputs):
    """Place source tensors on the backbone, casting only image pixels."""
    parameter = next(model.base.parameters())
    return {key: value.to(device=parameter.device, dtype=parameter.dtype if key == "pixel_values" else None)
            for key, value in inputs.items()}


def forward_endpoint(model, inputs, decoder, *, vanilla: bool = False):
    parameter = next(model.base.parameters())
    source = diagnostic_inputs(model, inputs)
    decoder = decoder.to(parameter.device)
    with torch.no_grad(), autocast_for(model), model.observe_payload() as events, model.transmission(None, bypass=vanilla,
            encoder_mask=source["attention_mask"], decoder_mask=torch.ones_like(decoder, dtype=torch.bool)):
        with model.attention_context():
            result = model.base(**source, decoder_input_ids=decoder, use_cache=False, return_dict=True)
    if not vanilla and model.memory_codec is not None and not any(event["stream"] == "memory" for event in events):
        raise RuntimeError("Decoder receiver-memory codec was bypassed")
    return result.logits[0, -1].detach().float().cpu()


def generate_caption(model, processor, inputs, *, vanilla: bool = False):
    source = diagnostic_inputs(model, inputs)
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
