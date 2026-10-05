"""Retained EVENING_SMALL_SUITE_V1 scoring, replay and intervention helpers.

The historical executor is retired; imports preserve scientific interpretation.
"""

if __name__ == "__main__":
    raise SystemExit(
        "evening_eval executor is retired; retained helpers are available by import."
    )

from contextlib import contextmanager
import time
from types import MethodType
from typing import Any

from scripts.frozen_prompt_gap import digest, tensor_batch

TOLERANCE = {"atol": 5e-4, "rtol": 1e-5}


def require(value, message):
    if not value:
        raise ValueError(message)


def replay_epsilon(z, mask, request_keys, seed, stream):
    """Valid coordinates are independent of padding, companions and other streams.

    Separate deterministic draws for valid/padding coordinates preserve physical
    AWGN on all allocated coordinates without allowing padding to advance payload
    RNG. Per-request stream keys are reconstructable from saved seed and IDs.
    """
    import torch
    from jscc.presentation import tensor_digest
    require(mask.shape == z.shape[:2], "Expected canonical [batch,tokens] mask")
    require(len(request_keys) == len(z), "Request/noise identity mismatch")
    epsilon = torch.empty_like(z)
    evidence = []
    for i, request in enumerate(request_keys):
        valid = mask[i].bool()
        for label, positions in (("valid", valid), ("padding", ~valid)):
            key = ["evening-valid-coordinates-v1", seed, stream, request, label]
            derived = int(digest(key)[:16], 16) % (2**63 - 1)
            generator = torch.Generator(device=z.device).manual_seed(derived)
            noise = torch.randn((int(positions.sum()), z.shape[-1]), device=z.device,
                                dtype=z.dtype, generator=generator)
            epsilon[i, positions] = noise
        evidence.append({"request": request, "stream": stream, "seed": seed,
                         "valid_epsilon_sha256": tensor_digest(epsilon[i, valid]),
                         "dtype": str(z.dtype), "device_type": z.device.type,
                         "valid_shape": [int(valid.sum()), z.shape[-1]],
                         "allocated_shape": list(z[i].shape)})
    return epsilon, evidence


class Intervention:
    """Intercept only the production transmission seam; preserve all model hooks."""
    def __init__(self, model, policies, seed, request_keys):
        self.model, self.policies, self.seed, self.request_keys = model, policies, seed, request_keys
        self.events = []
        self.active = None

    @contextmanager
    def install(self):
        from jscc.models.channel import AWGNChannel, _token_mask
        from jscc.presentation import tensor_digest
        model = self.model
        require(type(model.channel) is AWGNChannel, "Only production AWGN is supported")
        require(model.codec.film is None and
                (model.memory_codec is None or model.memory_codec.film is None), "FiLM must remain off")
        original_transmit, original_channel = model._transmit, model.channel.forward

        def channel(_channel, z, snr):
            event = self.active
            if event is None:
                raise ValueError("Channel outside stream scope")
            event["channel_calls"] += 1
            if event["noise_enabled"]:
                epsilon, proof = replay_epsilon(z, event.pop("_mask"), self.request_keys, self.seed, event["stream"])
                event["noise"] = proof
                return z + epsilon * (10.0 ** (6.0 / 20.0))
            event.pop("_mask", None)
            return original_channel(z, None)

        def transmit(_model, hidden, codec, stream, valid_mask=None, *, token_wise=False,
                     mask_representation=None):
            policy = self.policies[stream]
            require(policy in {"C", "O", "A"}, "Unknown stream intervention")
            mask = _token_mask(hidden, valid_mask, representation=mask_representation)
            if mask is None:
                raise ValueError("Missing stream payload mask")
            width = codec.config["bottleneck_dim"]
            event = {"stream": stream, "policy": policy, "codec_calls": 0,
                     "channel_calls": 0, "oracle_calls": int(policy == "O"),
                     "noise_enabled": policy == "A", "snr_db": -6.0 if policy == "A" else None,
                     "input_sha256": tensor_digest(hidden),
                     "allocated_per_request": [0 if policy == "O" else hidden.shape[1] * width] * len(hidden),
                     "valid_per_request": [0 if policy == "O" else int(m.sum()) * width for m in mask],
                     "_mask": mask}
            self.events.append(event)
            if policy == "O":
                event.pop("_mask")
                return hidden
            event["codec_calls"] += 1
            self.active = event
            try:
                return original_transmit(hidden, codec, stream, valid_mask, token_wise=token_wise,
                                         mask_representation=mask_representation)
            finally:
                self.active = None

        model._transmit = MethodType(transmit, model)
        model.channel.forward = MethodType(channel, model.channel)
        try:
            yield self
        finally:
            model._transmit = original_transmit
            model.channel.forward = original_channel


def compare(reference, candidate):
    import torch
    a, b = torch.tensor(reference, dtype=torch.float64), torch.tensor(candidate, dtype=torch.float64)
    require(a.shape == b.shape, "Replay score shape mismatch")
    return {"passed": bool(torch.allclose(a, b, atol=TOLERANCE["atol"], rtol=TOLERANCE["rtol"])),
            "maximum_absolute_difference": float((a - b).abs().max()), **TOLERANCE}


def score(adapter, model, batch, budget, cell, phase, policies=None, seed=None):
    import torch
    from jscc.harness_payload import ContinuationLengths
    from jscc.presentation import tensor_digest
    device = next(model.base.parameters()).device
    tensors = tensor_batch(batch, device)
    adapter._payload_lengths = ContinuationLengths([
        ((r["context"], r["continuation"]), r["context_token_ids"], r["continuation_token_ids"])
        for r in batch], 2048)
    native = model.base.prepare_decoder_input_ids_from_labels(tensors["labels"])
    native_observed = []

    def capture(_module, args, kwargs):
        actual = kwargs.get("input_ids", args[0] if args else None)
        if actual is None or not torch.equal(actual, native):
            raise ValueError("Native decoder input mismatch")
        cache = kwargs.get("past_key_values")
        require(cache is None or cache.get_seq_length() == 0, "Reused decoder cache")
        native_observed.append({"input_ids": actual.tolist(),
                                "attention_mask": None if kwargs.get("attention_mask") is None
                                else kwargs["attention_mask"].tolist(),
                                "position_ids": None if kwargs.get("position_ids") is None
                                else kwargs["position_ids"].tolist(),
                                "cache_initial_length": 0})

    hook = model.base.get_decoder().register_forward_pre_hook(capture, with_kwargs=True)
    intervention = Intervention(model, policies, seed, [[r["sample_id"], r["candidate"]] for r in batch]) if policies else None
    from contextlib import nullcontext
    started = time.monotonic()
    try:
        budget.reserve(64, cell, phase)
        with torch.inference_mode(), model.transmission(None, bypass=cell == "vanilla"), (
            intervention.install() if intervention else nullcontext()
        ):
            logits = adapter._model_call(tensors["input_ids"], attn_mask=tensors["attention_mask"], labels=tensors["labels"])
            require(logits.dtype == next(model.base.parameters()).dtype, "Forward dtype mismatch")
            logp = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
            # Preserve production's sum over only the actual continuation.
            values = torch.stack([
                logp[i, :len(row["continuation_token_ids"])]
                .gather(-1, tensors["labels"][i, :len(row["continuation_token_ids"]), None])
                .sum(dtype=torch.float32) for i, row in enumerate(batch)
            ])
            require(bool(torch.isfinite(values).all()), "Nonfinite candidate scores")
            values = values.tolist()
        require(len(native_observed) == 1, "Expected one decoder call per batch")
        if intervention:
            expected_streams = {"hidden", "memory"} if model.memory_codec is not None else {"hidden"}
            require(len(intervention.events) == len(expected_streams)
                    and {e["stream"] for e in intervention.events} == expected_streams,
                    "Actual transmission branch counts changed")
            for event in intervention.events:
                oracle = event["policy"] == "O"
                require(event["codec_calls"] == event["channel_calls"] == int(not oracle)
                        and event["oracle_calls"] == int(oracle), "Unexpected codec/channel/oracle calls")
                require(sum(event["allocated_per_request"]) == model.channel_uses_allocated[event["stream"]]
                        and sum(event["valid_per_request"]) == model.valid_payload_counts[event["stream"]],
                        "Per-request payload differs from measured production counters")
        evidence = {"cell": cell, "phase": phase, "request_keys": [[r["sample_id"], r["candidate"]] for r in batch],
                    "scores": values, "elapsed_seconds": time.monotonic() - started,
                    "allocated": dict(model.channel_uses_allocated), "valid": dict(model.valid_payload_counts),
                    "events": intervention.events if intervention else [],
                    "native_sha256": tensor_digest(native)}
        native_record: dict[str, Any] = {k: v.tolist() for k, v in tensors.items()}
        native_record.update(native_decoder=native_observed[0], request_keys=evidence["request_keys"],
                             rule="HFLM seq2seq labels; native BOS2 preparation; right PAD0; fresh cache each forward")
        return values, evidence, native_record
    finally:
        hook.remove()
        adapter._payload_lengths = None
