"""Carry real seq2seq request lengths through HFLM without changing its batching.

Lengths come from unpadded requests, never from PAD token values. The pinned
harness keeps sorting, token preparation, caching and scoring ownership.
"""
from collections import defaultdict
from typing import Any

import torch


class ContinuationLengths:
    def __init__(self, requests, max_length):
        self.by_context = defaultdict(set)
        for _, context, continuation in requests:
            self.by_context[tuple(context[-max_length:])].add(tuple(continuation[-max_length:]))

    def mask(self, inps, attention_mask, labels):
        if attention_mask is None or labels is None or labels.ndim != 2:
            raise ValueError("Payload accounting requires seq2seq attention mask and labels")
        lengths = []
        for context, visible, padded in zip(inps.tolist(), attention_mask.tolist(), labels.tolist(), strict=True):
            key = tuple(token for token, valid in zip(context, visible, strict=True) if valid)
            matches = {len(target) for target in self.by_context.get(key, ())
                       if len(target) <= len(padded)
                       and tuple(padded[:len(target)]) == target
                       and all(token == 0 for token in padded[len(target):])}
            if len(matches) != 1:
                raise ValueError(f"Missing or ambiguous actual continuation length: {sorted(matches)}")
            lengths.append(matches.pop())
        return torch.arange(labels.shape[1], device=labels.device)[None, :] < torch.tensor(
            lengths, device=labels.device)[:, None]


def payload_harness_class(harness_class: type[Any]) -> type[Any]:
    """Wrap only the two request/forward seams; leave HFLM numerical code intact."""
    class PayloadHFLM(harness_class):
        def __init__(self, *args, communication_model, **kwargs):
            super().__init__(*args, **kwargs)
            if self.backend != "seq2seq" or self.batch_size == "auto":
                raise ValueError("Payload accounting requires fixed-batch seq2seq evaluation")
            self.communication_model = communication_model
            self._payload_lengths = None

        def _loglikelihood_tokens(self, requests, disable_tqdm=False, override_bs=None):
            previous = self._payload_lengths
            self._payload_lengths = ContinuationLengths(requests, self.max_length)
            try:
                return super()._loglikelihood_tokens(requests, disable_tqdm=disable_tqdm, override_bs=override_bs)
            finally:
                self._payload_lengths = previous

        def _model_call(self, inps, attn_mask=None, labels=None):
            if self._payload_lengths is None:
                raise ValueError("Decoder payload forward has no actual request lengths")
            mask = self._payload_lengths.mask(inps, attn_mask, labels)
            with self.communication_model.decoder_payload_validity(mask):
                return super()._model_call(inps, attn_mask=attn_mask, labels=labels)

    return PayloadHFLM
