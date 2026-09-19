"""Bounded production-adapter same-logits check; run only within owner GPU budget.

Input is an existing 16-32-document JSONL with exact_candidate_arguments and
candidate_tokens (native policy fixture format). All supplied companions are
kept. Fixed 64-request batches; 2 passes (no_noise and vanilla), at most 256
candidate requests. No training, no download-specific fallback, no matrix.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch

from jscc.evaluation_policy import file_digest, scoring_kwargs
from jscc.harness_payload import ContinuationLengths, payload_harness_class
from jscc.models.split_model import build_model
from jscc.runtime import configure_training_determinism, isolated_rng, source_state


def check(model, tokenizer, rows, *, bypass=False):
    from lm_eval.api.registry import get_model
    cls = payload_harness_class(cast(type[Any], get_model("hf")))
    adapter = cls(communication_model=model, pretrained=model.base, tokenizer=tokenizer,
                  backend="seq2seq", batch_size=64,
                  **scoring_kwargs({"scoring_policy": "fp32-v1"}, model))
    encoded = []
    for row in rows:
        for i, pair in enumerate(row["exact_candidate_arguments"]):
            context, continuation = adapter._encode_pair(*pair)
            saved = row["candidate_tokens"][i]
            if context[-adapter.max_length:] != saved["context"] or continuation != saved["target"]:
                raise ValueError("Saved tokenization changed; do not silently rebuild this fixture")
            encoded.append((tuple(pair), context, continuation))
    lengths = ContinuationLengths(encoded, adapter.max_length)
    expected = {}
    original = adapter._model_call
    count = 0

    def same_logits(inps, attn_mask=None, labels=None):
        nonlocal count
        assert attn_mask is not None and labels is not None
        count += len(inps)
        if count > len(encoded):
            raise RuntimeError("Candidate request budget exceeded")
        logits = original(inps, attn_mask=attn_mask, labels=labels)
        valid = lengths.mask(inps, attn_mask, labels)
        # Debug-only bounded check: no vocabulary hashes or CPU logits copies.
        for i, length in enumerate(valid.sum(1).tolist()):
            target = labels[i, :length]
            selected = logits[i, :length].float().log_softmax(-1).gather(-1, target[:, None])
            key = (tuple(inps[i][attn_mask[i].bool()].tolist()), tuple(target.tolist()))
            expected[key] = float(selected.sum(dtype=torch.float32))
        return logits

    adapter._model_call = same_logits
    requests = [SimpleNamespace(args=request, task_name="hellaswag") for request, _, _ in encoded]
    noise = cast(Any, model.channel).replay(0, "evaluation-v1") if hasattr(model.channel, "replay") else nullcontext()
    with isolated_rng(0), noise, model.transmission(None, bypass=bypass):
        scores = adapter.loglikelihood(requests, disable_tqdm=True)
        payload = dict(model.valid_payload_counts)
    oracle = [expected[(tuple(context[-adapter.max_length:]), tuple(target))] for _, context, target in encoded]
    actual = [score for score, _ in scores]
    torch.testing.assert_close(torch.tensor(actual), torch.tensor(oracle), rtol=0, atol=1e-5)
    assert count == len(encoded) and all(torch.isfinite(torch.tensor(actual)))
    width = model.codec.config["bottleneck_dim"]
    expected_hidden = 0 if bypass else width * sum(
        len(target) if model.split["stack"] == "dec" else len(context[-adapter.max_length:])
        for _, context, target in encoded)
    assert payload["hidden"] == expected_hidden
    if bypass:
        assert payload == {"hidden": 0, "memory": 0}
    for start in range(0, len(actual), 4):
        assert max(range(4), key=lambda j: actual[start+j]) == max(range(4), key=lambda j: oracle[start+j])
    return {"status": "PASS_SCOPED", "candidate_requests": count, "batch_size": 64,
            "bypass": bypass, "scores": actual, "same_logits_fp32_scores": oracle,
            "payload": payload, "forward_dtypes": sorted(adapter.forward_logits_dtypes)}


@torch.no_grad()
def original_companions(model, tokenizer, path):
    """Replay preserved tensor batches; separate from fixed64 harness regression."""
    from lm_eval.api.registry import get_model
    fixtures = json.loads(Path(path).read_text())["fixtures"]
    requests = sum(len(x["batch"]["input_ids"]) for x in fixtures)
    if requests > 512:
        raise ValueError("companion fixture exceeds reserved512 requests")
    adapter = cast(type[Any], get_model("hf"))(pretrained=model.base, tokenizer=tokenizer,
        backend="seq2seq", batch_size=64, **scoring_kwargs({"scoring_policy":"fp32-v1"}, model))
    device = next(model.base.parameters()).device
    records = []
    with isolated_rng(0), model.transmission(None, bypass=True):
        for index, f in enumerate(fixtures):
            batch = f["batch"]
            tensors = {k:torch.tensor(batch[k], device=device) for k in ("input_ids","attention_mask","labels")}
            logits = adapter._model_call(tensors["input_ids"], attn_mask=tensors["attention_mask"], labels=tensors["labels"])
            logp = logits.log_softmax(-1, dtype=adapter.softmax_dtype)
            for i, companion in enumerate(batch["companions"]):
                length = companion["continuation_length"]
                target = tensors["labels"][i,:length]
                assert target.tolist() == companion["target_ids"]
                score = logp[i,:length].gather(-1,target[:,None]).sum(dtype=torch.float32)
                assert torch.isfinite(score)
                records.append({"fixture":index,"companion":companion,"fp32_score":float(score)})
        assert dict(model.valid_payload_counts) == {"hidden":0,"memory":0}
    return {"scope":"exact original padded tensor batches, BF16 forward/FP32 score; not fixed64 batches",
            "fixture_sha256":file_digest(path),"candidate_requests":requests,"records":records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-fixtures", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.samples.read_text().splitlines() if line.strip()]
    if not 16 <= len(rows) <= 32 or any(len(row["exact_candidate_arguments"]) != 4 for row in rows):
        raise ValueError("Require 16-32 complete four-candidate documents; retain all fixture companions")
    args.output.mkdir(parents=True, exist_ok=False)
    configure_training_determinism({"deterministic_algorithms": True})
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    state = torch.load(args.run / args.checkpoint, map_location="cpu", weights_only=True)
    processor, model = build_model(state["config"])
    model.load_communication_state(state)
    model.eval()
    results = []
    for bypass in (False, True):
        results.append(check(model, processor, rows, bypass=bypass))
        (args.output / "results.json").write_text(json.dumps({
            "source": source_state(), "checkpoint_sha256": file_digest(args.run / args.checkpoint),
            "step": state["step"], "fixture_sha256": file_digest(args.samples),
            "doc_ids": [row["doc_id"] for row in rows], "split": state["config"]["split"],
            "candidate_requests": sum(result["candidate_requests"] for result in results),
            "results": results}, indent=2, allow_nan=False) + "\n")

    if args.tensor_fixtures:
        companion_results = original_companions(model, processor, args.tensor_fixtures)
        (args.output / "original-companions.json").write_text(json.dumps(companion_results, indent=2)+"\n")


if __name__ == "__main__":
    main()
