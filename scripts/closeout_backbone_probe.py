"""Localize a recorded vanilla anomaly using original batch companions, without retraining."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from jscc.models.split_model import build_model, stack_module


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--evaluator", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    _, model = build_model(state["config"])
    model.load_communication_state(state)
    model.eval()
    records = []
    for precision in ["bfloat16", "float32"]:
        if precision == "float32":
            model.base.float()
        reference = None
        for batchsize in [8, 16, 32]:
            saved = json.loads(
                (args.evaluator / f"b{batchsize}-r0/vanilla.json").read_text()
            )
            cap = next(
                x for x in saved["captures"] if x["doc_id"] == 343 and x["choice"] == 2
            )
            batch = saved["batches"][cap["batch_index"]]
            index = next(
                i
                for i, x in enumerate(batch["companions"])
                if x["doc_id"] == 343 and x["choice"] == 2
            )
            inputs = {
                k: torch.tensor(batch[k], device="cuda")
                for k in ["input_ids", "attention_mask", "labels"]
            }
            activations = {}
            handles = []
            for stack in ["enc", "dec"]:
                module = stack_module(model.base, stack)
                length = len(
                    cap["context_ids"] if stack == "enc" else cap["target_ids"]
                )
                for i, layer in enumerate(module.layers):

                    def hook(m, a, o, key=f"{stack}_{i}", n=length):
                        value = o[0] if isinstance(o, tuple) else o
                        activations[key] = value[index, :n].detach().float().cpu()

                    handles.append(layer.register_forward_hook(hook))
            with torch.no_grad(), model.transmission(None, bypass=True):
                logits = model.base(**inputs).logits
            for handle in handles:
                handle.remove()
            target = torch.tensor(cap["target_ids"], device="cuda")
            chosen = logits[index, : len(target)]
            selected = (
                F.log_softmax(chosen.float(), dim=-1)
                .gather(1, target[:, None])
                .squeeze(1)
            )
            if reference is None:
                reference = activations
            diffs = {}
            for name, value in activations.items():
                a = reference[name]
                diff = value - a
                diffs[name] = {
                    "max_abs": float(diff.abs().max()),
                    "relative_l2": float(diff.norm() / a.norm().clamp_min(1e-30)),
                }
            records.append(
                {
                    "precision": precision,
                    "batch_size": batchsize,
                    "doc_id": 343,
                    "choice": 2,
                    "input_shape": list(inputs["input_ids"].shape),
                    "forward_requests": len(inputs["input_ids"]),
                    "score_fp32": float(selected.sum()),
                    "selected_log_probs_fp32": selected.tolist(),
                    "layer_differences_vs_batch8": diffs,
                    "attention_backend": str(model.base.config._attn_implementation),
                    "weight_policy": "same BF16-loaded weights; float32 diagnostic casts those values without reloading",
                }
            )
            del logits, chosen, selected, inputs
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
