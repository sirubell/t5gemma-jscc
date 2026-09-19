"""Opt-in fresh-run presentation and paired-randomness evidence helpers."""
import hashlib
import gzip
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Sampler


def derived_seed(seed, namespace):
    return int.from_bytes(hashlib.sha256(f"{seed}:{namespace}".encode()).digest()[:8], "little") % (2**63 - 1)


def tensor_digest(value):
    value = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str((tuple(value.shape), str(value.dtype))).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class PresentationSampler(Sampler[int]):
    """Concatenated CPU randperm epochs; no discarded or short epoch tails."""
    def __init__(self, row_ids, total, seed=0):
        self.row_ids = list(row_ids)
        if not self.row_ids or total <= 0:
            raise ValueError("presentation stream requires nonempty rows and a positive budget")
        self.seed, self.total = seed, total
        generator = torch.Generator(device="cpu").manual_seed(seed)
        chunks = []
        remaining = total
        while remaining:
            chunk = torch.randperm(len(row_ids), generator=generator)[:remaining]
            chunks.append(chunk)
            remaining -= len(chunk)
        self.indices = torch.cat(chunks)
        self.actual_ids = torch.tensor(self.row_ids, dtype=torch.int64)[self.indices]

    def __iter__(self):
        return iter(self.indices.tolist())

    def __len__(self):
        return self.total

    def save(self, directory: Path, batch_size):
        if self.total % batch_size:
            raise ValueError("presentation budget must comprise full batches")
        with gzip.open(directory / "presentation_ids.json.gz", "wt", encoding="utf-8") as output:
            json.dump(self.actual_ids.tolist(), output, separators=(",", ":"))
        manifest = {"policy": "epoch-permutations-v1", "algorithm": "torch CPU Generator randperm concatenated epochs",
                    "seed": self.seed, "torch_version": str(torch.__version__), "total_presentations": self.total,
                    "microbatch_size": batch_size, "pool_ids": self.row_ids,
                    "actual_ids_file": "presentation_ids.json.gz", "actual_ids_sha256": tensor_digest(self.actual_ids),
                    "actual_ids_bytes_sha256": hashlib.sha256(self.actual_ids.numpy().astype("<i8").tobytes()).hexdigest(),
                    "consumed_digest_format": "concatenated little-endian signed int64 actual row IDs",
                    "segment_size": batch_size, "segment_sha256": [tensor_digest(chunk) for chunk in self.actual_ids.split(batch_size)],
                    "resume": "fresh-only; exact resume unsupported"}
        (directory / "presentations.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest


def initialization_evidence(model, directory):
    """Save communication-only initial weights; hash unchanged cores separately."""
    components = {"hidden": model.codec, "memory": model.memory_codec}
    states, evidence = {}, {}
    for name, codec in components.items():
        if codec is None:
            continue
        if codec.film is not None:
            raise ValueError("paired outer-LN study requires FiLM disabled")
        internal = sum(isinstance(module, nn.LayerNorm) for prefix in (codec.encoder, codec.decoder) for module in prefix.modules())
        expected = 2 * codec.config["n_res_blocks"]
        if internal != expected:
            raise ValueError("internal residual LayerNorm count changed")
        state = {key: value.detach().cpu().clone() for key, value in codec.state_dict().items()}
        states[name] = state
        core = {key: tensor_digest(value) for key, value in state.items() if not key.startswith(("input_norm.", "output_norm."))}
        for module in (codec.input_norm, codec.output_norm):
            if isinstance(module, nn.LayerNorm):
                if not torch.equal(module.weight, torch.ones_like(module.weight)) or not torch.equal(module.bias, torch.zeros_like(module.bias)):
                    raise ValueError("outer LayerNorm must use standard affine initialization")
        evidence[name] = {"core_sha256": hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest(),
                          "core_parameter_sha256": core, "internal_layernorm_count": internal,
                          "outer_layernorm": codec.config["layernorm"], "film": False,
                          "parameter_count": sum(parameter.numel() for parameter in codec.parameters())}
    torch.save(states, directory / "initial_communication.pt")
    (directory / "initialization.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence


def feature_statistics(value, valid_mask):
    """Token feature mean/RMS/std distributions, reducing only valid FP32 tokens."""
    tokens = value.detach().float()[valid_mask.to(device=value.device, dtype=torch.bool)]
    if not tokens.numel():
        raise ValueError("feature summary has no valid tokens")
    stats = {"mean": tokens.mean(-1), "rms": tokens.square().mean(-1).sqrt(),
             "std": tokens.std(-1, correction=0)}
    return {"valid_tokens": tokens.shape[0], "reduction_dtype": "float32",
            **{name: {"mean": values.mean().item(), "min": values.min().item(), "max": values.max().item(),
                      "quantiles_0_25_50_75_100": torch.quantile(values, values.new_tensor([0, .25, .5, .75, 1])).tolist()}
               for name, values in stats.items()}}


def training_source_digest():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    for name in ("train.py", "uv.lock", "pyproject.toml"):
        digest.update(name.encode())
        digest.update((root.parent / name).read_bytes())
    return digest.hexdigest()
