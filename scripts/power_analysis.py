"""Arm 0 of the power-vs-split-norm wave: what power normalization discards, measured on trained checkpoints.

    python scripts/power_analysis.py OUT.json RUN_DIR [RUN_DIR ...]

Forward passes only (no training, ``jscc/`` unchanged). Each run is loaded as ``experiment.py evaluate`` does
(its ``config.yaml`` and ``best_distillation.pt``; a missing ``data.ids`` or ``codec.site_stats`` path, as in runs moved
from another machine, is taken from this checkout's ``configs/data/``) and scored on the validation selection batches (the 256
held-out rows, the trainer's masks) at every site it was trained on, under no noise and -6 dB, twice:
``normal`` and ``const_scale`` (the per-sample / per-token sqrt(P) replaced by the per-site, per-stream
constant sqrt(mean P) of the normal no-noise pass). Hooks: forward hook on ``codec.encoder`` (z before power
normalization; at decoder sites the first call is the hidden stream, the second the memory) and forward
pre-hook on ``channel`` (the transmitted signal x).

Per run x site x condition x stream: sqrt(P) mean and coefficient of variation (per sample, per valid token
for the decoder hidden stream), DC share of transmitted power ||E x||^2 / E||x||^2 over valid tokens and the
implied SNR offset 10 log10(1 - share), transmitted power in dB, pooled nMSE sum||h^ - h||^2 / sum||h||^2,
centred nMSE (denominator sum||h - mu||^2, mu from configs/data/hellaswag-site-stats.pt), ||E(h^ - h)||^2 /
E||h||^2, RMS(h^) / RMS(h), and the KL per valid token (mean over batches, as the trainer's validation).
Per run: Frobenius norms of the codec Linear weights and biases, ||decoder bias|| vs ||mu|| per stream and
max |g|, |beta| of a learned site norm. The best validation entry of metrics.jsonl is copied for comparison.
"""
import json
import math
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import jscc.model as jm  # noqa: E402
from jscc.config import load_config  # noqa: E402
from jscc.tasks import load_task_data  # noqa: E402
from jscc.train import site_loss, teacher_pass  # noqa: E402

CONDITIONS = ["no_noise", -6]
STATS = torch.load(REPO / "configs/data/hellaswag-site-stats.pt", map_location="cpu", weights_only=True)["stats"]


class Acc:
    """Sums for one (site, condition, stream)."""

    def __init__(self):
        self.sqrt_p, self.x_sum, self.x_sq, self.x_n, self.x_dim = [], torch.zeros((), dtype=torch.float64), 0.0, 0, 1
        self.err, self.sig, self.cen, self.rec, self.h_n = 0.0, 0.0, 0.0, 0.0, 0
        self.err_sum = torch.zeros((), dtype=torch.float64)

    def result(self):
        out = {}
        if self.sqrt_p:
            s = torch.cat(self.sqrt_p)
            out |= {"sqrt_p_mean": s.mean().item(), "sqrt_p_cv": (s.std() / s.mean()).item(),
                    "p_mean": s.pow(2).mean().item()}
        if self.x_n:
            mean_sq = self.x_sq / self.x_n  # E||x||^2 per token
            share = (self.x_sum / self.x_n).pow(2).sum().item() / mean_sq
            out |= {"dc_share": share, "snr_offset_db": 10 * math.log10(max(1 - share, 1e-12)),
                    "tx_power_db": 10 * math.log10(mean_sq / self.x_dim)}
        if self.h_n:
            out |= {"nmse": self.err / self.sig, "nmse_centred": self.err / self.cen,
                    "mean_error_share": (self.err_sum / self.h_n).pow(2).sum().item() / (self.sig / self.h_n),
                    "rms_ratio": math.sqrt(self.rec / self.sig)}
        return out


def _valid(t, mask):
    return t.float()[mask]  # [n_valid_tokens, D]


def analyse(run: Path, device: str):
    config = load_config(run / "config.yaml")
    config["model"]["device"] = device
    for section, key in (("data", "ids"), ("codec", "site_stats")):  # runs moved from another machine
        path = config.get(section, {}).get(key)
        if path and not Path(path).exists():
            config[section][key] = str(REPO / "configs/data" / Path(path).name)
            print(f"{run.name}: {section}.{key} {path} missing, using {config[section][key]}", flush=True)
    processor, model = jm.build_model(config, local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1")
    state = torch.load(run / "best_distillation.pt", map_location="cpu", weights_only=False)["communication"]
    model.load_communication_state(state)
    model.eval()
    data = load_task_data(config, processor)
    sites, cfg = list(config["training"]["sites"]), config["training"]
    current = {}  # per forward: masks and which stream the next encoder call carries
    acc = {}

    def stream_key():
        return "hidden" if current["calls"] == 0 else "memory"

    def on_encode(module, args, z):
        stream = stream_key()
        a = acc[(current["site"], current["cond"], stream)]
        token_wise = stream == "hidden" and model.is_decoder_site
        mask = current["dec"] if token_wise else current["enc"]
        zf = z.float()
        if token_wise:
            p = zf.pow(2).mean(-1)[mask]
        else:
            m = mask.unsqueeze(-1).float()
            p = (zf.pow(2) * m).sum((1, 2)) / (mask.sum(1).clamp_min(1) * zf.shape[-1])
        a.sqrt_p.append(p.sqrt().cpu())

    def on_channel(module, args):
        stream = stream_key()
        current["calls"] += 1
        a = acc[(current["site"], current["cond"], stream)]
        token_wise = stream == "hidden" and model.is_decoder_site
        x = _valid(args[0], current["dec"] if token_wise else current["enc"])
        a.x_sum = a.x_sum + x.sum(0).double()
        a.x_sq += x.double().pow(2).sum().item()
        a.x_n += x.shape[0]
        a.x_dim = x.shape[-1]

    handles = [model.codec.encoder.register_forward_hook(on_encode),
               model.channel.register_forward_pre_hook(on_channel)]
    constant = {}  # (site, stream) -> sqrt(mean P) of the normal no-noise pass
    original_scale = jm.power_scale

    def const_scale(z, mask=None, *, token_wise=False):
        z, _ = original_scale(z, mask, token_wise=token_wise)
        stream = "hidden" if (token_wise or not model.is_decoder_site or current["calls"] == 0) else "memory"
        return z, torch.tensor(constant[(current["site"], stream)], device=z.device, dtype=z.dtype)

    results = {}
    try:
        for mode in ("normal", "const_scale"):
            if mode == "const_scale":
                if not model.normalize:
                    break
                jm.power_scale = const_scale
            acc.clear()
            kl = {(s, c): [] for s in sites for c in CONDITIONS}
            torch.manual_seed(cfg["validation"].get("seed", 0))  # the trainer's validation noise
            with torch.no_grad():
                for batch in data.selection_batches():
                    inputs, labels = data.model_inputs(batch, model)
                    logits, prefix = teacher_pass(model, data, inputs, labels, sites)
                    enc, dec = inputs["attention_mask"].bool(), labels != -100
                    for site in sites:
                        model.set_site(site)
                        for c in CONDITIONS:
                            current.update(site=site, cond=c, calls=0, enc=enc, dec=dec)
                            for stream in ("hidden", "memory"):
                                acc.setdefault((site, c, stream), Acc())
                            snr = None if c == "no_noise" else torch.full((labels.shape[0], 1, 1), float(c),
                                                                          device=labels.device)
                            _, st = site_loss(model, data, inputs, labels, logits, prefix, snr, cfg, "distillation")
                            kl[(site, c)].append(st["kl"])
                            pairs = [("hidden", model.reconstruction, model.activation, dec if model.is_decoder_site else enc, site)]
                            if model.memory_reconstruction is not None:
                                pairs.append(("memory", model.memory_reconstruction, model.memory_activation, enc, "memory"))
                            for stream, rec, ref, mask, stats_key in pairs:
                                a = acc[(site, c, stream)]
                                r, h = _valid(rec, mask), _valid(ref, mask)
                                mu = STATS[stats_key]["mean"].to(h.device)
                                a.err += (r - h).double().pow(2).sum().item()
                                a.sig += h.double().pow(2).sum().item()
                                a.cen += (h - mu).double().pow(2).sum().item()
                                a.err_sum = a.err_sum + (r - h).sum(0).double()
                                a.rec += r.double().pow(2).sum().item()
                                a.h_n += h.shape[0]
            out = {}
            for site in sites:
                for c in CONDITIONS:
                    cell = {"kl": sum(kl[(site, c)]) / len(kl[(site, c)])}
                    for stream in ("hidden", "memory"):
                        r = acc[(site, c, stream)].result()
                        if r:
                            cell[stream] = r
                            if mode == "normal" and c == "no_noise" and "p_mean" in r:
                                constant[(site, stream)] = math.sqrt(r["p_mean"] + 1e-8)
                    out[f"{site}|{c}"] = cell
            results[mode] = out
    finally:
        jm.power_scale = original_scale
        for h in handles:
            h.remove()
    return config, state, results


def weights(config, state):
    codec = state["codec"]
    out = {k: torch.linalg.norm(v.float()).item() for k, v in codec.items() if k.endswith(("weight", "bias"))}
    sites = list(config["training"]["sites"])
    streams = sites + (["memory"] if any(s.startswith("dec_") for s in sites) else [])
    out["mu_norm"] = {s: torch.linalg.norm(STATS[s]["mean"]).item() for s in streams}
    norm = state.get("site_norm") or {}
    for key in ("affine.g_in", "affine.b_in", "affine.g_out", "affine.b_out"):
        if key in norm:
            out[f"max_abs_{key.split('.')[1]}"] = norm[key].abs().max().item()
    return out


def best_validation(run: Path):
    best = None
    if not (run / "metrics.jsonl").exists():  # some moved runs kept only checkpoints and results
        return None
    for line in (run / "metrics.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("type") == "validation" and r.get("phase") == "distillation" and (best is None or r["mean"] < best["mean"]):
            best = r
    return best and {"update": best["update"], "mean": best["mean"], "per_site": best["per_site"]}


def main(argv):
    out_path, runs = Path(argv[0]), [Path(a).resolve() for a in argv[1:]]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = json.loads(out_path.read_text()) if out_path.exists() else {}
    for run in runs:
        if run.name in report:
            print("skip (done)", run.name, flush=True)
            continue
        config, state, results = analyse(run, device)
        report[run.name] = {"run": str(run), "normalize_power": config["channel"].get("normalize_power", True),
                            "restore_power": config["channel"].get("restore_power", False),
                            "site_norm": config["codec"].get("site_norm", "none"), "weights": weights(config, state),
                            "logged_best_validation": best_validation(run), **results}
        out_path.write_text(json.dumps(report, indent=1) + "\n")  # after every run, so a stop keeps the done ones
        print("done", run.name, flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:])
