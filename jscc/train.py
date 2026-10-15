"""The single trainer: one codec trained over an ordered list of sites.

A per-site codec is a run with one site; the shared codec is the same run with
all sites. Each update trains one site and sites rotate round-robin, so the
k-th update of every site uses training batch k (the same data in both arms).

Phases (each stops by early stopping, or at its update cap):
  reconstruction (optional): nMSE between clean and reconstructed features
  distillation: KL(teacher || student) + weighted nMSE
The optimizer and LR schedule restart at the phase boundary.
"""
import json
import math
import random
import time
from pathlib import Path

import torch

from .losses import kl_stats, nmse_stats
from .model import SplitModel


class StopTraining(Exception):
    pass


# ---- objective --------------------------------------------------------
def _weights(model: SplitModel, cfg, phase):
    w = cfg["loss_weights"]["decoder" if model.is_decoder_site else "encoder"]
    kl = 0.0 if phase == "reconstruction" else cfg["loss_weights"]["kl"]
    hidden = w["hidden"] if phase == "distillation" else 1.0
    memory = w.get("memory", 0.0) if phase == "distillation" else (1.0 if model.is_decoder_site else 0.0)
    return kl, hidden, memory


def site_loss(model: SplitModel, data, batch, snr, cfg, phase, teacher_logits=None):
    """Return (loss, stats) for one batch at the model's current site."""
    inputs, labels = data.model_inputs(batch, model)
    enc_mask, dec_mask = inputs["attention_mask"], labels != -100
    with data.autocast(model):
        if teacher_logits is None and phase == "distillation":
            with torch.no_grad(), model.transmission(bypass=True, encoder_mask=enc_mask, decoder_mask=dec_mask):
                teacher_logits = model(**inputs).logits
        with model.transmission(snr, encoder_mask=enc_mask, decoder_mask=dec_mask):
            student = model(**inputs).logits
        kl_num, kl_den = (kl_stats(student, teacher_logits, labels, cfg.get("temperature", 1.0))
                          if phase == "distillation" else (student.new_zeros(()), labels.new_zeros(())))
        h_mask = dec_mask if model.is_decoder_site else enc_mask.bool()
        h_num, h_den = nmse_stats(model.reconstruction, model.activation, h_mask)
        m_num = m_den = None
        if model.memory_reconstruction is not None:
            m_num, m_den = nmse_stats(model.memory_reconstruction, model.memory_activation, enc_mask.bool())
    w_kl, w_h, w_m = _weights(model, cfg, phase)
    kl = kl_num / kl_den.clamp_min(1)
    hidden = h_num / h_den.clamp_min(1)
    memory = m_num / m_den.clamp_min(1) if m_num is not None and m_den is not None else hidden.new_zeros(())
    loss = w_kl * kl + w_h * hidden + w_m * memory
    aux = getattr(model.channel, "aux_loss", None)
    if aux is not None:
        loss = loss + cfg.get("channel_aux_weight", 0.0) * aux
    return loss, {"kl": kl.item(), "nmse_hidden": hidden.item(),
                  "nmse_memory": None if m_num is None else memory.item(), "loss": loss.item()}


def sample_snr(cfg_channel, batch_size, device, generator=None, per_batch=False):
    if not cfg_channel.get("train_noise", True) or cfg_channel["type"] == "identity":
        return None
    low, high = cfg_channel["train_snr_range"]
    if per_batch:
        return float(torch.empty(()).uniform_(low, high, generator=generator))
    return torch.empty((batch_size, 1, 1)).uniform_(low, high, generator=generator).to(device)


# ---- validation -------------------------------------------------------
@torch.no_grad()
def validate(model: SplitModel, data, sites, cfg, phase):
    """Mean validation metric per site and condition on the held-out selection batches.

    Teacher logits are computed once per batch and reused across sites/conditions.
    Noise uses a fixed seed so validations are comparable over time.
    """
    was_training = model.training
    model.eval()
    conditions = cfg["validation"]["conditions"]
    sums = {(s, c): [0.0, 0] for s in sites for c in conditions}
    cpu_rng, cuda_rng = torch.get_rng_state(), (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    try:
        torch.manual_seed(cfg["validation"].get("seed", 0))
        for batch in data.selection_batches():
            inputs, labels = data.model_inputs(batch, model)
            teacher = None
            if phase == "distillation":
                with data.autocast(model), model.transmission(bypass=True, encoder_mask=inputs["attention_mask"],
                                                              decoder_mask=labels != -100):
                    teacher = model(**inputs).logits
            for site in sites:
                model.set_site(site)
                for c in conditions:
                    snr = None if c == "no_noise" else torch.full((labels.shape[0], 1, 1), float(c),
                                                                  device=labels.device)
                    if snr is not None and getattr(model.channel, "snr_mode", "per_sample") == "per_batch":
                        snr = float(c)
                    _, stats = site_loss(model, data, batch, snr, cfg, phase, teacher_logits=teacher)
                    metric = stats["kl"] if phase == "distillation" else stats["nmse_hidden"]
                    sums[(site, c)][0] += metric
                    sums[(site, c)][1] += 1
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        model.train(was_training)
    per = {f"{s}|{c}": v[0] / max(1, v[1]) for (s, c), v in sums.items()}
    return {"per_site": per, "mean": sum(per.values()) / len(per)}


# ---- checkpoints ------------------------------------------------------
def atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _rng_state():
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _set_rng_state(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


# ---- trainer ----------------------------------------------------------
class Trainer:
    def __init__(self, model: SplitModel, data, config, run_dir: Path):
        self.model, self.data, self.config, self.run = model, data, config, Path(run_dir)
        self.cfg = config["training"]
        self.sites = list(self.cfg["sites"])
        self.phases = (["reconstruction"] if self.cfg.get("reconstruction", {}).get("enabled") else []) + ["distillation"]
        self.params = [p for p in model.parameters() if p.requires_grad]
        if any(p.requires_grad for p in model.base.parameters()):
            raise RuntimeError("backbone must stay frozen")
        self.phase_index = 0
        self.update = 0          # updates within the current phase
        self.total_updates = 0
        self.site_updates = {s: 0 for s in self.sites}
        self.best = math.inf
        self.bad = 0
        self.history = []
        self.generator = torch.Generator().manual_seed(config["seed"])
        self._new_optimizer()

    @property
    def phase(self):
        return self.phases[self.phase_index]

    def _phase_cfg(self):
        return self.cfg["reconstruction"] if self.phase == "reconstruction" else self.cfg

    def _new_optimizer(self):
        c = self._phase_cfg()
        self.optimizer = torch.optim.AdamW(self.params, lr=c["lr"], weight_decay=self.cfg["weight_decay"],
                                           betas=tuple(self.cfg.get("betas", (0.9, 0.999))),
                                           eps=self.cfg.get("eps", 1e-8))
        warmup = c.get("warmup_updates", 0)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda u: min(1.0, (u + 1) / warmup) if warmup else 1.0)
        self.best, self.bad = math.inf, 0

    # one optimizer update at one site
    def step(self):
        site = self.sites[self.total_updates % len(self.sites)]
        k = self.site_updates[site]
        self.model.set_site(site)
        self.model.train()
        batch = self.data.train_batch(k)
        per_batch = getattr(self.model.channel, "snr_mode", "per_sample") == "per_batch"
        snr = sample_snr(self.config["channel"], batch["labels"].shape[0], self.data.device,
                         self.generator, per_batch=per_batch)
        self.optimizer.zero_grad(set_to_none=True)
        loss, stats = site_loss(self.model, self.data, batch, snr, self.cfg, self.phase)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at update {self.total_updates + 1} ({site})")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(self.params, self.cfg["grad_clip"])
        if not torch.isfinite(grad):
            raise FloatingPointError(f"nonfinite gradient at update {self.total_updates + 1} ({site})")
        lr = self.optimizer.param_groups[0]["lr"]
        self.optimizer.step()
        self.scheduler.step()
        self.update += 1
        self.total_updates += 1
        self.site_updates[site] = k + 1
        return {"type": "update", "phase": self.phase, "update": self.update, "total": self.total_updates,
                "site": site, "site_update": k + 1, "lr": lr, "grad_norm": float(grad), **stats}

    def _log(self, record):
        record["time"] = time.time()
        with (self.run / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")

    def _checkpoint_state(self):
        return {"communication": self.model.communication_state(), "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(), "phase_index": self.phase_index, "update": self.update,
                "total_updates": self.total_updates, "site_updates": self.site_updates, "best": self.best,
                "bad": self.bad, "history": self.history, "generator": self.generator.get_state(),
                "rng": _rng_state(), "config": self.config}

    def save(self, name):
        atomic_save(self._checkpoint_state(), self.run / name)

    def load(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_communication_state(state["communication"])
        self.phase_index = state["phase_index"]
        self._new_optimizer()
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        for key in ("update", "total_updates", "site_updates", "best", "bad", "history"):
            setattr(self, key, state[key])
        self.generator.set_state(state["generator"])
        _set_rng_state(state["rng"])

    def run_phase(self):
        c = self._phase_cfg()
        es = c["early_stopping"]
        log_every = self.cfg.get("log_every", 50)
        while True:
            if self.update >= c["max_updates"]:
                return "max_updates"
            record = self.step()
            if self.update % log_every == 0 or self.update == 1:
                self._log(record)
            if self.update % es["every"] == 0:
                result = validate(self.model, self.data, self.sites, self.cfg, self.phase)
                improved = result["mean"] < self.best * (1 - es["min_delta"])
                if improved:
                    self.best, self.bad = result["mean"], 0
                    self.save(f"best_{self.phase}.pt")
                else:
                    self.bad += 1
                entry = {"type": "validation", "phase": self.phase, "update": self.update,
                         "total": self.total_updates, "improved": improved, "bad": self.bad, **result}
                self.history.append({k: entry[k] for k in ("phase", "update", "mean", "improved")})
                self._log(entry)
                self.save("last.pt")
                if self.bad >= es["patience"]:
                    return "early_stopping"

    def train(self):
        self.run.mkdir(parents=True, exist_ok=True)
        stops = {}
        while self.phase_index < len(self.phases):
            stops[self.phase] = self.run_phase()
            best = self.run / f"best_{self.phase}.pt"
            if best.exists():  # continue from the best weights of the finished phase
                self.model.load_communication_state(
                    torch.load(best, map_location="cpu", weights_only=False)["communication"])
            self._log({"type": "phase_end", "phase": self.phase, "reason": stops[self.phase],
                       "update": self.update, "best": self.best})
            if self.phase_index == len(self.phases) - 1:
                self.save("last.pt")
                return stops
            self.phase_index += 1
            self.update = 0
            self._new_optimizer()
        return stops
