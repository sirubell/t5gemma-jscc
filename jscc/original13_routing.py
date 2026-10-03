"""Original13 universal routes and objectives under the released M1 contract.

The integration owner constructs/resets optimizers and checkpoint envelopes.
Only the codec body is exported here; the frozen backbone stays in the model.
"""
from copy import deepcopy
from dataclasses import asdict
from typing import cast

import torch

from .activation_replay import canonical_digest
from .baseline_protocol import BaselineLearner
from .encoder_online_reconstruction import original13_reconstruction_batch_values
from .encoder_stages_plan import ORIGINAL13_SITES, Original13Plan
from .experiment_schedule import noise_namespace
from .models.channel import AWGNChannel
from .models.split_model import resolve_original13_site
from .presentation import derived_seed
from .sharing_protocol import SharingLearner
from .sharing_schedule import NATIVE16_WEIGHT_DECAY, SharingExposureLedger, LearnerSchedule
from .training import batch_losses


class Original13Learner(SharingLearner):
    """Reuse fail-stop numerical updates and the complete-view exposure ledger."""
    def __init__(self, model, *, plan, metadata, event_sink):
        if not isinstance(plan, Original13Plan):
            raise TypeError("original13 learner requires its released plan")
        config = model.codec.config
        if (config["architecture"] != "direct_affine" or config["layernorm"] != "none"
                or config["snr_film"] or config["n_res_blocks"] != 0 or config["dropout"] != 0
                or config["bottleneck_dim"] != plan.width or model.numerical_policy != "native"
                or not model.channel_config["normalize_power"] or not isinstance(model.channel, AWGNChannel)):
            raise ValueError("original13 requires the approved affine/native normalized AWGN recipe")
        sites = {name: resolve_original13_site(model.base, name, metadata["model_revision"])
                 for name in ORIGINAL13_SITES}
        if any(site.hidden_dim != model.codec.encoder[0].weight.shape[1] for site in sites.values()):
            raise ValueError("all original13 roles must match the canonical codec width")
        # Reuse baseline initialization before opting into the universal memory
        # alias. Its enc_fn-only guard continues to protect legacy callers.
        BaselineLearner.__init__(self, model, run_id=f"b{plan.width}_r{plan.ratio}", task="hellaswag",
            identity={"source": metadata["source_identity"], "config": metadata["config_identity"],
                      "data": metadata["data_identity"], "parent": None}, pairing_id=plan.study_pairing_id,
            final_step=plan.horizon, effective_batch=16, weight_decay=NATIVE16_WEIGHT_DECAY,
            synthetic=True, event_sink=event_sink, audit_policy="sparse_first_final")
        model.enable_universal_codec()
        self.synthetic = plan.synthetic
        self.schedule = plan
        self.sites = sites
        self.exposure = SharingExposureLedger(cast(LearnerSchedule, plan), sites=ORIGINAL13_SITES)
        self.microbatch_size = None
        self.active_site = self.active_local = self.pending_update = None
        self.noise_identity = canonical_digest({"pairing_id": plan.study_pairing_id,
            "schema": "original13-role-awgn-v1", "streams": ["hidden", "memory"],
            "torch": str(torch.__version__), "device": str(next(model.base.parameters()).device),
            "dtype": str(next(model.base.parameters()).dtype), "snr": "per-sequence-uniform[-6,18]"})
        # Placeholder only; the integration state installs the phase scheduler.
        stage = plan.stages[0]
        from .sharing_schedule import sharing_lr_factor
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda index: sharing_lr_factor(index, stage.horizon))

    def _objective_settings(self, kind):
        return {"temperature": 1.0, "loss_weights": self.schedule.objective_at(self.completed),
                "requires_memory": self.model.requires_memory_transmission}

    def _values(self, batch, kind, step, micro, purpose, condition: str | int = "uniform", *,
                capture=True, bound_identity=None):
        if purpose != "train" or micro != 0 or self.pending_update is None or self.active_site is None:
            raise ValueError("original13 requires one scheduled complete native training view")
        key = self.pending_update.noise_key
        namespace = noise_namespace(key)
        device = next(self.model.base.parameters()).device
        generator = torch.Generator(device=device).manual_seed(derived_seed(0, namespace + ":snr"))
        snr = torch.empty((len(batch["labels"]), 1, 1), device=device).uniform_(-6, 18, generator=generator)
        settings = self._objective_settings(kind)
        with self.model.channel.replay(0, namespace, capture=capture):
            if self.schedule.stage_at(self.completed).objective_kind == "reconstruction":
                values = original13_reconstruction_batch_values(self.model, batch, snr, settings,
                                                                site=self.sites[self.active_site])
            else:
                values = batch_losses(self.model, batch, settings, snr, return_stats=True, valid_only_kl=True)
        return values, snr, {"key": asdict(key), "draws": list(self.model.channel.draw_summaries)}

    def update(self, batches, **kwargs):
        if self.failed or self.completed >= self.schedule.horizon:
            raise RuntimeError("learner is terminal; no retries")
        if self.phase_start_step != self.schedule.stage_at(self.completed).start:
            raise ValueError("declared phase reset must precede the next update")
        return super().update(batches, **kwargs)

    def _event(self, event_type, phase, payload):
        stage = next(s for s in self.schedule.stages if s.start == self.phase_start_step)
        if event_type == "update":
            payload["objective"]["kind"] = stage.objective_kind
        return super()._event(event_type, stage.phase_id, payload)


def create_original13_learner(model, *, plan, metadata, event_sink):
    return Original13Learner(model, plan=plan, metadata=metadata, event_sink=event_sink)


def canonical_codec(model):
    if not model.universal_codec or model.memory_codec is not model.codec:
        raise ValueError("model does not have one original13 canonical parameter owner")
    if any(p.requires_grad for p in model.base.parameters()):
        raise ValueError("backbone must remain frozen")
    parameters = [p for p in model.parameters() if p.requires_grad]
    if {id(p) for p in parameters} != {id(p) for p in model.codec.parameters()}:
        raise ValueError("trainable parameters differ from the canonical codec")
    return model.codec


def export_codec(model):
    """Flat codec state_dict: encoder.0.{weight,bias}, decoder.0.{weight,bias}."""
    return {key: tensor.detach().clone() for key, tensor in canonical_codec(model).state_dict().items()}


def load_codec(model, weights):
    codec = canonical_codec(model)
    expected = codec.state_dict()
    if not isinstance(weights, dict) or set(weights) != set(expected):
        raise ValueError("wrong-kind or duplicate body; expected a flat canonical codec state")
    for key, value in weights.items():
        if (not torch.is_tensor(value) or value.shape != expected[key].shape
                or value.dtype != expected[key].dtype or not bool(torch.isfinite(value).all())):
            raise ValueError("invalid canonical codec tensor: " + key)
    codec.load_state_dict(weights, strict=True)
    canonical_codec(model)


_RUNTIME_FIELDS = ("snr_db", "bypass", "_encoder_valid_mask", "_decoder_valid_mask",
    "_active_encoder_valid_mask", "_encoder_mask_representation", "_decoder_mask_representation",
    "_active_encoder_mask_representation", "channel_uses", "channel_uses_valid")


def _require_idle(model):
    canonical_codec(model)
    if (model._routing_active or model._forward_active or model._payload_observer is not None
            or model.channel._replay is not None):
        raise ValueError("runtime-state export/restore requires an idle route and noise context")


def observation_state(model):
    """Detached idle state, including modes/masks/noise audits/payload counters."""
    _require_idle(model)
    return deepcopy({"schema": "original13-idle-runtime-v1", "split": model.split,
        "fields": {key: getattr(model, key) for key in _RUNTIME_FIELDS},
        "modes": {name: module.training for name, module in model.named_modules()},
        "channel_stream": model.channel._stream, "draw_summaries": model.channel.draw_summaries,
        "memory_hook_count": len(model._memory_handles)})


def restore_runtime_state(model, state):
    _require_idle(model)
    modes = dict(model.named_modules())
    if (not isinstance(state, dict) or state.get("schema") != "original13-idle-runtime-v1"
            or set(state) != {"schema", "split", "fields", "modes", "channel_stream", "draw_summaries", "memory_hook_count"}
            or state["split"] != model.split or state["memory_hook_count"] != len(model._memory_handles)
            or set(state["fields"]) != set(_RUNTIME_FIELDS) or set(state["modes"]) != set(modes)
            or any(type(mode) is not bool for mode in state["modes"].values())
            or any(state["modes"][name] for name in modes if name == "base" or name.startswith("base."))):
        raise ValueError("unsupported original13 idle runtime state")
    for key, value in state["fields"].items():
        setattr(model, key, deepcopy(value))
    for name, mode in state["modes"].items():
        modes[name].training = mode
    model.channel._stream = state["channel_stream"]
    model.channel.draw_summaries = deepcopy(state["draw_summaries"])
    model.activation = model.reconstruction = None
    model.memory_activation = model.memory_reconstruction = None
