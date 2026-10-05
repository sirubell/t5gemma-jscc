"""Current reconstruction bodies; saved record meanings preserved."""

from __future__ import annotations
import torch
from .models.split_model import ResolvedSite, resolve_encoder_site, stack_module
from .runtime import autocast_for
from typing import cast
from .checkpoint_readers import PILOT_SITES as SITES
from .losses import reconstruction_loss_stats
from .training_objectives import BatchValues, objective_components
from .presentation import derived_seed


class _EncoderBoundaryReached(Exception):
    """Unwind the frozen prefix before executing an encoder suffix."""


def _clear_activations(model):
    model.activation = model.reconstruction = None
    model.memory_activation = model.memory_reconstruction = None


@torch.no_grad()
def capture_online_encoder_activation(model, batch: dict, site: ResolvedSite) -> torch.Tensor:
    """Acquire one clean, native-dtype source feature at its exact encoder site.

    ``enc_emb`` is the first layer input, after backbone embedding processing;
    intermediate sites precede final normalization; ``enc_fn`` follows it.
    Native batch preparation owns image scatter and decoder-input semantics;
    the boundary exception prevents decoder execution and teacher logits.
    Text-only diagnostic callers retain the native encoder-only path.
    """
    if not isinstance(site, ResolvedSite):
        raise TypeError("online reconstruction requires a resolved encoder site")
    if resolve_encoder_site(model.base, site.split, site.model_revision) != site:
        raise ValueError("online reconstruction site differs from the backbone")
    if model.split != site.split or model.requires_memory_transmission:
        raise ValueError("route the model to the selected encoder site before capture")
    if model.base.training or any(p.requires_grad for p in model.base.parameters()):
        raise ValueError("online feature acquisition requires a frozen eval backbone")
    parameter = next(model.base.parameters())
    native_batch = "pixel_values" in batch and "labels" in batch
    if native_batch:
        from .runtime import model_inputs
        inputs, _labels = model_inputs(batch, model)
    else:
        inputs = {key: batch[key].to(parameter.device) for key in ("input_ids", "attention_mask")}
        if "pixel_values" in batch:
            pixels = batch["pixel_values"]
            inputs["pixel_values"] = (pixels.flatten(0, 1) if pixels.ndim == 5 else pixels).to(
                device=parameter.device, dtype=parameter.dtype)
    mask = inputs["attention_mask"]
    if mask.ndim != 2 or mask.shape != inputs["input_ids"].shape:
        raise ValueError("online reconstruction requires a complete 2D source mask")
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("online reconstruction requires a binary source mask")
    stack = stack_module(model.base, "enc")
    observed = []

    def stop(value):
        observed.append(value.detach().clone())
        raise _EncoderBoundaryReached

    def stop_input(_module, args, kwargs):
        stop(args[0] if args else kwargs["hidden_states"])

    def stop_output(_module, _args, output):
        stop(output[0] if isinstance(output, tuple) else output)

    if site.where == "after_embed":
        handle = stack.layers[0].register_forward_pre_hook(stop_input, with_kwargs=True, prepend=True)
    else:
        module = stack.norm if site.where == "after_final_norm" else stack.layers[site.index]
        handle = module.register_forward_hook(stop_output, prepend=True)
    try:
        try:
            with autocast_for(model), model.attention_context(), model.transmission(
                bypass=True, encoder_mask=mask
            ):
                if native_batch:
                    model.base(**inputs, return_dict=True)
                else:
                    model.base.get_encoder()(**inputs, return_dict=True)
        except _EncoderBoundaryReached:
            pass
        if len(observed) != 1:
            raise RuntimeError("online encoder boundary was not reached exactly once")
        activation = observed.pop()
        if activation.ndim != 3 or activation.shape != (*mask.shape, site.hidden_dim):
            raise RuntimeError("online activation differs from the complete padded source view")
        return activation
    finally:
        handle.remove()
        observed.clear()
        _clear_activations(model)


def online_reconstruction_batch_values(model, batch: dict, snr_db, settings: dict | None = None,
                                       *, site: ResolvedSite) -> BatchValues:
    """Return effective-batch-compatible statistics for exactly ``0.1 * nMSE``.

    The clean prefix is detached, while the local transmission retains codec and
    channel autograd. Hidden denominators count source-valid samples, never
    target tokens. Model activation references are cleared even on failure.
    """
    if site.site_id not in SITES:
        raise ValueError("online reconstruction supports the original six encoder sites")
    settings = settings or {"kl_weight": 0.0, "mse_weight": 0.1}
    explicit = settings.get("loss_weights")
    weights = explicit or {"kl": settings["kl_weight"], "hidden": settings["mse_weight"], "memory": 0.0}
    if (weights["kl"] != 0.0 or weights["hidden"] != 0.1 or weights.get("memory", 0.0) != 0.0):
        raise ValueError("online reconstruction requires the 0.1*nMSE objective without KL")
    activation = capture_online_encoder_activation(model, batch, site)
    mask = batch["attention_mask"].to(activation.device)
    try:
        with autocast_for(model), model.transmission(snr_db, encoder_mask=mask):
            model._roundtrip(activation)
            reconstructed = model.reconstruction
            numerator, denominator = reconstruction_loss_stats(reconstructed, activation, mask)
        hidden = numerator / denominator.clamp_min(1).to(numerator.dtype)
        zero = numerator.detach().new_zeros(())
        components = objective_components(zero, hidden, None, settings)
        return cast(BatchValues, {**components, "kl": zero, "nmse": hidden,
                    "kl_numerator": zero, "kl_denominator": denominator.new_zeros(()),
                    "hidden_numerator": numerator, "hidden_denominator": denominator})
    finally:
        _clear_activations(model)


@torch.no_grad()
def capture_clean_targets(model, batch, site):
    """Clean decoder prefix and encoder memory; unwind before the receiver."""
    from .models.split_model import ResolvedDecoderSite, resolve_training_site
    from .runtime import model_inputs
    if isinstance(site, ResolvedSite):
        # The canonical memory alias exists even when this route has one role.
        return capture_online_encoder_activation(model, batch, site), None
    if not isinstance(site, ResolvedDecoderSite):
        raise TypeError("original13 capture requires a resolved site")
    if (resolve_training_site(model.base, site.site_id, site.model_revision) != site
            or model.split != site.split or not model.universal_codec):
        raise ValueError("capture requires the selected universal decoder route")
    if model.base.training or any(p.requires_grad for p in model.base.parameters()):
        raise ValueError("online feature acquisition requires a frozen eval backbone")
    kwargs, labels = model_inputs(batch, model)
    mask = kwargs["attention_mask"]
    if (mask.ndim != 2 or mask.shape != kwargs["input_ids"].shape
            or not bool(((mask == 0) | (mask == 1)).all())):
        raise ValueError("online reconstruction requires a complete binary source mask")
    if labels.ndim != 2 or labels.shape[0] != mask.shape[0]:
        raise ValueError("online reconstruction requires a complete 2D native target view")
    observed, memories = [], []

    def capture_memory(_module, _args, output):
        memories.append(output.last_hidden_state.detach().clone())

    def stop(_module, _args, output):
        observed.append((output[0] if isinstance(output, tuple) else output).detach().clone())
        raise _EncoderBoundaryReached

    stack = stack_module(model.base, "dec")
    handle = stack.layers[site.index].register_forward_hook(stop, prepend=True)
    memory_handle = model.base.get_encoder().register_forward_hook(capture_memory)
    try:
        with autocast_for(model), model.attention_context(), model.transmission(
            bypass=True, encoder_mask=mask, decoder_mask=labels != -100
        ):
            try:
                # Native forward owns multimodal scatter and transmitter decoder memory.
                model.base(**kwargs, return_dict=True)
            except _EncoderBoundaryReached:
                pass
        if len(observed) != 1 or len(memories) != 1:
            raise RuntimeError("online decoder boundary/memory was not reached exactly once")
        hidden, memory = observed.pop(), memories.pop()
        if hidden.shape != (*labels.shape, site.hidden_dim) or memory.shape != (*mask.shape, site.hidden_dim):
            raise RuntimeError("online targets differ from complete native source/target views")
        return hidden, memory
    finally:
        handle.remove()
        memory_handle.remove()
        observed.clear()
        memories.clear()
        _clear_activations(model)


def reconstruction_batch_values(model, batch, snr_db, settings, *, site):
    """Local hidden/memory roundtrips with separate valid-sample reductions."""
    hidden, memory = capture_clean_targets(model, batch, site)
    source_mask = batch["attention_mask"].to(hidden.device)
    target_mask = (batch["labels"] != -100).to(hidden.device)
    try:
        with autocast_for(model), model.transmission(
            snr_db, encoder_mask=source_mask, decoder_mask=target_mask
        ):
            reconstructed = model._roundtrip(hidden)
            hidden_mask = target_mask if model.split["stack"] == "dec" else source_mask
            hn, hd = reconstruction_loss_stats(reconstructed, hidden, hidden_mask)
            mn = md = None
            if memory is not None:
                model.memory_activation = memory
                model.memory_reconstruction = model._transmit(memory, model.memory_codec, "memory",
                    source_mask, mask_representation="binary")
                mn, md = reconstruction_loss_stats(model.memory_reconstruction, memory, source_mask)
        h = hn / hd.clamp_min(1)
        m = mn / md.clamp_min(1) if mn is not None and md is not None else None
        zero = hn.detach().new_zeros(())
        components = objective_components(zero, h, m, settings)
        return cast(BatchValues, {**components, "kl": zero, "nmse": h if m is None else (h + m) / 2,
            "kl_numerator": zero, "kl_denominator": hd.new_zeros(()),
            "hidden_numerator": hn, "hidden_denominator": hd,
            "memory_numerator": mn, "memory_denominator": md})
    finally:
        _clear_activations(model)


def local_batch_values(model, shard: dict, snr_db, settings: dict) -> BatchValues:
    """Run the existing communication path only; no backbone forward or logits."""
    device = next(model.base.parameters()).device
    activation = shard["activation"].to(device)
    mask = shard["attention_mask"].to(device)
    with autocast_for(model), model.transmission(snr_db, encoder_mask=mask):
        model._roundtrip(activation)
        reconstructed = model.reconstruction
        numerator, denominator = reconstruction_loss_stats(reconstructed, activation, mask)
    zero = numerator.detach().new_zeros(())
    hidden = numerator / denominator.clamp_min(1).to(numerator.dtype)
    components = objective_components(zero, hidden, None, settings)
    return cast(BatchValues, {**components, "loss": components["loss"], "kl": zero, "nmse": hidden,
            "kl_numerator": zero, "kl_denominator": zero,
            "hidden_numerator": numerator, "hidden_denominator": denominator})




def paired_snr(model, config: dict, step: int, microbatch: int, batch_size: int):
    channel = config["channel"]
    if not channel["train_noise"]:
        return None
    paired = config["training"].get("paired_randomness")
    low, high = channel["train_snr_range"]
    device = next(model.base.parameters()).device
    if paired:
        generator = torch.Generator(device=device).manual_seed(
            derived_seed(paired["seed"], f"train:{step}:{microbatch}:snr"))
        return torch.empty((batch_size, 1, 1), device=device).uniform_(low, high, generator=generator)
    from .runtime import sample_snr_db
    return sample_snr_db(model, low, high, batch_size)
