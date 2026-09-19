"""Insert codec -> channel -> codec at one frozen encoder/decoder layer."""
from contextlib import contextmanager

import torch
from torch import nn

from ..config import resolve_codec_configs
from .channel import build_channel, normalize_power, valid_payload_count
from .codec import Codec


def stack_module(base, stack):
    if stack == "enc":
        encoder = base.get_encoder()
        return getattr(encoder, "text_model", encoder)
    if stack == "dec":
        return base.get_decoder()
    raise ValueError("split.stack must be enc or dec")


class SplitModel(nn.Module):
    def __init__(self, base, codec, channel, split, channel_config):
        super().__init__()
        self.base = base.requires_grad_(False).eval()
        self.codec = codec
        self.channel = channel
        self.split = split
        self.channel_config = channel_config
        self.snr_db = None
        self.bypass = False
        self.activation = None
        self.reconstruction = None
        self.memory_activation = None
        self.memory_reconstruction = None
        self.memory_codec: Codec | None = None
        self.channel_uses = {"hidden": 0, "memory": 0}
        # ``channel_uses`` is retained as the allocated execution-tensor
        # count.  The separate map excludes padded token positions and is
        # useful for reporting an effective payload count.
        self.channel_uses_valid = {"hidden": 0, "memory": 0}
        self._encoder_valid_mask = None
        self._decoder_valid_mask = None
        self._active_encoder_valid_mask = None
        self._encoder_mask_representation = None
        self._decoder_mask_representation = None
        self._active_encoder_mask_representation = None
        self._encoder_mask_handle = None
        stack = stack_module(base, split["stack"])
        # Capture the original encoder attention mask before Transformers
        # expands it into an internal attention mask.  This also covers
        # evaluation adapters that call ``model.base`` directly instead of
        # going through SplitModel.forward().
        encoder_stack = stack_module(base, "enc")
        self._encoder_mask_handle = encoder_stack.layers[0].register_forward_pre_hook(
            self._capture_encoder_mask, with_kwargs=True
        )
        where = split["where"]
        if where in ("after_embed", "before_first_layer"):
            # At encoder input, image features have already replaced image tokens.
            self.handle = stack.layers[0].register_forward_pre_hook(self._pre_hook, with_kwargs=True)
        elif where == "after_layer":
            self.handle = stack.layers[split["index"]].register_forward_hook(self._hook)
        elif where == "after_final_norm":
            self.handle = stack.norm.register_forward_hook(self._hook)
        else:
            raise ValueError(f"unknown split.where: {where}")
        if split["stack"] == "dec":
            first_receiver = (split["index"] + 1 if where == "after_layer" else
                              len(stack.layers) if where == "after_final_norm" else 0)
            if first_receiver < len(stack.layers):
                _, memory_config = resolve_codec_configs(codec.config)
                self.memory_codec = Codec(codec.encoder[0].weight.shape[1], memory_config)
                stack.register_forward_pre_hook(self._begin_decoder, with_kwargs=True)
                for layer in stack.layers[first_receiver:]:
                    layer.self_attn.register_forward_pre_hook(self._receiver_memory, with_kwargs=True)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()  # Freezing weights alone does not disable backbone dropout.
        return self

    @contextmanager
    def transmission(self, snr_db=None, *, bypass=False, encoder_mask=None, decoder_mask=None,
                     encoder_mask_representation=None, decoder_mask_representation=None):
        """Set channel condition and optional stream-valid masks.

        ``encoder_mask`` and ``decoder_mask`` are optional so existing callers
        remain unchanged.  The training loop can supply the decoder label mask
        here when its model input intentionally omits ``decoder_attention_mask``;
        the encoder mask is also used for receiver-memory transmission.
        """
        if (encoder_mask is not None and encoder_mask.is_floating_point()
                and encoder_mask.ndim == 4 and encoder_mask_representation is None):
            raise ValueError("explicit representation required for a public 4D encoder mask")
        if decoder_mask is not None and decoder_mask.ndim != 2:
            raise ValueError("decoder payload validity requires a 2D token mask, not a causal attention mask")
        previous = (self.snr_db, self.bypass, self._encoder_valid_mask,
                    self._decoder_valid_mask, self._active_encoder_valid_mask,
                    self._encoder_mask_representation, self._decoder_mask_representation,
                    self._active_encoder_mask_representation)
        self.snr_db, self.bypass = snr_db, bypass
        if encoder_mask is not None:
            self._encoder_valid_mask = encoder_mask
            self._active_encoder_valid_mask = encoder_mask
            self._encoder_mask_representation = (
                encoder_mask_representation
                if encoder_mask_representation is not None
                else self._infer_mask_representation(encoder_mask)
            )
            self._active_encoder_mask_representation = self._encoder_mask_representation
        if decoder_mask is not None:
            self._decoder_valid_mask = decoder_mask
            self._decoder_mask_representation = (
                decoder_mask_representation
                if decoder_mask_representation is not None
                else self._infer_mask_representation(decoder_mask)
            )
        self.channel_uses = {"hidden": 0, "memory": 0}
        self.channel_uses_valid = {"hidden": 0, "memory": 0}
        try:
            yield
        finally:
            (self.snr_db, self.bypass, self._encoder_valid_mask,
             self._decoder_valid_mask, self._active_encoder_valid_mask,
             self._encoder_mask_representation, self._decoder_mask_representation,
             self._active_encoder_mask_representation) = previous

    @contextmanager
    def decoder_payload_validity(self, mask):
        """Per-forward accounting mask; do not reset counts or change attention/noise."""
        if mask.ndim != 2 or mask.dtype != torch.bool:
            raise ValueError("decoder payload validity requires a 2D boolean mask")
        previous = self._decoder_valid_mask, self._decoder_mask_representation
        self._decoder_valid_mask, self._decoder_mask_representation = mask, "binary"
        try:
            yield
        finally:
            self._decoder_valid_mask, self._decoder_mask_representation = previous

    @property
    def channel_uses_allocated(self):
        """Allocated latent coordinates sent through each stream."""
        return self.channel_uses

    @property
    def valid_payload_counts(self):
        """Valid latent coordinates, excluding masked/padded positions."""
        return self.channel_uses_valid

    def _canonical_mask(self, mask, hidden):
        if mask is None:
            return None
        # ``normalize_power`` performs the same conversion.  Keep the raw
        # mask here so custom channels/hooks can still inspect its provenance.
        if torch.is_tensor(mask) and mask.ndim == hidden.ndim - 1:
            return mask.to(device=hidden.device)
        return mask

    @staticmethod
    def _infer_mask_representation(mask):
        """Resolve the caller's mask contract without guessing from zeros.

        Public model inputs use a two-dimensional binary validity mask.  A
        four-dimensional floating mask observed by the encoder hook is the
        Transformer's expanded additive attention mask.  Other floating
        shapes are ambiguous and are rejected by ``normalize_power`` unless a
        caller supplies an explicit representation.
        """
        if mask is None:
            return None
        if not torch.is_tensor(mask):
            return None
        if mask.dtype == torch.bool or not mask.is_floating_point():
            return "binary"
        if mask.ndim == 2:
            return "binary"
        if mask.ndim == 4:
            return "additive"
        return None

    def _transmit(self, hidden, codec, stream, valid_mask=None, *, token_wise=False,
                  mask_representation=None):
        # Training wraps the complete teacher/student pass in autocast.  The
        # HellaSwag evaluator calls ``model.base`` directly through lm-eval,
        # so the communication hook must preserve the same BF16-backbone /
        # FP32-codec boundary on its own.
        backbone_parameter = next(self.base.parameters())
        autocast_enabled = backbone_parameter.dtype in (torch.bfloat16, torch.float16)
        with torch.autocast(device_type=backbone_parameter.device.type,
                            dtype=backbone_parameter.dtype,
                            enabled=autocast_enabled):
            z = codec.encode(hidden)
            self.channel_uses[stream] += z.numel()
            self.channel_uses_valid[stream] += valid_payload_count(
                z, valid_mask, mask_representation=mask_representation
            )
            if self.channel_config["normalize_power"]:
                z = normalize_power(
                    z, valid_mask, token_wise=token_wise,
                    mask_representation=mask_representation,
                )
            received = self.channel(z, self.snr_db)
            film_snr = self.snr_db
            if film_snr is None:
                film_snr = self.channel_config["clean_film_snr"]
            # LayerNorm and residual blocks remain FP32 parameters in the
            # corrected precision contract.  Return the reconstructed stream
            # in the backbone activation dtype before downstream BF16 linear
            # layers consume it.
            return codec.decode(received, film_snr).to(dtype=hidden.dtype)

    def _roundtrip(self, hidden):
        self.activation = hidden.detach()
        valid_mask = self._active_encoder_valid_mask if self.split["stack"] == "enc" else self._decoder_valid_mask
        mask_representation = (
            self._active_encoder_mask_representation
            if self.split["stack"] == "enc"
            else self._decoder_mask_representation
        )
        self.reconstruction = None if self.bypass else self._transmit(
            hidden, self.codec, "hidden", self._canonical_mask(valid_mask, hidden),
            token_wise=self.split["stack"] == "dec",
            mask_representation=mask_representation,
        )
        return hidden if self.bypass else self.reconstruction

    def _begin_decoder(self, module, args, kwargs):
        self.memory_activation = self.memory_reconstruction = None

    def _receiver_memory(self, module, args, kwargs):
        if self.bypass:
            return
        cache = kwargs.get("past_key_values")
        if cache is not None and cache.is_updated.get(module.layer_idx, False):
            # These receiver K/V tensors were built from transmitted memory.
            return
        if self.memory_reconstruction is None:
            memory = kwargs["encoder_hidden_states"]
            self.memory_activation = memory.detach()
            self.memory_reconstruction = self._transmit(
                memory, self.memory_codec, "memory",
                self._canonical_mask(self._active_encoder_valid_mask, memory),
                mask_representation=self._active_encoder_mask_representation,
            )
        return args, {**kwargs, "encoder_hidden_states": self.memory_reconstruction}

    def load_communication_state(self, state):
        self.codec.load_state_dict(state["codec"])
        self.channel.load_state_dict(state["channel"])
        memory = state.get("memory_codec")
        if self.memory_codec is not None:
            if memory is None:
                raise ValueError("Decoder checkpoint has no receiver-memory codec. Use its historical source "
                                 "for diagnostic evaluation; train a new checkpoint for the corrected boundary.")
            self.memory_codec.load_state_dict(memory)
        elif memory is not None:
            raise ValueError("Checkpoint memory codec does not match this split")

    def _hook(self, module, inputs, output):
        if isinstance(output, tuple):
            return (self._roundtrip(output[0]), *output[1:])
        return self._roundtrip(output)

    def _pre_hook(self, module, args, kwargs):
        if args:
            return (self._roundtrip(args[0]), *args[1:]), kwargs
        return args, {**kwargs, "hidden_states": self._roundtrip(kwargs["hidden_states"])}

    @staticmethod
    def _mask_from_hook_args(args, kwargs):
        candidate = kwargs.get("attention_mask")
        if candidate is None and len(args) > 2:
            # T5Gemma text layers receive (hidden, position_embeddings,
            # expanded_attention_mask, position_ids, ...).
            candidate = args[2]
        return candidate

    def _capture_encoder_mask(self, module, args, kwargs):
        candidate = self._encoder_valid_mask
        if candidate is None:
            candidate = self._mask_from_hook_args(args, kwargs)
        self._active_encoder_valid_mask = candidate
        self._active_encoder_mask_representation = (
            self._encoder_mask_representation if self._encoder_valid_mask is not None
            else self._infer_mask_representation(candidate)
        )

    def _set_input_masks(self, kwargs):
        if "attention_mask" in kwargs and self._encoder_valid_mask is None:
            self._encoder_valid_mask = kwargs["attention_mask"]
            self._active_encoder_valid_mask = self._encoder_valid_mask
            self._encoder_mask_representation = self._infer_mask_representation(kwargs["attention_mask"])
            self._active_encoder_mask_representation = self._encoder_mask_representation
        if "decoder_attention_mask" in kwargs and self._decoder_valid_mask is None:
            self._decoder_valid_mask = kwargs["decoder_attention_mask"]
            self._decoder_mask_representation = self._infer_mask_representation(kwargs["decoder_attention_mask"])

    def forward(self, **kwargs):
        previous = (self._encoder_valid_mask, self._decoder_valid_mask,
                    self._active_encoder_valid_mask, self._encoder_mask_representation,
                    self._decoder_mask_representation, self._active_encoder_mask_representation)
        self._set_input_masks(kwargs)
        try:
            return self.base(**kwargs)
        finally:
            (self._encoder_valid_mask, self._decoder_valid_mask,
             self._active_encoder_valid_mask, self._encoder_mask_representation,
             self._decoder_mask_representation, self._active_encoder_mask_representation) = previous

    def generate(self, **kwargs):
        if self.memory_codec is not None:
            generation_config = kwargs.get("generation_config") or self.base.generation_config
            beams = kwargs.get("num_beams", generation_config.num_beams) or 1
            if beams != 1:
                raise ValueError("Receiver-memory transmission currently supports num_beams=1 only")
        previous = (self._encoder_valid_mask, self._decoder_valid_mask,
                    self._active_encoder_valid_mask, self._encoder_mask_representation,
                    self._decoder_mask_representation, self._active_encoder_mask_representation)
        self._set_input_masks(kwargs)
        try:
            return self.base.generate(**kwargs)
        finally:
            (self._encoder_valid_mask, self._decoder_valid_mask,
             self._active_encoder_valid_mask, self._encoder_mask_representation,
             self._decoder_mask_representation, self._active_encoder_mask_representation) = previous


def build_model(config):
    # Imports are delayed so --check and tensor-only tests need no Transformers.
    from transformers import AutoModelForSeq2SeqLM, AutoProcessor, AutoTokenizer
    model_config = config["model"]
    kwargs = {"revision": model_config["revision"]}
    processor_cls = AutoProcessor if config["task"] == "coco" else AutoTokenizer
    processor = processor_cls.from_pretrained(model_config["name"], **kwargs)
    dtype = getattr(torch, model_config["dtype"])
    base = AutoModelForSeq2SeqLM.from_pretrained(
        model_config["name"], dtype=dtype, low_cpu_mem_usage=True, **kwargs
    )
    input_dim = int(base.config.decoder.hidden_size)
    config["codec"]["input_dim"] = input_dim
    codec = Codec(input_dim, config["codec"])
    wrapper = SplitModel(base, codec, build_channel(config["channel"]),
                         config["split"], config["channel"])
    # Keep the frozen backbone at the requested execution dtype, while the
    # trainable communication modules remain FP32.  Calling ``wrapper.to``
    # with ``dtype`` here would silently cast codec parameters (and any
    # trainable channel parameters) to BF16.  ``build_model`` is also used by
    # evaluation, where preserving the checkpoint's FP32 codec values avoids
    # a load-time round trip through BF16.
    wrapper.to(device=model_config["device"])
    return processor, wrapper
