"""Frozen T5Gemma-2 with codec -> channel -> codec inserted at one split site.

Sites: ``enc_emb`` (input of encoder layer 0, after image scatter),
``enc_l{k}`` (output of encoder layer k), ``enc_fn`` (after encoder final norm),
``dec_l{k}`` (output of decoder layer k). At a decoder site the transmitter
layers 0..k use the clean encoder memory; the receiver layers k+1.. use encoder
memory that is itself sent once per forward through the memory codec.
"""
import re
from contextlib import contextmanager, nullcontext

import torch
from torch import nn

from .channel import build_channel, normalize_power, token_mask
from .codec import Codec


def encoder_stack(base):
    encoder = base.get_encoder()
    return getattr(encoder, "text_model", encoder)


def decoder_stack(base):
    return base.get_decoder()


def parse_site(site: str):
    if site in ("enc_emb", "enc_fn"):
        return "enc", site[4:], None
    match = re.fullmatch(r"(enc|dec)_l(\d+)", site)
    if match is None:
        raise ValueError(f"unknown site {site!r}")
    return match[1], "layer", int(match[2])


class SplitModel(nn.Module):
    def __init__(self, base, codec: Codec, channel: nn.Module, channel_config: dict, *,
                 memory_codec: Codec | None = None, sdpa_flash_math: bool = False):
        super().__init__()
        self.base = base.requires_grad_(False).eval()
        self.codec = codec
        # Receiver-memory codec: the same module (shared) unless a separate one is given.
        self.memory_codec = memory_codec if memory_codec is not None else codec
        self.channel = channel
        self.normalize = channel_config.get("normalize_power", True) and not getattr(channel, "normalizes_power", False)
        self.sdpa_flash_math = sdpa_flash_math
        self.site = None
        self._handles = []
        self.snr_db = None
        self.bypass = False
        self._encoder_mask = None
        self._decoder_mask = None
        self._captured_encoder_mask = None
        self.activation = self.reconstruction = None
        self.memory_activation = self.memory_reconstruction = None
        self.uses = {"hidden": 0, "memory": 0}
        # Evaluators may call ``base`` directly; capture the encoder mask from the first layer.
        encoder_stack(base).layers[0].register_forward_pre_hook(self._capture_encoder_mask, with_kwargs=True)

    # ---- site routing -------------------------------------------------
    def set_site(self, site: str):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        stack_name, where, index = parse_site(site)
        if stack_name == "enc":
            stack = encoder_stack(self.base)
            if where == "emb":
                self._handles.append(stack.layers[0].register_forward_pre_hook(self._pre_hook, with_kwargs=True))
            elif where == "fn":
                self._handles.append(stack.norm.register_forward_hook(self._hook))
            else:
                assert index is not None
                if not 0 <= index < len(stack.layers):
                    raise ValueError(f"{site} is outside the encoder")
                self._handles.append(stack.layers[index].register_forward_hook(self._hook))
        else:
            stack = decoder_stack(self.base)
            assert index is not None
            if not 0 <= index < len(stack.layers) - 1:
                raise ValueError(f"{site} needs at least one receiver layer after it")
            self._handles.append(stack.layers[index].register_forward_hook(self._hook))
            self._handles.append(stack.register_forward_pre_hook(self._begin_decoder, with_kwargs=True))
            for layer in stack.layers[index + 1:]:
                self._handles.append(layer.self_attn.register_forward_pre_hook(self._receiver_memory, with_kwargs=True))
        self.site = site
        return self

    @property
    def is_decoder_site(self):
        return self.site is not None and self.site.startswith("dec_")

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()  # frozen backbone never runs dropout
        return self

    # ---- transmission state ------------------------------------------
    @contextmanager
    def transmission(self, snr_db=None, *, bypass=False, encoder_mask=None, decoder_mask=None):
        """Set the channel condition for the enclosed forwards.

        ``bypass=True`` skips codec and channel entirely (the teacher / Vanilla).
        ``snr_db=None`` keeps codec and power normalization but adds no noise.
        """
        previous = (self.snr_db, self.bypass, self._encoder_mask, self._decoder_mask)
        self.snr_db, self.bypass = snr_db, bypass
        self._encoder_mask, self._decoder_mask = encoder_mask, decoder_mask
        self.uses = {"hidden": 0, "memory": 0}
        self.activation = self.reconstruction = None
        self.memory_activation = self.memory_reconstruction = None
        try:
            yield self
        finally:
            self.snr_db, self.bypass, self._encoder_mask, self._decoder_mask = previous

    def attention_context(self):
        if not self.sdpa_flash_math:
            return nullcontext()
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH])

    def _capture_encoder_mask(self, module, args, kwargs):
        mask = kwargs.get("attention_mask")
        if mask is None and len(args) > 2:
            mask = args[2]  # T5Gemma2 text layers: (hidden, position_embeddings, attention_mask, ...)
        self._captured_encoder_mask = mask

    def _source_mask(self, reference):
        mask = self._encoder_mask if self._encoder_mask is not None else self._captured_encoder_mask
        return token_mask(mask, reference)

    # ---- the communication path --------------------------------------
    def transmit(self, hidden, codec, stream, mask=None, *, token_wise=False):
        dtype = next(self.base.parameters()).dtype
        with torch.autocast(device_type=hidden.device.type, dtype=dtype,
                            enabled=dtype in (torch.bfloat16, torch.float16)):
            z = codec.encode(hidden)
            self.uses[stream] += int(z.numel() if mask is None else mask.sum().item() * z.shape[-1])
            if self.normalize:
                z = normalize_power(z, mask, token_wise=token_wise)
            received = self.channel(z, self.snr_db)
            return codec.decode(received, self.snr_db).to(dtype=hidden.dtype)

    def _roundtrip(self, hidden):
        self.activation = hidden.detach()
        if self.bypass:
            self.reconstruction = None
            return hidden
        if self.is_decoder_site:
            mask = token_mask(self._decoder_mask, hidden) if self._decoder_mask is not None else None
            self.reconstruction = self.transmit(hidden, self.codec, "hidden", mask, token_wise=True)
        else:
            self.reconstruction = self.transmit(hidden, self.codec, "hidden", self._source_mask(hidden))
        return self.reconstruction

    def _hook(self, module, inputs, output):
        if isinstance(output, tuple):
            return (self._roundtrip(output[0]), *output[1:])
        return self._roundtrip(output)

    def _pre_hook(self, module, args, kwargs):
        if args:
            return (self._roundtrip(args[0]), *args[1:]), kwargs
        return args, {**kwargs, "hidden_states": self._roundtrip(kwargs["hidden_states"])}

    def _begin_decoder(self, module, args, kwargs):
        self.memory_activation = self.memory_reconstruction = None

    def _receiver_memory(self, module, args, kwargs):
        if self.bypass:
            return None
        cache = kwargs.get("past_key_values")
        if cache is not None and cache.is_updated.get(module.layer_idx, False):
            return None  # cross-attention K/V were already built from the transmitted memory
        if self.memory_reconstruction is None:
            memory = kwargs["encoder_hidden_states"]
            self.memory_activation = memory.detach()
            self.memory_reconstruction = self.transmit(memory, self.memory_codec, "memory", self._source_mask(memory))
        return args, {**kwargs, "encoder_hidden_states": self.memory_reconstruction}

    # ---- forwards -------------------------------------------------------
    def forward(self, **kwargs):
        with self.attention_context():
            return self.base(**kwargs)

    def generate(self, **kwargs):
        if self.is_decoder_site and (kwargs.get("num_beams") or 1) != 1:
            raise ValueError("receiver-memory transmission supports greedy decoding only")
        with self.attention_context():
            return self.base.generate(**kwargs)

    def communication_state(self):
        state = {"codec": self.codec.state_dict(), "channel": self.channel.state_dict()}
        if self.memory_codec is not self.codec:
            state["memory_codec"] = self.memory_codec.state_dict()
        return state

    def load_communication_state(self, state):
        self.codec.load_state_dict(state["codec"])
        self.channel.load_state_dict(state.get("channel", {}))
        if self.memory_codec is not self.codec:
            self.memory_codec.load_state_dict(state["memory_codec"])


def build_model(config, *, local_files_only=False):
    """Return ``(processor, SplitModel)``; the site is set by the caller."""
    from transformers import AutoModelForSeq2SeqLM, AutoProcessor, AutoTokenizer
    m = config["model"]
    kwargs = {"revision": m["revision"], "local_files_only": local_files_only}
    processor_cls = AutoProcessor if config["task"] == "coco" else AutoTokenizer
    processor = processor_cls.from_pretrained(m["name"], **kwargs)
    model_kwargs = dict(kwargs)
    if "attn_implementation" in m:
        model_kwargs["attn_implementation"] = m["attn_implementation"]
    base = AutoModelForSeq2SeqLM.from_pretrained(m["name"], dtype=getattr(torch, m["dtype"]),
                                                 low_cpu_mem_usage=True, **model_kwargs)
    model = wrap(base, config)
    model.to(device=m["device"])  # codec/channel stay FP32; backbone keeps its dtype
    return processor, model


def wrap(base, config):
    width = int(base.config.decoder.hidden_size)
    codec = Codec(width, config["codec"])
    memory = Codec(width, config["codec"]) if config["codec"].get("memory") == "separate" else None
    return SplitModel(base, codec, build_channel(config["channel"]), config["channel"],
                      memory_codec=memory, sdpa_flash_math=config["model"].get("sdpa_flash_math", False))
