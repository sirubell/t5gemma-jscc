"""Insert codec -> channel -> codec at one frozen encoder/decoder layer."""
from contextlib import contextmanager

import torch
from torch import nn

from .channel import build_channel, normalize_power
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
        stack = stack_module(base, split["stack"])
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
                self.memory_codec = Codec(codec.encoder[0].weight.shape[1], codec.config)
                stack.register_forward_pre_hook(self._begin_decoder, with_kwargs=True)
                for layer in stack.layers[first_receiver:]:
                    layer.self_attn.register_forward_pre_hook(self._receiver_memory, with_kwargs=True)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()  # Freezing weights alone does not disable backbone dropout.
        return self

    @contextmanager
    def transmission(self, snr_db=None, *, bypass=False):
        previous = self.snr_db, self.bypass
        self.snr_db, self.bypass = snr_db, bypass
        self.channel_uses = {"hidden": 0, "memory": 0}
        try:
            yield
        finally:
            self.snr_db, self.bypass = previous

    def _transmit(self, hidden, codec, stream):
        z = codec.encode(hidden)
        self.channel_uses[stream] += z.numel()
        if self.channel_config["normalize_power"]:
            z = normalize_power(z)
        received = self.channel(z, self.snr_db)
        film_snr = self.snr_db
        if film_snr is None:
            film_snr = self.channel_config["clean_film_snr"]
        return codec.decode(received, film_snr)

    def _roundtrip(self, hidden):
        self.activation = hidden.detach()
        self.reconstruction = None if self.bypass else self._transmit(hidden, self.codec, "hidden")
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
            self.memory_reconstruction = self._transmit(memory, self.memory_codec, "memory")
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

    def forward(self, **kwargs):
        return self.base(**kwargs)

    def generate(self, **kwargs):
        if self.memory_codec is not None:
            generation_config = kwargs.get("generation_config") or self.base.generation_config
            beams = kwargs.get("num_beams", generation_config.num_beams) or 1
            if beams != 1:
                raise ValueError("Receiver-memory transmission currently supports num_beams=1 only")
        return self.base.generate(**kwargs)


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
    wrapper.to(device=model_config["device"], dtype=dtype)
    return processor, wrapper
