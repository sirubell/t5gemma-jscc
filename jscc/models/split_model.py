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

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()  # Freezing weights alone does not disable backbone dropout.
        return self

    @contextmanager
    def transmission(self, snr_db=None, *, bypass=False):
        previous = self.snr_db, self.bypass
        self.snr_db, self.bypass = snr_db, bypass
        try:
            yield
        finally:
            self.snr_db, self.bypass = previous

    def _roundtrip(self, hidden):
        self.activation = hidden.detach()
        if self.bypass:
            self.reconstruction = None
            return hidden
        z = self.codec.encode(hidden)
        if self.channel_config["normalize_power"]:
            z = normalize_power(z)
        received = self.channel(z, self.snr_db)
        film_snr = self.snr_db
        if film_snr is None:
            film_snr = self.channel_config["clean_film_snr"]
        reconstructed = self.codec.decode(received, film_snr)
        self.reconstruction = reconstructed
        return reconstructed

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
