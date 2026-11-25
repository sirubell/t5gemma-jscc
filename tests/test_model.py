from contextlib import nullcontext
from typing import Any

import pytest
import torch

from jscc.channel import normalize_power, power_scale
from jscc.codec import SiteNorm
from jscc.model import encoder_stack, wrap
from tests.helpers import CHANNEL, CODEC, image_batch, site_stats_file, text_batch, tiny_backbone, tiny_model

SITES = ["enc_emb", "enc_l0", "enc_l1", "enc_l2", "enc_fn", "dec_l0", "dec_l1"]


def logits(model, batch, **kw):
    with model.transmission(encoder_mask=batch["attention_mask"], decoder_mask=batch["labels"] != -100, **kw):
        return model(**batch).logits, model.reconstruction, model.memory_reconstruction


@pytest.mark.parametrize("site", SITES)
def test_every_site_transmits_and_bypass_is_vanilla(site):
    model = tiny_model()
    batch = text_batch()
    vanilla = model.base(**batch).logits  # no site installed yet
    model.set_site(site)
    bypass, rec, _ = logits(model, batch, bypass=True)
    assert rec is None and torch.equal(bypass, vanilla)
    coded, rec, mem = logits(model, batch, snr_db=None)
    assert rec is not None and not torch.allclose(coded, vanilla)
    assert (mem is not None) == site.startswith("dec_")


def test_decoder_site_needs_receiver_layer():
    with pytest.raises(ValueError):
        tiny_model().set_site("dec_l2")


def test_noise_changes_output_and_none_is_deterministic():
    model = tiny_model().set_site("enc_l1")
    batch = text_batch()
    a, _, _ = logits(model, batch, snr_db=None)
    b, _, _ = logits(model, batch, snr_db=None)
    c, _, _ = logits(model, batch, snr_db=torch.full((2, 1, 1), -6.0))
    assert torch.equal(a, b) and not torch.allclose(a, c)


def test_power_normalization_ignores_padding():
    z = torch.randn(2, 5, 4)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    out = normalize_power(z, mask)
    assert torch.allclose(out[0, :3].pow(2).mean(), torch.tensor(1.0), atol=1e-4)
    assert torch.allclose(out[1].pow(2).mean(), torch.tensor(1.0), atol=1e-4)


class Quantized(torch.nn.Module):
    normalizes_power = True

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def forward(self, z, snr_db):
        self.aux_loss = z.pow(2).mean()
        return z * self.scale


def test_custom_channel_owns_power_and_aux_loss():
    model = tiny_model(channel={"type": "tests.test_model:Quantized", "normalize_power": True}).set_site("enc_l1")
    assert not model.normalize
    out, _, _ = logits(model, text_batch(), snr_db=None)
    out.sum().backward()
    assert model.channel.scale.grad is not None and model.channel.aux_loss.ndim == 0


def test_qam_example_trains_through_the_split_model():
    model = tiny_model(channel={"type": "examples.qam_channel:LearnedQAMChannel", "normalize_power": True}).set_site("dec_l0")
    batch = text_batch()
    out, _, _ = logits(model, batch, snr_db=3.0)
    (out.sum() + model.channel.aux_loss).backward()
    channel: Any = model.channel
    assert channel.mapper[0].weight.grad is not None and model.codec.encoder[0].weight.grad is not None


PREFIX_SITES = ["enc_emb", "enc_l0", "enc_l1", "enc_l3", "enc_fn", "dec_l0", "dec_l1", "dec_l2"]


def _student(model, inputs, masks, autocast, prefix):
    """Logits, nMSE targets and codec gradients of one noisy student pass (full forward if ``prefix`` is None)."""
    for p in model.codec.parameters():
        p.grad = None
    torch.manual_seed(123)
    snr = torch.full((inputs["attention_mask"].shape[0], 1, 1), 3.0)
    with autocast, model.transmission(snr, encoder_mask=masks[0], decoder_mask=masks[1]):
        with model.from_prefix(inputs, prefix) if prefix is not None else nullcontext(inputs) as kwargs:
            out = model(**kwargs).logits
        loss = out.float().square().mean() + model.reconstruction.float().square().mean()
        if model.memory_reconstruction is not None:
            loss = loss + model.memory_reconstruction.float().square().mean()
    loss.backward()
    return [out.detach(), model.activation, model.memory_activation, *[p.grad.clone() for p in model.codec.parameters()]]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("images", [False, True])
def test_student_from_teacher_prefix_is_exact(images, dtype):
    """Guard for the frozen-prefix skip (pinned transformers internals): starting the student at the split
    from the teacher's states gives the same logits, nMSE targets and codec gradients as the full forward."""
    model = tiny_model(layers=4)
    model.base.to(dtype)
    model.train()
    batch = image_batch() if images else text_batch(batch=3, src=7, tgt=5)
    labels = batch["labels"]
    inputs: dict[str, Any] = {k: v for k, v in batch.items() if k != "labels"}
    inputs.update(decoder_input_ids=model.base.prepare_decoder_input_ids_from_labels(labels=labels), use_cache=False)
    if images:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    masks = (inputs["attention_mask"], labels != -100)
    autocast = torch.autocast("cpu", dtype=dtype, enabled=dtype != torch.float32)
    vision = []
    model.base.get_encoder().vision_tower.register_forward_hook(lambda *a: vision.append(1))
    with torch.no_grad(), autocast, model.transmission(bypass=True, encoder_mask=masks[0], decoder_mask=masks[1]), \
            model.capture_prefix(PREFIX_SITES) as prefix:
        model(**inputs)
    assert sorted(prefix, key=str) == [-1, 0, 1, 3, "memory"]  # enc_fn starts from the last layer's output
    for site in PREFIX_SITES:
        model.set_site(site)
        ref = _student(model, inputs, masks, autocast, None)
        vision.clear()
        new = _student(model, inputs, masks, autocast, prefix)
        assert not vision
        assert all(a is b is None or (a is not None and b is not None and torch.equal(a, b)) for a, b in zip(ref, new)), site
        assert not any("forward" in vars(layer) for layer in encoder_stack(model.base).layers)


@pytest.mark.parametrize("mode", ["fixed", "learned"])
def test_site_norm_standardizes_each_stream(tmp_path, mode):
    g = torch.Generator().manual_seed(0)
    stats = {s: {"mean": torch.randn(16, generator=g), "std": torch.rand(16, generator=g) + 0.5}
             for s in ("enc_l1", "dec_l0", "memory", "enc_fn")}
    torch.save({"stats": stats}, tmp_path / "stats.pt")
    config = {"codec": {**CODEC, "site_norm": mode, "site_stats": str(tmp_path / "stats.pt")}, "channel": dict(CHANNEL),
              "model": {}, "training": {"sites": ["enc_l1", "dec_l0"]}}
    model = wrap(tiny_backbone(), config)
    assert model.site_norm is not None and model.site_norm.streams == ["enc_l1", "dec_l0", "memory"]
    trainable = sum(p.numel() for p in model.site_norm.parameters() if p.requires_grad)
    assert trainable == (4 * 3 * 16 if mode == "learned" else 0)
    seen: list[Any] = []
    model.codec.encoder.register_forward_hook(lambda m, args, out: seen.append(("in", args[0])))
    model.codec.output_norm.register_forward_hook(lambda m, args, out: seen.append(("out", out)))  # decoder output
    batch = text_batch()
    for site, streams in (("enc_l1", ["enc_l1"]), ("dec_l0", ["dec_l0", "memory"])):
        seen.clear()
        model.set_site(site)
        _, rec, mem = logits(model, batch, snr_db=None)
        inputs = [x for kind, x in seen if kind == "in"]
        outputs = [x for kind, x in seen if kind == "out"]
        assert len(inputs) == len(outputs) == len(streams)
        sources = {"enc_l1": model.activation, "dec_l0": model.activation, "memory": model.memory_activation}
        received = {"enc_l1": rec, "dec_l0": rec, "memory": mem}
        for stream in streams:  # matched by content: the streams are coded in forward order, not list order
            s, h, r = stats[stream], sources[stream], received[stream]
            assert h is not None and r is not None
            matches = [j for j, x in enumerate(inputs) if x.shape == h.shape and torch.allclose(
                x.float(), (h.float() - s["mean"]) / s["std"], atol=1e-5)]  # identity affine at init
            assert matches, (site, stream)
            assert torch.allclose(r.float(), outputs[matches[0]].float() * s["std"] + s["mean"], atol=1e-4)
    if mode == "learned":  # the per-site parameters get gradients only for the streams that were coded
        assert rec is not None and mem is not None
        (rec.float().sum() + mem.float().sum()).backward()
        grad = model.site_norm.affine["b_out"].grad
        assert grad is not None and grad[1:].abs().sum() > 0 and grad[0].abs().sum() == 0
    state = model.communication_state()
    assert {"mean", "std"} <= set(state["site_norm"])
    other = wrap(tiny_backbone(), config)
    other.load_communication_state(state)
    assert other.site_norm is not None
    assert all(torch.equal(a, b) for a, b in zip(other.site_norm.state_dict().values(), state["site_norm"].values()))


def _coded(model, site, snr_db=None):
    """Source hidden state, the first stream's encoder output / channel output / decoder input, and the
    reconstruction of one forward at ``site`` (noise seeded)."""
    model.set_site(site)
    seen: dict[str, Any] = {}

    def keep(key, pick):
        def hook(*args):  # returns None, so the hooked value is not replaced
            seen.setdefault(key, pick(args))
        return hook
    hooks = [model.codec.encoder.register_forward_hook(keep("z", lambda a: a[2])),
             model.channel.register_forward_hook(keep("channel", lambda a: a[2])),
             model.codec.decoder[0].register_forward_pre_hook(keep("received", lambda a: a[1][0]))]
    torch.manual_seed(5)
    rec = logits(model, text_batch(), snr_db=snr_db)[1]
    for hook in hooks:
        hook.remove()
    return model.activation, seen, rec


def test_default_power_path_is_unchanged():
    """Default: the channel gets z / sqrt(mean power over valid tokens + 1e-8) and the receiver decodes its output as is."""
    model = tiny_model()
    assert not model.restore_power
    _, seen, _ = _coded(model, "enc_l1", -6.0)
    mask = text_batch()["attention_mask"].unsqueeze(-1)
    z = seen["z"].float()
    power = (z.pow(2) * mask).sum(dim=(1, 2), keepdim=True) / (mask.sum(dim=(1, 2), keepdim=True) * z.shape[-1])
    sent = z / torch.sqrt(power + 1e-8)
    torch.manual_seed(5)
    assert torch.allclose(seen["channel"], sent + torch.randn_like(sent) * 10.0 ** (6.0 / 20.0), atol=1e-6)
    assert torch.equal(seen["received"], seen["channel"])


@pytest.mark.parametrize("site", ["enc_l1", "dec_l0"])
def test_restore_power_gives_the_receiver_the_transmit_scale(site):
    model = tiny_model(channel={**CHANNEL, "restore_power": True})
    decoder_site = site.startswith("dec_")
    _, seen, _ = _coded(model, site, -6.0)
    mask = None if decoder_site else text_batch()["attention_mask"].bool()
    _, scale = power_scale(seen["z"], mask, token_wise=decoder_site)
    assert torch.equal(seen["received"], seen["channel"] * scale)  # noise added at unit power, then rescaled
    h, _, rec = _coded(model, site)
    streams = [(h, rec)] + ([(model.memory_activation, model.memory_reconstruction)] if decoder_site else [])
    for source, out in streams:  # noise-free: the same as the codec without power normalization
        assert source is not None and out is not None
        with torch.no_grad():
            assert torch.allclose(out, model.codec.decode(model.codec.encode(source)), atol=1e-5)


@pytest.mark.parametrize("channel", [{"type": "tests.test_model:Quantized"}, {"type": "awgn", "normalize_power": False}])
def test_restore_power_needs_caller_power_normalization(channel):
    with pytest.raises(ValueError, match="restore_power"):
        tiny_model(channel={**channel, "restore_power": True})


@pytest.mark.parametrize("learned", [False, True])
def test_site_norm_center_and_scale_switches(tmp_path, learned):
    path, stats = site_stats_file(tmp_path / "stats.pt")
    s, h = stats["enc_l1"], torch.randn(2, 3, 16)
    expected = {(True, True): ((h - s["mean"]) / s["std"], h * s["std"] + s["mean"]),
                (True, False): (h - s["mean"], h + s["mean"]),
                (False, True): (h / s["std"], h * s["std"])}
    for (center, scale), (into, out) in expected.items():  # learned: identity affine at init
        norm = SiteNorm(stats, ["enc_l1"], learned=learned, center=center, scale=scale)
        assert torch.equal(norm.normalize(h, "enc_l1"), into) and torch.equal(norm.denormalize(h, "enc_l1"), out)
        receiver = SiteNorm(stats, ["enc_l1"], learned=learned, sides="output", center=center, scale=scale)
        assert torch.equal(receiver.normalize(h, "enc_l1"), h) and torch.equal(receiver.denormalize(h, "enc_l1"), out)
    with pytest.raises(ValueError, match="site_norm_center"):
        SiteNorm(stats, ["enc_l1"], learned=learned, center=False, scale=False)
    codec = {**CODEC, "site_norm": "learned" if learned else "fixed", "site_stats": path, "site_norm_scale": False}
    model = wrap(tiny_backbone(), {"codec": codec, "channel": dict(CHANNEL), "model": {}, "training": {"sites": ["enc_l1"]}})
    assert model.site_norm is not None and model.site_norm.center and not model.site_norm.scale
    inputs: list[Any] = []
    model.codec.encoder.register_forward_pre_hook(lambda m, a: inputs.append(a[0]))
    h, _, _ = _coded(model, "enc_l1")
    assert h is not None and torch.equal(inputs[0], h.float() - s["mean"])
