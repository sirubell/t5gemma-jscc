"""COCO image routes on tiny real T5Gemma-2; CPU only, no downloads."""
import pytest
import torch

from jscc.data.coco import validate_ids
from jscc.models.channel import AWGNChannel
from jscc.models.codec import Codec
from jscc.models.split_model import SplitModel
from jscc.training import batch_losses
from model_helpers import tiny_backbone


class RecordingAWGN(AWGNChannel):
    def forward(self, z, snr_db):
        self.sent = z.detach().clone()
        received = super().forward(z, snr_db)
        self.received = received.detach().clone()
        return received


def coco_model(stack="enc", where="after_embed"):
    codec = Codec(16, {"hidden_dim": 16, "bottleneck_dim": 8, "n_res_blocks": 1,
                       "activation": "gelu", "dropout": 0.0, "layernorm": "none",
                       "snr_film": True, "film_hidden": 4})
    return SplitModel(tiny_backbone(), codec, RecordingAWGN(),
                      {"stack": stack, "where": where, "index": 0},
                      {"normalize_power": True, "clean_film_snr": 18.0})


def image_batch():
    # The tiny vision encoder produces four patches for each real pixel tensor.
    return {"input_ids": torch.tensor([[2, 31, 31, 31, 31, 4]]),
            "attention_mask": torch.ones(1, 6, dtype=torch.long),
            "pixel_values": torch.randn(1, 3, 8, 8)}


@torch.no_grad()
def assert_image_route(model, batch):
    """Detect the old embedding hook: image scatter overwrites its reconstruction."""
    observed = []

    def capture(module, args, kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]
        observed.append(hidden.detach().clone())

    observer = model.base.get_encoder().text_model.layers[0].register_forward_pre_hook(
        capture, with_kwargs=True)
    try:
        with model.transmission(None):
            model.base.get_encoder()(**batch)
        clean_images = observed[-1][:, 1:5].clone()
        with model.transmission(0.0):
            model.base.get_encoder()(**batch)
        # The first text layer must receive the codec result at image positions,
        # not fresh clean vision features scattered over that result.
        torch.testing.assert_close(observed[-1][:, 1:5], model.reconstruction[:, 1:5])
        assert not torch.equal(model.channel.sent[:, 1:5], model.channel.received[:, 1:5])
        assert not torch.equal(clean_images, observed[-1][:, 1:5])
        encoded_images = model.channel.sent[:, 1:5].clone()
        changed = {**batch, "pixel_values": torch.randn_like(batch["pixel_values"])}
        with model.transmission(0.0):
            model.base.get_encoder()(**changed)
        assert not torch.equal(encoded_images, model.channel.sent[:, 1:5]), (
            "Real image pixels must influence the image positions transmitted through the channel")
    finally:
        observer.remove()


@pytest.mark.parametrize("where", ["after_embed", "before_first_layer"])
def test_image_features_reach_channel_after_vision_scatter(where):
    torch.manual_seed(7)
    assert_image_route(coco_model(where=where).eval(), image_batch())


@torch.no_grad()
def test_decoder_receiver_memory_crosses_channel_but_sender_stays_clean():
    torch.manual_seed(7)
    model = coco_model("dec", "after_layer").eval()
    batch = {**image_batch(), "decoder_input_ids": torch.tensor([[0, 5]]), "use_cache": False}
    memories = []
    receiver, sender = [], []
    def record(destination):
        def hook(module, args, kwargs):
            destination.append(kwargs["encoder_hidden_states"].detach().clone())
        return hook
    sender_handle = model.base.get_decoder().layers[0].self_attn.register_forward_pre_hook(
        record(sender), with_kwargs=True)
    receiver_handle = model.base.get_decoder().layers[1].self_attn.register_forward_pre_hook(
        record(receiver), with_kwargs=True)
    handle = model.base.get_encoder().register_forward_hook(
        lambda module, args, output: memories.append(output.last_hidden_state.detach().clone()))
    try:
        with model.transmission(None):
            clean = model(**batch).logits
        with model.transmission(0.0):
            noisy = model(**batch).logits
        torch.testing.assert_close(memories[0], memories[1], rtol=0, atol=0)
        torch.testing.assert_close(sender[1], memories[1], rtol=0, atol=0)
        torch.testing.assert_close(receiver[1], model.memory_reconstruction, rtol=0, atol=0)
        assert not torch.equal(receiver[1], memories[1])
        assert not torch.equal(receiver[0], receiver[1])
        assert model.channel_uses == {"hidden": 2 * 8, "memory": 6 * 8}
        assert not torch.equal(clean, noisy)
    finally:
        handle.remove()
        sender_handle.remove()
        receiver_handle.remove()


@pytest.mark.parametrize("stack,where", [("enc", "after_embed"), ("enc", "after_layer"),
                                         ("enc", "after_final_norm"), ("dec", "after_layer")])
def test_coco_forward_backward_and_generation(stack, where):
    torch.manual_seed(7)
    model = coco_model(stack, where)
    batch = {"input_ids": torch.tensor([[2, 31, 31, 31, 31, 4], [2, 31, 31, 31, 31, 5]]),
             "attention_mask": torch.ones(2, 6, dtype=torch.long),
             "pixel_values": torch.randn(2, 1, 3, 8, 8),
             "labels": torch.tensor([[5, 6, 1], [7, 1, -100]])}
    loss = batch_losses(model, batch, {"temperature": 1.0, "kl_weight": 1.0, "mse_weight": 0.1}, 0.0)
    loss["loss"].backward()
    grad = model.codec.encoder[0].get_parameter("weight").grad
    assert grad is not None and grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.base.parameters())
    with torch.no_grad(), model.transmission(None):
        ids = model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                             pixel_values=batch["pixel_values"].flatten(0, 1),
                             max_new_tokens=3, decoder_start_token_id=0)
    assert ids.shape[0] == 2


def test_coco_split_excludes_train_and_demos():
    ids = {"train_ids": [1, 2], "demo_ids": [3], "selection_ids": [4], "report_ids": [5]}
    validate_ids(ids, {4, 5}, {6})
    with pytest.raises(ValueError, match="disjoint"):
        validate_ids({**ids, "report_ids": [4]}, {4, 5}, {6})
    with pytest.raises(ValueError, match="overlap"):
        validate_ids({**ids, "train_ids": [6]}, {4, 5}, {6})
