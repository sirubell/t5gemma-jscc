"""Tiny real T5Gemma-2 for CPU tests."""
import torch
from transformers import T5Gemma2Config, T5Gemma2ForConditionalGeneration

from jscc.model import wrap

CODEC = {"architecture": "linear", "bottleneck_dim": 8}
CHANNEL = {"type": "awgn", "normalize_power": True}


def tiny_backbone(layers=3):
    torch.manual_seed(0)
    text = {"vocab_size": 32, "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": layers,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8, "query_pre_attn_scalar": 8,
            "max_position_embeddings": 64, "layer_types": ["full_attention"] * layers}
    vision = {"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
              "num_attention_heads": 2, "image_size": 8, "patch_size": 4}
    config = T5Gemma2Config(encoder={"text_config": text, "vision_config": vision, "mm_tokens_per_image": 4,
                                     "boi_token_index": 29, "eoi_token_index": 30},
                            decoder=text, image_token_index=31)
    return T5Gemma2ForConditionalGeneration(config).eval()


def tiny_model(codec=CODEC, channel=CHANNEL, layers=3):
    config = {"codec": dict(codec), "channel": dict(channel), "model": {}}
    return wrap(tiny_backbone(layers), config)


def text_batch(batch=2, src=6, tgt=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(1, 28, (batch, src), generator=g)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[0, -2:] = 0
    labels = torch.randint(1, 28, (batch, tgt), generator=g)
    labels[1, -1] = -100
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def image_batch(batch=2, images=2, text=3, tgt=4, seed=1):
    """Batch with ``images`` images per item: [BOI, 4 image tokens, EOI] per image, then text; last item padded."""
    g = torch.Generator().manual_seed(seed)
    rows = [sum(([29, 31, 31, 31, 31, 30] for _ in range(images)), []) + torch.randint(1, 28, (text,), generator=g).tolist()
            for _ in range(batch)]
    input_ids = torch.tensor([r + [0, 0] for r in rows])
    attention_mask = torch.ones_like(input_ids)
    attention_mask[:-1, -2:] = 0
    input_ids[:-1, -2:] = 0
    input_ids[-1, -2:] = torch.randint(1, 28, (2,), generator=g)
    labels = torch.randint(1, 28, (batch, tgt), generator=g)
    labels[-1, -1] = -100
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
            "pixel_values": torch.randn(batch * images, 3, 8, 8, generator=g)}
