"""Small real Transformers model shared by both task test suites."""
import torch
from transformers import T5Gemma2Config, T5Gemma2ForConditionalGeneration


def tiny_backbone(num_hidden_layers=2):
    text = {"vocab_size": 32, "hidden_size": 16, "intermediate_size": 32,
            "num_hidden_layers": num_hidden_layers, "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 8, "query_pre_attn_scalar": 8, "max_position_embeddings": 64,
            "layer_types": ["full_attention"] * num_hidden_layers}
    vision = {"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
              "num_attention_heads": 2, "image_size": 8, "patch_size": 4}
    config = T5Gemma2Config(
        encoder={"text_config": text, "vision_config": vision, "mm_tokens_per_image": 4,
                 "boi_token_index": 29, "eoi_token_index": 30},
        decoder=text, image_token_index=31,
    )
    model = T5Gemma2ForConditionalGeneration(config)
    # HF initializes this projector to zero. A zero projector erases all pixel
    # differences and makes image-route tests meaningless; pretrained weights
    # replace it when loading the real model. Use an identity in this fixture.
    with torch.no_grad():
        model.get_encoder().multi_modal_projector.mm_input_projection_weight.copy_(torch.eye(16))
    return model
