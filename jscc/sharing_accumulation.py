"""Explicit effective64 partitions preserving every parent padding coordinate."""
from copy import deepcopy
from dataclasses import replace

import torch

from .activation_replay import canonical_digest
from .baseline_protocol import _check_batches, batch_identity


def partition_policy(microbatch_size):
    if type(microbatch_size) is not int or microbatch_size not in (16, 32):
        raise ValueError('explicit microbatch_size must be 32 or 16')
    return {'microbatch_size': microbatch_size, 'gradient_accumulation': 64 // microbatch_size,
            'padding': 'preserve-parent-v1', 'noise': 'sharing-parent-micro-v1'}


def partition_native64(batch, *, microbatch_size):
    """Clone complete sequence rows; never trim source/decoder/role widths."""
    partition_policy(microbatch_size)
    _check_batches([batch], 'combined', 64)
    if any(key in batch for key in ('pixel_values', 'activation')):
        raise ValueError('sharing partitions require online text inputs')
    source_shape, target_shape = batch['attention_mask'].shape, batch['labels'].shape
    for key, value in batch.items():
        if torch.is_tensor(value) and (value.ndim == 0 or value.shape[0] != 64):
            raise ValueError('sequence tensor must have 64 rows: ' + key)
        if isinstance(value, (list, tuple)) and len(value) != 64:
            raise ValueError('sequence metadata must have 64 rows: ' + key)
    for key in ('decoder_input_ids', 'decoder_attention_mask'):
        if key in batch and (not torch.is_tensor(batch[key]) or batch[key].shape != target_shape):
            raise ValueError('decoder layout differs from labels: ' + key)
    if 'token_positions' in batch and batch['token_positions'].shape != source_shape:
        raise ValueError('token positions differ from source layout')
    for key in ('token_roles', 'position_roles'):
        if key in batch and any(not isinstance(row, (list, tuple)) or len(row) != source_shape[1]
                                for row in batch[key]):
            raise ValueError('token roles differ from source layout: ' + key)
    for mask in (batch['attention_mask'], batch['labels'] != -100):
        expected = torch.arange(mask.shape[1], device=mask.device)[None] < mask.bool().sum(1)[:, None]
        if not torch.equal(mask.bool(), expected):
            raise ValueError('partition requires contiguous right padding')
    if 'decoder_attention_mask' in batch:
        mask = batch['decoder_attention_mask']
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError('decoder mask must be binary')
        if not torch.equal(mask.bool(), batch['labels'] != -100):
            raise ValueError('decoder mask differs from valid target positions')
    return [{key: value[start:start + microbatch_size].clone() if torch.is_tensor(value)
             else deepcopy(value[start:start + microbatch_size]) if isinstance(value, (list, tuple))
             else deepcopy(value) for key, value in batch.items()}
            for start in range(0, 64, microbatch_size)]


def micro_noise_key(parent_key, batch, *, micro, microbatch_size):
    """Keep scheduled parent/site/local identity and bind exact physical partition."""
    policy = partition_policy(microbatch_size)
    if type(micro) is not int or not 0 <= micro < policy['gradient_accumulation']:
        raise ValueError('micro index outside explicit partition')
    if len(batch['labels']) != microbatch_size:
        raise ValueError('micro size differs from explicit partition')
    return replace(parent_key, microbatch_layout=canonical_digest({
        'policy': policy, 'parent_layout': parent_key.microbatch_layout,
        'micro': micro, 'shape': list(batch['attention_mask'].shape),
        'view': batch_identity(batch)}))
