"""Channel input/output and transmit-power policies.

The caller normalizes transmit power per sample. ``snr_db=None`` means no
noise.  Custom Python channels implement the same ``forward(z, snr_db)``
method.

``normalize_power`` keeps its original whole-sample behavior by default.  A
``valid_mask`` excludes padded token positions from the sequence-wide power
estimate, while ``token_wise=True`` gives each token its own power estimate.
The latter is the causal policy used for decoder hidden streams: a later
token cannot change an earlier token's normalization factor.
"""
import importlib

import torch
from torch import nn


def _token_mask(z, valid_mask, *, representation=None):
    """Convert a token-valid mask to ``z``'s leading dimensions.

    The public model path passes a ``[batch, tokens]`` mask.  The small amount
    of handling for expanded attention masks keeps this helper usable from a
    Transformer hook as well, where masks may be ``[batch, heads, query,
    key]`` or additive floating-point masks.
    """
    if valid_mask is None:
        return None
    mask = torch.as_tensor(valid_mask, device=z.device)
    if mask.ndim == 0:
        mask = mask.reshape(1)
    if representation not in {None, "binary", "additive"}:
        raise ValueError("mask representation must be binary or additive")
    if mask.dtype != torch.bool:
        if mask.is_floating_point():
            if representation is None:
                # A two-dimensional float attention mask is the public
                # 0/1 validity contract.  A four-dimensional float mask is
                # deliberately rejected here: all-zero binary and additive
                # masks have different meanings and cannot be inferred from
                # values alone.
                values = torch.unique(mask.detach())
                if mask.ndim == z.ndim - 1 and bool(torch.all((values == 0) | (values == 1))):
                    representation = "binary"
                elif mask.ndim == 4 and z.ndim == 3:
                    raise ValueError(
                        "ambiguous four-dimensional floating attention mask; "
                        "pass representation='binary' or 'additive' explicitly"
                    )
                else:
                    raise ValueError(
                        "floating validity masks require an explicit representation "
                        "unless they are a two-dimensional 0/1 mask"
                    )
            if representation == "binary":
                mask = mask != 0
            else:
                # Additive attention masks use finite, non-negative values
                # for visible positions and a large negative value/-inf for
                # blocked keys.  The caller must select this representation.
                mask = torch.isfinite(mask) & (mask > -1e4)
        else:
            mask = mask != 0
    if mask.ndim == z.ndim - 1:
        return mask
    if mask.ndim == 4 and z.ndim == 3:
        # Collapse head and query dimensions to a key-token validity mask.
        return mask.any(dim=1).any(dim=-2)
    while mask.ndim > z.ndim - 1:
        if mask.shape[1] == 1:
            mask = mask.squeeze(1)
        else:
            mask = mask.any(dim=1)
    if mask.ndim != z.ndim - 1:
        raise ValueError(f"valid_mask shape {tuple(mask.shape)} is incompatible with latent shape {tuple(z.shape)}")
    return mask


def valid_payload_count(z, valid_mask=None, *, mask_representation=None):
    """Return the number of valid latent coordinates in ``z``.

    The allocated tensor count remains ``z.numel()``.  For a token mask, every
    valid token contributes all coordinates in the final bottleneck dimension.
    """
    mask = _token_mask(z, valid_mask, representation=mask_representation)
    if mask is None:
        return int(z.numel())
    return int(mask.to(dtype=torch.int64).sum().item() * z.shape[-1])


def normalize_power(z, valid_mask=None, *, token_wise=False, mask_representation=None):
    """Normalize each sample's latent power under an explicit mask policy.

    With the default policy, power is averaged over all non-batch dimensions,
    preserving the historical behavior.  With ``valid_mask``, only valid
    token positions contribute to the sequence-wide estimate.  With
    ``token_wise=True``, each token is normalized independently over its
    bottleneck coordinates and ``valid_mask`` is not used to couple tokens.
    """
    if z.ndim < 2:
        raise ValueError("latent tensor must have a batch dimension and a payload dimension")
    if token_wise:
        power = z.pow(2).mean(dim=-1, keepdim=True)
    else:
        mask = _token_mask(z, valid_mask, representation=mask_representation)
        if mask is None:
            # Normalize over all token/channel dimensions within each sample.
            power = z.pow(2).mean(dim=tuple(range(1, z.ndim)), keepdim=True)
        else:
            weighted = z.pow(2) * mask.to(dtype=z.dtype).unsqueeze(-1)
            total = weighted.sum(dim=tuple(range(1, z.ndim)), keepdim=True)
            count = mask.to(dtype=z.dtype).sum(dim=tuple(range(1, mask.ndim)), keepdim=True)
            count = count.reshape(z.shape[0], *([1] * (z.ndim - 1))) * z.shape[-1]
            power = total / count.clamp_min(1.0)
    return z / torch.sqrt(power + 1e-8)


class AWGNChannel(nn.Module):
    def forward(self, z, snr_db):
        if snr_db is None:
            return z
        return z + torch.randn_like(z) * (10.0 ** (-snr_db / 20.0))


class IdentityChannel(nn.Module):
    def forward(self, z, snr_db):
        return z


def build_channel(config):
    name = config["type"]
    if name == "awgn":
        cls = AWGNChannel
    elif name == "identity":
        cls = IdentityChannel
    else:
        module, cls_name = name.split(":")
        cls = getattr(importlib.import_module(module), cls_name)
    return cls(**config.get("kwargs", {}))
