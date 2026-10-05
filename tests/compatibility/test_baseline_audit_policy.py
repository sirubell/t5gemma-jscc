"""Diagnostic audit capture changes bookkeeping only, with bound prepared inputs."""

from dataclasses import FrozenInstanceError
import pytest
import torch
from jscc.shared_inputs import batch_identity, bind_batch_identity
from contracts.test_core import batch








def test_binding_is_complete_and_rejects_tensor_and_metadata_mutations():
    b = {**batch(), "pixel_values": torch.randn(2, 3, 4, 4),
         "activation": torch.randn(2, 3, 8), "site_id": "enc_fn", "view": {"crop": [0, 1]}}
    binding = bind_batch_identity(b, expected_identity=batch_identity(b))
    assert binding.validate(b) == batch_identity(b)
    with pytest.raises(FrozenInstanceError):
        setattr(binding, "view_sha256", "changed")
    changed = {**b, "activation": b["activation"].clone()}
    with pytest.raises(ValueError, match="changed"):
        binding.validate(changed)
    alias = b["pixel_values"].detach()
    alias.add_(1)
    with pytest.raises(ValueError, match="changed"):
        binding.validate(b)
    fresh = bind_batch_identity(b)
    b["view"]["crop"].append(2)
    with pytest.raises(ValueError, match="changed"):
        fresh.validate(b)
    with pytest.raises(ValueError, match="mismatch"):
        bind_batch_identity(b, expected_identity="wrong")
