"""Clean complete-sequence geometry capture from a frozen native encoder.

This adapter preserves supplied tokenization/provenance; it cannot establish that
an upstream tokenizer retained a complete five-shot prompt. Geometry sampling
happens only after capture and never changes model input or receiver tensors.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any

import torch

from .activation_geometry import MAX_PAIRED_ROWS, paired_geometry
from .activation_replay import canonical_digest
from .models.split_model import resolve_encoder_site, stack_module
from .presentation import tensor_digest
from .sharing_state import verify_study_freeze


def _verify_heldout(include_heldout, freeze_reference, obligation):
    if not include_heldout:
        return None
    if freeze_reference is None or not isinstance(obligation, str) or not obligation:
        raise ValueError(
            "heldout geometry requires a file-backed freeze and named obligation"
        )
    receipt = verify_study_freeze(freeze_reference, obligation=obligation)
    if obligation not in receipt["obligation_plan"]["geometry"]:
        raise ValueError("heldout obligation must be declared as geometry")
    return receipt


def _batch_metadata(batch):
    ids, mask = batch["input_ids"], batch["attention_mask"]
    if not isinstance(ids, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise ValueError("native input_ids and attention_mask must be tensors")
    if ids.ndim != 2 or mask.shape != ids.shape or not ids.numel():
        raise ValueError("complete nonempty [B,T] inputs and masks are required")
    if ids.dtype not in (torch.int32, torch.int64) or not bool(
        ((mask == 0) | (mask == 1)).all()
    ):
        raise ValueError("native input IDs must be integers and mask binary")
    query_ids, view_ids = batch["query_ids"], batch["view_ids"]
    if (
        len(query_ids) != len(ids)
        or len(view_ids) != len(ids)
        or any(not isinstance(v, str) or not v for v in [*query_ids, *view_ids])
    ):
        raise ValueError("each complete sequence requires query_id and view_id")
    if len(set(zip(query_ids, view_ids))) != len(ids):
        raise ValueError("duplicate query/view in capture batch")
    positions = torch.as_tensor(batch["token_positions"])
    if positions.shape != ids.shape or positions.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("absolute token positions must be integer [B,T]")
    if bool((positions < 0).any()) or any(
        len(torch.unique(row)) != len(row) for row in positions
    ):
        raise ValueError("absolute token positions must be unique and nonnegative")
    roles = batch["token_roles"]
    if len(roles) != len(ids) or any(len(row) != ids.shape[1] for row in roles):
        raise ValueError("token_roles must preserve the complete padded layout")
    for row_roles, row_mask in zip(roles, mask):
        if any(
            role not in ("query", "demo")
            for role, valid in zip(row_roles, row_mask.tolist())
            if valid
        ):
            raise ValueError("every valid token must be labeled query or demo")
    return ids, mask, positions, roles


@torch.no_grad()
def capture_geometry_batch(
    model,
    batch: dict[str, Any],
    *,
    model_revision: str,
    include_heldout: bool = False,
    freeze_reference=None,
    obligation: str | None = None,
) -> dict[str, Any]:
    """Capture l9/l19/final norm/raw-last from one unchanged native microbatch.

    ``model`` is a SplitModel. Inputs require input_ids, attention_mask, query_ids,
    view_ids, token_positions and token_roles. Optional ``provenance`` is retained
    verbatim, never inferred. l14 validation precedes any encoder execution.
    """
    freeze = _verify_heldout(include_heldout, freeze_reference, obligation)
    ids, mask, positions, roles = _batch_metadata(batch)
    if model.split["stack"] != "enc" or model.numerical_policy != "native":
        raise ValueError("geometry capture requires native encoder SplitModel")
    if model._routing_active or model._forward_active:
        raise RuntimeError("geometry capture cannot nest inside routing or a forward")
    if any(p.requires_grad for p in model.base.parameters()) or any(
        m.training for m in model.base.modules()
    ):
        raise ValueError("geometry backbone must already be frozen and in eval mode")
    stack = stack_module(model.base, "enc")
    last_index = len(stack.layers) - 1
    if last_index == 14 and not include_heldout:
        raise ValueError("raw-final resolves to heldout l14 before freeze")
    names = list(
        dict.fromkeys(
            ["enc_l9", "enc_l19", "enc_fn", f"enc_l{last_index}"]
            + (["enc_l14"] if include_heldout else [])
        )
    )
    sites = {}
    for name in names:
        spec = (
            {"stack": "enc", "where": "after_final_norm"}
            if name == "enc_fn"
            else {"stack": "enc", "where": "after_layer", "index": int(name[5:])}
        )
        sites[name] = resolve_encoder_site(model.base, spec, model_revision)
    if freeze is not None:
        expected_policy = {
            name: {
                "site": asdict(sites[name]),
                "role": "heldout_after_freeze" if name == "enc_l14" else "trained",
            }
            for name in ("enc_l9", "enc_l19", "enc_fn", "enc_l14")
        }
        if freeze.get("site_policy") != expected_policy:
            raise ValueError("freeze site policy differs from actual capture backbone")
    found: dict[str, torch.Tensor] = {}
    handles = []
    # transmission restores masks/condition; preserve observable capture/accounting
    # state as well because its normal training API intentionally resets counters.
    saved = {
        name: getattr(model, name)
        for name in (
            "activation",
            "reconstruction",
            "channel_uses",
            "channel_uses_valid",
        )
    }
    try:
        for name, site in sites.items():
            module = stack.norm if name == "enc_fn" else stack.layers[site.index]

            def observe(_module, _args, output, name=name, site=site):
                value = output[0] if isinstance(output, tuple) else output
                if name in found:
                    raise RuntimeError("geometry site executed more than once")
                if (
                    value.ndim != 3
                    or value.shape[:2] != mask.shape
                    or value.shape[-1] != site.hidden_dim
                ):
                    raise ValueError(
                        "geometry activation changed complete native layout"
                    )
                found[name] = value.detach().cpu().clone()

            handles.append(module.register_forward_hook(observe))
        device = next(model.base.parameters()).device
        with (
            model.transmission(bypass=True, encoder_mask=mask.to(device)),
            model.attention_context(),
        ):
            model.base.get_encoder()(
                input_ids=ids.to(device),
                attention_mask=mask.to(device),
                return_dict=True,
            )
        if set(found) != set(sites):
            raise RuntimeError("not all geometry sites executed exactly once")
        if any(model.channel_uses.values()) or any(model.channel_uses_valid.values()):
            raise RuntimeError(
                "clean geometry unexpectedly transmitted channel coordinates"
            )
    finally:
        for handle in handles:
            handle.remove()
        for name, value in saved.items():
            setattr(model, name, value)
    records = {}
    for site, values in found.items():
        records[site] = [
            {
                "query_id": batch["query_ids"][i],
                "view_id": batch["view_ids"][i],
                "activations": values[i],
                "mask": mask[i].detach().cpu().clone(),
                "token_positions": positions[i].detach().cpu().clone(),
                "token_roles": list(roles[i]),
            }
            for i in range(len(ids))
        ]
    receipt = {
        "model_revision": model_revision,
        "resolved_sites": {name: asdict(site) for name, site in sites.items()},
        "raw_final_site": f"enc_l{last_index}",
        "shape": list(ids.shape),
        "input_sha256": tensor_digest(ids),
        "mask_sha256": tensor_digest(mask),
        "token_positions_sha256": tensor_digest(positions),
        "query_ids": list(batch["query_ids"]),
        "view_ids": list(batch["view_ids"]),
        "token_roles": deepcopy(roles),
        "capture_dtype": {name: str(x.dtype) for name, x in found.items()},
        "activation_sha256": {name: tensor_digest(x) for name, x in found.items()},
        "provenance": deepcopy(batch.get("provenance")),
        "freeze_reference": deepcopy(freeze_reference) if freeze else None,
        "freeze_obligation": obligation if freeze else None,
        "phase": "post_freeze" if freeze else "pre_freeze",
        "freeze_synthetic": freeze.get("synthetic") if freeze else None,
        "codec_channel_bypassed": True,
    }
    receipt["identity"] = canonical_digest(receipt)
    return {"records": records, "receipt": receipt}


def capture_geometry_atlas(
    model,
    partitions: dict[str, list[dict[str, Any]]],
    *,
    model_revision: str,
    include_heldout: bool = False,
    freeze_reference=None,
    obligation: str | None = None,
    sample_cap: int = MAX_PAIRED_ROWS,
    sampling_seed: int = 0,
) -> dict[str, Any]:
    """Capture prepared complete batches, then run descriptive paired geometry."""
    if set(partitions) != {"train", "selection"} or any(
        not batches for batches in partitions.values()
    ):
        raise ValueError("atlas requires nonempty train and selection batch lists")
    if type(sample_cap) is not int or not 1 <= sample_cap <= MAX_PAIRED_ROWS:
        raise ValueError("sample_cap must be an integer in [1, 32768]")
    if type(sampling_seed) is not int:
        raise ValueError("sampling_seed must be an integer")
    # Reject cross-partition leakage and malformed metadata before model work.
    query_sets = {}
    for partition, batches in partitions.items():
        queries: set[str] = set()
        views: set[tuple[str, str]] = set()
        for batch in batches:
            _batch_metadata(batch)
            pairs = set(zip(batch["query_ids"], batch["view_ids"]))
            if views & pairs:
                raise ValueError("duplicate query/view across atlas batches")
            views |= pairs
            queries.update(batch["query_ids"])
        query_sets[partition] = queries
    if query_sets["train"] & query_sets["selection"]:
        raise ValueError("training and selection queries must be disjoint")
    captures: dict[str, dict[str, list[dict[str, Any]]]] = {}
    receipts = {}
    for partition, batches in partitions.items():
        captures[partition], receipts[partition] = {}, []
        for batch in batches:
            result = capture_geometry_batch(
                model,
                batch,
                model_revision=model_revision,
                include_heldout=include_heldout,
                freeze_reference=freeze_reference,
                obligation=obligation,
            )
            receipts[partition].append(result["receipt"])
            for site, records in result["records"].items():
                captures[partition].setdefault(site, []).extend(records)
    _verify_heldout(include_heldout, freeze_reference, obligation)
    analysis = paired_geometry(
        captures,
        sample_cap=sample_cap,
        sampling_seed=sampling_seed,
        verified_freeze=include_heldout,
    )
    return {"analysis": analysis, "capture_receipts": receipts}
