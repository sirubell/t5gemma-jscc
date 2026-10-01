"""Descriptive paired CPU geometry; never returns receiver transforms.

Records contain complete sequences (including padding): query_id, view_id,
activations [T,D], mask [T], token_positions [T], and token_roles [T]. Inputs
may be numpy arrays or CPU torch tensors. The caller must verify the full study
freeze before passing verified_freeze=True for enc_l14; this flag is not proof
of a freeze and this module does not validate checkpoint/run inventories.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any

import numpy as np
import torch


MAX_PAIRED_ROWS = 32768


def _array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("geometry accepts CPU tensors only")
        return value.detach().to(torch.float64).numpy().copy()
    return np.asarray(value).copy()


def _summary(values: Any) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    return {
        "count": int(values.size),
        "available_count": int(finite.size),
        "unavailable_count": int(values.size - finite.size),
        "unavailable_reason": "nonfinite_or_degenerate"
        if finite.size < values.size
        else ("empty" if not values.size else None),
        "mean": float(finite.mean()) if finite.size else None,
        "quantiles": dict(
            zip(
                ("min", "q25", "median", "q75", "max"),
                np.quantile(finite, [0, 0.25, 0.5, 0.75, 1]).tolist(),
            )
        )
        if finite.size
        else None,
    }


def _cosine(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    denominator = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    values = np.full(len(x), np.nan)
    np.divide(np.sum(x * y, axis=1), denominator, out=values, where=denominator > 0)
    return _summary(values)


def _spectrum(x: np.ndarray) -> dict[str, Any]:
    if len(x) < 2:
        return {
            "eigenvalues": None,
            "cumulative_variance": None,
            "effective_rank": None,
            "constant_coordinate_count": None,
            "unavailable_reason": "fewer_than_two_finite_paired_rows",
        }
    centered = x - x.mean(axis=0)
    eigenvalues = np.maximum(
        np.linalg.eigvalsh(centered.T @ centered / (len(x) - 1)), 0
    )[::-1]
    total = eigenvalues.sum()
    p = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)
    positive = p[p > 0]
    return {
        "eigenvalues": eigenvalues.tolist(),
        "cumulative_variance": np.cumsum(p).tolist() if total > 0 else None,
        "effective_rank": float(np.exp(-np.sum(positive * np.log(positive))))
        if total > 0
        else None,
        "constant_coordinate_count": int(np.sum(np.all(centered == 0, axis=0))),
        "unavailable_reason": None if total > 0 else "zero_total_variance",
    }


def paired_geometry(
    partitions: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    sample_cap: int = MAX_PAIRED_ROWS,
    sampling_seed: int = 0,
    verified_freeze: bool = False,
) -> dict[str, Any]:
    """Analyze paired train/selection records without modifying any input.

    Sampling ranks SHA256(seed, query, view, position) before reading outcomes.
    Nonfinite sampled rows are excluded jointly across sites, without replacement.
    Train-centered cosine uses means of those finite paired training sample rows.
    Other centered statistics use their own analyzed partition's sample mean.
    Actual/expected query counts are reported; tiny synthetic atlases are allowed.
    Full-sequence/mask/provenance capture validation remains the caller's duty.
    """
    if type(sample_cap) is not int or not 1 <= sample_cap <= MAX_PAIRED_ROWS:
        raise ValueError("sample_cap must be an integer in [1, 32768]")
    if type(sampling_seed) is not int:
        raise ValueError("sampling_seed must be an integer")
    if set(partitions) != {"train", "selection"}:
        raise ValueError("exactly train and selection partitions are required")
    sites = sorted(partitions["train"])
    if not sites or set(partitions["selection"]) != set(sites):
        raise ValueError("both partitions require the same nonempty site set")
    if "enc_l14" in sites and verified_freeze is not True:
        raise ValueError("enc_l14 requires caller-verified study freeze")
    output: dict[str, Any] = {
        "precision": "float64_cpu",
        "sampling_seed": sampling_seed,
        "sample_cap": sample_cap,
        "sampling": "sha256_identity_rank_v1",
        "covariance_estimator": "sample_covariance_ddof_1_partition_centered",
        "centered_cosine_mean": "finite_paired_training_sample_rows",
        "receiver_transforms": "none",
        "nonfinite_policy": "exclude jointly after identity sampling; no replacement",
        "scale_variance_estimator": "population_variance_ddof_0",
        "partitions": {},
    }
    samples: dict[str, dict[str, np.ndarray]] = {}
    query_sets: dict[str, set[str]] = {}
    width: int | None = None
    for partition in ("train", "selection"):
        site_records = partitions[partition]
        reference = site_records[sites[0]]
        identities: list[tuple[str, str, int]] = []
        sequence_keys: set[tuple[str, str]] = set()
        query_sets[partition] = set()
        masks = []
        positions = []
        roles = []
        for record in reference:
            query, view = record["query_id"], record["view_id"]
            if (
                not isinstance(query, str)
                or not query
                or not isinstance(view, str)
                or not view
            ):
                raise ValueError("query_id and view_id must be nonempty strings")
            if (query, view) in sequence_keys:
                raise ValueError("duplicate query/view identity")
            sequence_keys.add((query, view))
            query_sets[partition].add(query)
            mask = _array(record["mask"])
            position = _array(record["token_positions"])
            role = np.asarray(record["token_roles"])
            if mask.ndim != 1 or not np.isin(mask, [0, 1]).all():
                raise ValueError("mask must be a binary vector")
            mask = mask.astype(bool)
            if (
                position.shape != mask.shape
                or not np.isfinite(position).all()
                or not np.equal(position, np.floor(position)).all()
            ):
                raise ValueError(
                    "token_positions must be an integer vector matching mask"
                )
            if len(set(position.tolist())) != len(position) or (position < 0).any():
                raise ValueError("token positions must be unique and nonnegative")
            if (
                role.shape != mask.shape
                or not np.isin(role[mask], ["query", "demo"]).all()
            ):
                raise ValueError("valid token roles must label query/demo positions")
            masks.append(mask)
            positions.append(position)
            roles.append(role)
            identities.extend((query, view, int(p)) for p in position[mask])
        order = sorted(
            range(len(identities)),
            key=lambda i: (
                hashlib.sha256(
                    json.dumps(
                        [sampling_seed, *identities[i]], separators=(",", ":")
                    ).encode()
                ).digest(),
                identities[i],
            ),
        )[:sample_cap]
        rows: dict[str, np.ndarray] = {}
        site_stats = {}
        for site in sites:
            records = site_records[site]
            if len(records) != len(reference):
                raise ValueError("sites must contain identical complete sequences")
            chunks = []
            summaries = []
            offset = 0
            for i, (record, ref) in enumerate(zip(records, reference)):
                if any(
                    record[key] != ref[key] for key in ("query_id", "view_id")
                ) or any(
                    not np.array_equal(_array(record[key]), _array(ref[key]))
                    for key in ("mask", "token_positions", "token_roles")
                ):
                    raise ValueError(
                        "query/view/mask/position/role pairing differs across sites"
                    )
                x = _array(record["activations"]).astype(np.float64)
                if x.ndim != 2 or x.shape[0] != len(masks[i]) or x.shape[1] < 1:
                    raise ValueError("activations must have complete [T,D] shape")
                if width is None:
                    width = x.shape[1]
                if x.shape[1] != width:
                    raise ValueError(
                        "coordinate geometry requires equal feature widths"
                    )
                valid = x[masks[i]]
                chunks.append(valid)
                finite = valid[np.isfinite(valid).all(axis=1)]
                selected = sum(offset <= j < offset + len(valid) for j in order)
                offset += len(valid)
                summaries.append(
                    {
                        "query_id": record["query_id"],
                        "view_id": record["view_id"],
                        "valid_rows": len(valid),
                        "padded_rows": len(x) - len(valid),
                        "sampled_rows": selected,
                        "finite_rows": len(finite),
                        "nonfinite_rows": len(valid) - len(finite),
                        "query_rows": int(np.sum(roles[i][masks[i]] == "query")),
                        "demo_rows": int(np.sum(roles[i][masks[i]] == "demo")),
                        "raw_rms": float(np.sqrt(np.mean(finite**2)))
                        if len(finite)
                        else None,
                        "unavailable_reason": None
                        if len(finite)
                        else "empty_or_nonfinite",
                        "token_norm": _summary(np.linalg.norm(valid, axis=1)),
                        "coordinate_mean": _summary(
                            finite.mean(axis=0) if len(finite) else []
                        ),
                        "coordinate_variance": _summary(
                            finite.var(axis=0) if len(finite) else []
                        ),
                    }
                )
            rows[site] = np.concatenate(chunks) if chunks else np.empty((0, width or 0))
            site_stats[site] = {
                "sequences": summaries,
                "valid_rows": len(rows[site]),
                "nonfinite_rows": int((~np.isfinite(rows[site]).all(axis=1)).sum()),
                "finite_rows": int(np.isfinite(rows[site]).all(axis=1).sum()),
                "empty_sequences": sum(s["valid_rows"] == 0 for s in summaries),
                "token_norm": _summary(np.linalg.norm(rows[site], axis=1)),
                "raw_rms": _summary([s["raw_rms"] for s in summaries]),
            }
        sampled = {site: rows[site][order] for site in sites}
        finite_mask = np.ones(len(order), dtype=bool)
        for site in sites:
            finite_mask &= np.isfinite(sampled[site]).all(axis=1)
        samples[partition] = {site: sampled[site][finite_mask] for site in sites}
        finite_coverage: dict[tuple[str, str], int] = {}
        for j, is_finite in zip(order, finite_mask):
            key = identities[j][:2]
            finite_coverage[key] = finite_coverage.get(key, 0) + int(is_finite)
        for site in sites:
            for sequence in site_stats[site]["sequences"]:
                sequence["finite_paired_sampled_rows"] = finite_coverage.get(
                    (sequence["query_id"], sequence["view_id"]), 0
                )

        output["partitions"][partition] = {
            "expected_query_count": 128,
            "actual_query_count": len(query_sets[partition]),
            "sequence_count": len(reference),
            "full_valid_rows": len(identities),
            "sampled_rows": len(order),
            "finite_paired_rows": int(finite_mask.sum()),
            "nonfinite_paired_rows": int((~finite_mask).sum()),
            "sampled_identities": [list(identities[j]) for j in order],
            "sites": site_stats,
            "pairs": {},
        }
    if query_sets["train"] & query_sets["selection"]:
        raise ValueError("training and selection query identities must be disjoint")
    for partition in ("train", "selection"):
        report = output["partitions"][partition]
        for site in sites:
            report["sites"][site]["spectrum"] = _spectrum(samples[partition][site])
        for left, right in itertools.combinations(sites, 2):
            x, y = samples[partition][left], samples[partition][right]
            if len(x):
                xc, yc = x - x.mean(axis=0), y - y.mean(axis=0)
                denom = np.linalg.norm(xc, axis=0) * np.linalg.norm(yc, axis=0)
                pearson = np.full(x.shape[1], np.nan)
                np.divide(np.sum(xc * yc, axis=0), denom, out=pearson, where=denom > 0)
                cka_den = np.linalg.norm(xc.T @ xc) * np.linalg.norm(yc.T @ yc)
                cka = float(np.sum((xc.T @ yc) ** 2) / cka_den) if cka_den > 0 else None
            else:
                pearson, cka = np.full(width or 0, np.nan), None
            train_x, train_y = samples["train"][left], samples["train"][right]
            centered = (
                _cosine(x - train_x.mean(axis=0), y - train_y.mean(axis=0))
                if len(train_x)
                else {
                    "mean": None,
                    "count": len(x),
                    "available_count": 0,
                    "unavailable_count": len(x),
                    "unavailable_reason": "no_finite_training_rows",
                    "quantiles": None,
                }
            )
            ratios = []
            for a, b in zip(
                report["sites"][left]["sequences"], report["sites"][right]["sequences"]
            ):
                ratios.append(
                    a["raw_rms"] / b["raw_rms"]
                    if a["raw_rms"] is not None and b["raw_rms"]
                    else np.nan
                )
            report["pairs"][f"{left}/{right}"] = {
                "raw_cosine": _cosine(x, y),
                "train_centered_cosine": centered,
                "coordinate_pearson": _summary(pearson),
                "coordinate_pearson_values": [
                    float(v) if np.isfinite(v) else None for v in pearson
                ],
                "linear_cka": cka,
                "cka_unavailable_reason": None
                if cka is not None
                else "empty_or_zero_variance",
                "sequence_rms_ratio_left_over_right": _summary(ratios),
            }
    return output
