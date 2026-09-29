"""Portable statistical comparison: exact discrete evidence, bounded float roundoff."""

import math

METADATA_PATHS = {("numpy_version",), ("method", "numpy_version")}


def compare(reference, actual, path=()):
    """Return environment differences; raise for unapproved structural/value changes."""
    if path in METADATA_PATHS:
        if not isinstance(reference, str) or not isinstance(actual, str):
            raise ValueError(f"Invalid metadata at {path}")
        return (
            []
            if reference == actual
            else [{"path": ".".join(path), "reference": reference, "actual": actual}]
        )
    if type(reference) is not type(actual):
        raise ValueError(f"Type mismatch at {path}")
    if isinstance(reference, dict):
        if set(reference) != set(actual):
            raise ValueError(f"Keys differ at {path}")
        return [
            d
            for key in reference
            for d in compare(reference[key], actual[key], path + (key,))
        ]
    if isinstance(reference, list):
        if len(reference) != len(actual):
            raise ValueError(f"Length differs at {path}")
        return [
            d
            for i, (a, b) in enumerate(zip(reference, actual))
            for d in compare(a, b, path + (str(i),))
        ]
    if isinstance(reference, float):
        if not (
            math.isfinite(reference)
            and math.isfinite(actual)
            and math.isclose(reference, actual, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise ValueError(f"Float mismatch at {path}")
    elif reference != actual:
        raise ValueError(f"Exact evidence differs at {path}")
    return []
