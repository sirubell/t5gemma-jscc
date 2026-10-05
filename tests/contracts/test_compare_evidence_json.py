import pytest
from scripts.compare_evidence_json import compare


def test_only_version_and_float_roundoff_are_allowed():
    assert compare(
        {"method": {"numpy_version": "2.4"}, "p": 0.0001},
        {"method": {"numpy_version": "2.3"}, "p": 0.0001 + 1e-17},
    )
    for candidate in [{"count": 2}, {"count": 1.0}, {"count": 1, "extra": 0}]:
        with pytest.raises(ValueError):
            compare({"count": 1}, candidate)
    with pytest.raises(ValueError):
        compare({"other_version": "a"}, {"other_version": "b"})
    with pytest.raises(ValueError):
        compare({"p": 0.05}, {"p": 0.051})
    with pytest.raises(ValueError):
        compare({"p": 1.0}, {"p": float("nan")})
