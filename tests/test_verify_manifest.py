from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import verify_manifest


SCRIPT = Path(verify_manifest.__file__).resolve()
DIGEST = hashlib.sha256(b"payload").hexdigest()


def run_manifest(tmp_path: Path, manifest: object) -> subprocess.CompletedProcess[str]:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(manifest_path)],
        capture_output=True, text=True, check=False,
    )


def test_valid_contained_manifest_streams_and_accepts_auxiliary_fields(tmp_path: Path):
    payload = b"payload" * 300_000
    path = tmp_path / "nested" / "data.bin"
    path.parent.mkdir()
    path.write_bytes(payload)
    result = run_manifest(tmp_path, {
        "description": "auxiliary metadata",
        "files": [{"path": "nested/data.bin", "sha256": hashlib.sha256(payload).hexdigest(),
                   "bytes": len(payload), "label": "data"}],
    })
    assert result.returncode == 0, result.stderr
    assert result.stdout == "verified 1 files\n"


def test_explicit_empty_manifest(tmp_path: Path):
    result = run_manifest(tmp_path, {"files": []})
    assert result.returncode == 0
    assert result.stdout == "verified 0 files\n"


def test_optional_size_uppercase_digest_and_contained_symlink(tmp_path: Path):
    (tmp_path / "data").write_bytes(b"payload")
    (tmp_path / "link").symlink_to("data")
    result = run_manifest(tmp_path, {"files": [{"path": "link", "sha256": DIGEST.upper()}]})
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize(("payload", "size", "expected"), [
    (None, 7, ["MISSING data"]),
    (b"mutated", 7, ["HASH_MISMATCH data"]),
    (b"payload", 8, ["SIZE_MISMATCH data"]),
    (b"changed-size", 7, ["SIZE_MISMATCH data", "HASH_MISMATCH data"]),
])
def test_integrity_diagnostics(tmp_path: Path, payload: bytes | None, size: int,
                               expected: list[str]):
    if payload is not None:
        (tmp_path / "data").write_bytes(payload)
    result = run_manifest(tmp_path, {"files": [{"path": "data", "sha256": DIGEST, "bytes": size}]})
    assert result.returncode == 1
    assert result.stdout.splitlines() == expected


@pytest.mark.parametrize("manifest", [
    None, [], "manifest", 1, {}, {"files": None}, {"files": {}}, {"files": "data"},
    {"files": [None]}, {"files": [[]]}, {"files": [{}]},
    {"files": [{"path": "", "sha256": DIGEST}]},
    {"files": [{"path": 1, "sha256": DIGEST}]},
    {"files": [{"path": "data\x00", "sha256": DIGEST}]},
    {"files": [{"path": "data"}]},
    *[{"files": [{"path": "data", "sha256": digest}]}
      for digest in (None, 1, "", "a" * 63, "a" * 65, "g" * 64, "a" * 63 + "\n")],
    *[{"files": [{"path": "data", "sha256": DIGEST, "bytes": size}]}
      for size in (None, True, False, -1, 7.0, "7")],
])
def test_invalid_schema_rejected_by_cli(tmp_path: Path, manifest: object):
    (tmp_path / "data").write_bytes(b"payload")
    result = run_manifest(tmp_path, manifest)
    assert result.returncode == 1
    assert result.stdout.startswith("INVALID_MANIFEST ")
    assert "verified" not in result.stdout


def test_zero_bytes_is_valid(tmp_path: Path):
    (tmp_path / "empty").touch()
    result = run_manifest(tmp_path, {"files": [{
        "path": "empty", "sha256": hashlib.sha256(b"").hexdigest(), "bytes": 0,
    }]})
    assert result.returncode == 0, result.stdout


def test_malformed_json_rejected_by_cli(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{invalid")
    result = subprocess.run([sys.executable, str(SCRIPT), str(manifest)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert result.stdout.startswith("INVALID_MANIFEST ")


@pytest.mark.parametrize("second", ["data", "./data", "nested/../data", "link"])
def test_duplicate_or_traversal_sources_rejected(tmp_path: Path, second: str):
    (tmp_path / "data").write_bytes(b"payload")
    (tmp_path / "link").symlink_to("data")
    result = run_manifest(tmp_path, {"files": [
        {"path": "data", "sha256": DIGEST}, {"path": second, "sha256": DIGEST},
    ]})
    assert result.returncode == 1
    expected = "invalid relative path" if ".." in second else "duplicate source path"
    assert expected in result.stdout


@pytest.mark.parametrize("kind", ["absolute", "traversal", "symlink", "directory-symlink"])
def test_escape_rejected_before_any_hash_or_escaped_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
):
    root = tmp_path / "handoff"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"payload")
    (root / "data").write_bytes(b"payload")
    if kind == "absolute":
        escaped = str(outside)
    elif kind == "traversal":
        escaped = "../outside"
    elif kind == "symlink":
        (root / "link").symlink_to(outside)
        escaped = "link"
    else:
        (root / "link").symlink_to(tmp_path, target_is_directory=True)
        escaped = "link/outside"
    manifest = {"files": [
        {"path": "data", "sha256": DIGEST}, {"path": escaped, "sha256": DIGEST},
    ]}
    result = run_manifest(root, manifest)
    assert result.returncode == 1
    assert result.stdout.startswith("INVALID_MANIFEST ")

    original_open = Path.open
    opened = []

    def guarded_open(path: Path, *args, **kwargs):
        opened.append(path)
        assert path.resolve() != outside, "escaped bytes were opened"
        return original_open(path, *args, **kwargs)

    def forbidden_hash(path: Path):
        pytest.fail(f"hashed {path} before completing manifest validation")

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(verify_manifest, "sha256", forbidden_hash)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), str(root / "manifest.json")])
    with pytest.raises(SystemExit, match="1"):
        verify_manifest.main()
    assert outside not in opened
