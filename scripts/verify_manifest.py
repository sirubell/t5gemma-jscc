#!/usr/bin/env python3
"""Verify a JSON object with an explicit files list of contained relative paths.

Each entry requires path and a 64-digit hexadecimal sha256, with optional
nonnegative integer bytes. Source paths must be unique; absolute paths, parent
traversal and symlinks escaping the manifest directory are rejected before hashing.
Unknown auxiliary fields and an explicitly empty files list are accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_files(manifest: object, root: Path) -> list[tuple[str, Path, str, int | None]]:
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        raise ValueError("expected an object with an explicit files list")
    files = []
    seen = set()
    for index, item in enumerate(manifest["files"]):
        if not isinstance(item, dict):
            raise ValueError(f"files[{index}] must be an object")
        name = item.get("path")
        if not isinstance(name, str) or not name or "\x00" in name:
            raise ValueError(f"files[{index}].path must be a nonempty relative path")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid relative path: {name}")
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"path escapes manifest directory: {name}")
        if path in seen:
            raise ValueError(f"duplicate source path: {name}")
        seen.add(path)
        digest = item.get("sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
            raise ValueError(f"files[{index}].sha256 must be a 64-digit hexadecimal digest")
        size = item.get("bytes")
        if "bytes" in item and (type(size) is not int or size < 0):
            raise ValueError(f"files[{index}].bytes must be a nonnegative integer")
        files.append((name, path, digest.lower(), size))
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    root = manifest_path.parent
    try:
        manifest = json.loads(manifest_path.read_text())
        files = _manifest_files(manifest, root)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"INVALID_MANIFEST {error}")
        raise SystemExit(1) from error
    errors = []
    for name, path, expected, size in files:
        if not path.is_file():
            errors.append(f"MISSING {name}")
            continue
        if size is not None and path.stat().st_size != size:
            errors.append(f"SIZE_MISMATCH {name}")
        actual = sha256(path)
        if actual != expected:
            errors.append(f"HASH_MISMATCH {name}")
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print(f"verified {len(files)} files")


if __name__ == "__main__":
    main()
