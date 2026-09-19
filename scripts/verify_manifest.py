#!/usr/bin/env python3
"""Verify relative paths and SHA-256 entries in a handoff manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    errors = []
    for item in manifest.get("files", []):
        path = root / item["path"]
        if not path.is_file():
            errors.append(f"MISSING {item['path']}")
            continue
        if "bytes" in item and path.stat().st_size != item["bytes"]:
            errors.append(f"SIZE_MISMATCH {item['path']}")
        actual = sha256(path)
        if actual != item.get("sha256"):
            errors.append(f"HASH_MISMATCH {item['path']}")
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print(f"verified {len(manifest.get('files', []))} files")


if __name__ == "__main__":
    main()
