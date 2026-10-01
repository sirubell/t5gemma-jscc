#!/usr/bin/env python3
"""Verify/preview a sharing package or explicitly execute its finite lifecycle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jscc.sharing_preparation import load_prepared, preview, require_execution_ready


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview", type=Path, metavar="PREPARED")
    mode.add_argument("--execute", type=Path, metavar="PREPARED")
    parser.add_argument("--output", type=Path, help="fresh execution output directory")
    args = parser.parse_args(argv)
    path = (args.preview or args.execute).resolve()
    manifest = load_prepared(path)
    if args.preview:
        if args.output is not None:
            parser.error("--output is for execution only")
        result = preview(manifest)
    else:
        if args.output is None:
            parser.error("--execute requires --output")
        output = args.output.resolve()
        if output.exists():
            raise ValueError("execution requires a fresh output directory")
        require_execution_ready(manifest)
        from jscc.sharing_run import execute_prepared
        result = execute_prepared(manifest, path.parent, output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
