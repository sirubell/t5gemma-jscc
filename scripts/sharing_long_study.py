#!/usr/bin/env python3
"""Prepare/preview, train or evaluate one exact native16 long-study width."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jscc.sharing_long_run import configure_long_entry, evaluate_long
from jscc.sharing_preparation import load_prepared, preview
from jscc.sharing_run import execute_prepared


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['preview', 'train', 'evaluate'])
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--training-run', type=Path)
    parser.add_argument('--approved-source')
    parser.add_argument('--timeout-seconds', type=int)
    args = parser.parse_args(argv)
    manifest = load_prepared(args.prepared)
    if args.mode == 'preview':
        result = preview(manifest)
    else:
        if args.output is None or args.timeout_seconds is None:
            parser.error('execution requires --output and --timeout-seconds')
        if args.output.exists():
            raise ValueError('fresh output directory required; no automatic retry')
        if args.mode == 'evaluate' and args.training_run is None:
            parser.error('evaluate requires the exact --training-run')
        configure_long_entry(manifest, approved_source=args.approved_source,
                             timeout_seconds=args.timeout_seconds)
        if args.mode == 'train':
            result = execute_prepared(manifest, args.prepared.resolve().parent, args.output)
        else:
            result = evaluate_long(manifest, args.prepared.resolve().parent, args.training_run, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    main()
