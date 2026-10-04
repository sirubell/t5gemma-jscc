#!/usr/bin/env python3
"""Export compact recorded results and verify local copies without model/network access."""
import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from jscc.result_collection import collect_results  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--snapshot', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--select', action='append', default=[], help='Explicit additional source-relative file')
    parser.add_argument('--source-uri', help='Exact retrieval root, e.g. ssh://host/absolute/run; no connection occurs')
    args = parser.parse_args(argv)
    result = collect_results(args.run, args.output, snapshot=args.snapshot,
                             selected=args.select, source_uri=args.source_uri)
    print(json.dumps({key: result[key] for key in ('status', 'counts', 'snapshot_id', 'receipt')}, indent=2))
    if result['status'] != 'complete':
        raise SystemExit(1)
    return result


if __name__ == '__main__':
    main()
