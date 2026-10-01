#!/usr/bin/env python3
"""Validate a production contract or enter its externally bounded controller."""
from pathlib import Path
import argparse
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jscc.sharing_controller import external_command, run_controller


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preview', type=Path)
    mode.add_argument('--controller', type=Path)
    mode.add_argument('--preflight', type=Path)
    args = parser.parse_args()
    if args.preflight:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        from jscc.sharing_startup import preflight
        print(json.dumps(preflight(args.preflight)))
        return 0
    if args.preview:
        print(json.dumps({'command': external_command(args.preview), 'launched': False}))
        return 0
    return run_controller(args.controller)


if __name__ == '__main__':
    raise SystemExit(main())
