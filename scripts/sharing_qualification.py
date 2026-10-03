#!/usr/bin/env python3
"""Preview, CPU-preflight, or enter an owner-admitted no-norm diagnostic."""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("preview", "preflight", "controller", "worker"):
        modes.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.preflight:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    from jscc import sharing_qualification

    for name in ("preview", "preflight", "controller", "worker"):
        path = getattr(args, name)
        if path:
            print(json.dumps(getattr(sharing_qualification, name)(path), indent=2))
            return


if __name__ == "__main__":
    main()
