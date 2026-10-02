#!/usr/bin/env python3
"""Prepare exact q10000 native16 inputs offline; never bind or execute GPU jobs."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from jscc.encoder_stages_preparation import WIDTHS, prepare_inputs  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prefix-prepared', 'pool-manifest', 'assets', 'arrow', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('prefix-sha256', 'pool-sha256'):
        parser.add_argument('--' + name, required=True)
    for width in WIDTHS:
        parser.add_argument(f'--initializer-{width}', type=Path, required=True)
        parser.add_argument(f'--initializer-{width}-sha256', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    path = prepare_inputs(prefix_prepared=args.prefix_prepared, prefix_sha256=args.prefix_sha256,
        pool_manifest=args.pool_manifest, pool_sha256=args.pool_sha256, assets=args.assets,
        arrow=args.arrow, output=args.output,
        initializers={str(width): (getattr(args, f'initializer_{width}'),
            getattr(args, f'initializer_{width}_sha256')) for width in WIDTHS})
    print(path)


if __name__ == '__main__':
    main()
