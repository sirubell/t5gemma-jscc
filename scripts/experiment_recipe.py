#!/usr/bin/env python3
"""Resolve a YAML recipe using an existing local backbone config.json."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jscc.experiment_recipe import resolve_recipe


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--model-config', required=True)
    parser.add_argument('--output', required=True, help='Fresh complete resolved JSON destination')
    args = parser.parse_args(argv)
    result = resolve_recipe(args.config, model_config_path=args.model_config)
    with Path(args.output).open('x') as handle:
        handle.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'identity': result['identity'], 'matrix_counts': result['matrix_counts'],
                      'trajectories': len(result['trajectories']), 'launch_authorized': False}, indent=2))
    return result


if __name__ == '__main__':
    main()
