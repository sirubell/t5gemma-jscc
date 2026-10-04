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
    parser.add_argument('--output', required=True, help='Fresh resolved JSON or prepared receipt')
    parser.add_argument('--assets', help='JSON with acquired cache, initializers, templates and pairing ID')
    args = parser.parse_args(argv)
    if args.assets:
        from jscc.recipe_workflow import bind_recipe_workflow
        assets = json.loads(Path(args.assets).read_text())
        result = bind_recipe_workflow(args.config, model_config_path=args.model_config, output=args.output, **assets)
        print(json.dumps({'prepared_identity': result['manifest_identity'], 'launch_authorized': False}))
        return result
    result = resolve_recipe(args.config, model_config_path=args.model_config)
    with Path(args.output).open('x') as handle:
        handle.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'identity': result['identity'], 'matrix_counts': result['matrix_counts'],
                      'trajectories': len(result['trajectories']), 'launch_authorized': False}, indent=2))
    return result


if __name__ == '__main__':
    main()
