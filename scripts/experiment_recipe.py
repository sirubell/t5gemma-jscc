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
    parser.add_argument('--config')
    parser.add_argument('--model-config')
    parser.add_argument('--output', required=True, help='Fresh resolved JSON or prepared receipt')
    parser.add_argument('--assets', help='JSON with acquired cache, initializers, templates and pairing ID')
    parser.add_argument('--acquire-model', action='store_true', help='Hash an existing model once at acquisition')
    parser.add_argument('--model-path')
    parser.add_argument('--model-revision')
    parser.add_argument('--synthetic-cpu', action='store_true', help='Explicit local tiny fixture acquisition')
    args = parser.parse_args(argv)
    if args.acquire_model:
        if not args.model_path or not args.model_revision or args.config or args.model_config or args.assets:
            parser.error('model acquisition requires model path/revision and cannot combine with recipe options')
        from jscc.original13_provenance import bind_model_assets
        binding = bind_model_assets(args.model_path, args.model_revision, production=not args.synthetic_cpu)
        if binding['status'] != 'complete':
            raise ValueError('model acquisition requires existing complete local assets')
        with Path(args.output).open('x') as stream:
            stream.write(json.dumps(binding, indent=2)+'\n')
        print(json.dumps({'path': args.output, 'status': 'acquired', 'launch_authorized': False}))
        return binding
    if args.model_path or args.model_revision or args.synthetic_cpu or not args.config or not args.model_config:
        parser.error('recipe resolution requires config/model-config; acquisition options require --acquire-model')
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
