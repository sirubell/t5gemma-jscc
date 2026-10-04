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
    parser.add_argument('--registrations', help='Directory of exact acquired registrations; automatically prepare assets')
    parser.add_argument('--register-assets', action='store_true', help='Explicitly register existing pinned acquisition receipts')
    parser.add_argument('--register-from-prepared', help='Register retained prepared assets without modifying the receipt')
    parser.add_argument('--model-name', help='Exact task YAML model name for registration')
    parser.add_argument('--model-binding')
    parser.add_argument('--cache')
    parser.add_argument('--observation-template')
    parser.add_argument('--terminal-template')
    parser.add_argument('--synthetic-cpu', action='store_true', help='Explicit local tiny fixture acquisition')
    args = parser.parse_args(argv)
    registration_options = (args.model_binding, args.cache, args.observation_template, args.terminal_template)
    if args.register_assets or args.register_from_prepared:
        if (not args.model_name or args.acquire_model or args.config or args.model_config or args.assets
                or args.model_path or args.model_revision or args.registrations
                or (args.register_assets and args.register_from_prepared)):
            parser.error('registration requires exact model-name and cannot combine with acquisition/preparation')
        from jscc.recipe_assets import register_recipe_assets, register_prepared_assets
        if args.register_from_prepared:
            if any(registration_options) or args.synthetic_cpu:
                parser.error('retained registration uses the existing prepared acquisition fields')
            result = register_prepared_assets(args.register_from_prepared, model_name=args.model_name, output=args.output)
        else:
            if not all(registration_options):
                parser.error('registration requires model-binding/cache/observation-template/terminal-template')
            from jscc.activation_replay import file_digest
            def reference(path):
                path = Path(path).resolve()
                return {'path': str(path), 'sha256': file_digest(path)}
            result = register_recipe_assets(model_name=args.model_name,
                model_binding=reference(args.model_binding), cache=reference(args.cache),
                observation_template=reference(args.observation_template),
                terminal_template=reference(args.terminal_template), output=args.output, synthetic=args.synthetic_cpu)
        print(json.dumps({'identity': result['identity'], 'status': 'registered', 'launch_authorized': False}))
        return result
    if args.model_name or any(registration_options):
        parser.error('registration options require --register-assets or --register-from-prepared')
    if args.registrations:
        if (not args.config or args.acquire_model or args.model_config or args.assets
                or args.model_path or args.model_revision or args.synthetic_cpu):
            parser.error('automatic preparation requires config/registrations and a fresh output directory')
        from jscc.recipe_assets import prepare_recipe_assets
        result = prepare_recipe_assets(args.config, registrations=args.registrations, output=args.output)
        print(json.dumps({'prepared_identity': result['manifest_identity'], 'launch_authorized': False}))
        return result
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
