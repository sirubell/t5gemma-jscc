#!/usr/bin/env python3
"""Bind/check original13 inputs or run one explicit fresh/continuation segment."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jscc.activation_replay import file_digest
from jscc.encoder_stages import (initialize_original13, load_original13_prepared, original13_execution_guard,
                                run_original13_training, verify_original13_execution_source)
from jscc.encoder_stages_evaluation import _open_bound_reference
from jscc.encoder_stages_plan import Original13Plan
from jscc.encoder_stages_preparation import bind_original13_inputs
from jscc.experiment_state import CheckpointRef


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('bind', 'check', 'train', 'resume'))
    parser.add_argument('--inputs')
    parser.add_argument('--inputs-sha256')
    parser.add_argument('--model-path')  # Bind-time only; never mutate a saved training recipe.
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--prepared-sha256')
    parser.add_argument('--width', type=int)
    parser.add_argument('--ratio', type=int)
    parser.add_argument('--output')
    parser.add_argument('--approved-source')
    parser.add_argument('--timeout-seconds', type=float)
    parser.add_argument('--device-uuid')
    parser.add_argument('--resume-checkpoint')
    parser.add_argument('--resume-sha256')
    parser.add_argument('--resume-size', type=int)
    parser.add_argument('--recipe', help='YAML recipe; retained packages require exact compatibility')
    parser.add_argument('--resolved', help='Bound submission bundle; complete resolved recipe is checked')
    parser.add_argument('--resolved-identity')
    parser.add_argument('--trajectory', type=int)
    args = parser.parse_args(argv)
    if args.resolved:
        if args.recipe or not args.resolved_identity or args.trajectory is None or args.mode == 'bind':
            parser.error('--resolved requires identity/trajectory and cannot combine with YAML or bind')
    elif args.resolved_identity is not None or args.trajectory is not None:
        parser.error('resolved identity/trajectory require --resolved')
    if args.mode == 'bind':
        if not args.inputs or not args.inputs_sha256:
            parser.error('bind requires --inputs and --inputs-sha256')
        manifest = bind_original13_inputs(args.inputs, inputs_sha256=args.inputs_sha256,
                                          output=args.prepared, model_path=args.model_path)
    else:
        if args.model_path:
            parser.error('--model-path is bind-time only; execution uses the recorded model')
        if not args.prepared_sha256:
            parser.error('exact --prepared-sha256 is required')
        raw_prepared = json.loads(Path(args.prepared).read_text())
        if raw_prepared.get('schema') == 'recipe-prepared-v1':
            if args.mode != 'check':
                raise ValueError('generalized prepared train/resume retired; use train.py --config RECIPE --output ROOT')
            from jscc.recipe_workflow import training_cli
            result = training_cli(args)
            print(json.dumps(result, indent=2))
            return result
        if args.width is None or args.ratio is None:
            if args.mode != 'check':
                parser.error('train/resume require exact --width and --ratio')
            if file_digest(Path(args.prepared)) != args.prepared_sha256:
                raise ValueError('exact original13 prepared SHA required')
            raw = json.loads(Path(args.prepared).read_text())
            selected = Original13Plan.from_resolved(raw['plans'][0])
            args.width, args.ratio = selected.width, selected.ratio
        manifest, config, plan, batches, initial, metadata, data_ids = load_original13_prepared(
            args.prepared, checksum=args.prepared_sha256, width=args.width, ratio=args.ratio,
            load_batches=args.mode != 'check', allow_pending_assets=args.mode == 'check')
        if args.recipe:
            from jscc.experiment_recipe import resolve_recipe, validate_retained_recipe
            resolved = resolve_recipe(args.recipe, model_config_path=Path(config['model']['name'])/'config.json')
            matches = [item for item in resolved['trajectories']
                       if item['width'] == plan.width and item['ratio'] == plan.ratio]
            if len(matches) != 1:
                raise ValueError('recipe must declare the exact retained width and ratio')
            validate_retained_recipe(matches[0], plan, config, manifest['observation_request'])
        if args.resolved:
            from jscc.recipe_submission import validate_bundle
            from jscc.experiment_recipe import validate_retained_recipe
            from copy import deepcopy
            bundle = validate_bundle(json.loads(Path(args.resolved).read_text()))
            if bundle['identity'] != args.resolved_identity:
                raise ValueError('explicit resolved identity differs')
            if not 0 <= args.trajectory < len(bundle['resolved']['trajectories']):
                raise ValueError('resolved trajectory index differs')
            description = deepcopy(bundle['resolved']['trajectories'][args.trajectory])
            if (description['width'], description['ratio']) != (plan.width, plan.ratio):
                raise ValueError('resolved trajectory differs')
            if bundle['prepared']['sha256'] != args.prepared_sha256:
                raise ValueError('resolved prepared digest differs')
            if args.mode != 'check' and bundle['source']['commit'] != args.approved_source:
                raise ValueError('resolved runtime source differs')
            # Login/compute aliases name the same model; compare canonical paths.
            from jscc.recipe_submission import normalize_path
            if normalize_path(bundle['model']['path']) != normalize_path(config['model']['name']):
                raise ValueError('resolved model differs')
            description['config']['model']['name'] = config['model']['name']
            validate_retained_recipe(description, plan, config, manifest['observation_request'])
        if args.mode != 'check':
            if not args.output or not args.approved_source:
                parser.error('execution requires --output and --approved-source')
            verify_original13_execution_source(manifest, args.approved_source)
            guard = original13_execution_guard(config, synthetic=plan.synthetic,
                                               timeout_seconds=args.timeout_seconds, device_uuid=args.device_uuid)
            resume = None
            if args.mode == 'resume':
                if not args.resume_checkpoint or not args.resume_sha256 or not args.resume_size:
                    parser.error('resume requires an exact checkpoint path, SHA256 and size')
                resume = CheckpointRef(args.resume_checkpoint, args.resume_sha256, args.resume_size)
                state = _open_bound_reference(resume)
                for key, expected in {**metadata, 'plan_identity': plan.identity}.items():
                    if state.payload['metadata'].get(key) != expected:
                        raise ValueError('original13 resume identity differs: ' + key)
            elif args.resume_checkpoint or args.resume_sha256 or args.resume_size:
                parser.error('fresh train cannot take continuation fields')
            guard()
            processor, model = initialize_original13(config, initial, synthetic=plan.synthetic)
            if config['model']['device'] == 'cuda':
                import torch
                if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
                    raise ValueError('original13 requires one assigned BF16-capable GPU')
            guard()
            result = run_original13_training(model, processor, plan=plan, batches=batches, metadata=metadata,
                observation_request=manifest['observation_request'], output=args.output, guard=guard, resume=resume,
                config=config, data_ids=data_ids)
            print(json.dumps(result, indent=2))
            return result
    plans = [Original13Plan.from_resolved(row) for row in manifest['plans']]
    print(json.dumps({'plans': [{'width': p.width, 'ratio': p.ratio, 'horizon': p.horizon,
                                'plan_identity': p.identity} for p in plans], 'launch_authorized': False,
                      'model_assets_status': next(iter(manifest['configs'].values()))['model']['asset_binding']['status']}, indent=2))
    return manifest


if __name__ == '__main__':
    main()
