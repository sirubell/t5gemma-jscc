"""Train either task with uv:

uv run --locked python train.py --config configs/tasks/coco.yaml
uv run --locked python train.py --config configs/tasks/hellaswag.yaml
"""
import argparse
from pathlib import Path

import yaml

from jscc.config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", help="exact logical experiment root for a task or recipe")
    parser.add_argument("--resume", help="exact saved checkpoint from an interrupted run")
    parser.add_argument("--check", action="store_true", help="print resolved YAML, without loading a model")
    parser.add_argument("--run-path-file", help="write the completed run path for a following evaluation")
    parser.add_argument("--case", help="recipe case, e.g. width-32-ratio-50")
    parser.add_argument("--attempt", default="0001", help="explicit recipe training attempt name")
    parser.add_argument("--registrations", help="directory of explicitly acquired pinned recipe asset registrations")
    parser.add_argument("--prepare-only", action="store_true", help="bind a fresh recipe root without executing cases")
    parser.add_argument("--timeout-seconds", type=float, default=86400, help="per-case recipe execution deadline")
    parser.add_argument("--device-uuid", help="optional exact CUDA_VISIBLE_DEVICES assignment for a recipe case")
    args = parser.parse_args(argv)
    bound_recipe = args.output and (Path(args.output)/"recipe-run.json").is_file()
    authored = None if bound_recipe else yaml.safe_load(Path(args.config).read_text())
    if bound_recipe or (isinstance(authored, dict) and authored.get("schema") == "shared-codec-recipe-v1"):
        from jscc.recipe_runs import train_recipe, recipe_run_record
        if not args.output and not args.check:
            parser.error("recipe requires --output at an exact logical root")
        if args.prepare_only and args.resume:
            parser.error("recipe preparation cannot resume training")
        if args.check and (args.resume or args.prepare_only):
            parser.error("recipe --check cannot resume or prepare a run")
        if args.check and bound_recipe:
            record = recipe_run_record(args.output)
            if args.case and args.case not in [row['case_id'] for row in record['cases']]:
                raise ValueError('unknown recipe case')
            print(yaml.safe_dump(record, sort_keys=False))
            return record
        if args.check:
            from jscc.recipe_assets import discover_recipe_assets
            from jscc.recipe_workflow import _read_reference
            from jscc.asset_paths import asset_file_path
            from jscc.experiment_recipe import resolve_recipe
            import json
            if not args.registrations:
                parser.error("recipe --check requires --registrations for pinned model topology")
            acquired = discover_recipe_assets(args.config, args.registrations)
            assets = json.loads(_read_reference(acquired['model_binding']).read_text())
            resolved = resolve_recipe(args.config, model_config_path=asset_file_path(assets['path'], 'config.json'))
            from jscc.recipe_runs import case_id
            if args.case and args.case not in [case_id(item) for item in resolved['trajectories']]:
                raise ValueError('unknown recipe case')
            print(yaml.safe_dump(resolved, sort_keys=False))
            return resolved
        run = train_recipe(args.config, args.output, registrations=args.registrations, selected=args.case,
            attempt_id=args.attempt, resume=args.resume, prepare_only=args.prepare_only,
            timeout_seconds=args.timeout_seconds, device_uuid=args.device_uuid)
        if args.run_path_file:
            Path(args.run_path_file).parent.mkdir(parents=True, exist_ok=True)
            Path(args.run_path_file).write_text(str(run)+"\n")
        return recipe_run_record(run)
    if (args.case or args.attempt != "0001" or args.registrations or args.prepare_only or args.device_uuid
            or args.timeout_seconds != 86400):
        parser.error("recipe execution options require a shared-codec recipe")
    config = load_config(args.config)
    if args.check:
        print(yaml.safe_dump(config, sort_keys=False))
        return
    if args.run_path_file:
        Path(args.run_path_file).parent.mkdir(parents=True, exist_ok=True)
    from jscc.training import train
    if args.output:
        from jscc.ordinary_runs import train_task
        run = train_task(config, args.output, authored_config=args.config, resume=args.resume)
    else:
        run = train(config, resume=args.resume)
    if args.run_path_file:
        Path(args.run_path_file).write_text(str(run) + "\n")


if __name__ == "__main__":
    main()
