"""Evaluate either task with uv; the run checkpoint identifies the task:

uv run --locked python evaluate.py --run runs/<coco-run>
uv run --locked python evaluate.py --run runs/<hellaswag-run>
"""
import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--attempt", default="0001", help="explicit fresh evaluation attempt ID for a logical root")
    parser.add_argument("--checkpoint", help="checkpoint filename within the run")
    parser.add_argument("--config", help="optional YAML containing evaluation options only")
    parser.add_argument("--expected-step", type=int, help="require this exact saved optimizer step")
    parser.add_argument("--case", help="one declared recipe case (default: all declared cases)")
    parser.add_argument("--scope", choices=("initial", "supplemental", "development"), default="initial",
                        help="recipe policy scope; development requires an exact configured observation checkpoint")
    parser.add_argument("--timeout-seconds", type=float, default=86400, help="finite recipe evaluation timeout")
    parser.add_argument("--device-uuid", help="explicit recipe GPU assignment, when supplied")
    parser.add_argument("--output-path-file", help="write completed evaluation directory for a prepared study")
    args = parser.parse_args(argv)
    if args.output_path_file:
        if Path(args.output_path_file).exists():
            parser.error("evaluation output link already exists")
    if args.expected_step is not None and not args.checkpoint:
        parser.error("expected-step requires an explicit checkpoint")
    from jscc.evaluation import evaluate
    from jscc.ordinary_runs import evaluate_run, is_recorded_root
    from jscc.recipe_evaluation import evaluate_recipe_run, is_recipe_run
    recipe = is_recipe_run(args.run)
    if recipe:
        output = evaluate_recipe_run(args.run, selected=args.case, attempt_id=args.attempt,
            checkpoint=args.checkpoint, expected_step=args.expected_step, scope=args.scope,
            overrides_path=args.config, timeout_seconds=args.timeout_seconds, device_uuid=args.device_uuid)
    elif args.case is not None or args.scope != "initial":
        parser.error("case/scope selectors require a recorded recipe run")
    elif is_recorded_root(args.run):
        output = evaluate_run(args.run, args.checkpoint, args.config,
                              expected_step=args.expected_step, attempt_id=args.attempt)
    else:
        output = evaluate(args.run, args.checkpoint, args.config, expected_step=args.expected_step)
    if args.output_path_file:
        Path(args.output_path_file).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.output_path_file).open("x") as handle:
            handle.write(str(output) + "\n")
    if recipe:
        import json
        result = json.loads((output / "results.json").read_text())
        print(json.dumps({"output": str(output), "status": result["status"], "counts": result["counts"]}))
        if result["status"] != "complete":
            raise SystemExit(1)
    return output


if __name__ == "__main__":
    main()
