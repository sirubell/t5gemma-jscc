"""Evaluate either task with uv; the run checkpoint identifies the task:

uv run --locked python evaluate.py --run runs/<coco-run>
uv run --locked python evaluate.py --run runs/<hellaswag-run>
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint", help="checkpoint filename within the run")
    parser.add_argument("--config", help="optional YAML containing evaluation options only")
    parser.add_argument("--expected-step", type=int, help="require this exact saved optimizer step")
    args = parser.parse_args()
    if args.expected_step is not None and not args.checkpoint:
        parser.error("expected-step requires an explicit checkpoint")
    from jscc.evaluation import evaluate
    evaluate(args.run, args.checkpoint, args.config, expected_step=args.expected_step)


if __name__ == "__main__":
    main()
