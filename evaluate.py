"""Evaluate either task with uv; the run checkpoint identifies the task:

uv run --locked python evaluate.py --run runs/<coco-run>
uv run --locked python evaluate.py --run runs/<hellaswag-run>
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint", default="best.pt", help="checkpoint filename within the run")
    parser.add_argument("--config", help="optional YAML containing evaluation options only")
    args = parser.parse_args()
    from jscc.evaluation import evaluate
    evaluate(args.run, args.checkpoint, args.config)


if __name__ == "__main__":
    main()
