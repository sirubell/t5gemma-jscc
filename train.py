"""Train either task with uv:

uv run --locked python train.py --config configs/tasks/coco.yaml
uv run --locked python train.py --config configs/tasks/hellaswag.yaml
"""
import argparse

from jscc.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="last.pt from an interrupted run")
    parser.add_argument("--check", action="store_true", help="print resolved YAML, without loading a model")
    parser.add_argument("--run-path-file", help="write the completed run path for a following evaluation")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.check:
        import yaml
        print(yaml.safe_dump(config, sort_keys=False))
        return
    from jscc.training import train
    run = train(config, resume=args.resume)
    if args.run_path_file:
        from pathlib import Path
        Path(args.run_path_file).write_text(str(run) + "\n")


if __name__ == "__main__":
    main()
