"""Single entry point.

    python experiment.py train CONFIG.yaml [--name NAME]     # new run under run.root
    python experiment.py train --resume RUN_DIR              # continue from RUN_DIR/last.pt
    python experiment.py evaluate RUN_DIR [--checkpoint best|last|PATH]

``train`` evaluates the best checkpoint afterwards when ``evaluation.after_training`` is true.
Every stage writes RUN_DIR/summary.json + SUMMARY.md and one line in <run.root>/INDEX.md.
"""
import argparse
import signal
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent


def _interrupt(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


def _load_model(config):
    from jscc.model import build_model
    import os
    return build_model(config, local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1")


def _evaluate(run, config, checkpoint, processor=None, model=None, data=None):
    import torch
    from jscc.evaluate import evaluate
    from jscc.report import write_summary
    if model is None:
        processor, model = _load_model(config)
    if config["task"] == "coco" and data is None:
        from jscc.tasks import CocoData
        data = CocoData(config, processor)
    path = Path(checkpoint)
    if not path.is_absolute() and not path.exists():
        path = run / ({"best": "best_distillation.pt", "last": "last.pt"}.get(str(checkpoint), str(checkpoint)))
    model.load_communication_state(torch.load(path, map_location="cpu", weights_only=False)["communication"])
    sites = config["evaluation"].get("sites") or config["training"]["sites"]
    results = evaluate(model, processor, config, sites, run / "eval" / path.stem, data=data)
    old = {}
    import json
    if (run / "summary.json").exists():
        old = json.loads((run / "summary.json").read_text()).get("evaluations", {})
    write_summary(run, config, "evaluated", evaluations={**old, path.stem: results})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("config", nargs="?")
    t.add_argument("--name")
    t.add_argument("--resume", type=Path)
    e = sub.add_parser("evaluate")
    e.add_argument("run", type=Path)
    e.add_argument("--checkpoint", default="best")
    args = parser.parse_args(argv)
    signal.signal(signal.SIGTERM, _interrupt)  # Slurm time limit -> summary says "interrupted"

    from jscc.config import load_config, save_config
    from jscc.report import append_index, source_state, write_summary

    if args.command == "evaluate":
        run = args.run.resolve()
        config = load_config(run / "config.yaml")
        stage = "evaluate"
    elif args.resume:
        run = args.resume.resolve()
        config = load_config(run / "config.yaml")
        stage = "train"
    else:
        config = load_config(args.config)
        name = args.name or f"{config['run']['name']}-{time.strftime('%Y%m%d-%H%M%S')}"
        run = Path(config["run"]["root"]).resolve() / name
        run.mkdir(parents=True, exist_ok=False)
        config["run"]["root"] = str(Path(config["run"]["root"]).resolve())
        save_config(config, run / "config.yaml")
        stage = "train"
    source = source_state(REPO)
    write_summary(run, config, f"{stage}_running", source=source)
    append_index(run, config, stage, "started", f"{source['commit'][:10]}{' dirty' if source['dirty'] else ''}")
    try:
        if stage == "evaluate":
            results = _evaluate(run, config, args.checkpoint)
        else:
            import torch
            from jscc.tasks import load_task_data
            from jscc.train import Trainer
            torch.manual_seed(config["seed"])
            processor, model = _load_model(config)
            data = load_task_data(config, processor)
            trainer = Trainer(model, data, config, run)
            if args.resume:
                trainer.load(run / "last.pt")
            stops = trainer.train()
            write_summary(run, config, "trained", stops=stops)
            append_index(run, config, "train", "done", ", ".join(f"{k}: {v}" for k, v in stops.items()))
            results = None
            if config["evaluation"].get("after_training", True):
                stage = "evaluate"
                append_index(run, config, stage, "started", "best checkpoint")
                results = _evaluate(run, config, "best", processor, model, data if config["task"] == "coco" else None)
        if results is not None:
            from jscc.report import score_table
            append_index(run, config, "evaluate", "done", f"see {run.name}/SUMMARY.md")
            print(score_table(results, config["task"]))
        return 0
    except BaseException as exc:
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        write_summary(run, config, status, error="".join(traceback.format_exception(exc))[-4000:])
        append_index(run, config, stage, status, f"{type(exc).__name__}: {str(exc)[:120]}")
        raise


if __name__ == "__main__":
    sys.exit(main())
