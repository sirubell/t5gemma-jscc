"""Single entry point.

    python experiment.py train CONFIG.yaml [--name NAME]     # new run under run.root
    python experiment.py train --resume RUN_DIR              # continue from RUN_DIR/last.pt
    python experiment.py evaluate RUN_DIR [--checkpoint best|early_stop|last|PATH]
    python experiment.py evaluate RUN_DIR --conditions -10 0 6 --out DIR   # extra panels into DIR only

``train`` evaluates ``evaluation.checkpoints`` (default ``[best]``) afterwards when ``evaluation.after_training``
is true. Checkpoint names: best (best_distillation.pt), best_normalized (best_distillation_normalized.pt),
last (last.pt, the final weights), or a path.
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


def _evaluate(run, config, checkpoint, processor=None, model=None, data=None, out=None):
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
        path = run / ({"best": "best_distillation.pt", "best_normalized": "best_distillation_normalized.pt",
                      "early_stop": "best_distillation_early_stop.pt", "last": "last.pt"}.get(str(checkpoint), str(checkpoint)))
    model.load_communication_state(torch.load(path, map_location="cpu", weights_only=False)["communication"])
    sites = config["evaluation"].get("sites") or config["training"]["sites"]
    results = evaluate(model, processor, config, sites, out or run / "eval" / path.stem, data=data)
    if out is not None:
        return results
    old = {}
    import json
    if (run / "summary.json").exists():
        old = json.loads((run / "summary.json").read_text()).get("evaluations", {})
    write_summary(run, config, "evaluated", evaluations={**old, path.stem: results})
    return results


def _condition(text):
    """A condition from the command line: vanilla, no_noise, or an SNR in dB (int when integral)."""
    if text in ("vanilla", "no_noise"):
        return text
    value = float(text)
    return int(value) if value.is_integer() else value


def _evaluate_elsewhere(args):
    """``evaluate --out``: score a run's checkpoint into another directory; the run dir is not written."""
    from jscc.config import load_config, save_config
    from jscc.report import score_table, source_state
    run, out = args.run.resolve(), args.out.resolve()
    config = load_config(run / "config.yaml")
    if args.conditions:
        config["evaluation"]["conditions"] = [_condition(c) for c in args.conditions]
    out.mkdir(parents=True, exist_ok=True)
    save_config(config, out / "config.yaml")
    import json
    (out / "source.json").write_text(json.dumps({"run": str(run), "checkpoint": args.checkpoint,
                                                 "source": source_state(REPO)}, indent=1) + "\n")
    print(score_table(_evaluate(run, config, args.checkpoint, out=out), config["task"]))
    return 0


def _report(run, config, results):
    from jscc.report import append_index, score_table
    append_index(run, config, "evaluate", "done", f"see {run.name}/SUMMARY.md")
    print(score_table(results, config["task"]))


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
    e.add_argument("--conditions", nargs="+", help="replace evaluation.conditions (needs --out)")
    e.add_argument("--out", type=Path, help="write results.json/items.npz here and leave the run dir untouched")
    args = parser.parse_args(argv)
    if args.command == "evaluate" and args.conditions and not args.out:
        parser.error("--conditions needs --out")
    # Slurm time limit -> summary says "interrupted". During training the trainer handles SIGTERM itself:
    # it saves last.pt at the next safe point, then raises KeyboardInterrupt.
    signal.signal(signal.SIGTERM, _interrupt)

    from jscc.config import load_config, save_config
    from jscc.report import append_index, source_state, write_summary

    if args.command == "evaluate" and args.out:
        return _evaluate_elsewhere(args)
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
            _report(run, config, _evaluate(run, config, args.checkpoint))
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
            if config["evaluation"].get("after_training", True):
                stage = "evaluate"
                for checkpoint in config["evaluation"].get("checkpoints", ["best"]):
                    if trainer.stop_requested:  # SIGTERM arrived while the last phase was finishing
                        raise KeyboardInterrupt("SIGTERM before evaluation")
                    append_index(run, config, stage, "started", f"{checkpoint} checkpoint")
                    _report(run, config, _evaluate(run, config, checkpoint, processor, model,
                                                   data if config["task"] == "coco" else None))
        return 0
    except BaseException as exc:
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        write_summary(run, config, status, error="".join(traceback.format_exception(exc))[-4000:])
        append_index(run, config, stage, status, f"{type(exc).__name__}: {str(exc)[:120]}")
        raise


if __name__ == "__main__":
    sys.exit(main())
