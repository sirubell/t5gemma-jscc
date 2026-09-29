"""Explicit local COCO query/endpoint diagnostic; never submits a job."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jscc.coco_diagnostic import prepare_manifest, run, verify_plan


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    action = value.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true", help="Create and verify a local identity manifest without models")
    action.add_argument("--check-only", action="store_true", help="Verify files and identities without loading models or data")
    action.add_argument("--run", action="store_true", help="Execute the bounded offline diagnostic")
    value.add_argument("--selection", type=Path, required=True)
    value.add_argument("--checkpoint-index", type=Path, required=True)
    value.add_argument("--manifest", type=Path, required=True,
                       help="Verified identity manifest (new output path with --prepare)")
    value.add_argument("--bindings", type=Path,
                       help="Input paths JSON required with --prepare")
    value.add_argument("--output", type=Path, help="New directory for append-only run evidence")
    value.add_argument("--max-seconds", type=float, help="Required hard wall-clock budget in run mode")
    value.add_argument("--reserve-seconds", type=float, default=60,
                       help="Time reserved for flushing; no requests start in this interval")
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    if args.prepare and args.bindings is None:
        parser().error("--prepare requires --bindings")
    if args.run and (args.output is None or args.max_seconds is None):
        parser().error("--run requires --output and --max-seconds")
    repo = Path(__file__).resolve().parents[1]
    if args.prepare:
        plan = prepare_manifest(repo=repo, selection_path=args.selection,
                                checkpoint_index_path=args.checkpoint_index,
                                bindings_path=args.bindings, manifest_path=args.manifest)
    else:
        plan = verify_plan(repo=repo, selection_path=args.selection,
                           checkpoint_index_path=args.checkpoint_index, manifest_path=args.manifest)
    if args.prepare or args.check_only:
        summary = {"status": "identities_verified_not_executed", "logical_requests": 1024,
                   "included_timing_requests": 128, "model_loaded": False,
                   "manifest_sha256": plan["manifest_sha256"],
                   "selection_sha256": plan["selection_sha256"]}
    else:
        summary = run(plan, args.output, args.max_seconds, args.reserve_seconds)
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0 if summary["status"] in ("identities_verified_not_executed", "complete") else 2


if __name__ == "__main__":
    raise SystemExit(main())
