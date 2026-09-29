"""Prepare/check or explicitly run the fixed COCO visited-trajectory diagnostic."""
import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jscc.coco_diagnostic import prepare_manifest, verify_plan
from jscc.coco_trajectory import candidate, run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--check-only", action="store_true")
    actions.add_argument("--run", action="store_true")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--checkpoint-index", type=Path)
    parser.add_argument("--endpoint-manifest", type=Path)
    parser.add_argument("--bindings", type=Path, help="Existing checkpoint/archive path bindings; prepare a fresh endpoint manifest")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--reserve-seconds", type=float, default=60)
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    plan = None
    if args.bindings is not None and not args.prepare:
        parser.error("--bindings is only valid with --prepare")
    if args.endpoint_manifest is not None or args.bindings is not None:
        if any(value is None for value in (args.selection, args.checkpoint_index, args.endpoint_manifest)):
            parser.error("Target identity verification requires selection, checkpoint-index and endpoint-manifest")
        kwargs: dict[str, Any] = dict(repo=repo, selection_path=args.selection, checkpoint_index_path=args.checkpoint_index,
                      manifest_path=args.endpoint_manifest)
        plan = (prepare_manifest(**kwargs, bindings_path=args.bindings) if args.bindings is not None
                else verify_plan(**kwargs))
    expected = candidate(repo, args.evidence)
    if plan is not None:
        expected["endpoint_manifest_sha256"] = plan["manifest_sha256"]
    if args.prepare:
        with args.manifest.open("x") as stream:
            json.dump(expected, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    elif json.loads(args.manifest.read_text()) != expected:
        parser.error("Manifest differs from current source, evidence, workload or runtime")
    if args.run:
        if any(value is None for value in (args.selection, args.checkpoint_index, args.endpoint_manifest, args.output, args.max_seconds)):
            parser.error("Run requires selection, checkpoint-index, endpoint-manifest, output, and max-seconds")
        assert plan is not None
        expected = run(plan, args.evidence, expected, args.output, args.max_seconds, args.reserve_seconds)
    print(json.dumps(expected, indent=2, sort_keys=True))
    return 0 if expected["status"] in ("complete", "prepared_not_executed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
