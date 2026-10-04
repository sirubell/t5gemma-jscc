#!/usr/bin/env python3
"""Prepare/check/render/submit a YAML recipe job; real submission is explicit.

Prepare: --job job.yaml --limits approved-limits.yaml --output fresh/bundle.json
Check/render: --bundle fresh/bundle.json [--output fresh/job.sbatch]
Mock submit: --bundle bundle.json --mock-results scheduler-results.json
Later authorized live: --bundle bundle.json --live --authorization approval.json
Both submission paths use the same bundle, payload, and training entry point.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jscc.recipe_submission import (prepare_job, render_payload, run_job, submit_job,
                                    validate_bundle, validate_prepared)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'check', 'render', 'submit', 'run'))
    parser.add_argument('--job')
    parser.add_argument('--limits')
    parser.add_argument('--bundle')
    parser.add_argument('--output')
    parser.add_argument('--identity')
    submission = parser.add_mutually_exclusive_group()
    submission.add_argument('--mock-results')
    submission.add_argument('--live', action='store_true')
    parser.add_argument('--authorization')
    args = parser.parse_args(argv)
    if args.mode != 'submit' and (args.live or args.mock_results or args.authorization):
        parser.error('scheduler options require submit mode')
    if args.mode == 'prepare':
        if not args.job or not args.limits or not args.output:
            parser.error('prepare requires --job --limits --output')
        result = prepare_job(args.job, args.limits)
        with Path(args.output).open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    else:
        if not args.bundle:
            parser.error('mode requires --bundle')
        bundle = validate_bundle(json.loads(Path(args.bundle).read_text()))
        if args.mode in ('check', 'render', 'submit'):
            validate_prepared(bundle)
        if args.mode == 'render':
            if not args.output:
                parser.error('render requires fresh --output')
            with Path(args.output).open('x') as stream:
                stream.write(render_payload(bundle, args.bundle))
            result = {'payload': args.output, 'bundle_identity': bundle['identity'], 'launch_authorized': False}
        elif args.mode == 'check':
            result = {'bundle_identity': bundle['identity'], 'trajectories': len(bundle['resolved']['trajectories']),
                      'launch_authorized': False, 'runtime_acceptance': 'requires authorized runtime check'}
        elif args.mode == 'submit':
            if args.mock_results:
                if args.authorization:
                    parser.error('mock submission cannot take live authorization')
                responses = iter(json.loads(Path(args.mock_results).read_text()))

                def scheduler(command):
                    row = next(responses)
                    return subprocess.CompletedProcess(command, row['returncode'], row['stdout'], row['stderr'])

                result = submit_job(bundle, bundle['output'], scheduler=scheduler)
            elif args.live and args.authorization:
                approval = json.loads(Path(args.authorization).read_text())
                result = submit_job(bundle, bundle['output'], authorization=approval)
            else:
                parser.error('submit requires --mock-results or --live --authorization')
        else:
            if not args.identity:
                parser.error('run requires explicit --identity')
            result = run_job(bundle, args.identity)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    main()
