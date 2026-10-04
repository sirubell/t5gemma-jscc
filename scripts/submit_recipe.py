#!/usr/bin/env python3
"""Prepare/check/render/submit a YAML recipe job; real submission is explicit.

One reviewed case graph: action --job job.yaml --limits approved-limits.yaml
    --output fresh/bundle.json [--mock-results scheduler-results.json]
Prepare: --job job.yaml --limits approved-limits.yaml --output fresh/bundle.json
Check/render: --bundle fresh/bundle.json [--output fresh/job.sbatch]
Mock submit: --bundle bundle.json --mock-results scheduler-results.json
Later authorized live: --bundle bundle.json --live --authorization approval.json
Both submission paths use the same bundle, payload, and training entry point.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jscc.recipe_submission import (prepare_job, preview_job, render_payload, run_job, submit_job,
                                    submission_preflight,
                                    validate_bundle, write_json)


def _submit(bundle, args):
    if args.mock_results:
        if args.authorization:
            raise ValueError('mock submission cannot take live authorization')
        responses = iter(json.loads(Path(args.mock_results).read_text()))

        def scheduler(command):
            row = next(responses)
            return subprocess.CompletedProcess(command, row['returncode'], row['stdout'], row['stderr'])

        return submit_job(bundle, bundle['output'], scheduler=scheduler)
    if args.live and args.authorization:
        approval = json.loads(Path(args.authorization).read_text())
        return submit_job(bundle, bundle['output'], authorization=approval)
    raise ValueError('submission requires --mock-results or --live --authorization')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('action', 'prepare', 'preview', 'check', 'render', 'submit', 'run'))
    parser.add_argument('--job')
    parser.add_argument('--limits')
    parser.add_argument('--bundle')
    parser.add_argument('--output')
    parser.add_argument('--identity')
    parser.add_argument('--group', type=int, help='exact reviewed case-array group at launch')
    submission = parser.add_mutually_exclusive_group()
    submission.add_argument('--mock-results')
    submission.add_argument('--live', action='store_true')
    parser.add_argument('--authorization')
    args = parser.parse_args(argv)
    if args.mode not in ('action', 'submit') and (args.live or args.mock_results or args.authorization):
        parser.error('scheduler options require action or submit mode')
    if args.mode in ('prepare', 'action'):
        if not args.job or not args.limits or not args.output:
            parser.error('preparation requires --job --limits --output')
        result = prepare_job(args.job, args.limits)
        with Path(args.output).open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
        if args.mode == 'action':
            preview = preview_job(result, args.output)
            print(json.dumps(preview, indent=2), flush=True)
            if args.mock_results or args.live or args.authorization:
                result = _submit(result, args)
            else:
                result = preview
    else:
        if not args.bundle:
            parser.error('mode requires --bundle')
        # A launch may fail to parse/validate its bound bundle. Its path is
        # supplied by the reviewed payload, so record failure beside that path
        # without trusting an output path inside a mutated bundle.
        try:
            bundle = validate_bundle(json.loads(Path(args.bundle).read_text()))
            if args.mode == 'run' and args.identity and bundle['identity'] != args.identity:
                raise ValueError('explicit bundle identity differs')
        except (ValueError, KeyError, OSError) as error:
            if args.mode in ('run', 'submit'):
                task = os.environ.get('SLURM_ARRAY_TASK_ID', 'unknown')
                task = task if task.isdigit() else 'unknown'
                group = str(args.group) if args.group is not None and args.group >= 0 else 'unknown'
                if args.mode == 'run':
                    name = (f'group-{group}-task-{task}.bundle-failure.json' if args.group is not None
                            else f'trajectory-{task}.bundle-failure.json')
                else:
                    name = 'submission.bundle-failure.json'
                marker = Path(args.bundle).parent/name
                with marker.open('x'):
                    pass
                write_json(marker, {'status': 'failed', 'stage': 'bundle-preflight',
                                    'identity': args.identity, 'bundle_path': str(Path(args.bundle).resolve()),
                                    'failure': {'type': type(error).__name__, 'message': str(error)}})
            raise
        if args.mode in ('check', 'render'):
            submission_preflight(bundle)
        if args.mode == 'preview':
            result = preview_job(bundle, args.bundle)
        elif args.mode == 'render':
            if not args.output:
                parser.error('render requires fresh --output')
            with Path(args.output).open('x') as stream:
                stream.write(render_payload(bundle, args.bundle))
            result = {'payload': args.output, 'bundle_identity': bundle['identity'], 'launch_authorized': False}
        elif args.mode == 'check':
            result = {'bundle_identity': bundle['identity'], 'launch_authorized': False,
                      'runtime_acceptance': 'requires authorized runtime check'}
            if bundle['schema'] == 'recipe-case-submission-v1':
                result.update(selected_cases=sum('training' in case['commands'] for case in bundle['cases']),
                              declared_cases=len(bundle['cases']))
            else:
                result['trajectories'] = len(bundle['resolved']['trajectories'])
        elif args.mode == 'submit':
            result = _submit(bundle, args)
        else:
            if not args.identity:
                parser.error('run requires explicit --identity')
            if bundle['schema'] == 'recipe-case-submission-v1':
                if args.group is None:
                    parser.error('case-array run requires explicit --group')
                result = run_job(bundle, args.identity, args.group)
            else:
                result = run_job(bundle, args.identity)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    main()
