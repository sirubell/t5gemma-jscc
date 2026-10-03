#!/usr/bin/env python3
"""Bind original13 terminal requests, check them, or render descriptive receipts.

Execution uses run_original13_evaluation with an already constructed canonical
model/processor. Canonical model construction belongs to the integration owner.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jscc.activation_replay import file_digest
from jscc.baseline_protocol import write_manifest
from jscc.encoder_stages_evaluation import (
    _validate_request, open_original13_terminal, original13_report, terminal_original13_request,
)


def _read(path, checksum):
    if file_digest(Path(path)) != checksum:
        raise ValueError('explicit input file checksum differs')
    return json.loads(Path(path).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    bind = commands.add_parser('request')
    bind.add_argument('--training-run', required=True)
    bind.add_argument('--template', required=True)
    bind.add_argument('--template-sha256', required=True)
    bind.add_argument('--scope', choices=['initial', 'supplemental'], default='initial')
    bind.add_argument('--output', required=True)
    check = commands.add_parser('check')
    check.add_argument('--request', required=True)
    check.add_argument('--request-sha256', required=True)
    report = commands.add_parser('report')
    report.add_argument('--training-run', required=True)
    report.add_argument('--initial')
    report.add_argument('--supplemental')
    report.add_argument('--initial-request')
    report.add_argument('--supplemental-request')
    report.add_argument('--observation', action='append', default=[])
    report.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.command == 'check':
        request = _read(args.request, args.request_sha256)
        state, _ = open_original13_terminal(request['training_run'])
        _validate_request(state, request, observation=False)
        print(json.dumps({'status': 'valid', 'checkpoint': state.reference.sha256}))
        return
    if args.command == 'request':
        value = terminal_original13_request(args.training_run,
            _read(args.template, args.template_sha256), scope=args.scope)
    else:
        value = original13_report(args.training_run, initial=args.initial,
            supplemental=args.supplemental, observations=args.observation,
            initial_request=json.loads(Path(args.initial_request).read_text()) if args.initial_request else None,
            supplemental_request=json.loads(Path(args.supplemental_request).read_text()) if args.supplemental_request else None)
    # Reserve the output before publishing; no clobbering prior requests/reports.
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x'):
        pass
    write_manifest(path, value)
    print(json.dumps({'path': str(path.resolve()), 'sha256': file_digest(path)}))


if __name__ == '__main__':
    main()
