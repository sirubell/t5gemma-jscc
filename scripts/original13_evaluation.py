#!/usr/bin/env python3
"""Bind original13 terminal requests, check them, or render descriptive receipts.

Execution loads the source-bound prepared model locally and evaluates the exact
frozen terminal request. Supplemental panels require their own explicit request.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jscc.activation_replay import file_digest
from jscc.baseline_protocol import write_manifest
from jscc.encoder_stages import (initialize_original13, load_original13_prepared, original13_execution_guard,
                                verify_original13_execution_source)
from jscc.encoder_stages_plan import Original13Plan
from jscc.encoder_stages_evaluation import (
    _validate_request, open_original13_terminal, original13_report, terminal_original13_request,
    run_original13_evaluation,
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
    run = commands.add_parser('run')
    run.add_argument('--request', required=True)
    run.add_argument('--request-sha256', required=True)
    run.add_argument('--prepared', required=True)
    run.add_argument('--prepared-sha256', required=True)
    run.add_argument('--approved-source', required=True)
    run.add_argument('--timeout-seconds', type=float, required=True)
    run.add_argument('--device-uuid')
    run.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.command == 'run':
        request = _read(args.request, args.request_sha256)
        state, _ = open_original13_terminal(request['training_run'])
        _validate_request(state, request, observation=False)
        own_plan = Original13Plan.from_resolved(state.payload['metadata']['original13_plan'])
        manifest, config, plan, _, initial, metadata, _ = load_original13_prepared(args.prepared,
            checksum=args.prepared_sha256, width=own_plan.width, ratio=own_plan.ratio, load_batches=False)
        verify_original13_execution_source(manifest, args.approved_source)
        for key, expected in {**metadata, 'plan_identity': plan.identity}.items():
            if state.payload['metadata'].get(key) != expected:
                raise ValueError('original13 evaluation model/prepared identity differs: ' + key)
        guard = original13_execution_guard(config, synthetic=plan.synthetic,
            timeout_seconds=args.timeout_seconds, device_uuid=args.device_uuid)
        guard()
        processor, model = initialize_original13(config, initial, evaluation=True, synthetic=plan.synthetic)
        if config['model']['device'] == 'cuda':
            import torch
            if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
                raise ValueError('original13 evaluation requires one assigned BF16-capable GPU')
        guard()
        result = run_original13_evaluation(model, processor, request=request, output=args.output, guard=guard)
        print(json.dumps(result, indent=2))
        return result
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
