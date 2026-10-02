#!/usr/bin/env python3
"""Exact-source, per-stage-timeout entry for the approved original-six pilot."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jscc.activation_replay import file_digest
from jscc.encoder_shared_pilot import (initialize, load_pilot, metadata_for, open_terminal,
                                      readiness, readiness_view_indices, run_evaluation, run_training)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'readiness', 'train', 'evaluate'))
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--prepared-sha', required=True)
    parser.add_argument('--approved-source', required=True)
    parser.add_argument('--output')
    parser.add_argument('--training-run')
    parser.add_argument('--readiness-run')
    parser.add_argument('--timeout-seconds', type=int)
    args = parser.parse_args(argv)
    if file_digest(Path(args.prepared)) != args.prepared_sha:
        raise ValueError('exact reviewed pilot manifest SHA required')
    source = Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip() != args.approved_source:
        raise ValueError('exact approved pilot source commit required')
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--'], cwd=source, check=True)
    started = time.monotonic()
    limits = {'readiness': 300, 'train': 21600, 'evaluate': 3600}
    if args.mode != 'check' and (not args.output or type(args.timeout_seconds) is not int
                                or not 0 < args.timeout_seconds <= limits[args.mode]):
        raise ValueError('explicit fresh output and approved per-stage timeout required')

    def guard():
        if args.mode != 'check' and time.monotonic() - started >= args.timeout_seconds:
            raise TimeoutError('pilot per-stage timeout; partial evidence retained, no retry')

    guard()
    manifest, config, plan, batches, validation, template, initial = load_pilot(args.prepared)
    if args.mode == 'train' and plan.q != 6400 and args.timeout_seconds > 14400:
        raise ValueError('short pilot retains its four-hour training limit')
    tracked = set(subprocess.check_output(['git', 'ls-files'], cwd=source, text=True).splitlines())
    if not set(manifest['source_inventory']) <= tracked:
        raise ValueError('pilot execution inventory must belong to approved Git commit')
    metadata = metadata_for(manifest)
    guard()
    if args.mode == 'check':
        result = {'status': 'CPU_verified', 'prepared_identity': manifest['manifest_identity'],
                  'q_per_site': plan.q, 'horizon': plan.horizon, 'GPU_launched': False}
    else:
        if manifest['synthetic_cpu']:
            raise ValueError('GPU entry requires real fixed pilot inputs')
        if os.environ.get('CUDA_VISIBLE_DEVICES') != manifest['device_uuid']:
            raise ValueError('pilot GPU assignment differs from prepared UUID')
        if args.mode == 'train':
            if not args.readiness_run:
                raise ValueError('training requires the exact completed six-site readiness result')
            receipt = json.loads((Path(args.readiness_run) / 'result.json').read_text())
            if (receipt.get('status') != 'pass' or receipt.get('prepared_identity') != manifest['manifest_identity']
                    or receipt.get('source_commit') != args.approved_source
                    or receipt.get('device_uuid') != manifest['device_uuid']):
                raise ValueError('readiness result is not complete for this exact source/package/GPU')
            expected_views = readiness_view_indices(batches)
            from jscc.encoder_pilot_plan import SITES
            if (receipt.get('scientific_updates') != 0
                    or {cycle['view'] for cycle in receipt['cycles']} != expected_views
                    or len(receipt['cycles']) != len(expected_views)
                    or any(not cycle['fresh_state_restored_exact']
                           or [row['site'] for row in cycle['sites']] != list(SITES)
                           for cycle in receipt['cycles'])):
                raise ValueError('readiness did not restore both worst-layout six-site cycles')
        state = freeze = None
        if args.mode == 'evaluate':
            if not args.training_run:
                raise ValueError('exact own training-run path required')
            state, freeze = open_terminal(args.training_run, metadata, plan)
        guard()
        processor, model = initialize(config, initial)
        import torch
        if torch.cuda.device_count() != 1 or '5090' not in torch.cuda.get_device_name(0):
            raise ValueError('assigned pilot device is not one RTX 5090')
        torch.set_num_threads(1)
        guard()
        if args.mode == 'readiness':
            result = readiness(model, plan=plan, batches=batches, metadata=metadata,
                               output=args.output, guard=guard)
            # The scientific train gate binds receipt bytes to this invocation.
            result.update(prepared_identity=manifest['manifest_identity'], source_commit=args.approved_source,
                          device_uuid=manifest['device_uuid'])
            from jscc.baseline_protocol import write_manifest
            guard()
            write_manifest(Path(args.output) / 'result.json', result)
        elif args.mode == 'train':
            result = run_training(model, plan=plan, batches=batches, validation=validation,
                                  metadata=metadata, output=args.output, guard=guard)
        else:
            result = run_evaluation(model, processor, state=state, freeze_reference=freeze,
                                    template=template, output=args.output, guard=guard)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


if __name__ == '__main__':
    main()
