#!/usr/bin/env python3
"""Original13 training-owned preparation and CPU contract inspection CLI.

Training enters run_original13_training only after routing/evaluation consumer
integration. This CLI currently rejects training rather than loading a model
through an unimplemented or legacy encoder-only path.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jscc.activation_replay import canonical_digest, file_digest
from jscc.encoder_stages import stages_source_inventory
from jscc.encoder_stages_plan import Original13Plan
from jscc.encoder_stages_preparation import bind_original13_inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('bind', 'check'))
    parser.add_argument('--inputs')
    parser.add_argument('--inputs-sha256')
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--prepared-sha256')
    args = parser.parse_args(argv)
    if args.mode == 'bind':
        if not args.inputs or not args.inputs_sha256:
            parser.error('bind requires --inputs and --inputs-sha256')
        manifest = bind_original13_inputs(args.inputs, inputs_sha256=args.inputs_sha256, output=args.prepared)
    else:
        if not args.prepared_sha256 or file_digest(Path(args.prepared)) != args.prepared_sha256:
            raise ValueError('exact original13 prepared SHA required')
        manifest = json.loads(Path(args.prepared).read_text())
        if (manifest.get('schema') != 'original13-prepared-v1'
                or manifest.get('manifest_identity') != canonical_digest(
                    {k: v for k, v in manifest.items() if k != 'manifest_identity'})
                or manifest['source_inventory'] != stages_source_inventory()):
            raise ValueError('original13 prepared source/manifest differs')
    plans = [Original13Plan.from_resolved(row) for row in manifest['plans']]
    print(json.dumps({'plans': [{'width': p.width, 'ratio': p.ratio, 'horizon': p.horizon,
                                'plan_identity': p.identity} for p in plans],
                      'launch_authorized': False}, indent=2))


if __name__ == '__main__':
    main()
