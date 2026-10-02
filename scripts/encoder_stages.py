#!/usr/bin/env python3
"""Pinned single-GPU shared A/E/R lifecycle; no scheduler or dispatch framework."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jscc.activation_replay import canonical_digest, file_digest
from jscc.baseline_protocol import write_manifest
from jscc.encoder_shared_pilot import initialize
from jscc.encoder_stages import (load_stages, open_stages_terminal, qualify_stages,
                                run_stages_evaluation, run_stages_training, stages_source_inventory)
from jscc.encoder_stages_plan import ENCODER_STAGES_PROTOCOL


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('bind', 'qualify', 'train', 'evaluate'))
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--prepared-sha')
    parser.add_argument('--approved-source', required=True)
    parser.add_argument('--inputs')
    parser.add_argument('--inputs-sha')
    parser.add_argument('--model-path')
    parser.add_argument('--width', type=int, choices=(512, 1152, 2304), default=512)
    parser.add_argument('--arm', choices=('A', 'E', 'R'), default='A')
    parser.add_argument('--output')
    parser.add_argument('--training-run')
    parser.add_argument('--timeout-seconds', type=int)
    parser.add_argument('--device-uuid')
    args = parser.parse_args(argv)
    source = Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip() != args.approved_source:
        raise ValueError('exact approved source commit required')
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--'], cwd=source, check=True)
    if args.mode == 'bind':
        if not args.inputs or not args.inputs_sha or file_digest(Path(args.inputs)) != args.inputs_sha:
            raise ValueError('exact inputs manifest SHA required')
        root = Path(args.inputs).resolve().parent
        if Path(args.prepared).resolve().parent != root or Path(args.prepared).exists():
            raise ValueError('fresh prepared manifest in inputs root required')
        inputs = json.loads(Path(args.inputs).read_text())
        if inputs.get('schema') != 'encoder-stages-inputs-v1':
            raise ValueError('encoder stages data manifest required')
        if inputs.get('configs'):
            from jscc.config import load_config
            from jscc.sharing_preparation import _reference
            import yaml
            configs = {}
            for width, reference in inputs['configs'].items():
                config = load_config(_reference(root, reference))
                if args.model_path:
                    config['model']['name'] = args.model_path
                config['protocol'] = ENCODER_STAGES_PROTOCOL
                config['run']['output_dir'] = str(root / 'runs')
                path = root / f'runtime-b{width}.yaml'
                if path.exists():
                    raise ValueError('fresh runtime configuration required')
                path.write_text(yaml.safe_dump(config, sort_keys=False))
                configs[width] = {'path': path.name, 'sha256': file_digest(path)}
            inputs['configs'] = configs
        manifest = {**inputs, 'schema': 'encoder-stages-prepared-v1', 'protocol_id': ENCODER_STAGES_PROTOCOL,
                    'synthetic_cpu': False, 'study_pairing_id': 'encoder-stages-seed0-fullpool-v1',
                    'source_inventory': stages_source_inventory(), 'inputs_manifest_sha256': args.inputs_sha}
        manifest.pop('manifest_identity', None)
        manifest['source_identity'] = canonical_digest(manifest['source_inventory'])
        manifest['manifest_identity'] = canonical_digest(manifest)
        write_manifest(Path(args.prepared), manifest)
        print(json.dumps({'prepared_sha256': file_digest(Path(args.prepared)),
                          'manifest_identity': manifest['manifest_identity']}), flush=True)
        return
    if (not args.prepared_sha or file_digest(Path(args.prepared)) != args.prepared_sha
            or not args.output or not args.timeout_seconds or args.timeout_seconds <= 0
            or not args.device_uuid or os.environ.get('CUDA_VISIBLE_DEVICES') != args.device_uuid):
        raise ValueError('exact prepared SHA, GPU UUID, fresh output and finite timeout required')
    if args.mode == 'qualify' and args.timeout_seconds > 1800:
        raise ValueError('qualification cannot exceed 30 minutes')
    started = time.monotonic()
    def guard():
        if time.monotonic() - started >= args.timeout_seconds:
            raise TimeoutError('finite stage deadline reached; no retries or extra updates')
    manifest, config, plan, batches, template, initial, metadata = load_stages(
        args.prepared, arm=args.arm, width=args.width)
    if manifest['synthetic_cpu']:
        raise ValueError('GPU entry requires real immutable full-pool inputs')
    state = freeze = None
    if args.mode == 'evaluate':
        if not args.training_run:
            raise ValueError('exact own training-run link required')
        state, freeze = open_stages_terminal(args.training_run, metadata, plan)
    guard()
    processor, model = initialize(config, initial)
    import torch
    torch.set_num_threads(1)
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise ValueError('one assigned BF16-capable CUDA GPU required')
    guard()
    if args.mode == 'qualify':
        if '5090' not in torch.cuda.get_device_name(0):
            raise ValueError('qualification is authorized only on RTX 5090')
        result = qualify_stages(model, plan=plan, batches=batches, metadata=metadata, output=args.output, guard=guard)
    elif args.mode == 'train':
        result = run_stages_training(model, plan=plan, batches=batches, metadata=metadata, output=args.output, guard=guard)
    else:
        result = run_stages_evaluation(model, processor, state=state, freeze_reference=freeze,
                                      template=template, output=args.output, guard=guard)
    print(json.dumps(result, sort_keys=True), flush=True)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
