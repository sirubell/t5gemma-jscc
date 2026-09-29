"""One synthetic replicated-real-image capacity update; never quality evidence."""
import argparse
from pathlib import Path
import resource
import signal
import sys
import time

import torch

from jscc import training
from jscc.config import load_config, save_config
from jscc.runtime import prepare_trainable_parameters, precision_telemetry, seed_everything, source_state
from scripts.coco_route_smoke import check_payload, sha, write
from scripts.five_shot_preflight import check_gradients


def replicated_batch(fixture, batch_size, stress_target=False):
    if not 1 <= batch_size <= 16:
        raise ValueError('Capacity batch must be 1..16')
    if tuple(fixture['pixel_values'].shape) != (1, 5, 3, 896, 896):
        raise ValueError('Require original B1 five-image 896px fixture')
    result = {}
    for key, value in fixture.items():
        if torch.is_tensor(value):
            if value.ndim == 0 or value.shape[0] != 1:
                raise ValueError(f'Unexpected non-B1 tensor: {key}')
            result[key] = value.repeat(batch_size, *([1] * (value.ndim - 1)))
    if result['labels'].shape != (batch_size, 64):
        raise ValueError('Require target length 64')
    if stress_target:
        labels = result['labels']
        valid = labels[0][labels[0] != -100]
        if not len(valid):
            raise ValueError('No valid target token')
        labels[labels == -100] = valid[-1]
    return result


def run(args):
    if not 0 < args.max_minutes <= 3 or not 1 <= args.batch_size <= 16:
        raise ValueError('Bounded to <=3 minutes, batch 1..16, one update')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = {'status': 'RUNNING', 'quality_evidence': False, 'updates': 0,
              'synthetic_replicated_fixture': True, 'synthetic_full_target_stress': args.stress_target,
              'batch_size': args.batch_size, 'accumulation': 1, 'snr': 'no_noise',
              'fixture_sha256': sha(args.fixture), 'config_sha256': sha(args.config),
              'source': source_state(), 'source_hashes': {str(p): sha(p) for p in
                  sorted(list(Path('jscc').glob('**/*.py')) + [Path(__file__)])},
              'limitations': ['Repeated one image/demo set; not throughput or quality evidence',
                             'One update does not bound all real-data shapes or all splits']}
    def expired(*_):
        raise TimeoutError('Three-minute capacity bound')
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, args.max_minutes * 60)
    try:
        config = load_config(args.config)
        if config['task'] != 'coco' or config['model'].get('sdpa_backend_policy') != 'flash_math':
            raise ValueError('Require COCO and flash_math')
        split = config['split']
        if not ((split['stack'], split['where']) == ('enc', 'after_embed') or
                (split['stack'], split['where'], split.get('index')) == ('dec', 'after_layer', 24)):
            raise ValueError('Only enc_emb and dec_l24 capacity routes are authorized')
        if config['codec']['snr_film'] or not config['channel']['normalize_power']:
            raise ValueError('Require FiLM off and power normalization')
        seed_everything(0)
        config['seed'] = 0
        config['training'].update(batch_size=args.batch_size, gradient_accumulation=1, max_steps=1)
        save_config(config, args.output / 'executed-config.yaml')
        batch = replicated_batch(torch.load(args.fixture, weights_only=True, map_location='cpu'),
                                 args.batch_size, args.stress_target)
        record['shapes'] = {k: list(v.shape) for k, v in batch.items()}
        record['valid_target_tokens'] = int((batch['labels'] != -100).sum())
        write(args.output / 'results.json', record)
        _, model = training.build_model(config)
        prepare_trainable_parameters(model)
        params = [p for p in model.parameters() if p.requires_grad]
        if any(p.dtype != torch.float32 for p in params):
            raise AssertionError('Codec must be FP32')
        if any(p.dtype != torch.bfloat16 for p in model.base.parameters() if p.is_floating_point()):
            raise AssertionError('Frozen backbone must be BF16')
        decoder = config['split']['stack'] == 'dec'
        if decoder != (model.memory_codec is not None):
            raise AssertionError('Receiver memory codec route mismatch')
        model.train()
        optimizer = torch.optim.AdamW(params, lr=config['training']['lr'],
                                      weight_decay=config['training']['weight_decay'])
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        update_started = time.monotonic()
        denominator = training.effective_batch_denominators(model, [batch], next(model.base.parameters()).device)
        values = training.batch_losses(model, batch, config['training'], None, return_stats=True, valid_only_kl=True)
        loss = training.scaled_batch_loss(values, config['training'], denominator)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite objective')
        loss.backward()
        record['gradient_norms'] = check_gradients(model)
        torch.nn.utils.clip_grad_norm_(params, config['training']['grad_clip'])
        optimizer.step()
        torch.cuda.synchronize()
        record.update(updates=1, update_seconds=time.monotonic() - update_started, loss=float(loss.detach()))
        check_payload(model.valid_payload_counts, decoder)
        record['valid_payload'] = dict(model.valid_payload_counts)
        record['allocated_payload'] = dict(model.channel_uses)
        record['precision'] = precision_telemetry(model, optimizer)
        if any(v.is_floating_point() and v.dtype != torch.float32 for state in optimizer.state.values()
               for v in state.values() if torch.is_tensor(v)):
            raise AssertionError('Non-FP32 Adam state')
        record['status'] = 'PASS_CAPACITY_ONLY'
    except BaseException as exc:
        record.update(status='BLOCKED', error={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        record['seconds'] = time.monotonic() - started
        record['peak_cpu_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        if torch.cuda.is_initialized():
            record['cuda'] = dict(peak_allocated=torch.cuda.max_memory_allocated(),
                peak_reserved=torch.cuda.max_memory_reserved(), free_bytes=torch.cuda.mem_get_info()[0],
                total_bytes=torch.cuda.mem_get_info()[1], name=torch.cuda.get_device_name())
        write(args.output / 'results.json', record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'fixture', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--stress-target', action='store_true')
    parser.add_argument('--max-minutes', type=float, default=3)
    run(parser.parse_args())
