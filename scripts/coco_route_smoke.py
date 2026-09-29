"""Disposable full-weight COCO route readiness; never quality evidence.

Run each route in a separate process with an external five-minute timeout too.
The frozen fixture must come from the reviewed real-image enc_fn preflight.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import signal
import time

import torch

from jscc import training
from jscc.config import load_config, save_config
from jscc.runtime import (autocast_for, model_inputs, precision_telemetry,
                          prepare_trainable_parameters, seed_everything, source_state)
from scripts.coco_native_preflight import communication_digest, communication_snapshot, perturb_and_reload
from scripts.five_shot_preflight import check_gradients


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def logit_comparison(full, cached, atol=1e-4, rtol=1e-4):
    if full.shape != cached.shape or not torch.isfinite(full).all() or not torch.isfinite(cached).all():
        raise ValueError('Nonfinite or mismatched logits')
    full, cached = full.detach().float().cpu(), cached.detach().float().cpu()
    delta = (full - cached).abs()
    return {'allclose': bool(torch.allclose(full, cached, atol=atol, rtol=rtol)),
            'atol': atol, 'rtol': rtol, 'max_abs_delta': float(delta.max()),
            'mean_abs_delta': float(delta.mean()),
            'argmax_equal': bool(torch.equal(full.argmax(-1), cached.argmax(-1)))}


def check_payload(counts, decoder):
    if counts.get('hidden', 0) <= 0:
        raise AssertionError('Main stream has no valid payload')
    if decoder and counts.get('memory', 0) <= 0:
        raise AssertionError('Decoder route has no receiver-memory payload')
    if not decoder and counts.get('memory', 0) != 0:
        raise AssertionError('Encoder route unexpectedly transmitted receiver memory')


def load_forced_prefix(path):
    value = json.loads(path.read_text())
    ids = value['cached_ids']
    if (not isinstance(ids, list) or len(ids) < 65 or ids[0] != 2
            or any(type(x) is not int or x < 0 for x in ids)):
        raise ValueError('Require at least 65 cached IDs beginning with native BOS 2')
    return ids[:64]


def prefix_replay(model, kwargs, output, precision, atol, rtol, *, forced_ids=None):
    """Same-prefix full-vs-incremental logits; force every intervening cache step."""
    from transformers.modeling_outputs import BaseModelOutput
    lengths = [1, 2, 3] if forced_ids is None else [1, 2, 3, 31, 32, 33, 34, 64]
    decoder = (kwargs['decoder_input_ids'][:, :3] if forced_ids is None else
               torch.tensor([forced_ids], dtype=torch.long, device=kwargs['decoder_input_ids'].device))
    if decoder.shape != (1, max(lengths)):
        raise ValueError('Require exactly one prefix of the planned length')
    rows, logits, cache, encoder = [], [], None, None
    with torch.no_grad(), autocast_for(model):
        for position in range(max(lengths)):
            selected = position + 1 in lengths
            left = None
            if selected:
                with model.transmission(None):
                    full = model(**{**kwargs, 'decoder_input_ids': decoder[:, :position + 1], 'use_cache': False})
                    left = full.logits[:, -1].detach().cpu()
                    del full
            inputs = {**kwargs, 'decoder_input_ids': decoder[:, position:position + 1], 'use_cache': True}
            if cache is not None:
                inputs.pop('input_ids')
                inputs.pop('pixel_values', None)
                inputs.update(encoder_outputs=encoder, past_key_values=cache)
            with model.transmission(None, encoder_mask=kwargs['attention_mask']):
                current = model(**inputs)
                cache = current.past_key_values
                encoder = BaseModelOutput(last_hidden_state=current.encoder_last_hidden_state)
                right = current.logits[:, -1].detach().cpu() if selected else None
                del current
            if selected:
                assert left is not None and right is not None
                rows.append({'position': position, 'prefix_length': position + 1,
                             'prefix': decoder[0, :position + 1].tolist(),
                             **logit_comparison(left, right, atol, rtol)})
                logits.append({'position': position, 'full': left, 'cached': right})
    torch.save(logits, output / f'{precision}-same-prefix-logits.pt')
    return rows


def run(args):
    if args.updates not in (1, 2) or not 0 < args.max_minutes <= 5:
        raise ValueError('Bounded to one/two updates and <= five minutes')
    if args.atol <= 0 or args.rtol <= 0:
        raise ValueError('Positive explicit tolerances required')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = {'status': 'RUNNING', 'quality_evidence': False, 'updates': 0,
              'fixture_sha256': sha(args.fixture), 'config_sha256': sha(args.config),
              'source': source_state(), 'checks': {},
              'runtime': {'torch': str(torch.__version__), 'cuda': torch.version.cuda},
              'source_hashes': {str(p.resolve().relative_to(Path.cwd())): sha(p) for p in
                  sorted(list(Path.cwd().glob('jscc/**/*.py')) + [Path(__file__)])},
              'limitations': ['One frozen real-image example; no dataset-wide capacity or quality claim',
                  'FP32 control casts the same BF16-loaded backbone; not original FP32 weights',
                  'Only selected same-prefix positions tested; BF16 free-running equality is not a structural gate']}
    def flush():
        record['seconds'] = time.monotonic() - started
        write(args.output / 'results.json', record)
    def expired(*_):
        raise TimeoutError('Route smoke five-minute bound')
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, args.max_minutes * 60)
    try:
        config = load_config(args.config)
        if config['task'] != 'coco' or config['codec']['snr_film'] or not config['channel']['normalize_power']:
            raise ValueError('Require COCO, FiLM off, constrained power')
        forced_ids = None
        if args.prefix_json is not None:
            forced_ids = load_forced_prefix(args.prefix_json)
            record['forced_prefix'] = {'source_path': str(args.prefix_json),
                'sha256': sha(args.prefix_json), 'ids': forced_ids,
                'scope': 'Shared frozen enc_fn control generation, not this route generated tokens',
                'comparison_lengths': [1, 2, 3, 31, 32, 33, 34, 64]}
            (args.output / 'forced-prefix-source.json').write_bytes(args.prefix_json.read_bytes())
        seed_everything(0)
        config['seed'] = 0
        save_config(config, args.output / 'executed-config.yaml')
        fixture = torch.load(args.fixture, map_location='cpu', weights_only=True)
        if fixture['input_ids'].shape[0] != 1 or 'pixel_values' not in fixture:
            raise ValueError('Require single-example real-image fixture')
        torch.save(fixture, args.output / 'fixture.pt')
        record['fixture_shapes'] = {k: list(v.shape) for k, v in fixture.items() if torch.is_tensor(v)}
        _, model = training.build_model(config)
        prepare_trainable_parameters(model)
        record['initial_communication_sha256'] = communication_digest(communication_snapshot(model))
        record['hardware'] = {'name': torch.cuda.get_device_name(),
            'total_memory_bytes': torch.cuda.get_device_properties(0).total_memory}
        kwargs, _ = model_inputs(fixture, model)
        record['native_prefix'] = kwargs['decoder_input_ids'][0, :3].tolist()
        if record['native_prefix'][0] != 2:
            raise AssertionError('Expected native T5Gemma 2 decoder BOS 2')
        decoder_route = config['split']['stack'] == 'dec'
        if decoder_route != (model.memory_codec is not None):
            raise AssertionError('Receiver-memory codec does not match route')
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
        values = training.batch_losses(model, fixture, config['training'], None, return_stats=True, valid_only_kl=True)
        values['kl'].backward()
        record['checks']['kl_only_gradients'] = check_gradients(model)
        optimizer.zero_grad(set_to_none=True)
        del values
        denominator = training.effective_batch_denominators(model, [fixture], next(model.base.parameters()).device)
        record['training'] = []
        for _ in range(args.updates):
            values = training.batch_losses(model, fixture, config['training'], None, return_stats=True, valid_only_kl=True)
            loss = training.scaled_batch_loss(values, config['training'], denominator)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training objective')
            loss.backward()
            gradients = check_gradients(model)
            torch.nn.utils.clip_grad_norm_(params, config['training']['grad_clip'])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            check_payload(model.valid_payload_counts, decoder_route)
            record['updates'] += 1
            record['training'].append({'loss': float(loss.detach()), 'gradient_norms': gradients,
                                      'valid_payload': dict(model.valid_payload_counts)})
            del loss, values
            flush()
        record['precision'] = precision_telemetry(model, optimizer)
        if any(p.dtype != torch.float32 for p in params):
            raise AssertionError('Non-FP32 trainable parameter')
        if any(v.is_floating_point() and v.dtype != torch.float32 for s in optimizer.state.values()
               for v in s.values() if torch.is_tensor(v)):
            raise AssertionError('Non-FP32 Adam state')
        record['checks']['reload'] = perturb_and_reload(model, args.output / 'communication.pt')
        model.eval()
        generation = {k: v for k, v in kwargs.items() if k not in ('decoder_input_ids', 'use_cache')}
        generated = []
        for replay in range(2):
            with torch.no_grad(), autocast_for(model), model.transmission(None), model.observe_payload() as events:
                ids = model.generate(**generation, max_new_tokens=64, do_sample=False, num_beams=1, use_cache=True)
                generated.append(ids.detach().cpu())
                check_payload(model.valid_payload_counts, decoder_route)
                write(args.output / f'generation-{replay}.json', {'ids': ids.tolist(), 'events': events,
                      'valid_payload': dict(model.valid_payload_counts), 'allocated_payload': dict(model.channel_uses)})
        if not torch.equal(*generated):
            raise AssertionError('Repeated cached generation tokens differ')
        record['checks']['cached_replay'] = True
        noisy = []
        for _ in range(2):
            seed_everything(20260921)
            with torch.no_grad(), autocast_for(model), model.transmission(-6):
                result = model(**kwargs)
                noisy.append(result.logits.detach().cpu())
                del result
        record['checks']['noisy_replay'] = logit_comparison(*noisy, atol=0, rtol=0)
        if not record['checks']['noisy_replay']['allclose']:
            raise AssertionError('Fixed-seed noisy forward replay mismatch')
        del noisy, optimizer, params
        record['bf16_same_prefix'] = prefix_replay(model, kwargs, args.output, 'bf16', args.atol, args.rtol)
        flush()
        gc.collect()
        torch.cuda.empty_cache()
        model.float()
        kwargs, _ = model_inputs(fixture, model)
        record['fp32_same_prefix'] = prefix_replay(model, kwargs, args.output, 'fp32', args.atol, args.rtol, forced_ids=forced_ids)
        if not all(row['allclose'] for row in record['fp32_same_prefix']):
            raise AssertionError('FP32 same-prefix cache equivalence outside explicit tolerance')
        record['status'] = 'PASS_ROUTE_READINESS_ONLY'
    except BaseException as exc:
        record.update(status='BLOCKED', error={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'fixture', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--prefix-json', type=Path)
    parser.add_argument('--updates', type=int, default=1)
    parser.add_argument('--max-minutes', type=float, default=5)
    parser.add_argument('--atol', type=float, default=1e-4)
    parser.add_argument('--rtol', type=float, default=1e-4)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
