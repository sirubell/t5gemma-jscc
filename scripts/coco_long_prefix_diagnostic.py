"""Bounded FP32 cache replay on one fixed 64-token prefix, with no training.

FP32 means upcast of the same loaded BF16 weights. Both coded and vanilla use
identical decoder input IDs, ignoring generation EOS to remove length confounds.
An external timeout of 600 seconds is required in addition to SIGALRM.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import signal
import time

import torch

from jscc import training
from jscc.config import load_config
from jscc.runtime import autocast_for, model_inputs, prepare_trainable_parameters, seed_everything
from scripts.coco_cache_diagnostic import write_json


def tensor_digest(tensor):
    data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def fixed_prefix(document):
    ids = document['cached_ids']
    if len(ids) < 65 or ids[0] != 2 or any(not isinstance(x, int) or x < 0 for x in ids):
        raise ValueError('Require at least 65 cached IDs beginning with native BOS2')
    return torch.tensor([ids[:-1][:64]], dtype=torch.long)


def compare_logits(full, cached, atol=1e-4):
    if full.shape != cached.shape or full.ndim != 3 or full.shape[0] != 1 or full.shape[1] != 64:
        raise ValueError('Expected equal [1,64,vocabulary] logits')
    if not torch.isfinite(full).all() or not torch.isfinite(cached).all():
        raise ValueError('Nonfinite raw logits')
    rows = []
    for position in range(64):
        dtype = torch.float64 if full.dtype == torch.float64 or cached.dtype == torch.float64 else torch.float32
        left, right = full[0, position].to(dtype), cached[0, position].to(dtype)
        delta = float((left - right).abs().max())
        rows.append({'position': position, 'max_abs_logit_delta': delta,
                     'allclose_atol_1e_4_rtol_0': bool(torch.allclose(left, right, atol=atol, rtol=0)),
                     'full_argmax': int(left.argmax()), 'cached_argmax': int(right.argmax()),
                     'argmax_equal': bool(left.argmax() == right.argmax())})
    return {'positions': rows, 'first_failure': next((r['position'] for r in rows if not r['allclose_atol_1e_4_rtol_0']), None),
            'maximum_absolute_delta': max(r['max_abs_logit_delta'] for r in rows),
            'all_argmax_equal': all(r['argmax_equal'] for r in rows)}



def compare_query_last(query_logits, cached, full, length):
    if not 1 <= length <= 64 or query_logits.ndim != 3 or query_logits.shape[0] != 1:
        raise ValueError("Invalid query logits or length")
    if cached.shape != full.shape or cached.ndim != 3 or cached.shape[:2] != (1, 64):
        raise ValueError("Reference logits must have matching [1,64,vocabulary] shapes")
    dtype = torch.float64 if query_logits.dtype == torch.float64 else torch.float32
    last = query_logits[:, -1].to(dtype)
    if last.shape != cached[:, length - 1].shape:
        raise ValueError("Query and reference vocabulary differ")
    if not torch.isfinite(last).all():
        raise ValueError("Nonfinite query logits")
    return {"length": length, "returned_logit_positions": query_logits.shape[1],
        "query_argmax": int(last.argmax(-1)[0]),
        "cached_argmax": int(cached[:, length - 1].argmax(-1)[0]),
        "full64_argmax": int(full[:, length - 1].argmax(-1)[0]),
        "max_abs_vs_cached": float((last - cached[:, length - 1]).abs().max()),
        "max_abs_vs_full64": float((last - full[:, length - 1]).abs().max())}



def source_padding_inputs(kwargs, mode):
    if mode not in ('original', 'trim', 'plus32'):
        raise ValueError('Unknown source padding mode')
    ids, mask = kwargs['input_ids'], kwargs['attention_mask']
    if ids.shape != mask.shape or ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError('Require one matching source IDs/mask row')
    changed = dict(kwargs)
    if mode == 'trim':
        nonzero = torch.nonzero(mask[0], as_tuple=False).flatten()
        if not len(nonzero):
            raise ValueError('Cannot trim empty source')
        stop = int(nonzero[-1]) + 1
        changed['input_ids'], changed['attention_mask'] = ids[:, :stop], mask[:, :stop]
    elif mode == 'plus32':
        changed['input_ids'] = torch.cat([ids, ids.new_zeros((1, 32))], dim=1)
        changed['attention_mask'] = torch.cat([mask, mask.new_zeros((1, 32))], dim=1)
    if not torch.equal(ids[mask.bool()], changed['input_ids'][changed['attention_mask'].bool()]):
        raise AssertionError('Source valid token sequence changed')
    digest = tensor_digest
    pixels = kwargs['pixel_values']
    if changed['pixel_values'] is not pixels:
        raise AssertionError('Source padding changed image tensor')
    receipt = {'mode': mode, 'original_shape': list(ids.shape), 'effective_shape': list(changed['input_ids'].shape),
        'original_ids_sha256': digest(ids), 'effective_ids_sha256': digest(changed['input_ids']),
        'original_mask_sha256': digest(mask), 'effective_mask_sha256': digest(changed['attention_mask']),
        'pixels_sha256': digest(pixels), 'valid_ids_exact': True, 'pixels_same_object': True}
    return changed, receipt


def cache_metadata(cache):
    if cache is None:
        return None
    result = {'class': type(cache).__name__, 'seq_length': int(cache.get_seq_length())}
    result['is_updated'] = {str(k): bool(v) for k, v in getattr(cache, 'is_updated', {}).items()}
    for name in ('self_attention_cache', 'cross_attention_cache'):
        component = getattr(cache, name, None)
        if component is None:
            continue
        layers = []
        for index, layer in enumerate(component.layers):
            layers.append({'index': index, 'class': type(layer).__name__,
                'seq_length': int(component.get_seq_length(index)),
                'keys_shape': list(layer.keys.shape) if getattr(layer, 'keys', None) is not None else None,
                'values_shape': list(layer.values.shape) if getattr(layer, 'values', None) is not None else None,
                'sliding_window': getattr(layer, 'sliding_window', None)})
        result[name] = {'class': type(component).__name__, 'layers': layers}
    return result


def run(args):
    if not 0 < args.max_minutes <= 10:
        raise ValueError('Maximum diagnostic budget is 10 minutes')
    if args.sdpa_backend not in ('auto', 'math'):
        raise ValueError('Backend must be auto or math')
    if args.precision not in ('bf16', 'fp32', 'fp64'):
        raise ValueError('Precision must be bf16, fp32 or fp64')
    config = load_config(args.config)
    if config['task'] != 'coco' or config['split'] != {'stack': 'enc', 'where': 'after_final_norm'}:
        raise ValueError('Only frozen COCO enc_fn is authorized')
    prefix = fixed_prefix(json.loads(args.prefix_json.read_text()))
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {'status': 'RUNNING', 'training_updates': 0, 'sdpa_backend': args.sdpa_backend, 'prefix_ids': prefix.tolist(), 'conditions': [],
              'precision': args.precision,
              'precision_scope': 'same BF16-loaded backbone and FP32 codec; selected FP32/FP64 mode upcasts those weights',
              'interpretation': 'teacher-forced cache correctness only; not task quality or generation acceptance',
              'input_hashes': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
                  [('config', args.config), ('state', args.state), ('fixture', args.fixture), ('prefix', args.prefix_json)]}}
    def timeout(*_):
        raise TimeoutError('Long-prefix diagnostic exceeded wall budget')
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, args.max_minutes * 60)
    try:
        seed_everything(config['seed'])
        _, model = training.build_model(config)
        prepare_trainable_parameters(model)
        model.load_communication_state(torch.load(args.state, map_location='cpu', weights_only=True))
        if args.precision == "fp32":
            model.float()
        elif args.precision == "fp64":
            model.double()
        model.eval()
        result["parameter_dtypes"] = {name: str(next(component.parameters()).dtype) for name, component in
            (("backbone", model.base), ("codec", model.codec))}
        fixture = torch.load(args.fixture, map_location='cpu', weights_only=True)
        kwargs, _ = model_inputs(fixture, model)
        kwargs, result["source_padding"] = source_padding_inputs(kwargs, args.source_padding_mode)
        prefix = prefix.to(next(model.base.parameters()).device)
        result['model_config'] = model.base.config.to_dict()
        result['generation_config'] = model.base.generation_config.to_dict()
        write_json(args.output / 'results.json', result)
        encoder_inputs = {k: v for k, v in kwargs.items() if k in ('input_ids', 'attention_mask', 'pixel_values')}
        for bypass in ((False,) if args.only_coded else (False, True)):
            name = 'vanilla' if bypass else 'coded'
            begin = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            backend = (torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH])
                       if args.sdpa_backend == 'math' else nullcontext())
            with backend, torch.no_grad(), autocast_for(model), model.transmission(None, bypass=bypass, encoder_mask=kwargs['attention_mask']):
                encoder = model.base.get_encoder()(**encoder_inputs, return_dict=True)
                encoder_digest = tensor_digest(encoder.last_hidden_state)
                full_output = model.base(encoder_outputs=encoder, attention_mask=kwargs['attention_mask'],
                    decoder_input_ids=prefix, use_cache=False, return_dict=True)
                full = full_output.logits.detach().cpu()
                del full_output
                cache, outputs, cache_records = None, [], []
                for position in range(64):
                    out = model.base(encoder_outputs=encoder, attention_mask=kwargs['attention_mask'],
                        decoder_input_ids=prefix[:, position:position + 1], past_key_values=cache,
                        use_cache=True, return_dict=True)
                    cache = out.past_key_values
                    outputs.append(out.logits.detach().cpu())
                    if position in (0, 1, 29, 30, 31, 32, 33, 34, 62, 63):
                        cache_records.append({'position': position, 'cache': cache_metadata(cache)})
                    del out
                cached = torch.cat(outputs, dim=1)
                torch.save({'full_logits': full, 'cached_logits': cached, 'prefix_ids': prefix.cpu()},
                           args.output / f'{name}-logits.pt')
                query_results = []
                if args.query_length_sweep:
                    for length in (1, 8, 16, 31, 32, 33, 34, 48, 63, 64):
                        for keep in (0, 1):
                            profiler = (torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])
                                if args.profile_attention and not bypass and length == 33 and keep == 0 else None)
                            with profiler if profiler is not None else nullcontext():
                                query = model.base(encoder_outputs=encoder, attention_mask=kwargs['attention_mask'],
                                    decoder_input_ids=prefix[:, :length], use_cache=False, logits_to_keep=keep,
                                    return_dict=True)
                            if profiler is not None:
                                result['attention_operator_evidence'] = {
                                    'scope': 'CPU dispatcher events only; not GPU kernel timing',
                                    'condition': name, 'query_length': 33, 'logits_to_keep': 0,
                                    'events': [{'name': event.key, 'count': event.count}
                                        for event in profiler.key_averages()
                                        if 'attention' in event.key.lower() or 'sdp' in event.key.lower()]}
                                write_json(args.output / 'attention-operators.json', result['attention_operator_evidence'])
                            query_logits = query.logits.detach().cpu()
                            row = compare_query_last(query_logits, cached, full, length)
                            row['logits_to_keep'] = keep
                            query_results.append(row)
                            torch.save(query_logits[:, -1].clone(), args.output / f'{name}-query-{length}-keep-{keep}.pt')
                            write_json(args.output / f'{name}-query-sweep.json', query_results)
                            del query, query_logits
                comparison = compare_logits(full, cached)
                comparison['query_length_sweep'] = query_results
                comparison.update(name=name, encoder_sha256=encoder_digest, encoder_computations=1,
                    seconds=time.monotonic() - begin, cache_records=cache_records,
                    peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
                write_json(args.output / f'{name}.json', comparison)
                result['conditions'].append({key: comparison[key] for key in
                    ('name', 'first_failure', 'maximum_absolute_delta', 'all_argmax_equal', 'seconds', 'peak_allocated_gib', 'peak_reserved_gib')})
                write_json(args.output / 'results.json', result)
                del encoder, cache, outputs, full, cached
        result['status'] = 'COMPLETED_DIAGNOSTIC_NOT_ACCEPTANCE'
    except BaseException as exc:
        result.update(status='BLOCKED', error={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        result['seconds'] = time.monotonic() - started
        write_json(args.output / 'results.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'state', 'fixture', 'prefix-json', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--precision', choices=('bf16', 'fp32', 'fp64'), default='fp32')
    parser.add_argument('--only-coded', action='store_true')
    parser.add_argument('--profile-attention', action='store_true')
    parser.add_argument('--source-padding-mode', choices=('original', 'trim', 'plus32'), default='original')
    parser.add_argument('--query-length-sweep', action='store_true')
    parser.add_argument('--sdpa-backend', choices=('auto', 'math'), default='auto')
    parser.add_argument('--max-minutes', type=float, default=10)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
