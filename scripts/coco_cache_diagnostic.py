"""Read-only model replay of a frozen COCO cache fixture; no optimizer updates.

Use an external process timeout in addition to the <=10 minute alarm here.
Generation scores are the post-logits-processor scores actually used for greedy
selection; output_logits records raw model logits separately.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import signal
import time
from typing import Any

import torch

from jscc import training
from jscc.config import load_config
from jscc.runtime import autocast_for, model_inputs, prepare_trainable_parameters, seed_everything


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def top_scores(scores, k=8):
    values, ids = scores.float().flatten().topk(min(k, scores.numel()))
    # Generation processors can mask tokens with -inf; JSON preserves these as null.
    return [{'token_id': int(i), 'score': float(v) if torch.isfinite(v) else None}
            for v, i in zip(values, ids)]


def compare_outputs(cached, uncached, output, stem):
    """Compare logits only while both decoding paths have an identical prefix."""
    if cached.sequences.shape[0] != 1 or uncached.sequences.shape[0] != 1:
        raise ValueError('Cache diagnostic requires a single fixed example')
    a, b = cached.sequences[0].cpu(), uncached.sequences[0].cpu()
    offsets = (len(a) - len(cached.scores), len(b) - len(uncached.scores))
    if offsets[0] != offsets[1] or not torch.equal(a[:offsets[0]], b[:offsets[1]]):
        raise ValueError('Generation decoder prefixes do not match')
    rows, first = [], None
    for step, (left, right) in enumerate(zip(cached.scores, uncached.scores)):
        left, right = left[0].detach().float().cpu(), right[0].detach().float().cpu()
        raw_left = cached.logits[step][0].detach().float().cpu()
        raw_right = uncached.logits[step][0].detach().float().cpu()
        finite = torch.isfinite(left) & torch.isfinite(right)
        raw_finite = torch.isfinite(raw_left) & torch.isfinite(raw_right)
        row = {'step': step, 'same_prefix_ids': a[:offsets[0] + step].tolist(),
               'cached_token': int(a[offsets[0] + step]), 'uncached_token': int(b[offsets[0] + step]),
               'score_finite_mask_equal': bool(torch.equal(torch.isfinite(left), torch.isfinite(right))),
               'max_abs_finite_score_delta': float((left[finite] - right[finite]).abs().max()) if finite.any() else None,
               'max_abs_raw_logit_delta': float((raw_left[raw_finite] - raw_right[raw_finite]).abs().max()) if raw_finite.any() else None,
               'cached_finite_score_count': int(torch.isfinite(left).sum()),
               'uncached_finite_score_count': int(torch.isfinite(right).sum()),
               'cached_top2_margin': float(left.topk(2).values.diff().neg()[0]) if torch.isfinite(left).sum() >= 2 else None,
               'uncached_top2_margin': float(right.topk(2).values.diff().neg()[0]) if torch.isfinite(right).sum() >= 2 else None}
        rows.append(row)
        if row['cached_token'] != row['uncached_token']:
            first = dict(row, cached_topk=top_scores(left), uncached_topk=top_scores(right),
                         cached_raw_topk=top_scores(raw_left), uncached_raw_topk=top_scores(raw_right))
            torch.save({'cached_scores': left, 'uncached_scores': right,
                        'cached_logits': raw_left, 'uncached_logits': raw_right},
                       output / f'{stem}-first-divergence.pt')
            break
    result = {'tokens_equal': bool(torch.equal(a, b)), 'cached_ids': a.tolist(), 'uncached_ids': b.tolist(),
              'common_prefix_comparisons': rows, 'first_divergence': first,
              'length_only_difference': first is None and len(a) != len(b),
              'score_semantics': 'scores post generation processors; logits raw model output'}
    write_json(output / f'{stem}.json', result)
    return result


def cache_generation_options(*, cached, cache_implementation="default", disable_compile=False):
    if cache_implementation not in ("default", "dynamic", "hybrid"):
        raise ValueError("Unsupported diagnostic cache implementation")
    if not cached:
        return {}
    options = {}
    if cache_implementation != "default":
        options["cache_implementation"] = cache_implementation
    if disable_compile:
        options["disable_compile"] = True
    return options


def cache_classes(cache):
    if cache is None:
        return None
    result: dict[str, Any] = {"class": type(cache).__name__}
    for name in ("self_attention_cache", "cross_attention_cache"):
        component = getattr(cache, name, None)
        if component is not None:
            result[name] = {"class": type(component).__name__,
                            "layer_classes": sorted({type(x).__name__ for x in getattr(component, "layers", [])})}
    return result


def generation_output(model, generation, max_tokens, *, cached, bypass=False,
                      cache_implementation="default", disable_compile=False):
    with torch.no_grad(), autocast_for(model), model.transmission(None, bypass=bypass):
        result = model.generate(**generation, max_new_tokens=max_tokens, do_sample=False,
                                num_beams=1, use_cache=cached, return_dict_in_generate=True,
                                output_scores=True, output_logits=True,
                                **cache_generation_options(cached=cached, cache_implementation=cache_implementation,
                                                           disable_compile=disable_compile))
    # Keep generations on CPU so three sequential runs do not accumulate GPU logits.
    result.sequences = result.sequences.detach().cpu()
    result.scores = tuple(x.detach().cpu() for x in result.scores)
    result.logits = tuple(x.detach().cpu() for x in result.logits)
    result.diagnostic_cache_classes = cache_classes(getattr(result, 'past_key_values', None))
    if hasattr(result, 'past_key_values'):
        result.past_key_values = None
    return result


def run(args):
    if not 0 < args.max_minutes <= 10:
        raise ValueError('Diagnostic maximum is 10 minutes')
    config = load_config(args.config)
    if config['task'] != 'coco' or config['split'] != {'stack': 'enc', 'where': 'after_final_norm'}:
        raise ValueError('Only the frozen COCO enc_fn diagnostic is authorized')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = {'status': 'RUNNING', 'training_updates': 0, 'conditions': [],
              'cache_implementation_override': args.cache_implementation, 'disable_compile': args.disable_compile,
              'input_hashes': {name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for name, path in [('config', args.config), ('state', args.state), ('fixture', args.fixture)]},
              'fp32_control_scope': 'cast same loaded BF16 backbone weights to FP32; not freshly loaded original FP32 weights'}
    def timeout(*_):
        raise TimeoutError('Diagnostic wall-clock bound reached')
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, args.max_minutes * 60)
    try:
        seed_everything(config['seed'])
        _, model = training.build_model(config)
        prepare_trainable_parameters(model)
        model.load_communication_state(torch.load(args.state, map_location='cpu', weights_only=True))
        model.eval()
        record["generation_config"] = model.base.generation_config.to_dict()
        fixture = torch.load(args.fixture, map_location='cpu', weights_only=True)
        for precision in ('bf16', 'fp32'):
            try:
                if precision == 'fp32':
                    gc.collect()
                    torch.cuda.empty_cache()
                    model.float()
                kwargs, _ = model_inputs(fixture, model)
                generation = {k: v for k, v in kwargs.items() if k not in ('decoder_input_ids', 'use_cache')}
                for bypass in (False, True):
                    stem = f'{precision}-' + ('vanilla' if bypass else 'coded')
                    begin = time.monotonic()
                    torch.cuda.reset_peak_memory_stats()
                    cached = generation_output(model, generation, config['evaluation']['max_new_tokens'], cached=True, bypass=bypass,
                        cache_implementation=args.cache_implementation, disable_compile=args.disable_compile)
                    uncached = generation_output(model, generation, config['evaluation']['max_new_tokens'], cached=False, bypass=bypass)
                    result = compare_outputs(cached, uncached, args.output, stem)
                    replay = generation_output(model, generation, config['evaluation']['max_new_tokens'], cached=True, bypass=bypass,
                        cache_implementation=args.cache_implementation, disable_compile=args.disable_compile)
                    replay_result = compare_outputs(cached, replay, args.output, stem + '-cached-replay')
                    record['conditions'].append({'name': stem, 'status': 'COMPLETED',
                        'tokens_equal': result['tokens_equal'], 'cached_replay_equal': replay_result['tokens_equal'],
                        'cached_cache_classes': cached.diagnostic_cache_classes,
                        'uncached_cache_classes': uncached.diagnostic_cache_classes,
                        'replay_cache_classes': replay.diagnostic_cache_classes,
                        'seconds': time.monotonic() - begin,
                        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                        'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30})
                    write_json(args.output / 'results.json', record)
                    del cached, uncached, replay
            except torch.OutOfMemoryError as exc:
                record['conditions'].append({'precision': precision, 'status': 'OOM', 'error': str(exc)})
                break
        record['status'] = 'COMPLETED_DIAGNOSTIC_NOT_ACCEPTANCE'
    except BaseException as exc:
        record.update(status='BLOCKED', error={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        record['seconds'] = time.monotonic() - started
        write_json(args.output / 'results.json', record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'state', 'fixture', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cache-implementation', choices=('default', 'dynamic', 'hybrid'), default='default')
    parser.add_argument('--disable-compile', action='store_true')
    parser.add_argument('--max-minutes', type=float, default=10)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
