"""Bounded no-noise pooled32 K/H/M probe for the main-weight early screen.

One invocation is one observation (12 VJPs). The orchestrator owns the shared
96-attempt cap and invokes shared initialization only once. Gradient bytes are
not exported; saved norms/dots permit offline algebra, not a fresh VJP replay.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
from typing import Any, cast

import torch

from jscc.data.hellaswag import Collator
from jscc.data.hellaswag_prompts import digest
from jscc.models.split_model import build_model
from jscc.runtime import configure_training_determinism, isolated_rng, source_state
from jscc.training import batch_losses, effective_batch_denominators
from scripts.experiments.evening_gradients import (
    Budget, component_vjps, feature_rows, file_hash, gradient_summary,
    selection_fixture, validate_fixture,
)

COMPONENTS = ('kl', 'hidden', 'memory')


def summarize_vectors(vectors, weights):
    result = gradient_summary(vectors, weights)
    total = sum((vectors[k].double() * weights[k] for k in COMPONENTS), torch.zeros_like(vectors['kl'], dtype=torch.float64))
    result['total_weighted_norm_before_clip'] = float(total.norm())
    result['weighted_component_to_kl_norm_ratios'] = {
        k: result['norms'][f'weighted_{k}'] / result['norms']['weighted_kl']
        if result['norms']['weighted_kl'] > 1e-12 else None
        for k in ('hidden', 'memory')
    }
    result['weighted_pairs'] = {
        f'{a}:{b}': {
            'dot': float(torch.dot(vectors[a].double(), vectors[b].double())) * weights[a] * weights[b],
            'cosine': result['pairs'][f'{a}:{b}']['cosine'],
            'reason': result['pairs'][f'{a}:{b}']['reason'],
        } for a, b in [('kl', 'hidden'), ('kl', 'memory'), ('hidden', 'memory')]
    }
    return result


def run_probe(model, rows, training, pad_id, hidden_weight, budget, write, identity):
    """Observe fixed32 rows without modifying parameter/grad/RNG/mode state."""
    if len(rows) != 32 or len({r['row_id'] for r in rows}) != 32:
        raise ValueError('probe requires exactly32 unique frozen selection rows')
    if hidden_weight not in (.05, .5, 5.):
        raise ValueError('unregistered hidden weight')
    if model.memory_codec is None or model.split['stack'] != 'dec':
        raise ValueError('probe requires legal decoder main and memory streams')
    if budget.cap - budget.calls < 12:
        raise ValueError('12 VJP attempts must be available before first forward')
    groups = {name: [(n, p) for n, p in module.named_parameters() if p.requires_grad]
              for name, module in [('main', model.codec), ('memory', model.memory_codec)]}
    weights = {'kl': 1., 'hidden': hidden_weight, 'memory': .05}
    batches = [Collator(pad_id)(rows[i:i+8]) for i in range(0, 32, 8)]
    device = next(model.base.parameters()).device
    denominators = effective_batch_denominators(model, batches, device)
    if any(float(denominators[k]) <= 0 for k in COMPONENTS):
        raise ValueError('empty pooled component denominator')
    modes = [(module, module.training) for module in model.modules()]
    counters = {key: copy.deepcopy(getattr(model, key)) for key in ('channel_uses', 'channel_uses_valid')}
    budget.write = write
    pooled, structural, totals = {}, {}, dict.fromkeys(COMPONENTS, 0.)

    def reserve_forward(module, args):
        budget.check()
        kind = 'teacher' if module.bypass else 'student'
        setattr(budget, kind + '_forwards', getattr(budget, kind + '_forwards') + 1)
        write('probe_attempts.jsonl', {**budget.identity, 'operation': kind + '_forward'})

    hook = model.register_forward_pre_hook(reserve_forward)
    try:
        with isolated_rng(20260921):
            model.eval()
            for index, batch in enumerate(batches):
                budget.check()
                row_ids = [r['row_id'] for r in rows[index*8:index*8+8]]
                common = {**identity, 'case': 'no_noise', 'microbatch_index': index, 'row_ids': row_ids}
                budget.identity = common
                budget.presentations += 8
                values = cast(dict[str, Any], batch_losses(model, batch, training, None, return_stats=True,
                                      valid_only_kl=training.get('valid_only_kl', False)))
                components = {k: values[k+'_numerator']/denominators[k] for k in COMPONENTS}
                losses = {}
                for key in COMPONENTS:
                    numerator = float(values[key+'_numerator'].detach())
                    denominator = float(values[key+'_denominator'])
                    totals[key] += numerator
                    losses[key] = {'numerator': numerator, 'denominator': denominator,
                                   'mean': numerator/denominator if denominator else None}
                write('probe_losses.jsonl', {**common, 'components': losses, 'weights': weights,
                      'allocated': dict(model.channel_uses), 'valid': dict(model.channel_uses_valid)})
                for stream, original, reconstructed, mask in [
                    ('hidden', model.activation, model.reconstruction, batch['labels'] != -100),
                    ('memory', model.memory_activation, model.memory_reconstruction, batch['attention_mask']),
                ]:
                    for row_id, stats in zip(row_ids, feature_rows(original, reconstructed, mask.to(device))):
                        write('probe_features.jsonl', {**common, 'row_id': row_id, 'stream': stream,
                                                      'statistics': stats})
                gradients = component_vjps(components, groups, budget)
                for group in groups:
                    for key in COMPONENTS:
                        vector = gradients[key][group]['vector']
                        zeros = set(gradients[key][group]['structural_zero_parameters'])
                        entry = group, key
                        if entry not in pooled:
                            pooled[entry], structural[entry] = vector.clone(), zeros
                        else:
                            pooled[entry].add_(vector)
                            structural[entry].intersection_update(zeros)
                del gradients, components, values
                model.activation = model.reconstruction = None
                model.memory_activation = model.memory_reconstruction = None
            global_square = 0.
            for group in groups:
                vectors = {k: pooled[group, k] for k in COMPONENTS}
                stats = summarize_vectors(vectors, weights)
                global_square += stats['total_weighted_norm_before_clip']**2
                write('probe_gradients.jsonl', {**identity, 'case': 'no_noise', 'scope': 'pooled32',
                      'group': group, 'row_ids': [r['row_id'] for r in rows], 'weights': weights,
                      'denominators': {k: float(denominators[k]) for k in COMPONENTS},
                      'structural_zero_parameters': {k: sorted(structural[group, k]) for k in COMPONENTS},
                      **stats})
            means = {k: totals[k]/float(denominators[k]) for k in COMPONENTS}
            clip_norm = global_square**.5
            write('probe_losses.jsonl', {**identity, 'case': 'no_noise', 'scope': 'pooled32',
                  'weights': weights, 'components': {k: {'numerator': totals[k],
                  'denominator': float(denominators[k]), 'mean': means[k]} for k in COMPONENTS},
                  'weighted_components': {k: means[k]*weights[k] for k in COMPONENTS},
                  'total': sum(means[k]*weights[k] for k in COMPONENTS),
                  'global_norm_before_clip': clip_norm, 'clip_threshold': 1.,
                  'hypothetical_clip_scale': min(1., 1./(clip_norm+1e-6)),
                  'clip_applied': False, 'optimizer_updates': 0})
    finally:
        hook.remove()
        for module, mode in modes:
            module.training = mode
        for key, value in counters.items():
            setattr(model, key, value)
        model.activation = model.reconstruction = None
        model.memory_activation = model.memory_reconstruction = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--fixture', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--arm', required=True, choices=['shared_init', 'W005', 'W05', 'W5'])
    parser.add_argument('--step', required=True, type=int, choices=[0, 500, 1000])
    parser.add_argument('--hidden-weight', required=True, type=float, choices=[.05, .5, 5.])
    parser.add_argument('--seconds', type=int, default=300)
    parser.add_argument('--deadline-epoch', required=True, type=float)
    args = parser.parse_args()
    expected_weight = {'shared_init': .05, 'W005': .05, 'W05': .5, 'W5': 5.}[args.arm]
    if args.hidden_weight != expected_weight or (args.arm == 'shared_init') != (args.step == 0):
        parser.error('arm/weight/step are not an approved observation')
    if not 1 <= args.seconds <= 600:
        parser.error('seconds must be in1..600')
    budget = Budget(12, args.seconds)
    budget.deadline = min(budget.deadline, time.monotonic()+args.deadline_epoch-time.time())
    budget.check()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = {'arm': args.arm, 'checkpoint_step': args.step, 'checkpoint': str(args.checkpoint),
               'checkpoint_sha256': file_hash(args.checkpoint), 'fixture_file_sha256': file_hash(args.fixture),
               'source': source_state(), 'source_file_sha256': file_hash(__file__),
               'gradient_bytes_exported': False, 'completion': 'FAILED'}

    def write(name, value):
        with (args.output/name).open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False)+'\n')

    try:
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if type(state.get('step')) is not int or state['step'] != args.step:
            raise ValueError('checkpoint internal step mismatch')
        config = state['config']
        if tuple(config['split'].get(k) for k in ('stack', 'where', 'index')) != ('dec', 'after_layer', 20):
            raise ValueError('expected dec_l20 split')
        if config.get('protocol') != 'corrected-baseline-v2-native-decoder-inputs':
            raise ValueError('native-v2 checkpoint required')
        if config['codec'].get('snr_film', False) or config['codec'].get('memory', {}).get('snr_film', False):
            raise ValueError('FiLM must remain off')
        configure_training_determinism(config['training'])
        tokenizer, model = build_model(config)
        model.load_communication_state(state)
        payload = json.loads(args.fixture.read_text())
        supplied = payload['rows'] if isinstance(payload, dict) else payload
        expected_rows = {r['row_id']: r for r in selection_fixture(config, state['data_ids'], tokenizer)}
        for row in supplied:
            for key in ['input_ids', 'attention_mask', 'label_ids']:
                if row[key] != expected_rows[row['row_id']][key]:
                    raise ValueError('E3 frozen fixture token mismatch')
        rows = validate_fixture(supplied, state['data_ids'])[:32]
        receipt['row_ids'] = [r['row_id'] for r in rows]
        receipt['fixture_sha256'] = digest(rows)
        (args.output/'fixture32.json').write_text(json.dumps({'rows': rows}))
        identity = {k: receipt[k] for k in ('arm', 'checkpoint_step', 'checkpoint_sha256', 'fixture_sha256')}
        run_probe(model, rows, config['training'], tokenizer.pad_token_id, args.hidden_weight,
                  budget, write, identity)
        if budget.calls != 12:
            raise RuntimeError('completed observation must have12 VJPs')
        receipt['completion'] = 'COMPLETE'
    except Exception as exc:
        receipt['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        receipt.update(component_vjp_calls=budget.calls, teacher_forwards=budget.teacher_forwards,
                       student_forwards=budget.student_forwards, presentations=budget.presentations,
                       optimizer_updates=0, wall_seconds=time.monotonic()-started,
                       checkpoint_sha256_after=file_hash(args.checkpoint))
        (args.output/'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False))
        if receipt['checkpoint_sha256_after'] != receipt['checkpoint_sha256']:
            raise RuntimeError('checkpoint bytes changed during probe')


if __name__ == '__main__':
    main()
