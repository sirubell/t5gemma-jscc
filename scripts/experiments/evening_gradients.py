"""Retained E3 loss, feature, component-VJP and selection helpers; no optimizer.

The historical executor is retired; imports preserve scientific interpretation.
"""
from __future__ import annotations

if __name__ == "__main__":
    raise SystemExit(
        "evening_gradients executor is retired; retained helpers are available by import."
    )

from contextlib import nullcontext
import hashlib
from pathlib import Path
import random
import time

import torch

from jscc.data.hellaswag import Collator
from jscc.data.hellaswag_prompts import PromptBuilder
from jscc.models.channel import AWGNChannel
from jscc.presentation import derived_seed, tensor_digest
from jscc.training import batch_losses, effective_batch_denominators

SEED = 20260921


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Budget:
    def __init__(self, cap=256, seconds=2400):
        if not 1 <= cap <= 256:
            raise ValueError('VJP cap must be in 1..256')
        self.cap, self.calls = cap, 0
        self.deadline = time.monotonic() + seconds
        self.teacher_forwards = self.student_forwards = self.presentations = 0
        self.write = lambda name, value: None
        self.identity = {}

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError('E3 deadline reached before dispatch')

    def reserve_vjp(self, component=None):
        self.check()
        if self.calls >= self.cap:
            raise RuntimeError('component VJP budget exhausted')
        self.calls += 1  # Count attempted calls, including failures.
        self.write("e3_attempts.jsonl", {**self.identity, "operation": "component_vjp",
                                        "component": component, "attempt": self.calls})


class ExampleAWGN(AWGNChannel):
    """Fixed row/stream epsilon, generated on CPU in canonical FP32 coordinates."""
    def __init__(self):
        super().__init__()
        self.row_ids = []

    def forward(self, z, snr_db):
        if snr_db is None:
            return z
        if len(self.row_ids) != z.shape[0]:
            raise ValueError('noise row identity count differs from latent batch')
        samples = []
        for row_id, sample in zip(self.row_ids, z):
            seed = derived_seed(SEED, f'e3:{row_id}:{self._stream}')
            generator = torch.Generator().manual_seed(seed)
            epsilon = torch.randn(sample.shape, generator=generator, dtype=torch.float32)
            samples.append(epsilon)
            self.draw_summaries.append({'row_id': row_id, 'stream': self._stream,
                                        'seed': seed, 'shape': list(sample.shape),
                                        'epsilon_sha256': tensor_digest(epsilon)})
        noise = torch.stack(samples).to(device=z.device, dtype=z.dtype)
        return z + noise * (10.0 ** (-snr_db / 20.0))


def sample_snrs(row_ids):
    return [random.Random(derived_seed(SEED, f'e3-snr:{row_id}')).uniform(-6, 18)
            for row_id in row_ids]


def feature_rows(original, reconstructed, mask):
    """Per-example raw-coordinate sufficient statistics, excluding padded tokens."""
    rows = []
    for x, y, valid in zip(original.detach(), reconstructed.detach(), mask):
        x, y = x[valid.bool()].double(), y[valid.bool()].double()
        if x.numel() == 0:
            rows.append({'valid_coordinates': 0, 'reason': 'empty_valid_stream'})
            continue
        power, error = x.square().sum(), (y-x).square().sum()
        nx, ny = x.norm(), y.norm()
        rows.append({'valid_coordinates': x.numel(), 'valid_tokens': x.shape[0],
                     'original_mean': x.mean().item(), 'original_rms': x.square().mean().sqrt().item(),
                     'original_std': x.std(unbiased=False).item(),
                     'reconstructed_mean': y.mean().item(), 'reconstructed_rms': y.square().mean().sqrt().item(),
                     'reconstructed_std': y.std(unbiased=False).item(),
                     'original_energy': power.item(), 'error_energy': error.item(),
                     'nmse': (error / (power + x.numel()*1e-8)).item(),
                     'cosine': (torch.sum(x*y)/(nx*ny)).item() if min(nx, ny) > 1e-12 else None,
                     'cosine_reason': None if min(nx, ny) > 1e-12 else 'near_zero_feature_norm'})
    return rows


def component_vjps(components, groups, budget):
    """Differentiate each scalar once against every trainable codec tensor."""
    parameters = [p for group in groups.values() for _, p in group]
    result = {}
    for index, (name, loss) in enumerate(components.items()):
        budget.reserve_vjp(name)
        grads = torch.autograd.grad(loss, parameters, retain_graph=index < len(components)-1,
                                    allow_unused=True)
        offset, grouped = 0, {}
        for group_name, group in groups.items():
            values = grads[offset:offset+len(group)]
            grouped[group_name] = {
                'vector': torch.cat([torch.zeros(p.numel()) if g is None else g.detach().float().cpu().reshape(-1)
                                     for (_, p), g in zip(group, values)]),
                'structural_zero_parameters': [n for (n, _), g in zip(group, values) if g is None],
            }
            if not torch.isfinite(grouped[group_name]['vector']).all():
                raise FloatingPointError(f'nonfinite gradient: {name}/{group_name}')
            offset += len(group)
        result[name] = grouped
    return result


def gradient_summary(vectors, weights, near_zero=1e-12):
    """Norms/dots are computed after vector pooling, never by averaging cosines."""
    raw = {key: value.double() for key, value in vectors.items()}
    weighted = {key: value * weights[key] for key, value in raw.items()}
    reconstruction = sum((value for key, value in weighted.items() if key != 'kl'),
                         torch.zeros_like(weighted['kl']))
    all_vectors = {**raw, **{f'weighted_{k}': v for k, v in weighted.items()},
                   'weighted_reconstruction': reconstruction}
    norms = {key: value.norm().item() for key, value in all_vectors.items()}
    pairs = {}
    for left, right in [('kl', 'hidden'), ('kl', 'memory'), ('hidden', 'memory'),
                        ('weighted_kl', 'weighted_reconstruction')]:
        if left not in all_vectors or right not in all_vectors:
            continue
        dot = torch.dot(all_vectors[left], all_vectors[right]).item()
        small = min(norms[left], norms[right]) <= near_zero
        pairs[f'{left}:{right}'] = {'dot': dot, 'cosine': None if small else dot/(norms[left]*norms[right]),
                                   'reason': 'near_zero_gradient_norm' if small else None}
    small = norms['weighted_kl'] <= near_zero
    return {'norms': norms, 'pairs': pairs, 'near_zero_absolute_threshold': near_zero,
            'weighted_reconstruction_to_kl_norm_ratio': None if small else norms['weighted_reconstruction']/norms['weighted_kl'],
            'ratio_reason': 'near_zero_weighted_kl_norm' if small else None,
            'kl_dot_total_weighted_gradient': torch.dot(raw['kl'], weighted['kl']+reconstruction).item(),
            'direction_interpretation': 'local first-order gradient direction; not AdamW next-step quality'}


def selection_fixture(config, ids, tokenizer):
    from datasets import load_dataset
    raw = load_dataset(config['data']['name'], revision=config['data']['revision'])['train']
    builder = PromptBuilder(raw, ids, config['data']['prompt_policy'])
    selection = selection_ids(ids)
    selected = raw.select(selection)
    tokens = builder.tokenize(selected[:], selection, tokenizer)
    return [dict({key: values[i] for key, values in tokens.items()}, row_id=row_id)
            for i, row_id in enumerate(selection)]


def selection_ids(ids):
    """Freeze 128 rows before any scores, from the recorded train holdout."""
    expected = list(ids['validation_rows'])
    if ids.get('validation_split') != 'train' or len(expected) < 128 or len(set(expected)) != len(expected):
        raise ValueError('E3 requires at least128 unique saved train selection rows')
    if set(expected) & set(ids.get('train_rows', [])):
        raise ValueError('selection and optimization IDs overlap')
    if len(expected) > 128:
        random.Random(SEED).shuffle(expected)
    return expected[:128]


def validate_fixture(rows, ids):
    expected = selection_ids(ids)
    if len(rows) != 128 or len({r['row_id'] for r in rows}) != 128 or set(r['row_id'] for r in rows) != set(expected):
        raise ValueError('fixture must exactly match frozen selection128 IDs')
    by_id = {row['row_id']: row for row in rows}
    order = list(expected)
    random.Random(SEED).shuffle(order)
    return [by_id[row_id] for row_id in order]


def run_diagnostic(model, rows, training, pad_id, budget, write, identity):
    budget.write = write
    def reserve_forward(module, args):
        budget.check()
        kind = 'teacher' if module.bypass else 'student'
        if kind == 'teacher':
            budget.teacher_forwards += 1
        else:
            budget.student_forwards += 1
        write('e3_attempts.jsonl', {**budget.identity, 'operation': kind+'_forward',
              'teacher_forwards': budget.teacher_forwards, 'student_forwards': budget.student_forwards})
    forward_hook = model.register_forward_pre_hook(reserve_forward)
    groups = {'main': [(n, p) for n, p in model.codec.named_parameters() if p.requires_grad]}
    if model.memory_codec is not None:
        groups['memory'] = [(n, p) for n, p in model.memory_codec.named_parameters() if p.requires_grad]
    weights = {'kl': training['kl_weight'], 'hidden': training['mse_weight']/len(groups)}
    if 'memory' in groups:
        weights['memory'] = training['mse_weight']/len(groups)
    batches = [Collator(pad_id)(rows[i:i+8]) for i in range(0, len(rows), 8)]
    device = next(model.base.parameters()).device
    denominators = effective_batch_denominators(model, batches[:8], device)
    selection_denominators = effective_batch_denominators(model, batches, device)
    channel = ExampleAWGN()
    model.channel = channel
    model.eval()
    for case in ['no_noise', 'fixed_per_sample_uniform_awgn']:
        case_started = time.monotonic()
        pooled, structural = {}, {}
        totals = {key: 0.0 for key in weights}
        for batch_index, batch in enumerate(batches):
            budget.check()
            row_ids = [r['row_id'] for r in rows[batch_index*8:batch_index*8+8]]
            channel.row_ids = row_ids
            snrs = sample_snrs(row_ids) if case != 'no_noise' else None
            snr = torch.tensor(snrs, device=device).reshape(-1, 1, 1) if snrs else None
            common = {**identity, 'case': case, 'microbatch_index': batch_index, 'row_ids': row_ids}
            budget.identity = common
            budget.presentations += len(row_ids)
            context = nullcontext() if batch_index < 8 else torch.no_grad()
            with context, channel.replay(SEED, 'e3', capture=True):
                values = batch_losses(model, batch, training, snr, return_stats=True,
                                      valid_only_kl=training.get('valid_only_kl', False))
                components = {key: values[f'{key}_numerator']/denominators[key] for key in weights}
                losses = {}
                for key in weights:
                    numerator = values[f'{key}_numerator'].detach().item()
                    denominator = values[f'{key}_denominator'].item()
                    totals[key] += numerator
                    losses[key] = {'numerator': numerator, 'denominator': denominator,
                                   'mean': numerator/denominator if denominator else None}
                write('e3_losses.jsonl', {**common, 'components': losses, 'weights': weights,
                                        'snr_db': snrs, 'noise_draws': channel.draw_summaries,
                                        'source_tokens_valid': int(batch['attention_mask'].sum()),
                                        'source_tokens_allocated': batch['attention_mask'].numel(),
                                        'target_tokens_valid': int((batch['labels'] != -100).sum()),
                                        'target_tokens_allocated': batch['labels'].numel(),
                                        'allocated': dict(model.channel_uses), 'valid': dict(model.channel_uses_valid)})
                for stream, original, reconstructed in [
                    ('hidden', model.activation, model.reconstruction),
                    ('memory', model.memory_activation, model.memory_reconstruction),
                ]:
                    if original is None or reconstructed is None:
                        write('e3_features.jsonl', {**common, 'stream': stream, 'status': 'NOT_APPLICABLE'})
                        continue
                    mask = batch['labels'] != -100 if stream == 'hidden' and model.split['stack'] == 'dec' else batch['attention_mask']
                    features = feature_rows(original, reconstructed, mask.to(device))
                    for row_id, feature in zip(row_ids, features):
                        write('e3_features.jsonl', {**common, 'row_id': row_id, 'stream': stream,
                                                  'statistics': feature, 'demo_query_spans': 'NOT_MEASURED'})
                if batch_index < 8:
                    gradients = component_vjps(components, groups, budget)
                    for group in groups:
                        vectors = {key: gradients[key][group]['vector'] for key in weights}
                        # Report microbatch mean gradients, while accumulating contributions
                        # scaled to the full64 token/sample denominators.
                        local = {key: vector * float(denominators[key])/max(1, losses[key]['denominator'])
                                 for key, vector in vectors.items()}
                        zeros = {key: gradients[key][group]['structural_zero_parameters'] for key in weights}
                        write('e3_gradients.jsonl', {**common, 'scope': 'microbatch_mean', 'group': group,
                              'weights': weights, 'denominators': {key: losses[key]['denominator'] for key in weights},
                              'structural_zero_parameters': zeros, **gradient_summary(local, weights)})
                        for key, vector in vectors.items():
                            index = (group, key)
                            if index not in pooled:
                                pooled[index] = vector.clone()
                                structural[index] = set(zeros[key])
                            else:
                                pooled[index].add_(vector)
                                structural[index].intersection_update(zeros[key])
                    del gradients, components
            del values
            model.activation = model.reconstruction = None
            model.memory_activation = model.memory_reconstruction = None
        for group in groups:
            write('e3_gradients.jsonl', {**identity, 'case': case, 'scope': 'pooled64', 'group': group,
                  'row_ids': [r['row_id'] for r in rows[:64]], 'weights': weights,
                  'denominators': {k: float(denominators[k]) for k in weights},
                  'structural_zero_parameters': {k: sorted(structural[(group, k)]) for k in weights},
                  **gradient_summary({k: pooled[(group, k)] for k in weights}, weights)})
        write('e3_losses.jsonl', {**identity, 'case': case, 'scope': 'pooled128', 'weights': weights,
              'case_wall_seconds': time.monotonic()-case_started,
              'components': {k: {'numerator': v, 'denominator': float(selection_denominators[k]),
                                 'mean': v/float(selection_denominators[k])} for k, v in totals.items()}})

    forward_hook.remove()
