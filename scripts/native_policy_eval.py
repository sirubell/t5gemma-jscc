"""Fixed development-panel policy measurement; no final-validation evaluation."""
import argparse
import hashlib
import json
import platform
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
import torch.nn.functional as F

from jscc.checkpoint_policy import validate_checkpoint_step
from jscc.harness_payload import ContinuationLengths, payload_harness_class
from jscc.models.split_model import build_model
from jscc.runtime import configure_training_determinism, isolated_rng, source_state
from scripts.native_preflight_eval import digest


POLICIES = [('bfloat16', 64), ('bfloat16', 128), ('bfloat16', 256), ('float32', 64)]


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


class Evidence:
    def __init__(self, output, deadline, cap=8192):
        self.output, self.deadline, self.cap = Path(output), deadline, cap
        self.reserved = 0
        self.write_seconds = 0.0

    def write(self, name, value):
        start = time.perf_counter()
        with (self.output / name).open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')
            stream.flush()
        self.write_seconds += time.perf_counter() - start

    def reserve(self, count, identity):
        if count < 0 or self.reserved + count > self.cap:
            raise RuntimeError('candidate request cap exceeded before dispatch')
        if time.monotonic() >= self.deadline:
            raise TimeoutError('evaluation deadline reached before dispatch')
        self.reserved += count
        self.write('reservations.jsonl', {'identity': identity, 'requests': count,
                                        'cumulative_reserved': self.reserved})


def stratified_panel(rows, size=192):
    """Round-robin source x input quartile x continuation quartile, hash ties."""
    if len(rows) < size or len({r['doc_id'] for r in rows}) != len(rows):
        raise ValueError('insufficient or duplicate holdout identities')
    ranks = {}
    for field in ['input_length', 'continuation_length']:
        order = sorted(rows, key=lambda r: (r[field], r['doc_id']))
        ranks[field] = {r['doc_id']: min(3, i * 4 // len(rows)) for i, r in enumerate(order)}
    buckets = defaultdict(list)
    for row in rows:
        key = (row['source'], ranks['input_length'][row['doc_id']],
               ranks['continuation_length'][row['doc_id']])
        buckets[key].append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda r: digest(['panel-v1', 0, r['doc_id']]))
    selected = []
    keys = sorted(buckets)
    while len(selected) < size:
        for key in keys:
            if buckets[key] and len(selected) < size:
                row = dict(buckets[key].pop(0))
                row['stratum'] = list(key)
                selected.append(row)
    return selected


def prepare_panel(config, data_ids, tokenizer, base):
    from lm_eval.api.registry import get_model
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.tasks import TaskManager
    from lm_eval.tasks._yaml_loader import load_yaml
    if data_ids.get('validation_split') != 'train':
        raise ValueError('development panel must use recorded train holdout')
    holdout = data_ids['validation_rows']
    if set(holdout) & set(data_ids['train_rows']):
        raise ValueError('training and development identities overlap')
    recipe = TaskManager().task_index['hellaswag'].yaml_path
    if recipe is None:
        raise RuntimeError('missing installed HellaSwag task recipe')
    task_config = load_yaml(recipe, resolve_func=True)
    task_config['dataset_path'] = config['data']['name']
    task_config['dataset_kwargs'] = {'revision': config['data']['revision']}
    task = ConfigurableTask(config=task_config)
    task.set_fewshot_seed(1234)
    docs = task.training_docs()
    # Keep all actual row IDs in the resolved few-shot examples. Exclude holdout.
    pool = [dict(docs[i], dataset_row_id=i) for i in data_ids['train_rows']]
    task.sampler.replace_df(pool)
    sampled = []
    original_sample = task.sampler.sample

    def sample(*args, **kwargs):
        selected = original_sample(*args, **kwargs)
        sampled[:] = selected
        return selected

    task.sampler.sample = sample
    adapter = cast(type[Any], get_model('hf'))(pretrained=base, tokenizer=tokenizer,
                                               backend='seq2seq', batch_size=64)
    rows = []
    for row_id in sorted(holdout):
        doc = dict(docs[row_id])
        context = task.fewshot_context(doc, num_fewshot=config['evaluation']['num_fewshot'])
        instances = task.construct_requests(doc, context, metadata=('hellaswag', row_id, 1))
        if not isinstance(instances, list):
            raise ValueError('expected multiple-choice request list')
        pairs = [list(instance.args) for instance in instances]
        if len(pairs) != 4:
            raise ValueError('expected four candidate requests per document')
        tokens = []
        for pair in pairs:
            inp, target = adapter._encode_pair(*pair)
            tokens.append({'context': inp[-adapter.max_length:], 'target': target[-adapter.max_length:],
                           'untruncated_lengths': [len(inp), len(target)]})
        rows.append({'doc_id': row_id, 'dataset_split': 'train', 'doc': doc,
                     'source': str(doc.get('source_id', 'unknown')).split('~')[0],
                     'exact_candidate_arguments': pairs, 'candidate_tokens': tokens,
                     'fewshot_examples': list(sampled),
                     'input_length': max(len(t['context']) for t in tokens),
                     'continuation_length': max(len(t['target']) for t in tokens)})
    return stratified_panel(rows), {'max_length': adapter.max_length, 'fewshot_seed': 1234,
                                  'fewshot_pool': 'recorded optimization train_rows excluding holdout',
                                  'task_recipe_sha256': file_hash(recipe),
                                  'selection': 'source x input/continuation rank quartiles; round-robin sorted strata; SHA256 panel-v1 seed0 ties',
                                  'holdout_ids_sha256': digest(holdout)}


def paired_scores(values, target):
    fp32 = F.log_softmax(values.float(), dim=-1).gather(1, target[:, None]).squeeze(1)
    native = F.log_softmax(values, dim=-1).gather(1, target[:, None]).squeeze(1)
    return {'fp32_token_log_probs': fp32.tolist(), 'fp32_score': float(fp32.sum(dtype=torch.float32)),
            'native_token_log_probs': native.float().tolist(), 'native_score': float(native.sum()),
            'logits_dtype': str(values.dtype), 'native_reduction_dtype': str(native.dtype),
            'valid_logits_sha256': tensor_hash(values)}


def score_pass(model, tokenizer, rows, batch, precision, snr, evidence, identity):
    from lm_eval.api.registry import get_model
    start = time.perf_counter()
    cls = payload_harness_class(cast(type[Any], get_model('hf')))
    adapter = cls(pretrained=model.base, tokenizer=tokenizer, backend='seq2seq',
                  batch_size=batch, communication_model=model)
    encoded, lookup = [], defaultdict(list)
    for row in rows:
        for choice, pair in enumerate(row['exact_candidate_arguments']):
            inp, target = adapter._encode_pair(*pair)
            inp, target = inp[-adapter.max_length:], target[-adapter.max_length:]
            if inp != row['candidate_tokens'][choice]['context'] or target != row['candidate_tokens'][choice]['target']:
                raise RuntimeError('candidate token identity changed across policies')
            encoded.append((pair, inp, target))
            lookup[(tuple(inp), tuple(target))].append((row['doc_id'], choice))
    lengths = ContinuationLengths(encoded, adapter.max_length)
    requests = [SimpleNamespace(args=tuple(pair), task_name='hellaswag') for pair, _, _ in encoded]
    preparation = time.perf_counter() - start
    original = adapter._model_call
    scores_by_id, forward_times = {}, []
    actual = 0
    write_before = evidence.write_seconds

    def capture(inps, attn_mask=None, labels=None):
        nonlocal actual
        if time.monotonic() >= evidence.deadline:
            raise TimeoutError('deadline before forward')
        actual += len(inps)
        assert attn_mask is not None and labels is not None
        trace = {'policy': identity, 'batch_index': len(forward_times),
                 'input_ids': inps.tolist(), 'attention_mask': attn_mask.tolist(), 'labels': labels.tolist(),
                 'prepared_decoder_ids': model.base.prepare_decoder_input_ids_from_labels(labels=labels).tolist(),
                 'cuda_rng_before_sha256': tensor_hash(torch.cuda.get_rng_state()),
                 'snr_db': snr, 'noise_comparison': 'batch-dependent draws; not arithmetic equivalence'}
        evidence.write('batch-attempts.jsonl', trace)
        torch.cuda.synchronize()
        begin = time.perf_counter()
        logits = original(inps, attn_mask=attn_mask, labels=labels)
        torch.cuda.synchronize()
        forward_times.append(time.perf_counter() - begin)
        valid = lengths.mask(inps, attn_mask, labels)
        items = []
        for i, count in enumerate(valid.sum(1).tolist()):
            target = labels[i, :count]
            key = (tuple(inps[i][attn_mask[i].bool()].tolist()), tuple(target.tolist()))
            item = paired_scores(logits[i, :count], target)
            item['identities'] = lookup[key]
            for candidate_id in lookup[key]:
                scores_by_id[candidate_id] = item
            items.append(item)
        evidence.write('batch-results.jsonl', {'policy': identity, 'batch_index': len(forward_times)-1,
                       'forward_seconds': forward_times[-1], 'candidates': items,
                       'cuda_rng_after_sha256': tensor_hash(torch.cuda.get_rng_state())})
        return logits

    adapter._model_call = capture
    evidence.reserve(len(requests), identity)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with isolated_rng(0), model.transmission(snr):
        raw_scores = adapter.loglikelihood(requests, disable_tqdm=True)
    torch.cuda.synchronize()
    call_wall = time.perf_counter() - started
    compact = []
    for index, row in enumerate(rows):
        native = [raw_scores[index*4+j][0] for j in range(4)]
        fp32 = [scores_by_id[(row['doc_id'], j)]['fp32_score'] for j in range(4)]
        if not torch.isfinite(torch.tensor([native, fp32])).all():
            raise RuntimeError('nonfinite candidate score')
        denominators = [len(choice) for choice in row['doc']['choices']]
        normalized = [[score/n for score, n in zip(values, denominators, strict=True)] for values in [native, fp32]]
        item = {'policy': identity, 'doc_id': row['doc_id'], 'gold': row['doc']['gold'],
                'native_harness_scores': native, 'fp32_scores': fp32, 'denominators': denominators,
                'normalized_scores': normalized,
                'predictions': [max(range(4), key=lambda i: values[i]) for values in normalized],
                'margins': [sorted(values, reverse=True)[0]-sorted(values, reverse=True)[1] for values in normalized]}
        evidence.write('items.jsonl', item)
        compact.append(item)
    if actual != len(requests):
        raise RuntimeError('unexpected harness request count')
    result = {'policy': identity, 'status': 'MEASURED', 'precision': precision, 'batch_size': batch,
              'candidate_requests': actual, 'request_sha256': digest(encoded),
              'preparation_seconds': preparation, 'forward_seconds': forward_times,
              'forward_seconds_sum': sum(forward_times), 'whole_harness_call_seconds': call_wall,
              'evidence_write_seconds_inclusive': evidence.write_seconds-write_before,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
              'harness_softmax_dtype': str(adapter.softmax_dtype),
              'attention_backend': str(model.base.config._attn_implementation),
              'allocated': dict(model.channel_uses_allocated), 'valid': dict(model.valid_payload_counts)}
    evidence.write('policies.jsonl', result)
    return result


def replay_fixtures(model, tokenizer, fixtures, evidence, precision):
    from lm_eval.api.registry import get_model
    adapter = cast(type[Any], get_model('hf'))(pretrained=model.base, tokenizer=tokenizer,
                                               backend='seq2seq', batch_size=64)
    for index, fixture in enumerate(fixtures):
        batch = fixture['batch']
        identity = f'fixture-{index}-{precision}'
        evidence.reserve(len(batch['input_ids']), identity)
        tensors = {key: torch.tensor(batch[key], device=next(model.base.parameters()).device) for key in ['input_ids', 'attention_mask', 'labels']}
        evidence.write('fixture-attempts.jsonl', {'identity': identity, 'fixture': fixture})
        with isolated_rng(0), model.transmission(None, bypass=True), torch.no_grad():
            logits = adapter._model_call(tensors['input_ids'], attn_mask=tensors['attention_mask'], labels=tensors['labels'])
        for i, companion in enumerate(batch['companions']):
            count = companion.get('continuation_length', len(batch['labels'][i]))
            evidence.write('fixture-results.jsonl', {'identity': identity, 'companion': companion,
                           'includes_padding': 'continuation_length' not in companion,
                           **paired_scores(logits[i, :count], tensors['labels'][i, :count])})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--expected-step', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path)
    parser.add_argument('--seconds', type=int, default=1500)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 1500:
        parser.error('--seconds must be in 1..1500 (within shared 60-minute route allocation)')
    if args.expected_step != 80:
        parser.error('bounded benchmark requires explicit first64 final step80')
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    evidence = Evidence(args.output, time.monotonic()+args.seconds)
    checkpoint = args.run / args.checkpoint
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    validate_checkpoint_step(state, args.expected_step)
    config = state['config']
    if config.get('protocol') != 'corrected-baseline-v2-native-decoder-inputs':
        raise ValueError('native-v2 checkpoint required')
    configure_training_determinism(config['training'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer, model = build_model(config)
    model.load_communication_state(state)
    model.eval()
    if next(model.base.parameters()).dtype != torch.bfloat16:
        raise ValueError('must load BF16 backbone before FP32 upward cast')
    setup_seconds = time.perf_counter()-started
    prep_start = time.perf_counter()
    panel, metadata = prepare_panel(config, state['data_ids'], tokenizer, model.base)
    evidence.write('panel.jsonl', {'documents': panel, 'metadata': metadata})
    fixtures = json.loads(args.fixtures.read_text())['fixtures'] if args.fixtures else []
    if 2 * sum(len(f['batch']['input_ids']) for f in fixtures) > 512:
        raise ValueError('original fixture exceeds per-route 512 request allowance')
    identity = {'config': config, 'checkpoint_sha256': file_hash(checkpoint),
                'panel_sha256': digest(panel), 'tokenization_sha256': digest([r['candidate_tokens'] for r in panel]),
                'metadata': metadata, 'torch': torch.__version__, 'python': platform.python_version(),
                'source': source_state(),
                'source_manifest_sha256': file_hash(Path(__file__).resolve().parents[1] / 'policy-input/source.json') if (Path(__file__).resolve().parents[1] / 'policy-input/source.json').exists() else None,
                'runner_sha256': file_hash(__file__),
                'tokenizer_name': tokenizer.name_or_path,
                'tokenizer_revision': config['model'].get('revision'),
                'device': torch.cuda.get_device_name(), 'deterministic': torch.are_deterministic_algorithms_enabled(),
                'tf32': False, 'seed': 0, 'policies': POLICIES, 'conditions': [None, -6.0],
                'fixtures_sha256': file_hash(args.fixtures) if args.fixtures else None,
                'FP32_reference': 'same BF16-loaded backbone cast upward; not original FP32 weights',
                'formal_execution_authorized': False}
    evidence.write('identity.jsonl', {**identity, 'policy_hash': digest(identity)})
    prep_seconds = time.perf_counter()-prep_start
    results = []
    stop = False
    bf16_failed_batch = None
    for precision, batch in POLICIES:
        if stop:
            break
        if precision == 'bfloat16' and bf16_failed_batch is not None and batch > bf16_failed_batch:
            evidence.write('failures.jsonl', {'policy': f'{precision}-b{batch}', 'status': 'SKIPPED_AFTER_SMALLER_OOM'})
            continue
        if precision == 'float32':
            try:
                model.base.float()
            except torch.cuda.OutOfMemoryError as exc:
                evidence.write('failures.jsonl', {'policy': 'float32-cast', 'error': str(exc), 'retry': False})
                torch.cuda.empty_cache()
                break
        for snr in [None, -6.0]:
            name = f'{precision}-b{batch}-{snr}'
            try:
                results.append(score_pass(model, tokenizer, panel, batch, precision, snr, evidence, name))
            except (torch.cuda.OutOfMemoryError, TimeoutError) as exc:
                evidence.write('failures.jsonl', {'policy': name, 'error': str(exc), 'retry': False})
                torch.cuda.empty_cache()
                if precision == 'bfloat16':
                    bf16_failed_batch = batch
                stop = isinstance(exc, TimeoutError) or (precision == 'bfloat16' and batch == 64)
                break
            except RuntimeError as exc:
                evidence.write('failures.jsonl', {'policy': name, 'error': str(exc), 'status': 'STOP_ROUTE', 'retry': False})
                stop = True
                break
        if fixtures and (precision == 'float32' or batch == 64) and not stop:
            try:
                replay_fixtures(model, tokenizer, fixtures, evidence, precision)
            except (torch.cuda.OutOfMemoryError, TimeoutError) as exc:
                evidence.write('failures.jsonl', {'fixture_precision': precision, 'error': str(exc), 'retry': False})
                torch.cuda.empty_cache()
                stop = isinstance(exc, TimeoutError)
            except RuntimeError as exc:
                evidence.write('failures.jsonl', {'fixture_precision': precision, 'error': str(exc), 'status': 'STOP_ROUTE', 'retry': False})
                stop = True
    evidence.write('summary.jsonl', {'status': 'MEASUREMENT_ONLY', 'policies': results,
                   'reserved_candidate_requests': evidence.reserved, 'setup_seconds': setup_seconds,
                   'request_preparation_seconds': prep_seconds, 'total_wall_seconds': time.perf_counter()-started,
                   'evidence_write_seconds': evidence.write_seconds,
                   'timing': 'setup/preparation exclude scoring; forward nested in whole harness call; write spans overlap call; synchronize forward; first forward included, no warmup',
                   'fixtures': 'original exact tensor batches' if fixtures else 'BLOCKED: fixture not supplied',
                   'selection': 'no accuracy-driven policy winner; lower measured batch retained after OOM',
                   'noise': 'same seed, different batching changes channel draws; no numerical-equivalence claim'})


if __name__ == '__main__':
    main()
