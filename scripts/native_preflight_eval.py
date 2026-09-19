"""Bounded native preflight: decoder accounting or encoder precision/batch probes.

No full-panel or quality study. Load only the fresh preflight checkpoint. FP32
uses the same BF16-loaded backbone cast upward, not different pretrained weights.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
import torch.nn.functional as F

from jscc.harness_payload import ContinuationLengths, payload_harness_class
from jscc.models.split_model import build_model
from jscc.runtime import configure_training_determinism, isolated_rng


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def score_pass(model, tokenizer, rows, batch, precision, snr, masked, output):
    from lm_eval.api.registry import get_model
    cls = cast(type[Any], get_model('hf'))
    if masked:
        cls = payload_harness_class(cls)
    kwargs = {'communication_model': model} if masked else {}
    adapter = cls(pretrained=model.base, tokenizer=tokenizer, backend='seq2seq',
                  batch_size=batch, **kwargs)
    requests = [SimpleNamespace(args=tuple(pair), task_name='hellaswag')
                for row in rows for pair in row['exact_candidate_arguments']]
    encoded = []
    lookup = {}
    for row in rows:
        for choice, pair in enumerate(row['exact_candidate_arguments']):
            context, target = adapter._encode_pair(*pair)
            context, target = context[-adapter.max_length:], target[-adapter.max_length:]
            encoded.append((pair, context, target))
            lookup.setdefault((tuple(context), tuple(target)), []).append([row['doc_id'], choice])
    lengths = ContinuationLengths(encoded, adapter.max_length)
    traces, selected_scores, forward_seconds = [], {}, []
    original = adapter._model_call
    candidate_calls = 0

    def capture(inps, attn_mask=None, labels=None):
        nonlocal candidate_calls
        candidate_calls += len(inps)
        assert attn_mask is not None and labels is not None
        torch.cuda.synchronize()
        start = time.perf_counter()
        logits = original(inps, attn_mask=attn_mask, labels=labels)
        torch.cuda.synchronize()
        forward_seconds.append(time.perf_counter() - start)
        valid = lengths.mask(inps, attn_mask, labels)
        trace = {'input_ids': inps.tolist(), 'attention_mask': attn_mask.tolist(),
                 'labels': labels.tolist(), 'payload_mask': valid.tolist(),
                 'prepared_decoder_ids': model.base.prepare_decoder_input_ids_from_labels(labels=labels).tolist(),
                 'candidates': []}
        for i, count in enumerate(valid.sum(1).tolist()):
            target = labels[i, :count]
            values = logits[i, :count]
            selected = values.gather(1, target[:, None]).squeeze(1).float().tolist()
            fp32 = F.log_softmax(values.float(), -1).gather(1, target[:, None]).squeeze(1)
            key = (tuple(inps[i][attn_mask[i].bool()].tolist()), tuple(target.tolist()))
            item = {'identities': lookup[key], 'selected_logits': selected,
                    'fp32_token_log_probs': fp32.tolist(), 'fp32_score': float(fp32.sum()),
                    'valid_logits_sha256': hashlib.sha256(values.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()}
            for identity in lookup[key]:
                selected_scores[tuple(identity)] = item
            trace['candidates'].append(item)
        traces.append(trace)
        return logits

    adapter._model_call = capture
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        with isolated_rng(0), model.transmission(snr):
            scores = adapter.loglikelihood(requests, disable_tqdm=True)
        torch.cuda.synchronize()
        compact = []
        for index, row in enumerate(rows):
            raw = [scores[index * 4 + j][0] for j in range(4)]
            fp32 = [selected_scores[(row['doc_id'], j)]['fp32_score'] for j in range(4)]
            denominator = [len(x) for x in row['doc']['choices']]
            norm = [s / n for s, n in zip(raw, denominator)]
            norm32 = [s / n for s, n in zip(fp32, denominator)]
            compact.append({'doc_id': row['doc_id'], 'gold': row['doc']['gold'],
                            'raw_scores': raw, 'fp32_scoring_scores': fp32,
                            'denominators': denominator, 'normalized_scores': norm,
                            'prediction_norm': max(range(4), key=lambda i: norm[i]),
                            'prediction_fp32_scoring': max(range(4), key=lambda i: norm32[i]),
                            'margin': sorted(norm, reverse=True)[0] - sorted(norm, reverse=True)[1]})
        expected_hidden = sum(len(x[2]) for x in encoded) * model.codec.config['bottleneck_dim']
        result = {'status': 'PASS_SCOPED', 'precision': precision, 'batch': batch, 'snr': snr,
                  'payload_mask_enabled': masked, 'candidate_requests': candidate_calls,
                  'compact': compact, 'batches': traces, 'request_hash': digest(encoded),
                  'forward_seconds': forward_seconds, 'forward_seconds_sum': sum(forward_seconds),
                  'instrumented_wall_seconds': time.perf_counter() - start,
                  'allocated': dict(model.channel_uses_allocated), 'valid': dict(model.valid_payload_counts),
                  'expected_valid_hidden': expected_hidden,
                  'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                  'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30,
                  'attention_backend': str(model.base.config._attn_implementation),
                  'softmax_dtype': str(adapter.softmax_dtype),
                  'actual_forward_dtype': str(next(model.base.parameters()).dtype),
                  'weights': 'same BF16-loaded weights; FP32 casts them upward',
                  'timing_limit': 'single short pass, includes first-forward warmup; forward-only excludes evidence transfers/scoring; not steady-state full-panel timing'}
        assert candidate_calls == len(rows)*4
        assert all(torch.isfinite(torch.tensor(x['raw_scores'])).all() for x in compact)
        if masked:
            assert result['valid']['hidden'] == expected_hidden
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        return result
    except Exception as exc:
        output.write_text(json.dumps({'status': 'FAIL', 'error': str(exc), 'candidate_requests_attempted': candidate_calls,
                                     'precision': precision, 'batch': batch}, indent=2)+'\n')
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--samples', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['payload', 'precision'], required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    state = torch.load(args.run/'step_000020.pt', map_location='cpu', weights_only=True)
    from jscc.checkpoint_policy import validate_checkpoint_step
    validate_checkpoint_step(state, 20)
    assert state['config']['protocol'] == 'corrected-baseline-v2-native-decoder-inputs'
    configure_training_determinism(state['config']['training'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer, model = build_model(state['config'])
    model.load_communication_state(state)
    model.eval()
    rows = [json.loads(line) for line in args.samples.read_text().splitlines()]
    rows = rows[:8 if args.mode == 'payload' else 16]
    assert len(rows) == (8 if args.mode == 'payload' else 16)
    (args.output/'documents.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    if args.mode == 'payload':
        assert model.split['stack'] == 'dec'
        for snr in [None, -6.0]:
            key = 'no_noise' if snr is None else 'minus6'
            before = score_pass(model, tokenizer, rows, 8, 'bfloat16', snr, False, args.output/(key+'-before.json'))
            after = score_pass(model, tokenizer, rows, 8, 'bfloat16', snr, True, args.output/(key+'-after.json'))
            assert before['compact'] == after['compact']
            assert before['allocated'] == after['allocated']
            assert before['valid']['memory'] == after['valid']['memory']
            def hashes(result):
                return [x['valid_logits_sha256'] for b in result['batches'] for x in b['candidates']]
            assert hashes(before) == hashes(after)
            assert model._decoder_valid_mask is None
        summary = {'status':'PASS_SCOPED', 'candidate_requests':128, 'conditions':['no_noise','-6'],
                   'assertions':'same full valid-position logits hashes, scores, allocated and memory counts; actual length hidden count; restored mask'}
    else:
        assert model.split['stack'] == 'enc'
        for precision, batches in [('bfloat16',[8,16,32,64]), ('float32',[8,16])]:
            if precision == 'float32':
                model.base.float()
            for batch in batches:
                score_pass(model, tokenizer, rows, batch, precision, None, False,
                           args.output/f'{precision}-b{batch}.json')
        summary = {'status':'MEASURED_NOT_POLICY_ACCEPTANCE', 'candidate_requests':384,
                   'scope':'16 development docs; no_noise short-trained codec; not full-panel quality, not proof of final training quality; precision/batch policy still pending'}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__ == '__main__':
    main()
