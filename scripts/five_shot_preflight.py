"""Bounded, disposable capacity/timing and communication reload smoke for one split."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time

import torch

from jscc.config import load_config
from jscc.data.hellaswag import load_data
from jscc.evaluation_policy import scoring_kwargs
from jscc.harness_payload import ContinuationLengths, payload_harness_class
from jscc.models.split_model import build_model
from jscc.runtime import configure_training_determinism, isolated_rng, prepare_trainable_parameters, seed_everything
from scripts.frozen_prompt_gap import sha, tensor_batch, write
from scripts.prompt_alignment_benchmark import inventory, one_update


def check_gradients(model):
    result = {}
    for name, codec in [('hidden', model.codec), ('memory', model.memory_codec)]:
        if codec is None:
            continue
        grads = [p.grad for p in codec.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise ValueError(f'{name} gradients missing or nonfinite')
        norm = sum(float(g.float().square().sum()) for g in grads)**.5
        if norm <= 0:
            raise ValueError(f'{name} gradient norm is zero')
        result[name] = norm
    return result


def score(model, adapter, requests, snr, path, phase):
    tensors = tensor_batch(requests, next(model.base.parameters()).device)
    encoded = [((r['context'], r['continuation']), r['context_token_ids'], r['continuation_token_ids']) for r in requests]
    adapter._payload_lengths = ContinuationLengths(encoded, 2048)
    started = time.perf_counter()
    with isolated_rng(0), torch.inference_mode(), model.transmission(snr, bypass=False):
        logits = adapter._model_call(tensors['input_ids'], attn_mask=tensors['attention_mask'], labels=tensors['labels'])
        if logits.dtype != torch.bfloat16:
            raise ValueError('Expected BF16 forward logits')
        logp = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
        values = [float(logp[i, :len(r['continuation_token_ids'])].gather(
            -1, tensors['labels'][i, :len(r['continuation_token_ids']), None]).sum()) for i, r in enumerate(requests)]
        if not all(torch.isfinite(torch.tensor(values))):
            raise ValueError('Nonfinite inference scores')
        payload = dict(model.valid_payload_counts)
    hidden_tokens = sum(len(r['continuation_token_ids']) if model.split['stack'] == 'dec' else len(r['context_token_ids']) for r in requests)
    memory_tokens = sum(len(r['context_token_ids']) for r in requests) if model.memory_codec is not None else 0
    if payload != {'hidden': hidden_tokens*512, 'memory': memory_tokens*512}:
        raise ValueError(f'Unexpected valid stream payload {payload}')
    record = {'phase': phase, 'snr': snr, 'requests': len(requests), 'seconds': time.perf_counter()-started,
              'payload': payload, 'samples': [{**r, 'score': value} for r, value in zip(requests, values)]}
    write(path, record)
    return values


def run(config_path, prepared, output, microbatch):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    stage = 'setup'
    try:
        config = load_config(config_path)
        if config['data']['prompt_policy']['mode'] != 'five_shot' or microbatch not in (64, 128):
            raise ValueError('Preflight supports only five_shot, micro64/128')
        t = config['training']
        t['batch_size'], t['gradient_accumulation'] = microbatch, 128//microbatch
        configure_training_determinism(t)
        seed_everything(config['seed'])
        tokenizer, model = build_model(config)
        prepare_trainable_parameters(model)
        model.train()
        data = load_data(config, tokenizer)
        info = inventory(data, microbatch)
        info['stress_scope'] = 'actual prospective640k presentation stream; natural attention-cost proxy, not an exhaustive memory maximum'
        write(output/'inventory.json', info)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=.01)
        torch.cuda.reset_peak_memory_stats()
        stage = 'stress_update'
        stress = data.train.collate_fn([data.train.dataset[i] for i in info['stress_dataset_indices']])
        one_update(model, optimizer, [stress], config)
        check_gradients(model)
        iterator = iter(data.train)
        updates = []
        stage = 'timed_updates'
        for i in range(6):
            batches = [next(iterator) for _ in range(128//microbatch)]
            torch.cuda.synchronize()
            start = time.perf_counter()
            loss, norm = one_update(model, optimizer, batches, config)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            updates.append({'update': i+1, 'loss': loss, 'gradient_norm': norm, 'seconds': elapsed,
                            'stream_gradient_norms': check_gradients(model)})
        precision = {str(v.dtype) for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v)}
        if precision != {'torch.float32'}:
            raise ValueError(f'Unexpected optimizer precision: {precision}')
        metrics = {'config_sha256': sha(config_path), 'microbatch': microbatch, 'accumulation': 128//microbatch,
                   'effective_batch': 128, 'split': config['split'], 'optimizer_updates': 7,
                   'presentations': 768+microbatch, 'warmup_updates': 2, 'timed_updates': 4,
                   'median_seconds_per_update': statistics.median(x['seconds'] for x in updates[2:]),
                   'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                   'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30,
                   'updates': updates, 'optimizer_dtypes': sorted(precision)}
        write(output/'training.json', metrics)
        stage = 'checkpoint_and_evaluation'
        model.eval()
        from lm_eval.models.huggingface import HFLM
        adapter = payload_harness_class(HFLM)(communication_model=model, pretrained=model.base, tokenizer=tokenizer,
                    backend='seq2seq', batch_size=64, max_length=2048,
                    **scoring_kwargs({'scoring_policy': 'fp32-v1'}, model))
        fixture = prepared/'encoded-requests.jsonl'
        if sha(fixture) != '5422c0cb564ec405efcbfb1a2f265087286748418f803fa97fde15d9b346dd74':
            raise ValueError('Unexpected frozen evaluation fixture')
        requests = [json.loads(line) for line in fixture.read_text().splitlines() if json.loads(line)['shot'] == 5][:64]
        before = score(model, adapter, requests, None, output/'eval-before.json', 'before_reload')
        noisy_before = score(model, adapter, requests, -6., output/'eval-noisy-before.json', 'noisy_before_reload')
        checkpoint = output/'disposable-smoke.pt'
        torch.save({'codec': model.codec.state_dict(), 'channel': model.channel.state_dict(),
                    'memory_codec': model.memory_codec.state_dict() if model.memory_codec else None,
                    'optimizer': optimizer.state_dict()}, checkpoint)
        checkpoint_hash = sha(checkpoint)
        # Force an actual reload rather than merely comparing an unchanged live model.
        with torch.no_grad():
            for codec in [model.codec, model.memory_codec]:
                if codec is not None:
                    for parameter in codec.parameters():
                        parameter.zero_()
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_communication_state(state)
        after = score(model, adapter, requests, None, output/'eval-after.json', 'after_reload')
        if not torch.allclose(torch.tensor(before), torch.tensor(after), atol=1e-5, rtol=1e-5):
            raise ValueError('Checkpoint reload score mismatch')
        noisy_after = score(model, adapter, requests, -6., output/'eval-noisy.json', 'noisy_after_reload')
        if not torch.allclose(torch.tensor(noisy_before), torch.tensor(noisy_after), atol=1e-5, rtol=1e-5):
            raise ValueError('Noisy checkpoint reload score mismatch')
        write(output/'completion.json', {**metrics, 'status': 'PASS', 'quality_evidence': False,
                'elapsed_seconds': time.perf_counter()-started, 'evaluation_requests': 256,
                'checkpoint_sha256': checkpoint_hash, 'checkpoint_reload': 'codec/channel score replay; optimizer saved but not resume-tested',
                'memory_stream': model.memory_codec is not None})
        del state, model, optimizer, adapter
        gc.collect()
    except BaseException as exc:
        write(output/'failure.json', {'status': 'FAILED', 'stage': stage, 'error_type': type(exc).__name__,
                                     'error': str(exc), 'elapsed_seconds': time.perf_counter()-started})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--microbatch', type=int, choices=[64,128], required=True)
    a = p.parse_args()
    run(a.config, a.prepared, a.output, a.microbatch)


if __name__ == '__main__':
    main()
