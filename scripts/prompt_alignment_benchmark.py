"""Bounded natural-length capacity/throughput checks, never retained as quality runs."""
import argparse
import copy
import gc
import json
from pathlib import Path
import statistics
import time

import torch
from transformers import AutoTokenizer

from jscc.config import load_config
from jscc.data.hellaswag import load_data
from jscc.models.split_model import build_model
from jscc.runtime import configure_training_determinism, prepare_trainable_parameters, seed_everything, sample_snr_db
from jscc import training


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def inventory(data, microbatch):
    lengths = torch.tensor([(len(r['input_ids']), len(r['label_ids'])) for r in data.train.dataset])
    indices = data.train.sampler.indices
    grouped = lengths[indices].reshape(-1, microbatch, 2).amax(1)
    cost = microbatch * (grouped[:, 0]**2 + grouped[:, 1]**2 + grouped[:, 0]*grouped[:, 1])
    where = int(cost.argmax())
    selected = indices[where*microbatch:(where+1)*microbatch].tolist()
    return {'length_quantiles': torch.quantile(lengths.float(), torch.tensor([0., .5, .9, .99, 1.]), dim=0).tolist(),
            'worst_attention_proxy_microbatch': where, 'worst_lengths': grouped[where].tolist(),
            'stress_dataset_indices': selected, 'stress_row_ids': [data.ids['train_rows'][i] for i in selected],
            'stress_scope': 'actual prospective128k presentation stream, natural lengths; no synthetic padding',
            'prompt_summary': {part: {k: v for k, v in getattr(data, 'prompt_evidence')[part].items() if k != 'rows'} for part in ('train', 'selection')}}


def one_update(model, optimizer, batches, config):
    settings = config['training']
    optimizer.zero_grad(set_to_none=True)
    denominators = training.effective_batch_denominators(model, batches, next(model.base.parameters()).device)
    values = []
    for batch in batches:
        snr = sample_snr_db(model, -6., 18., batch['labels'].shape[0])
        stats = training.batch_losses(model, batch, settings, snr, return_stats=True, valid_only_kl=True)
        loss = training.scaled_batch_loss(stats, settings, denominators)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite benchmark loss')
        loss.backward()
        values.append(float(loss.detach()))
    parameters = [p for p in model.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
    if not torch.isfinite(norm) or float(norm) <= 0:
        raise FloatingPointError('nonfinite benchmark gradient')
    optimizer.step()
    return sum(values), float(norm)


def prepare(configs, output):
    output.mkdir(parents=True, exist_ok=False)
    for path in configs:
        config = load_config(path)
        model = config['model']
        tokenizer = AutoTokenizer.from_pretrained(model['name'], revision=model['revision'], local_files_only=True)
        data = load_data(config, tokenizer)
        mode = config['data']['prompt_policy']['mode']
        info = inventory(data, config['training']['batch_size'])
        write(output / f'{mode}-inventory.json', info)
        write(output / f'{mode}-prompt-evidence.json', getattr(data, 'prompt_evidence'))
        write(output / f'{mode}-data-ids.json', data.ids)
        print(mode, info['length_quantiles'], flush=True)
    write(output / 'completion.json', {'status': 'PASS_CPU_DATA_PREPARE', 'model_loaded': False})


def benchmark(configs, output, microbatch):
    output.mkdir(parents=True, exist_ok=False)
    results = []
    try:
        for path in configs:
            config = copy.deepcopy(load_config(path))
            t = config['training']
            t['batch_size'], t['gradient_accumulation'] = microbatch, 128 // microbatch
            configure_training_determinism(t)
            seed_everything(config['seed'])
            tokenizer, model = build_model(config)
            prepare_trainable_parameters(model)
            model.train()
            data = load_data(config, tokenizer)
            info = inventory(data, microbatch)
            mode = config['data']['prompt_policy']['mode']
            write(output / f'{mode}-inventory.json', info)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=.01)
            torch.cuda.reset_peak_memory_stats()
            stress = data.train.collate_fn([data.train.dataset[i] for i in info['stress_dataset_indices']])
            one_update(model, optimizer, [stress], config)
            iterator = iter(data.train)
            times = []
            losses = []
            batches = []
            for update in range(6):
                batches = [next(iterator) for _ in range(t['gradient_accumulation'])]
                torch.cuda.synchronize()
                start = time.perf_counter()
                loss, grad = one_update(model, optimizer, batches, config)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                if update >= 2:
                    times.append(elapsed)
                losses.append({'update': update+1, 'loss': loss, 'grad_norm': grad, 'seconds': elapsed})
            results.append({'mode': mode, 'microbatch': microbatch, 'accumulation': 128//microbatch,
                            'stress_presentations': microbatch, 'regular_presentations': 768,
                            'optimizer_updates': 7, 'measured_seconds_per_update': times,
                            'median_seconds_per_update': statistics.median(times),
                            'max_seconds_per_update': max(times), 'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                            'updates': losses, 'status': 'PASS_FINITE_NATURAL_CAPACITY'})
            write(output / 'results.json', results)
            del model, optimizer, tokenizer, data, batches, stress, iterator
            gc.collect()
            torch.cuda.empty_cache()
        write(output / 'completion.json', {'status': 'PASS_BENCHMARK', 'results': results,
                                          'quality_evidence': False, 'formal_weights': 'fresh reset required'})
    except BaseException as error:
        write(output / 'failure.json', {'type': type(error).__name__, 'message': str(error), 'completed_modes': results})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['prepare', 'benchmark'])
    p.add_argument('--configs', nargs=2, type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--microbatch', type=int, choices=[16, 32], default=32)
    a = p.parse_args()
    if a.phase == 'prepare':
        prepare(a.configs, a.output)
    else:
        benchmark(a.configs, a.output, a.microbatch)


if __name__ == '__main__':
    main()
