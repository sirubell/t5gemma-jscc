"""Bounded target-GPU capacity checks after approval; never a formal checkpoint."""
import argparse
import copy
import json
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from jscc import training
from jscc.config import load_config, save_config
from jscc.presentation import PresentationSampler


def inventory(data, batch_size):
    rows = data.train.dataset
    lengths = [(len(x['input_ids']), len(x['label_ids'])) for x in rows]
    ranked = sorted(range(len(rows)), key=lambda i: (sum(lengths[i]), lengths[i], i), reverse=True)
    tensor = torch.tensor(lengths, dtype=torch.float32)
    formal = PresentationSampler(data.ids["train_rows"], 640000, 0)
    batched = tensor[formal.indices].reshape(5000, batch_size, 2).amax(dim=1)
    cost = batch_size * (batched[:, 0]**2 + batched[:, 1]**2 + batched[:, 0]*batched[:, 1])
    worst = int(cost.argmax())
    worst_indices = formal.indices[worst*batch_size:(worst+1)*batch_size].tolist()
    stress = worst_indices + ranked[:batch_size*3]
    return stress, {'rows': len(rows), 'input_target_quantiles': tensor.quantile(torch.tensor([0., .5, .9, .99, 1.]), dim=0).tolist(),
                    'maximum_formal_padded_cost_batch': {'index': worst, 'input_target_max': batched[worst].tolist(), 'proxy': float(cost[worst]), 'formula': 'B*(source_max^2+target_max^2+source_max*target_max), not measured FLOPs'},
                    'stress_rule': 'first actual640k-stream maximum attention-cost-proxy batch, then top384 natural length rows; no synthetic/truncation change',
                    'stress_row_ids': [data.ids['train_rows'][i] for i in stress]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--indices', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    indices = [int(x) for x in args.indices.split(',')]
    if len(set(indices)) != len(indices) or len(indices) > 6:
        p.error('one bounded job supports at most six distinct configurations')
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.manifest.parent
    original = training.load_data
    results = []
    for index in indices:
        entry = manifest['runs'][index]
        config = copy.deepcopy(load_config(root/entry['config']))
        config['run'].update(name='capacity-'+entry['run_id'], output_dir=str(args.output/'runs'))
        t = config['training']
        assert t['batch_size'] == 128 and t['gradient_accumulation'] == 1
        t.update(max_steps=4, schedule_steps=4, warmup_ratio=.25, eval_every=4, save_steps=[4], validation_batches=1, log_every=1)
        t['presentation_stream']['total_presentations'] = 512
        t['paired_randomness']['audit_steps'] = [1, 4]
        t['feature_summary']['steps'] = [0, 4]
        save_config(config, args.output/(entry['run_id']+'.yaml'))

        def load(*a, **kw):
            data = original(*a, **kw)
            order, report = inventory(data, 128)
            sampler = PresentationSampler(data.ids['train_rows'], 512, 0)
            sampler.indices = torch.tensor(order[:512], dtype=torch.int64)
            sampler.actual_ids = torch.tensor(data.ids['train_rows'], dtype=torch.int64)[sampler.indices]
            data.train = DataLoader(data.train.dataset, batch_size=128, sampler=sampler,
                                    num_workers=4, collate_fn=data.train.collate_fn, pin_memory=True)
            report['scope'] = 'capacity-only explicit natural-long order; overrides production permutation; reset before fresh formal run'
            (args.output/(entry['run_id']+'-lengths.json')).write_text(json.dumps(report, indent=2)+'\n')
            return data

        try:
            with patch.object(training, 'load_data', load):
                run = training.train(config)
            presentation_path = run/'presentations.json'
            presentation = json.loads(presentation_path.read_text())
            presentation['policy'] = 'capacity-natural-long-explicit-ids-v1'
            presentation['algorithm'] = 'explicit IDs from the adjacent length inventory; production permutation overridden for capacity only'
            presentation_path.write_text(json.dumps(presentation, indent=2)+'\n')
            completion = json.loads((run/'completion.json').read_text())
            assert completion['step'] == 4 and completion['status'] == 'FULL_BUDGET_COMPLETED'
            metrics = [json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines()]
            updates = [x for x in metrics if x['phase'] == 'train' and x['step'] > 1]
            assert any(x['parameter_update_l2'] > 0 for x in updates)
            results.append({'index': index, 'run_id': entry['run_id'], 'status': 'PASS', 'run': str(run), 'presentations': 512, 'updates': 4})
        except Exception as exc:
            results.append({'index': index, 'run_id': entry['run_id'], 'status': 'FAIL', 'error': str(exc), 'no_retry': True})
            (args.output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
            raise
        (args.output/'results.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
