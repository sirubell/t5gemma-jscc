"""Current primitive assertions extracted without resurrecting legacy execution."""
import json
from pathlib import Path
import torch
from sharing_fixtures import sharing_batches
from jscc import evaluation


def task_template():
    return {'task': 'hellaswag', 'input_ids': [80, 81], 'source_family_ids': ['dev80', 'dev81'],
            'prompt_policy': 'synthetic-native', 'noise': {'seed': 0, 'namespace': 'same-eval'},
            'layout': 'tiny-2x3', 'backend': 'eager', 'precision': 'fp32', 'scorer': 'fixture',
            'reference_corpus': 'fixture-dev', 'expected_items': 2,
            'settings': {'num_samples': 2, 'numerical_policy': 'native'}, 'data_settings': {},
            'conditions': ['no_noise', -6, 0, 6, 12, 18]}


def install_adapter(monkeypatch):
    calls = []
    def adapter(model, processor, settings, output, condition, data_settings):
        calls.append((dict(model.split), condition))
        rows = [{'sample_id': i, 'source_id': f'dev{i}', 'tokens': [2, 1],
                 'raw_correct': 1, 'normalized_correct': 1, 'normalized_prediction': 0,
                 'normalized_scores': [-.1, -.2]} for i in [80, 81]]
        (Path(output) / f'compact_{condition}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return {'acc': 1., 'acc_norm': 1.}
    monkeypatch.setattr(evaluation, 'evaluate_hellaswag', adapter)
    return calls


def geometry_batches():
    result = {}
    for partition in ['train', 'selection']:
        batch: dict = sharing_batches()[0]
        batch.update(query_ids=[partition+'0', partition+'1'], view_ids=['v0', 'v1'],
                     token_positions=torch.tensor([[0, 1, 2], [0, 1, 2]]),
                     token_roles=[['demo', 'query', 'query'], ['demo', 'query', 'query']])
        result[partition] = [batch]
    return result
