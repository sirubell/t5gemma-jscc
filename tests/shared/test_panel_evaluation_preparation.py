import json

import pytest
import torch

from jscc.activation_replay import file_digest
from jscc.shared_inputs import batch_identity
from jscc.checkpoint_readers import PREPARATION_CONDITIONS as CONDITIONS, final_task, reuse_prefix
from jscc.presentation import PresentationSampler


def test_reuses_exact_tensor_bytes_and_epoch_order(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'output'
    source.mkdir()
    output.mkdir()
    sampler = PresentationSampler(list(range(32)), 48, seed=20260920)
    references = []
    for index in range(2):
        batch = {'input_ids': torch.ones(16, 4, dtype=torch.long),
                 'attention_mask': torch.ones(16, 4, dtype=torch.long),
                 'labels': torch.ones(16, 2, dtype=torch.long),
                 'row_ids': sampler.actual_ids[index * 16:(index + 1) * 16]}
        path = source / f'{index}.pt'
        torch.save(batch, path)
        references.append({'path': path.name, 'sha256': file_digest(path),
                           'view_sha256': batch_identity(batch)})
    retained, lengths = reuse_prefix(source, references, sampler, output)
    assert retained == references
    assert sum(item['source_valid_tokens'] for item in lengths) == 128
    assert all((source / ref['path']).stat().st_ino == (output / ref['path']).stat().st_ino
               for ref in retained)
    wrong = tmp_path / 'wrong'
    wrong.mkdir()
    with pytest.raises(ValueError, match='deterministic full-pool order'):
        reuse_prefix(source, references, PresentationSampler(list(range(32)), 48, seed=1), wrong)


def test_full_final_panel_keeps_membership_and_five_ordered_conditions():
    ids = list(range(10042))
    families = ['family'] * len(ids)
    template = {'expected_items': 10042, 'input_ids': ids, 'source_family_ids': families,
        'reference_corpus': {'split': 'validation'}, 'precision': 'bf16', 'backend': 'sdpa',
        'scorer': 'lm-eval-hellaswag-fp32-v1-acc-and-acc_norm', 'conditions': ['no_noise'],
        'settings': {'num_samples': 10042, 'num_fewshot': 5, 'batch_size': 16,
                     'numerical_policy': 'native', 'scoring_policy': 'fp32-v1',
                     'evidence_mode': 'compact-v1', 'vanilla': False, 'snrs': ['no_noise']}}
    result = final_task(template, {'final_validation': {'input_ids': ids, 'source_family_ids': families}})
    assert result['conditions'] == result['settings']['snrs'] == CONDITIONS
    assert result['input_ids'] == ids and result['source_family_ids'] == families
    assert template['conditions'] == ['no_noise']
    result['input_ids'] = list(reversed(ids))
    with pytest.raises(ValueError, match='validation identity'):
        final_task(result, {'final_validation': {'input_ids': ids, 'source_family_ids': families}})


def test_wrong_batch_digest_rejects_before_link(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'output'
    source.mkdir()
    output.mkdir()
    path = source / '0.pt'
    path.write_text(json.dumps({'not': 'a tensor batch'}))
    with pytest.raises(ValueError, match='checksum'):
        reuse_prefix(source, [{'path': '0.pt', 'sha256': 'wrong', 'view_sha256': 'wrong'}],
                     PresentationSampler(range(32), 48), output)
    assert not list(output.iterdir())
