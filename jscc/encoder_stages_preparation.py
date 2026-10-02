"""Offline native16 data extension for the paired encoder stage trajectories."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from .activation_replay import canonical_digest, file_digest
from .baseline_protocol import batch_identity, read_prepared_batch
from .config import load_config
from .data.hellaswag import Collator
from .data.hellaswag_prompts import PromptBuilder, digest
from .presentation import PresentationSampler
from .sharing_preparation import _reference, state_dict_identity

CONDITIONS = ['no_noise', -6, 0, 6, 18]
WIDTHS = (512, 1152, 2304)
SEED = 20260920


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _ref(path, root, *, batch=None):
    reference = {'path': str(Path(path).relative_to(root)), 'sha256': file_digest(path)}
    if batch is not None:
        reference['view_sha256'] = batch_identity(batch)
    return reference


def _link(root, reference, output):
    source = _reference(root, reference)
    target = output / reference['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, target)
    return copy.deepcopy(reference)


def reuse_prefix(root, references, sampler, output):
    """Verify actual native rows once, then retain the exact existing bytes."""
    if len(references) * 16 > len(sampler):
        raise ValueError('prefix exceeds requested presentation stream')
    retained, lengths = [], []
    for index, reference in enumerate(references):
        # Reading verifies both the tensor-file checksum and complete view identity.
        path = _reference(root, reference)
        batch = read_prepared_batch(reference, root)
        if (len(batch['input_ids']) != 16
                or batch['row_ids'].tolist() != sampler.actual_ids[index * 16:(index + 1) * 16].tolist()):
            raise ValueError('saved prefix differs from native16 deterministic full-pool order')
        target = output / reference['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, target)
        retained.append(copy.deepcopy(reference))
        lengths.append(_lengths(index, batch))
    return retained, lengths


def final_task(template, data_ids):
    """Keep the inspected official membership and change only the condition list."""
    task = copy.deepcopy(template)
    membership = data_ids['final_validation']
    if (task['expected_items'] != 10042 or task['input_ids'] != list(range(10042))
            or task['input_ids'] != membership['input_ids']
            or task['source_family_ids'] != membership['source_family_ids']
            or task['reference_corpus']['split'] != 'validation'
            or task['precision'] != 'bf16' or task['backend'] != 'sdpa'
            or task['scorer'] != 'lm-eval-hellaswag-fp32-v1-acc-and-acc_norm'):
        raise ValueError('full inspected official validation identity differs')
    settings = task['settings']
    if (settings['num_samples'] != 10042 or settings['num_fewshot'] != 5
            or settings['batch_size'] != 16 or settings['numerical_policy'] != 'native'
            or settings['scoring_policy'] != 'fp32-v1'
            or settings['evidence_mode'] != 'compact-v1' or settings['vanilla'] is not False):
        raise ValueError('fixed final scoring settings differ')
    task['conditions'] = list(CONDITIONS)
    settings['snrs'] = list(CONDITIONS)
    return task


def _lengths(index, batch):
    return {'index': index, 'source_padded': batch['input_ids'].shape[1],
            'target_padded': batch['labels'].shape[1],
            'source_valid_tokens': int(batch['attention_mask'].sum()),
            'target_valid_tokens': int(batch['labels'].ne(-100).sum()),
            'source_valid_max': int(batch['attention_mask'].sum(1).max()),
            'target_valid_max': int(batch['labels'].ne(-100).sum(1).max())}


def _native_batch(rows, index, raw, builder, tokenizer, config):
    """Use the established native prompt, target, offset and padding contract."""
    from transformers import T5Gemma2ForConditionalGeneration

    docs = [raw[row] for row in rows]
    prompts = [builder.build(doc, row) for doc, row in zip(docs, rows)]
    tokens = tokenizer([text for text, _ in prompts], truncation=False, return_offsets_mapping=True)
    labels = tokenizer([doc['endings'][int(doc['label'])] for doc in docs],
                       truncation=True, max_length=512)
    batch: dict = Collator(tokenizer.pad_token_id)([
        {'input_ids': token_ids[-2048:], 'attention_mask': mask[-2048:],
         'label_ids': target, 'row_id': row}
        for row, token_ids, mask, target in zip(
            rows, tokens['input_ids'], tokens['attention_mask'], labels['input_ids'])])
    width = batch['input_ids'].shape[1]
    roles, positions, source_views, view_ids = [], [], [], []
    for i, (row, doc, (text, demos)) in enumerate(zip(rows, docs, prompts)):
        full_ids = tokens['input_ids'][i]
        kept = full_ids[-2048:]
        query_start = len(text) - len(builder.format(doc)[0])
        full_roles = []
        for position, (start, end) in enumerate(tokens['offset_mapping'][i]):
            if start == end:
                if position != 0 or full_ids[position] != tokenizer.bos_token_id:
                    raise ValueError('unexpected zero-offset token')
                full_roles.append('demo')
            else:
                full_roles.append('query' if end > query_start else 'demo')
        demo_families = [builder.documents[d]['source_id'] for d in demos]
        meta = dict(demo_ids=demos, source_id=doc['source_id'], demo_source_ids=demo_families,
                    source_length=len(full_ids), input_length=len(kept), source_hash=digest(doc),
                    text_hash=digest(text), input_hash=digest(kept), target_hash=digest(labels['input_ids'][i]))
        roles.append(full_roles[-2048:] + ['padding'] * (width - len(kept)))
        positions.append(list(range(len(full_ids) - len(kept), len(full_ids) - len(kept) + width)))
        view_ids.append(canonical_digest({'row_id': row, 'native_view': meta}))
        source_views.append({'row_id': row, 'native_prefix': 'reference',
                             'demo_source_family_ids': demo_families,
                             'query_character_start': query_start, 'native_view': meta})
    batch.update(query_ids=[f'hellaswag/train/{row}' for row in rows], view_ids=view_ids,
                 source_family_ids=[doc['source_id'] for doc in docs],
                 demo_ids=[demos for _, demos in prompts], token_positions=torch.tensor(positions),
                 token_roles=roles, position_roles=roles, source_views=source_views,
                 prompt_policy=copy.deepcopy(builder.policy),
                 batch_view_id=canonical_digest({'rows': rows, 'view_ids': view_ids, 'native16_index': index}))
    batch['decoder_input_ids'] = T5Gemma2ForConditionalGeneration.prepare_decoder_input_ids_from_labels(
        SimpleNamespace(config=config), labels=batch['labels'])  # pyright: ignore[reportArgumentType]
    batch['decoder_attention_mask'] = batch['labels'].ne(-100)
    return batch


def prepare_inputs(*, prefix_prepared, prefix_sha256, pool_manifest, pool_sha256,
                   assets, arrow, initializers, output, q=10000):
    """Extend one matched per-site stream; the integrator binds execution source."""
    from datasets import Dataset
    from transformers import AutoConfig, AutoTokenizer

    prefix_path, pool_path, assets, arrow, output = map(Path,
        (prefix_prepared, pool_manifest, assets, arrow, output))
    prefix_path, pool_path, assets, arrow, output = (
        p.resolve() for p in (prefix_path, pool_path, assets, arrow, output))
    if q != 10000 or file_digest(prefix_path) != prefix_sha256 or file_digest(pool_path) != pool_sha256:
        raise ValueError('q10000 and exact approved prefix/pool manifests required')
    prefix, pool = json.loads(prefix_path.read_text()), json.loads(pool_path.read_text())
    ids = prefix['data_ids']
    if (len(prefix['updates']) != 6400 or len(ids['train_rows']) != 39058
            or ids['train_rows'] != pool['data_ids']['train_rows']
            or prefix['model_revision'] != pool['model_revision']
            or set(map(int, initializers)) != set(WIDTHS)):
        raise ValueError('approved full pool, q6400 prefix and all three width initializers required')
    if arrow.stat().st_size != pool['dataset_file']['bytes'] or file_digest(arrow) != pool['dataset_file']['sha256']:
        raise ValueError('pinned raw Arrow differs')
    for name, record in pool['tokenizer_assets']['files'].items():
        if (assets / name).stat().st_size != record['bytes'] or file_digest(assets / name) != record['sha256']:
            raise ValueError('pinned tokenizer/config differs: ' + name)
    sampler = PresentationSampler(ids['train_rows'], q * 16, seed=SEED)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'schema': 'encoder-stages-inputs-v1', 'q': q, 'batch_size': 16,
                'model_revision': prefix['model_revision'], 'data_ids': copy.deepcopy(ids),
                'presentation_stream': {'policy': 'epoch-permutations-v1', 'seed': SEED,
                    'total_presentations': q * 16, 'start_presentation': 0,
                    'pool_identity': canonical_digest(ids['train_rows'])},
                'prefix_origin': {'path': str(prefix_path), 'sha256': prefix_sha256},
                'pool_origin': {'path': str(pool_path), 'sha256': pool_sha256},
                'dataset_file': copy.deepcopy(pool['dataset_file']),
                'tokenizer_assets': copy.deepcopy(pool['tokenizer_assets']),
                'launch_authorized': False, 'initializations': {}, 'configs': {}}
    manifest['updates'], lengths = reuse_prefix(prefix_path.parent, prefix['updates'], sampler, output)
    manifest['validation'] = [_link(prefix_path.parent, ref, output) for ref in prefix['validation']]
    task = final_task(json.loads(_reference(prefix_path.parent, prefix['task_request']).read_text()), ids)
    _json(output / 'task.json', task)
    manifest['task_request'] = _ref(output / 'task.json', output)
    base_config = load_config(_reference(prefix_path.parent, prefix['config']))
    for width in WIDTHS:
        initializer_path, initializer_sha = initializers[str(width)]
        initializer_path = Path(initializer_path).resolve()
        if file_digest(initializer_path) != initializer_sha:
            raise ValueError('initializer package checksum differs')
        initializer = json.loads(initializer_path.read_text())
        if initializer.get('initialization_policy') != {
                'seed': 0, 'fresh_cpu_codec': True, 'baseline_state_reused': False}:
            raise ValueError('fresh seed0 initializer required, never a trained checkpoint')
        initializer_config = load_config(_reference(initializer_path.parent, initializer['config']))
        if initializer_config['codec']['bottleneck_dim'] != width:
            raise ValueError('initializer width differs')
        original_ref = initializer['initialization']
        state_path = _reference(initializer_path.parent, original_ref)
        if state_dict_identity(torch.load(state_path, map_location='cpu', weights_only=True)) != original_ref['state_identity']:
            raise ValueError('initializer tensor identity differs')
        initial_path = output / f'initial-b{width}.pt'
        os.link(state_path, initial_path)
        manifest['initializations'][str(width)] = {**_ref(initial_path, output),
            'state_identity': original_ref['state_identity'], 'seed': 0,
            'paired_stages': ['A', 'E', 'R'],
            'origin': {'path': str(initializer_path), 'sha256': initializer_sha}}
        config = copy.deepcopy(base_config)
        config['codec']['bottleneck_dim'] = width
        config['run']['name'] = f'encoder-stages-b{width}'
        config['training'].update(max_steps=6 * q, schedule_steps=6 * q,
                                  eval_every=6 * q, save_steps=[6 * q])
        config['training']['presentation_stream']['total_presentations'] = q * 16
        config['evaluation']['snrs'] = list(CONDITIONS)
        config_path = output / f'config-b{width}.yaml'
        config_path.write_text(yaml.safe_dump(config))
        manifest['configs'][str(width)] = {**_ref(config_path, output),
            'config_identity': canonical_digest(load_config(config_path))}
    raw = Dataset.from_file(str(arrow))
    tokenizer = AutoTokenizer.from_pretrained(assets, local_files_only=True)
    if not tokenizer.is_fast:
        raise ValueError('actual fast-tokenizer offsets required')
    config = AutoConfig.from_pretrained(assets, local_files_only=True)
    first = read_prepared_batch(prefix['updates'][0], prefix_path.parent)
    builder = PromptBuilder(raw, ids, first['prompt_policy'])
    for index in range(6400, q):
        rows = sampler.actual_ids[index * 16:(index + 1) * 16].tolist()
        batch = _native_batch(rows, index, raw, builder, tokenizer, config)
        path = output / f'native16/update-{index:05d}.pt'
        torch.save(batch, path)
        manifest['updates'].append(_ref(path, output, batch=batch))
        lengths.append(_lengths(index, batch))
        if (index + 1) % 100 == 0:
            print(f'Prepared {index + 1}/{q} exact full-pool native16 views', flush=True)
    sampler.save(output, 16)
    _json(output / 'lengths.json', lengths)
    manifest['stream_evidence'] = {'prefix_views_reused_exactly': 6400, 'new_suffix_views': q - 6400,
        'unique_pool_rows': len(set(sampler.actual_ids.tolist())),
        'actual_ids_sha256_le_int64': hashlib.sha256(sampler.actual_ids.numpy().astype('<i8').tobytes()).hexdigest(),
        'source_valid_tokens_per_site': sum(x['source_valid_tokens'] for x in lengths),
        'target_valid_tokens_per_site': sum(x['target_valid_tokens'] for x in lengths),
        'source_valid_max': max(x['source_valid_max'] for x in lengths),
        'target_valid_max': max(x['target_valid_max'] for x in lengths)}
    manifest['checks'] = {'models_constructed': 0, 'activation_cache': False,
                          'CUDA_initialized': torch.cuda.is_initialized(),
                          'final_split': 'already-inspected official validation; no untouched test claim'}
    manifest['manifest_identity'] = canonical_digest(manifest)
    _json(output / 'inputs.json', manifest)
    return output / 'inputs.json'
