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

from .original13_provenance import (  # noqa: F401
    model_asset_inventory, bind_model_assets, verify_model_assets,
)

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


def resolve_original13_plans(views, *, inputs, input_identity, synthetic=False):
    """Bind all nine plans to actual input/panel metadata, without model loading.

    Source/input-file hashes are checked by the caller at loading. BatchView
    objects must come from the verified native tensor inputs, never estimates.
    """
    from .encoder_stages_plan import ORIGINAL13_RATIOS, Original13Plan
    from .activation_replay import canonical_digest
    panel = inputs['data_ids']['development']
    if (panel.get('expected_items') != 256 or panel.get('input_ids') != list(range(256))
            or len(panel.get('source_family_ids', [])) != 256
            or panel.get('split') != 'validation'
            or panel.get('panel') != 'f3e5b8ffbaeaae3a775106937170d8f12cb8670b6ecb01f31efe3c7c33179bae'
            or panel.get('noise') != {'namespace': 'encfn-baseline-20260930-development-v2', 'seed': 0}):
        raise ValueError('original13 requires the exact existing256 development panel')
    if not synthetic:
        pool = inputs['data_ids']['train_rows']
        stream = {'policy': 'epoch-permutations-v1', 'seed': SEED, 'total_presentations': 160000,
                  'start_presentation': 0, 'pool_identity': canonical_digest(pool)}
        if len(pool) != 39058 or len(set(pool)) != 39058 or inputs['presentation_stream'] != stream:
            raise ValueError('original13 full training pool/stream differs')
    identity = canonical_digest(panel)
    if not synthetic and identity != 'd83b0856429b508910fab8a42453490b9d4bce3c52e78e17291b64046116980f':
        raise ValueError('original13 development source-family object differs')
    return tuple(Original13Plan(tuple(views), 'original13-seed0-fullpool-v1', ratio, width,
                                input_identity, identity, synthetic)
                 for width in (2304, 1152, 512) for ratio in ORIGINAL13_RATIOS)


def bind_original13_inputs(inputs_path, *, inputs_sha256, output, model_path=None):
    """Prepare source-bound plan/config identities from retained inputs only."""
    from .activation_replay import canonical_digest
    from .encoder_stages import stages_source_inventory
    from .encoder_stages_plan import ORIGINAL13_PROTOCOL
    from .sharing_protocol import batch_view
    path = Path(inputs_path).resolve()
    if file_digest(path) != inputs_sha256:
        raise ValueError('original13 exact inputs manifest hash required')
    inputs = json.loads(path.read_text())
    if (inputs.get('manifest_identity') != canonical_digest({k: v for k, v in inputs.items() if k != 'manifest_identity'})
            or inputs.get('synthetic_cpu', False) or inputs.get('q') != 10000):
        raise ValueError('original13 complete real q10000 inputs required')
    pool = inputs['data_ids']['train_rows']
    expected = PresentationSampler(pool, 160000, seed=SEED).actual_ids.tolist()
    views = []
    for i, ref in enumerate(inputs['updates']):
        batch = read_prepared_batch(ref, path.parent)
        if batch['row_ids'].tolist() != expected[i*16:(i+1)*16]:
            raise ValueError('original13 actual native input order differs')
        views.append(batch_view(batch))
    plans = resolve_original13_plans(tuple(views), inputs=inputs,
                                     input_identity=inputs_sha256)
    configurations, initializers = {}, {}
    for width in WIDTHS:
        config = load_config(_reference(path.parent, inputs['configs'][str(width)]))
        validate_original13_config(config, width)
        if (config['model']['revision'] != 'dd0a2683227859151b1730ca3a63087df5b5f39b'
                or config['model']['dtype'] != 'bfloat16' or config['model']['attn_implementation'] != 'sdpa'):
            raise ValueError('original13 pinned production model recipe differs')
        config = copy.deepcopy(config)
        if model_path is not None:
            config['model']['name'] = str(model_path)
        config['training'].update(max_steps=130000, schedule_steps=130000, eval_every=130000,
                                  save_steps=[])
        config['protocol'] = ORIGINAL13_PROTOCOL
        configurations[str(width)] = config
        ref = inputs['initializations'][str(width)]
        initial = torch.load(_reference(path.parent, ref), map_location='cpu', weights_only=True)
        if ref.get('seed') != 0 or state_dict_identity(initial) != ref['state_identity']:
            raise ValueError('original13 fresh width initializer differs')
        initializers[str(width)] = copy.deepcopy(ref)
    model_names = {c['model']['name'] for c in configurations.values()}
    if len(model_names) != 1:
        raise ValueError('original13 widths require the same backbone/tokenizer snapshot')
    assets = bind_model_assets(next(iter(model_names)), 'dd0a2683227859151b1730ca3a63087df5b5f39b',
                              tokenizer_files=inputs['tokenizer_assets']['files'], production=True)
    for config in configurations.values():
        config['model']['name'] = assets['path']
        config['model']['asset_binding'] = copy.deepcopy(assets)
    template = final_task(json.loads(_reference(path.parent, inputs['task_request']).read_text()), inputs['data_ids'])
    template['conditions'] = ['no_noise', -6]
    template['settings']['snrs'] = ['no_noise', -6]
    panel = copy.deepcopy(inputs['data_ids']['development'])
    observation = {**template, 'schema': 'original13-observation-request-v1', 'panel': panel,
                   'input_ids': panel['input_ids'], 'source_family_ids': panel['source_family_ids'],
                   'expected_items': 256, 'noise': panel['noise'],
                   'sites': ['enc_l9', 'enc_fn', 'dec_l0', 'dec_l24'], 'conditions': ['no_noise', -6],
                   'settings': {**template['settings'], 'num_samples': 256, 'snrs': ['no_noise', -6]}}
    inventory = stages_source_inventory()
    from .runtime import source_state
    manifest = {'schema': 'original13-prepared-v1', 'source_commit': source_state()['revision'],
                'source_inventory': inventory,
                'source_identity': canonical_digest(inventory), 'inputs_path': str(path),
                'inputs_sha256': inputs_sha256, 'data_identity': canonical_digest(inputs['data_ids']),
                'configs': configurations, 'initializations': initializers,
                'plans': [plan.resolved() for plan in plans], 'observation_request': observation,
                'terminal_request': template, 'launch_authorized': False}
    manifest['manifest_identity'] = canonical_digest(manifest)
    output = Path(output)
    if output.exists():
        raise ValueError('fresh original13 prepared destination required')
    from .baseline_protocol import write_manifest
    write_manifest(output, manifest)
    return manifest


def validate_original13_config(config, width):
    """Check the adopted cohort recipe at preparation and runtime load."""
    train, codec, channel = config['training'], config['codec'], config['channel']
    if (config['task'] != 'hellaswag' or config['seed'] != 0
            or codec['architecture'] != 'direct_affine' or codec['bottleneck_dim'] != width
            or codec['layernorm'] != 'none' or codec['n_res_blocks'] != 0 or codec['snr_film']
            or codec['dropout'] != 0 or train['batch_size'] != 16 or train['gradient_accumulation'] != 1
            or train['lr'] != 2e-4 or train['weight_decay'] != 0.002500001398541448
            or train['temperature'] != 1 or train['grad_clip'] != 1 or not train['valid_only_kl']
            or train['warmup_ratio'] != .05 or train['min_lr_ratio'] != 0
            or train.get('patience') is not None or train.get('loss_weights') is not None
            or train['kl_weight'] != 1 or train['mse_weight'] != .1
            or config['model'].get('numerical_policy', 'native') != 'native'
            or codec.get('memory') is not None
            or channel['type'] != 'awgn' or not channel['normalize_power'] or not channel['train_noise']
            or channel['train_snr_range'] != [-6.0, 18.0]):
        raise ValueError('inherited original13 configuration differs')
