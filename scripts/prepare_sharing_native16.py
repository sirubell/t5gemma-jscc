#!/usr/bin/env python3
"""CPU-only fresh16 input stream from the full frozen HellaSwag training pool.

Reuses existing objective/geometry/development panels. This is preparation,
not a GPU entrypoint, architecture selection, or final10042 leakage audit.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from jscc.activation_replay import canonical_digest, file_digest, producer_spec_v2, sequence_view_v2  # noqa: E402
from jscc.baseline_protocol import batch_identity, read_prepared_batch  # noqa: E402
from jscc.data.hellaswag import Collator  # noqa: E402
from jscc.data.hellaswag_prompts import PromptBuilder, digest  # noqa: E402
from jscc.presentation import PresentationSampler  # noqa: E402
from jscc.sharing_preparation import _geometry_contract, _production_data_contract, _reference  # noqa: E402
from jscc.sharing_schedule import NATIVE16_PROTOCOL, validate_native16_horizon  # noqa: E402
from jscc.sharing_accumulation import partition_native64  # noqa: E402


def prepare(*, template_path, template_sha256, assets, arrow, q, output):
    """Regenerate prompts/padding from original rows, without cycling saved batches."""
    from datasets import Dataset
    from transformers import AutoConfig, AutoTokenizer, T5Gemma2ForConditionalGeneration

    q = validate_native16_horizon(q)
    template_path, assets, arrow, output = map(lambda p: Path(p).resolve(),
                                             (template_path, assets, arrow, output))
    if file_digest(template_path) != template_sha256:
        raise ValueError('frozen template digest differs')
    original = json.loads(template_path.read_text())
    if original.get('schema') != 'sharing-production-inputs-v1' or original.get('status') != 'cpu_contracts_verified':
        raise ValueError('verified production input template required')
    dataset = original['dataset_file']
    if arrow.stat().st_size != dataset['bytes'] or file_digest(arrow) != dataset['sha256']:
        raise ValueError('raw train Arrow differs from frozen source')
    for name, record in original['tokenizer_assets']['files'].items():
        if (assets / name).stat().st_size != record['bytes'] or file_digest(assets / name) != record['sha256']:
            raise ValueError('pinned tokenizer/config differs: ' + name)
    raw = Dataset.from_file(str(arrow))
    tokenizer = AutoTokenizer.from_pretrained(assets, local_files_only=True)
    if not tokenizer.is_fast:
        raise ValueError('native token roles require actual fast-tokenizer offsets')
    config = AutoConfig.from_pretrained(assets, local_files_only=True)
    first = read_prepared_batch(original['updates'][0], template_path.parent)
    ids = original['data_ids']
    builder = PromptBuilder(raw, ids, first['prompt_policy'])
    sampler = PresentationSampler(ids['train_rows'], q * 16, seed=20260920)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {key: copy.deepcopy(original[key]) for key in (
        'schema', 'status', 'model_revision', 'data_ids', 'validation', 'geometry',
        'producers', 'task_template', 'dataset_file', 'tokenizer_assets', 'provenance_policy')}
    # Retain original panels for provenance, then explicitly split full rows to16.
    for reference in [*manifest['validation'], manifest['task_template'],
                      *manifest['geometry']['train'], *manifest['geometry']['selection'],
                      *(ref for role, ref in manifest['producers'].items() if role != 'optimization')]:
        source = _reference(template_path.parent, reference)
        target = output / reference['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
    manifest["original_panels"] = {key: copy.deepcopy(manifest[key]) for key in ("validation", "geometry", "producers")}
    for role, references in [("objective_validation", manifest["validation"]),
                             ("geometry_training", manifest["geometry"]["train"]),
                             ("geometry_selection", manifest["geometry"]["selection"])]:
        new_refs, new_views = [], []
        for parent_index, reference in enumerate(references):
            parent = read_prepared_batch(reference, output)
            for micro_index, micro in enumerate(partition_native64(parent, microbatch_size=16)):
                micro["batch_view_id"] = canonical_digest({"parent": parent["batch_view_id"], "micro": micro_index, "size": 16})
                path = output / f"native16/{role}-{parent_index}-{micro_index}.pt"
                path.parent.mkdir(exist_ok=True)
                torch.save(micro, path)
                new_refs.append({"path": str(path.relative_to(output)), "sha256": file_digest(path),
                                 "view_sha256": batch_identity(micro)})
                new_views.append(sequence_view_v2(micro))
        old = json.loads(_reference(template_path.parent, original["producers"][role]).read_text())
        producer = producer_spec_v2(source_root=Path(__file__).resolve().parents[1],
            **{key: old[key] for key in ("backbone", "environment", "data", "policy", "sites", "data_role", "disjointness")},
            views=new_views)
        path = output / f"native16/{role}-producer.json"
        path.write_text(json.dumps(producer, indent=2) + "\n")
        manifest["producers"][role] = {"path": str(path.relative_to(output)), "sha256": file_digest(path)}
        if role == "objective_validation":
            manifest["validation"] = new_refs
        else:
            manifest["geometry"]["train" if role == "geometry_training" else "selection"] = new_refs
    manifest.update(q=q, batch_size=16, protocol_id=NATIVE16_PROTOCOL,
        launch_authorized=False, architecture=None,
        template_origin={'path': str(template_path), 'sha256': template_sha256},
        preparation_script_sha256=file_digest(Path(__file__)),
        presentation_stream={'policy': 'epoch-permutations-v1', 'seed': 20260920,
            'total_presentations': q * 16, 'start_presentation': 0,
            'pool_identity': canonical_digest(ids['train_rows'])}, updates=[])
    views, lengths = [], []
    collate = Collator(tokenizer.pad_token_id)
    for index, chunk in enumerate(sampler.actual_ids.split(16)):
        rows = chunk.tolist()
        docs = [raw[row] for row in rows]
        prompts = [builder.build(doc, row) for doc, row in zip(docs, rows)]
        tokens = tokenizer([text for text, _ in prompts], truncation=False, return_offsets_mapping=True)
        labels = tokenizer([doc['endings'][int(doc['label'])] for doc in docs], truncation=True, max_length=512)
        batch: dict = collate([{'input_ids': token_ids[-2048:], 'attention_mask': mask[-2048:],
                          'label_ids': target, 'row_id': row}
                         for row, token_ids, mask, target in zip(rows, tokens['input_ids'], tokens['attention_mask'], labels['input_ids'])])
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
                'demo_source_family_ids': demo_families, 'query_character_start': query_start, 'native_view': meta})
        batch.update(query_ids=[f'hellaswag/train/{row}' for row in rows], view_ids=view_ids,
            source_family_ids=[doc['source_id'] for doc in docs], demo_ids=[demos for _, demos in prompts],
            token_positions=torch.tensor(positions), token_roles=roles, position_roles=roles,
            source_views=source_views, prompt_policy=first['prompt_policy'],
            batch_view_id=canonical_digest({'rows': rows, 'view_ids': view_ids, 'native16_index': index}))
        batch['decoder_input_ids'] = T5Gemma2ForConditionalGeneration.prepare_decoder_input_ids_from_labels(
            SimpleNamespace(config=config), labels=batch['labels'])  # pyright: ignore[reportArgumentType]
        batch['decoder_attention_mask'] = batch['labels'].ne(-100)
        path = output / f'native16/update-{index:05d}.pt'
        path.parent.mkdir(exist_ok=True)
        torch.save(batch, path)
        manifest['updates'].append({'path': str(path.relative_to(output)), 'sha256': file_digest(path),
                                    'view_sha256': batch_identity(batch)})
        views.append(sequence_view_v2(batch))
        lengths.append({'index': index, 'source_padded': width, 'target_padded': batch['labels'].shape[1],
                        'source_valid_tokens': int(batch['attention_mask'].sum())})
        if (index + 1) % 100 == 0:
            print(f'Prepared {index + 1}/{q} native16 batches', flush=True)
    # Probe tails are explicit diagnostics from the frozen400-parent stream,
    # never substituted into the scientific full-pool presentation sequence.
    extrema = []
    for reference in original['updates']:
        batch = read_prepared_batch(reference, template_path.parent)
        extrema.append((int(batch['attention_mask'].sum(1).max()),
                        int(batch['labels'].ne(-100).sum(1).max()), reference))
    cases = [{'label': 'representative', 'batch': manifest['updates'][0]}]
    for column, label in [(0, 'longest_source'), (1, 'longest_target')]:
        _, _, reference = max(extrema, key=lambda row: row[column])
        parent = read_prepared_batch(reference, template_path.parent)
        mask = parent['attention_mask'] if column == 0 else parent['labels'].ne(-100)
        micro_index = int(mask.sum(1).argmax()) // 16
        micro: dict = partition_native64(parent, microbatch_size=16)[micro_index]
        micro['batch_view_id'] = canonical_digest({'parent': parent['batch_view_id'], 'micro': micro_index, 'case': label})
        path = output / f'native16/{label}.pt'
        torch.save(micro, path)
        cases.append({'label': label, 'batch': {'path': str(path.relative_to(output)),
            'sha256': file_digest(path), 'view_sha256': batch_identity(micro)},
            'origin': {'template': template_sha256, 'parent': reference, 'micro_index': micro_index},
            'source_valid_max': int(micro['attention_mask'].sum(1).max()),
            'target_valid_max': int(micro['labels'].ne(-100).sum(1).max())})
    manifest['readiness_cases'] = cases
    old = json.loads(_reference(template_path.parent, original['producers']['optimization']).read_text())
    producer = producer_spec_v2(source_root=Path(__file__).resolve().parents[1],
        **{key: old[key] for key in ('backbone', 'environment', 'data', 'policy', 'sites', 'data_role', 'disjointness')},
        views=views)
    (output / 'native16/optimization-producer.json').write_text(json.dumps(producer, indent=2) + '\n')
    manifest['producers']['optimization'] = {'path': 'native16/optimization-producer.json',
        'sha256': file_digest(output / 'native16/optimization-producer.json')}
    task = json.loads((output / manifest['task_template']['path']).read_text())
    _production_data_contract(output, manifest, task)
    _geometry_contract(output, manifest['geometry'])
    manifest['checks'] = {'complete_pool_order_verified': True, 'fresh_presentations': q * 16,
                         'pool_rows': len(ids['train_rows']), 'models_constructed': 0, 'CUDA_initialized': torch.cuda.is_initialized()}
    (output / 'lengths.json').write_text(json.dumps(lengths, indent=2) + '\n')
    (output / 'production-inputs.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return output / 'production-inputs.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('template', 'assets', 'arrow', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--template-sha256', required=True)
    parser.add_argument('--q', type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    path = prepare(template_path=args.template, template_sha256=args.template_sha256,
                   assets=args.assets, arrow=args.arrow, q=args.q, output=args.output)
    print(path)


if __name__ == '__main__':
    main()
