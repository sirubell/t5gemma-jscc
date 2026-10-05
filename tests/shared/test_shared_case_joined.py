"""Retained local prepared-reader and asset boundaries, without retired launch fronts."""

import jscc
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import pytest
import torch
from jscc import site_routing as routing
from jscc.activation_replay import canonical_digest, file_digest
from jscc.experiment_records import write_manifest
from jscc.checkpoint_readers import Original13Plan, ORIGINAL13_PROTOCOL, load_original13_prepared, stages_source_inventory
from jscc.asset_identity import state_dict_identity, bind_model_assets, verify_model_assets
from jscc.shared_inputs import batch_view
from contracts.test_harness_payload import fixture_model
from shared.test_shared_case_routes import make_model, batch
from shared.test_shared_case_plan_state import panel
from shared.test_shared_case_evaluation import template
from shared.test_shared_case_evaluation import frozen as frozen



def cli(name):
    path = Path(jscc.__file__).resolve().parents[1] / 'scripts' / name
    spec = importlib.util.spec_from_file_location(name.removesuffix('.py'), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main

@pytest.fixture
def prepared(tmp_path):
    """Serialize real tiny backbone/tokenizer assets and exact native CPU views."""
    torch.set_num_threads(1)
    model = make_model()
    cache = tmp_path/'model'
    model.base.save_pretrained(cache)
    tokenizer, _ = fixture_model()
    tokenizer.save_pretrained(cache)
    data = []
    updates = []
    for i in range(400):
        item = batch(16)
        item['row_ids'] = torch.arange(i*16, (i+1)*16)
        path = tmp_path/f'view-{i}.pt'
        torch.save(item, path)
        data.append(item)
        updates.append({'path': path.name, 'sha256': file_digest(path), 'view_sha256': batch_view(item).view_sha256})
    initial = routing.export_codec(model)
    torch.save(initial, tmp_path/'initial.pt')
    ref = {'path': 'initial.pt', 'sha256': file_digest(tmp_path/'initial.pt'),
           'state_identity': state_dict_identity(initial), 'seed': 0}
    inputs = {'updates': updates, 'data_ids': {'development': panel()}, 'initializations': {'8': ref}}
    write_manifest(tmp_path/'inputs.json', inputs)
    input_hash = file_digest(tmp_path/'inputs.json')
    plan = Original13Plan(tuple(batch_view(b) for b in data), 'joined-cpu', 90, 8,
                          input_hash, canonical_digest(panel()), True)
    assets = bind_model_assets(cache, 'tiny-cpu')
    config = {'task': 'hellaswag', 'seed': 0, 'protocol': ORIGINAL13_PROTOCOL,
        'run': {'name': 'joined-tiny', 'output_dir': str(tmp_path/'runs')},
        'model': {'name': str(cache), 'revision': 'tiny-cpu', 'device': 'cpu', 'dtype': 'float32',
                  'attn_implementation': 'sdpa', 'numerical_policy': 'native', 'asset_binding': assets},
        'split': {'stack': 'enc', 'where': 'after_final_norm'}, 'codec': deepcopy(model.codec.config),
        'channel': {'type': 'awgn', 'kwargs': {}, 'normalize_power': True, 'train_noise': True,
                    'train_snr_range': [-6.0, 18.0], 'clean_film_snr': 18.0},
        'training': {'max_steps': plan.horizon, 'schedule_steps': plan.horizon, 'batch_size': 16,
                     'gradient_accumulation': 1, 'eval_every': plan.horizon, 'monitor': 'loss',
                     'validation_snrs': ['no_noise'], 'lr': 2e-4, 'weight_decay': 0.002500001398541448,
                     'temperature': 1., 'grad_clip': 1., 'valid_only_kl': True, 'warmup_ratio': .05,
                     'min_lr_ratio': 0, 'kl_weight': 1., 'mse_weight': .1, 'deterministic_algorithms': True},
        'evaluation': template()['settings'], 'data': {**template()['data_settings'],
            'max_length': 512, 'prompt_policy': {'version': 'prompt-alignment-v1', 'mode': 'five_shot',
                'seed': 0, 'source_max_length': 2048}}}
    observation = {**template(256), 'schema': 'original13-observation-request-v1', 'panel': panel(),
        'noise': panel()['noise'], 'sites': ['enc_l9', 'enc_fn', 'dec_l0', 'dec_l24'], 'conditions': ['no_noise', -6]}
    inventory = stages_source_inventory()
    manifest = {'schema': 'original13-prepared-v1', 'source_commit': 'a'*40,  # Synthetic fixture identity; no Git operation.
        'source_inventory': inventory, 'source_identity': canonical_digest(inventory),
        'inputs_path': str(tmp_path/'inputs.json'), 'inputs_sha256': input_hash,
        'data_identity': canonical_digest(inputs['data_ids']), 'configs': {'8': config},
        'initializations': {'8': ref}, 'plans': [plan.resolved()], 'observation_request': observation,
        'terminal_request': template(), 'launch_authorized': False}
    manifest['manifest_identity'] = canonical_digest(manifest)
    write_manifest(tmp_path/'prepared.json', manifest)
    return tmp_path/'prepared.json', manifest

def offline_dataset(monkeypatch):
    from datasets import Dataset, DatasetDict
    from lm_eval.tasks import TaskManager
    def rows(count):
        return [{'ctx_a': 'x', 'ctx_b': 'x', 'activity_label': 'x', 'endings': ['x', 'x x', 'x x x', 'x x x x'],
                 'label': str(i%4), 'source_id': f'family-{i}', 'ind': i} for i in range(count)]
    raw = DatasetDict({'train': Dataset.from_list(rows(8)), 'validation': Dataset.from_list(rows(256))})
    monkeypatch.setattr('datasets.load_dataset', lambda *a, **kw: raw)
    manager = TaskManager()
    monkeypatch.setattr('lm_eval.tasks.TaskManager', lambda: manager)

def execution_args(path, manifest, output):
    return ['--prepared', str(path), '--prepared-sha256', file_digest(path), '--width', '8', '--ratio', '90',
            '--approved-source', manifest['source_commit'], '--timeout-seconds', '600', '--output', str(output)]

def test_loader_identity_rejection_before_model_construction(prepared, monkeypatch):
    path, manifest = prepared
    changed = deepcopy(manifest)
    changed['configs']['8']['training']['lr'] = .01
    changed['manifest_identity'] = canonical_digest({k: v for k, v in changed.items() if k != 'manifest_identity'})
    write_manifest(path, changed)
    with pytest.raises(ValueError, match='configuration differs'):
        load_original13_prepared(path, checksum=file_digest(path), width=8, ratio=90)
    with pytest.raises(ValueError, match='prepared SHA'):
        load_original13_prepared(path, checksum='a'*64, width=8, ratio=90)

@pytest.mark.parametrize('asset', ['model.safetensors', 'tokenizer.json', 'config.json'])
def test_changed_cached_assets_rejected_before_transformers_load(prepared, monkeypatch, asset):
    path, _ = prepared
    _, config, _, _, initial, _, _ = load_original13_prepared(path, checksum=file_digest(path), width=8, ratio=90)
    target = Path(config['model']['name'])/asset
    target.write_bytes(target.read_bytes()+b'changed')
    def forbidden(*a, **kw):
        pytest.fail('changed assets reached the actual model loader')
    monkeypatch.setattr('jscc.models.split_model.build_model', forbidden)
    with pytest.raises(ValueError, match='asset bytes differ'):
        verify_model_assets(config)

def test_pending_assets_and_input_tokenizer_mismatch_fail_closed(tmp_path):
    from jscc.asset_identity import verify_model_assets
    binding = bind_model_assets(tmp_path/'missing', 'revision')
    config = {'model': {'name': binding['path'], 'revision': 'revision', 'asset_binding': binding}}
    assert verify_model_assets(config, require_complete=False)['status'] == 'pending'
    with pytest.raises(ValueError, match='pending'):
        verify_model_assets(config)
    root = tmp_path/'snapshots'/'pinned'
    root.mkdir(parents=True)
    (root/'config.json').write_text('{}')
    (root/'tokenizer.json').write_text('{}')
    (root/'model.safetensors').write_bytes(b'fixture')
    with pytest.raises(ValueError, match='retained inputs'):
        bind_model_assets(root, 'pinned', tokenizer_files={'config.json': {'bytes': 2, 'sha256': 'a'*64}}, production=True)


@pytest.mark.parametrize('mutation', ['coefficient', 'partial_component', 'denominator', 'payload', 'weighted_total', 'numerator', 'source_exposure', 'phase'])
def test_actual_stream_events_persist_and_reject_science_drift(frozen, mutation):
    from jscc import panel_evaluation as panels
    model, plan, root = frozen
    events = [json.loads(line) for line in (root/'metrics.jsonl').read_text().splitlines()]
    assert len(events) == plan.horizon
    assert events[0]['payload']['exposure']['memory_coordinates']['value'] is None
    assert events[0]['payload']['objective']['components']['memory']['raw']['value'] is None
    assert events[6]['payload']['exposure']['memory_coordinates'] > 0
    p = events[6]['payload']
    if mutation == 'coefficient':
        p['objective']['resolved_weights']['hidden'] = .1
    elif mutation == 'partial_component':
        p['objective']['components']['kl']['weighted'] = {'value': 0, 'reason': None}
    elif mutation == 'denominator':
        p['objective']['components']['memory']['denominator'] = 0
    elif mutation == 'payload':
        p['exposure']['memory_coordinates_allocated'] = 0
    elif mutation == 'weighted_total':
        p['objective']['total']['value'] += 1.
    elif mutation == 'numerator':
        p['objective']['components']['hidden']['numerator']['value'] += 1.
    elif mutation == 'source_exposure':
        p['exposure']['source_tokens'] += 1
    else:
        events[6]['phase_id'] = 'functional'
    originals = {name: (root/name).read_bytes() for name in ('metrics.jsonl','freeze.json','campaign.json')}
    try:
        (root/'metrics.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
        # Rebind transport checksums to probe science validators rather than byte integrity.
        freeze = json.loads(originals['freeze.json'])
        freeze['metrics']['sha256'] = file_digest(root/'metrics.jsonl')
        write_manifest(root/'freeze.json', freeze)
        campaign = json.loads(originals['campaign.json'])
        campaign['freeze_reference']['sha256'] = file_digest(root/'freeze.json')
        write_manifest(root/'campaign.json', campaign)
        with pytest.raises(ValueError):
            panels.recipe_training_rows(root)
    finally:
        for name, data in originals.items():
            (root/name).write_bytes(data)
