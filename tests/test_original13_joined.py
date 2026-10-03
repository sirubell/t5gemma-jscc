"""Final routing + real local loader/runner/scorer; no model/data downloads."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest
import torch

from jscc import original13_routing as routing, encoder_stages_evaluation as evaluation
from jscc.activation_replay import canonical_digest, file_digest
from jscc.baseline_protocol import bind_batch_identity, write_manifest
from jscc.encoder_stages import (Original13TrainingState, initialize_original13, load_original13_prepared,
                                stages_source_inventory)
from jscc.encoder_stages_plan import Original13Plan, ORIGINAL13_PROTOCOL
from jscc.experiment_state import capture_rng, restore_rng
from jscc.sharing_preparation import state_dict_identity
from jscc.sharing_protocol import batch_view
from jscc.sharing_qualification_guard import state_exact
from test_harness_payload import fixture_model
from test_original13_routes import make_model, batch
from test_original13_plan_state import panel
from test_original13_evaluation import template, score_stub


def cli(name):
    path = Path(__file__).resolve().parents[1] / 'scripts' / name
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
    from jscc.encoder_stages_preparation import bind_model_assets
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
        'evaluation': template()['settings'], 'data': template()['data_settings']}
    observation = {**template(256), 'schema': 'original13-observation-request-v1', 'panel': panel(),
        'noise': panel()['noise'], 'sites': ['enc_l9', 'enc_fn', 'dec_l0', 'dec_l24'], 'conditions': ['no_noise', -6]}
    inventory = stages_source_inventory()
    manifest = {'schema': 'original13-prepared-v1', 'source_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1], text=True).strip(),
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


def test_real_loader_train_resume_observe_and_evaluation_cli(prepared, tmp_path, monkeypatch):
    path, manifest = prepared
    offline_dataset(monkeypatch)
    train = cli('original13_training.py')
    result = train(['train', *execution_args(path, manifest, tmp_path/'trained')])
    assert result['status'] == 'training_frozen' and len(result['checkpoints']) == 11
    assert result['observations_status'] == 'complete'
    receipts = [json.loads(p.read_text()) for p in (tmp_path/'trained').glob('observation_*/result.json')]
    assert len(receipts) == 2 and sum(r['completed_panels'] for r in receipts) == 16
    assert sum(p['completed_items'] for r in receipts for p in r['panels']) == 4096
    assert (tmp_path/'trained'/'config.yaml').is_file() and (tmp_path/'trained'/'data_ids.json').is_file()
    terminal, _ = evaluation.open_original13_terminal(tmp_path/'trained')
    pre = next(c for c in result['checkpoints'] if c['role'] == 'pre_reset')
    resumed = train(['resume', *execution_args(path, manifest, tmp_path/'resumed'),
        '--resume-checkpoint', pre['path'], '--resume-sha256', pre['sha256'], '--resume-size', str(pre['size'])])
    second, _ = evaluation.open_original13_terminal(tmp_path/'resumed')
    assert state_exact(terminal.payload['model'], second.payload['model'])
    assert state_exact(terminal.payload['optimizer'], second.payload['optimizer'])
    assert resumed['segment_updates'] == 520
    assert resumed['observations'][0]['status'] == 'pending_prior_segment'
    assert resumed['observations'][1]['status'] == 'complete'
    # Full terminal lifecycle uses a scoring fixture, while observations above
    # exercised the real HellaSwag adapter through exact-final decoder routes.
    request = evaluation.terminal_original13_request(tmp_path/'trained', manifest['terminal_request'])
    write_manifest(tmp_path/'request.json', request)
    monkeypatch.setattr(evaluation, '_score_original13_panel', score_stub)
    run = cli('original13_evaluation.py')
    receipt = run(['run', '--request', str(tmp_path/'request.json'), '--request-sha256', file_digest(tmp_path/'request.json'),
        '--prepared', str(path), '--prepared-sha256', file_digest(path),
        '--approved-source', manifest['source_commit'], '--timeout-seconds', '600', '--output', str(tmp_path/'evaluated')])
    assert receipt['completed_panels'] == 26 and receipt['checkpoint']['sha256'] == terminal.reference.sha256
    assert not torch.cuda.is_initialized()


def test_real_update_restore_scoring_failure_and_fatal_restore(prepared, tmp_path, monkeypatch):
    path, _ = prepared
    manifest, config, plan, batches, initial, metadata, _ = load_original13_prepared(path,
        checksum=file_digest(path), width=8, ratio=90)
    assert batches is not None
    processor, model = initialize_original13(config, initial, synthetic=True)
    learner = routing.create_original13_learner(model, plan=plan, metadata=metadata, event_sink=None)
    machine = Original13TrainingState(learner, plan=plan, metadata=metadata, routing=routing)
    # Reach the real reconstruction endpoint; no simulated state counters/moments.
    while learner.completed < plan.reconstruction_stop:
        data = batches[learner.completed//13]
        learner.update([data], batch_identities=[bind_batch_identity(data)])
    state = machine.save(tmp_path/'pre.pt')
    request = {**manifest['observation_request'], 'plan_identity': plan.identity, 'step': learner.completed,
               'state_role': 'pre_reset', 'checkpoint_sha256': state.reference.sha256}
    rng = capture_rng()
    runtime = routing.observation_state(model)
    def failed(*a, **kw):
        torch.rand(3)
        raise evaluation.ObservationScoringError('injected scoring failure')
    monkeypatch.setattr(evaluation, '_score_original13_panel', failed)
    with pytest.raises(evaluation.ObservationScoringError):
        evaluation.observe_original13(model, processor, state=state, request=request, output=tmp_path/'failed', guard=lambda: None)
    assert state_exact(rng, capture_rng()) and state_exact(runtime, routing.observation_state(model))
    machine.reset(state)
    data = batches[learner.completed//13]
    event = learner.update([data], batch_identities=[bind_batch_identity(data)])
    expected_weights, expected_optimizer = routing.export_codec(model), deepcopy(learner.optimizer.state_dict())
    processor2, model2 = initialize_original13(config, initial, synthetic=True)
    fresh = routing.create_original13_learner(model2, plan=plan, metadata=metadata, event_sink=None)
    restored = Original13TrainingState(fresh, plan=plan, metadata=metadata, routing=routing)
    restored.restore(state.reference)
    restored.reset(state)
    replay = fresh.update([data], batch_identities=[bind_batch_identity(data)])
    assert event['payload']['objective'] == replay['payload']['objective']
    assert state_exact(expected_weights, routing.export_codec(model2))
    assert state_exact(expected_optimizer, fresh.optimizer.state_dict())
    restore_rng(rng)
    def broken(*a, **kw):
        raise ValueError('fatal runtime restoration')
    monkeypatch.setattr(routing, 'restore_runtime_state', broken)
    with pytest.raises(ValueError, match='fatal runtime restoration'):
        evaluation.observe_original13(model2, processor2, state=state, request=request,
                                      output=tmp_path/'fatal', guard=lambda: None)


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


def test_actual_stream_events_persist_and_reject_science_drift(tmp_path):
    from jscc.experiment_records import AppendOnlyEventWriter, RecordError, read_events, validate_event
    from test_original13_objectives import real_setup, advance
    learner, _, batches = real_setup()
    learner.event_sink = AppendOnlyEventWriter(tmp_path/'metrics.jsonl').append
    for _ in range(7):
        advance(learner, batches)
    events = read_events(tmp_path/'metrics.jsonl')
    assert len(events) == 7
    assert events[0]['payload']['exposure']['memory_coordinates']['value'] is None
    assert events[0]['payload']['objective']['components']['memory']['raw']['value'] is None
    assert events[-1]['payload']['exposure']['memory_coordinates'] > 0
    for mutation in ('coefficient', 'partial_component', 'denominator', 'payload'):
        event = deepcopy(events[-1])
        p = event['payload']
        if mutation == 'coefficient':
            p['objective']['resolved_weights']['hidden'] = .1
        elif mutation == 'partial_component':
            p['objective']['components']['kl']['weighted'] = {'value': 0, 'reason': None}
        elif mutation == 'denominator':
            p['objective']['components']['memory']['denominator'] = 0
        else:
            p['exposure']['memory_coordinates_allocated'] = 0
        with pytest.raises(RecordError):
            validate_event(event)


def test_execution_guard_rejects_unbounded_or_wrong_assignment_and_git_redirects(prepared, monkeypatch):
    from jscc.encoder_stages import original13_execution_guard, verify_original13_execution_source
    _, manifest = prepared
    config = manifest['configs']['8']
    for timeout in (None, 0, -1, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='finite positive'):
            original13_execution_guard(config, synthetic=True, timeout_seconds=timeout)
    with pytest.raises(ValueError, match='assigned CUDA'):
        original13_execution_guard(config, synthetic=False, timeout_seconds=10)
    gpu = deepcopy(config)
    gpu['model']['device'] = 'cuda'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'wrong')
    with pytest.raises(ValueError, match='GPU UUID'):
        original13_execution_guard(gpu, synthetic=False, timeout_seconds=10, device_uuid='GPU-required')
    # Git process redirects must not change attribution of this source root.
    monkeypatch.setenv('GIT_DIR', '/nonexistent/redirect.git')
    verify_original13_execution_source(manifest, manifest['source_commit'])
    assert not torch.cuda.is_initialized()


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
        initialize_original13(config, initial, synthetic=True)


def test_pending_assets_and_input_tokenizer_mismatch_fail_closed(tmp_path):
    from jscc.encoder_stages_preparation import bind_model_assets, verify_model_assets
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
