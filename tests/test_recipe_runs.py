"""Public recipe dispatch with real tiny CPU updates and durable continuation."""
import json
from pathlib import Path
from time import perf_counter

import pytest
import torch
import yaml

from jscc.activation_replay import file_digest
from jscc.sharing_qualification_guard import state_exact
from test_recipe_assets import registered
from test_recipe_workflow import workflow, terminal_payload
from test_original13_evaluation import score_stub

pytest_plugins = ['test_recipe_workflow']


def read(path):
    return json.loads(Path(path).read_text())


def public(recipe, registrations, root, *extra):
    from train import main
    record = main(['--config', str(recipe), '--registrations', str(registrations),
                   '--output', str(root), '--timeout-seconds', '600', *extra])
    assert isinstance(record, dict)
    return record


def result(root, case='width-8-ratio-50', attempt='0001'):
    path = root/'cases'/case/'attempts'/f'training-{attempt}'
    return path, read(path/'engine/campaign.json')


def register_retained(assets):
    from scripts.experiment_recipe import main
    _, _, recipe, prepared, _ = workflow(assets, 'retained', retained=True)
    registrations = assets['root']/'registrations'
    registrations.mkdir()
    main(['--register-from-prepared', str(prepared), '--model-name', str(assets['model']),
          '--output', str(registrations/'assets.json')])
    return recipe, registrations


def test_public_retained_recipe_counts_and_terminal(joined_assets, monkeypatch):
    # Explicit scoring stub: acceptance is real update/lifecycle parity, not quality.
    monkeypatch.setattr('jscc.encoder_stages_evaluation._score_original13_panel', score_stub)
    recipe, registrations = register_retained(joined_assets)
    root = joined_assets['root']/'public-retained'
    started = perf_counter()
    record = public(recipe, registrations, root)
    path, campaign = result(root)
    assert campaign['completed_updates'] == campaign['segment_updates'] == 520
    assert len(campaign['checkpoints']) == 11
    counts = record['cases'][0]['counts']
    assert counts['optimizer_updates'] == 520
    assert counts['base_stream_presentations'] == 640
    assert counts['processed_presentations'] == 8320
    terminal = read(root/'cases/width-8-ratio-50/checkpoint.json')
    assert terminal['role'] == 'terminal' and terminal['step'] == 520
    assert terminal['sha256'] == file_digest(terminal['path'])
    assert terminal['training_attempt'] == str(path)
    assert terminal_payload(campaign)['metadata']['completed_updates'] == 520
    (root/'test-cost.json').write_text(json.dumps({'seconds': perf_counter()-started,
        'optimizer_updates': 520, 'processed_presentations': 8320, 'scoring': 'explicit stub'}))


def test_changed_public_recipe_boundary_recovery_exact_state(joined_assets, monkeypatch):
    from jscc import original13_training
    recipe, registrations, _ = registered(joined_assets)
    authored = yaml.safe_load(recipe.read_text())
    authored['reconstruction_percentages'] = [50, 75]
    authored['locations'].reverse()
    authored['overrides']['training'].update(lr=.003, weight_decay=.2, warmup_ratio=.5,
                                            min_lr_ratio=.6, betas=[.7, .9], eps=1e-6)
    authored['save'].update(reconstruction_fractions=[.5, 1], functional_fractions=[.5, 1])
    recipe.write_text(yaml.safe_dump(authored))
    root = joined_assets['root']/'reference'
    started = perf_counter()
    record = public(recipe, registrations, root, '--case', 'width-8-ratio-50')
    _, campaign = result(root)
    final = terminal_payload(campaign)
    counts = record['cases'][0]['counts']
    assert counts['selected_rows'] == 12
    assert counts['optimizer_updates'] == 24
    assert counts['base_stream_presentations'] == 128
    assert counts['processed_presentations'] == 384
    assert campaign['segment_updates'] == 24
    assert {site: row['updates'] for site, row in final['metadata']['exposure'].items()} == {
        site: 8 for site in authored['locations']}
    events = [json.loads(line) for line in (result(root)[0]/'engine/metrics.jsonl').read_text().splitlines()]
    assert [row['payload']['site_id'] for row in events] == authored['locations']*8
    original = original13_training.run_recipe_training
    tested = []
    # Reconstruction step 6 already crosses the 12-row sampler cycle (native16).
    # Both functional saves are exercised; no completed reference root is resumed.
    boundaries = [(row['role'], torch.load(row['path'], weights_only=False)['metadata']['completed_updates'])
                  for row in campaign['checkpoints'] if row['role'] != 'terminal']
    for role, step in boundaries:
        interrupted = joined_assets['root']/f'interrupted-{role}-{step}'
        def stop_once(*args, **kwargs):
            publication = kwargs['on_checkpoint']
            def publish(saved, actual_role):
                publication(saved, actual_role)
                if actual_role == role and saved.payload['metadata']['completed_updates'] == step:
                    raise KeyboardInterrupt('fixture interruption after durable publication')
            kwargs['on_checkpoint'] = publish
            return original(*args, **kwargs)
        monkeypatch.setattr(original13_training, 'run_recipe_training', stop_once)
        with pytest.raises(KeyboardInterrupt):
            public(recipe, registrations, interrupted, '--case', 'width-8-ratio-50')
        attempt = interrupted/'cases/width-8-ratio-50/attempts/training-0001'
        assert read(attempt/'attempt.json')['status'] == 'interrupted'
        saved = read(attempt/'checkpoints.json')['checkpoints'][-1]
        assert saved['role'] == role and saved['step'] == step
        assert not (interrupted/'cases/width-8-ratio-50/.training-active').exists()
        with pytest.raises(ValueError, match='no implicit retry'):
            public(recipe, registrations, interrupted, '--case', 'width-8-ratio-50')
        with pytest.raises(ValueError, match='recorded nonterminal checkpoint from this case'):
            public(recipe, registrations, interrupted, '--case', 'width-8-ratio-75',
                   '--resume', saved['path'], '--attempt', '0002')
        monkeypatch.setattr(original13_training, 'run_recipe_training', original)
        public(recipe, registrations, interrupted, '--case', 'width-8-ratio-50',
               '--resume', saved['path'], '--attempt', '0002')
        _, resumed = result(interrupted, attempt='0002')
        terminal = terminal_payload(resumed)
        for field in ('model', 'optimizer', 'scheduler', 'scaler', 'rng', 'stream'):
            assert state_exact(terminal[field], final[field]), (role, step, field)
        assert resumed['segment_updates'] == 24-step
        before = file_digest(attempt/'checkpoints.json')
        with pytest.raises(ValueError, match='already completed'):
            public(recipe, registrations, interrupted, '--case', 'width-8-ratio-50',
                   '--resume', saved['path'], '--attempt', '0002')
        assert file_digest(attempt/'checkpoints.json') == before
        tested.append({'role': role, 'step': step, 'resumed_updates': 24-step})
    (root/'test-cost.json').write_text(json.dumps({'seconds': perf_counter()-started,
        'reference_updates': 24, 'boundaries': tested, 'selected_rows': 12, 'batch_size': 16}))


def test_multiple_cases_selection_frozen_yaml_and_record_integrity(joined_assets, monkeypatch):
    recipe, registrations, _ = registered(joined_assets)
    authored = yaml.safe_load(recipe.read_text())
    authored['reconstruction_percentages'] = [50, 75]
    recipe.write_text(yaml.safe_dump(authored))
    selected = joined_assets['root']/'selected'
    record = public(recipe, registrations, selected, '--prepare-only')
    assert [row['case_id'] for row in record['cases']] == ['width-8-ratio-50', 'width-8-ratio-75']
    assert record['totals']['optimizer_updates'] == 48
    recipe.write_text('invalid authored YAML after binding: [')
    public(recipe, registrations, selected, '--case', 'width-8-ratio-75')
    assert not (selected/'cases/width-8-ratio-50/attempts').exists()
    with pytest.raises(ValueError, match='already completed'):
        public(recipe, registrations, selected, '--case', 'width-8-ratio-75')
    with pytest.raises(ValueError, match='explicit named case'):
        public(recipe, registrations, selected)
    case_record = selected/'cases/width-8-ratio-50/counts.json'
    case_record.write_text('{}')
    with pytest.raises(ValueError, match='record changed'):
        public(recipe, registrations, selected, '--case', 'width-8-ratio-50')
    recipe.write_text(yaml.safe_dump(authored))
    from jscc import recipe_runs
    seen = []
    original = recipe_runs.train_recipe_case
    def traced(root, case, **kwargs):
        seen.append(case)
        assert not list(Path(root).glob('cases/*/.training-active'))
        return original(root, case, **kwargs)
    monkeypatch.setattr(recipe_runs, 'train_recipe_case', traced)
    sequential = joined_assets['root']/'sequential'
    public(recipe, registrations, sequential)
    assert seen == ['width-8-ratio-50', 'width-8-ratio-75']
    assert all(read(sequential/'cases'/case/'checkpoint.json')['role'] == 'terminal' for case in seen)


def test_public_prepare_reuses_assets_without_download_or_full_hash(joined_assets, monkeypatch):
    recipe, registrations, _ = registered(joined_assets)
    public(recipe, registrations, joined_assets['root']/'first', '--prepare-only')
    from jscc import recipe_assets, original13_provenance, encoder_stages_preparation
    hashes = []
    def measured(path):
        if Path(path).suffix in ('.safetensors', '.bin', '.pt', '.arrow'):
            hashes.append(str(path))
        return file_digest(path)
    for module in (recipe_assets, original13_provenance, encoder_stages_preparation):
        monkeypatch.setattr(module, 'file_digest', measured)
    monkeypatch.setattr('huggingface_hub.snapshot_download', lambda *a, **k: pytest.fail('download'))
    monkeypatch.setattr('transformers.AutoModelForSeq2SeqLM.from_pretrained', lambda *a, **k: pytest.fail('model construction'))
    started = perf_counter()
    public(recipe, registrations, joined_assets['root']/'second', '--prepare-only')
    assert hashes == []
    (joined_assets['root']/'public-preparation-cost.json').write_text(json.dumps({
        'seconds': perf_counter()-started, 'large_file_hashes': hashes, 'cases': 1, 'optimizer_updates': 0}))


@pytest.mark.parametrize('change,message', [
    ({'training': {'batch_size': 8}}, 'native16'),
    ({'training': {'gradient_accumulation': 2}}, 'native16'),
    ({'channel': {'type': 'rayleigh'}}, 'channel'),
])
def test_public_check_rejects_unsupported_choices(joined_assets, change, message):
    recipe, registrations, _ = registered(joined_assets)
    authored = yaml.safe_load(recipe.read_text())
    for section, values in change.items():
        authored['overrides'].setdefault(section, {}).update(values)
    recipe.write_text(yaml.safe_dump(authored))
    from train import main
    with pytest.raises(ValueError, match=message):
        main(['--config', str(recipe), '--registrations', str(registrations), '--check'])


def test_public_ordinary_dispatch_preserved(joined_assets, monkeypatch):
    from train import main
    calls = []
    monkeypatch.setattr('jscc.training.train', lambda config, resume=None: calls.append((config, resume)))
    main(['--config', str(joined_assets['task']), '--resume', 'ordinary.pt'])
    assert calls[0][0]['task'] == 'hellaswag' and calls[0][1] == 'ordinary.pt'


def test_public_check_rejects_unsupported_task(joined_assets):
    recipe, registrations, _ = registered(joined_assets)
    task = yaml.safe_load(joined_assets['task'].read_text())
    task['task'] = 'coco'
    joined_assets['task'].write_text(yaml.safe_dump(task))
    from train import main
    with pytest.raises(ValueError, match='HellaSwag'):
        main(['--config', str(recipe), '--registrations', str(registrations), '--check'])


def test_prepared_distinct_case_reservations_can_overlap(joined_assets, monkeypatch):
    """Mock engine proves the reservation graph; real CPU state tests are above."""
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    import threading
    from jscc import recipe_runs, original13_training
    recipe, registrations, _ = registered(joined_assets)
    authored = yaml.safe_load(recipe.read_text())
    authored['reconstruction_percentages'] = [50, 75]
    recipe.write_text(yaml.safe_dump(authored))
    root = joined_assets['root']/'concurrent'
    public(recipe, registrations, root, '--prepare-only')
    rows = read(root/'cases/width-8-ratio-50/data_selection.json')
    config = yaml.safe_load((root/'cases/width-8-ratio-50/config.yaml').read_text())
    monkeypatch.setattr(recipe_runs, 'load_recipe_workflow',
        lambda *a, **k: (None, config, object(), object(), {}, {}, rows))
    monkeypatch.setattr(recipe_runs, 'initialize_recipe', lambda *a, **k: (object(), object()))
    barrier = threading.Barrier(2)
    def engine(*args, output, on_checkpoint, **kwargs):
        case = output.parents[2]
        with pytest.raises(FileExistsError):
            recipe_runs.train_recipe_case(root, case.name)
        barrier.wait(timeout=10)
        assert len(list(root.glob('cases/*/.training-active'))) == 2
        # Keep both reservations live until both assertions have finished.
        barrier.wait(timeout=10)
        output.mkdir()
        checkpoint = output/'fixture-terminal.pt'
        checkpoint.write_bytes(b'explicit mock reservation checkpoint')
        on_checkpoint(SimpleNamespace(reference=SimpleNamespace(path=str(checkpoint),
            sha256=file_digest(checkpoint), size=checkpoint.stat().st_size),
            payload={'metadata': {'completed_updates': 24}}), 'terminal')
        return {'status': 'fixture-mock'}
    monkeypatch.setattr(original13_training, 'run_recipe_training', engine)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(recipe_runs.train_recipe_case, root, case)
                   for case in ('width-8-ratio-50', 'width-8-ratio-75')]
        attempts = [future.result(timeout=30) for future in futures]
    assert len(set(attempts)) == 2
    assert all(read(path/'attempt.json')['status'] == 'completed' for path in attempts)
    assert not list(root.glob('cases/*/.training-active'))
