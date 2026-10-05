"""Public recipe dispatch with real tiny CPU updates and durable continuation."""

import jscc

from jscc import shared_training as _body_shared_training
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import pytest
import torch
import yaml

from jscc.activation_replay import file_digest
from jscc.qualification import state_exact
from shared.test_recipe_assets import registered
from shared.test_recipe_workflow import workflow, terminal_payload
from shared.test_shared_case_evaluation import score_stub

pytest_plugins = ['shared.test_recipe_workflow']


def read(path):
    return json.loads(Path(path).read_text())


def public(recipe, registrations, root, *extra):
    from request_helpers import invoke
    from jscc.shared_config import read_recipe
    from jscc.shared_runs import recipe_run_record
    arguments = [value for value in extra if value != '--finalize-terminal']
    options = dict(zip(arguments[::2], arguments[1::2])) if '--prepare-only' not in extra else {}
    if not Path(root).exists():
        science = read_recipe(recipe)
        science.pop('schema')
        science['task_config'] = str((recipe.parent/science['task_config']).resolve())
        invoke(recipe.parent, 'shared', 'prepare', {'root': str(root), 'preparation': {'registrations': str(registrations)}}, science)
    if '--prepare-only' not in extra:
        ex = {'root': str(root), 'attempt': options.get('--attempt', '0001'), 'timeout_seconds': 600}
        if '--case' in options:
            ex['case'] = options['--case']
        if '--resume' in options:
            checkpoint = Path(options['--resume'])
            index = read(checkpoint.parents[1]/'checkpoints.json')
            saved = next(row for row in index['checkpoints'] if row['path'] == str(checkpoint))
            ex['continuation'] = {'path': str(checkpoint), 'sha256': saved['sha256'], 'size_bytes': saved['size'],
                                   'role': saved['role'], 'step': saved['step'], 'parent_attempt': str(checkpoint.parents[1])}
            if '--finalize-terminal' in extra:
                ex['continuation']['finalize_terminal'] = True
            invoke(recipe.parent, 'shared', 'resume', ex)
        else:
            invoke(recipe.parent, 'shared', 'train', ex)
    return recipe_run_record(root)


def result(root, case='width-8-ratio-50', attempt='0001'):
    path = root/'cases'/case/'attempts'/f'training-{attempt}'
    return path, read(path/'engine/campaign.json')


def register_retained(assets):
    from request_helpers import invoke
    from jscc.shared_preparation import bind_recipe_workflow
    root, authored, recipe, _, manifest = workflow(assets, 'retained', retained=True)
    authored['evaluation'].update(fewshot=5, candidate_batch=16)
    recipe.write_text(yaml.safe_dump(authored))
    terminal = read(root/'terminal.json')
    terminal['prompt_policy'] = 'five-shot'
    terminal['settings'].update(num_fewshot=5, batch_size=16)
    (root/'terminal.json').write_text(json.dumps(terminal))
    prepared = root/'matching-prepared.json'
    bind_recipe_workflow(recipe, model_config_path=assets['model']/'config.json',
        model_path=assets['model'], model_binding=assets['model_binding'],
        cache_manifest=manifest['cache']['path'], cache_sha256=manifest['cache']['sha256'],
        initializations=manifest['initializations'], observation_template=manifest['observation_template'],
        terminal_template={'path': str(root/'terminal.json'), 'sha256': file_digest(root/'terminal.json')},
        study_pairing_id='joined-tiny', output=prepared, synthetic=True)
    registrations = assets['root']/'registrations'
    registrations.mkdir()
    invoke(assets['root'], 'shared', 'register', {'registration': {'from_prepared': str(prepared),
           'model_name': str(assets['model']), 'output': str(registrations/'assets.json')}})
    return recipe, registrations


def test_public_retained_recipe_counts_and_terminal(joined_assets, monkeypatch):
    # Explicit scoring stub: acceptance is real update/lifecycle parity, not quality.
    monkeypatch.setattr('jscc.panel_evaluation._score_panel', score_stub)
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
    assert terminal['sha256'] == file_digest(Path(terminal['path']))
    assert terminal['training_attempt'] == str(path)
    assert terminal_payload(campaign)['metadata']['completed_updates'] == 520
    (root/'test-cost.json').write_text(json.dumps({'seconds': perf_counter()-started,
        'optimizer_updates': 520, 'processed_presentations': 8320, 'scoring': 'explicit stub'}))


def test_changed_public_recipe_boundary_recovery_exact_state(joined_assets, monkeypatch):
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
    original = _body_shared_training.run_recipe_training
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
        hard_death = (role, step) == boundaries[0]
        if hard_death:
            public(recipe, registrations, interrupted, '--prepare-only')
            child_code = '''import json, os, sys
from jscc import shared_training
from experiment import main
original = shared_training.run_recipe_training
role, step, argv = json.loads(sys.argv[1])
def kill_after_save(*args, **kwargs):
    publication = kwargs['on_checkpoint']
    def publish(saved, actual_role):
        publication(saved, actual_role)
        if actual_role == role and saved.payload['metadata']['completed_updates'] == step:
            os._exit(23)
    kwargs['on_checkpoint'] = publish
    return original(*args, **kwargs)
shared_training.run_recipe_training = kill_after_save
main(argv)
'''
            child_request = joined_assets['root']/'child-request.yaml'
            child_request.write_text(yaml.safe_dump({'schema': 'jscc-experiment-v1', 'workflow': 'shared', 'operation': 'train',
                'execution': {'root': str(interrupted), 'case': 'width-8-ratio-50', 'attempt': '0001', 'timeout_seconds': 600}}))
            child = subprocess.run([sys.executable, '-c', child_code, json.dumps([role, step,
                ['--config', str(child_request)]])], cwd=Path(jscc.__file__).resolve().parents[1],
                 capture_output=True, text=True, timeout=120)
            assert child.returncode == 23, child.stderr[-4000:]
        else:
            monkeypatch.setattr(_body_shared_training, 'run_recipe_training', stop_once)
            with pytest.raises(KeyboardInterrupt):
                public(recipe, registrations, interrupted, '--case', 'width-8-ratio-50')
        attempt = interrupted/'cases/width-8-ratio-50/attempts/training-0001'
        assert read(attempt/'attempt.json')['status'] == ('running' if hard_death else 'interrupted')
        saved = read(attempt/'checkpoints.json')['checkpoints'][-1]
        assert saved['role'] == role and saved['step'] == step
        assert (interrupted/'cases/width-8-ratio-50/.training-active').exists() == hard_death
        if not hard_death:
            with pytest.raises(ValueError, match='no implicit retry'):
                public(recipe, registrations, interrupted, '--case', 'width-8-ratio-50')
        with pytest.raises(ValueError, match='recorded nonterminal checkpoint from this case'):
            public(recipe, registrations, interrupted, '--case', 'width-8-ratio-75',
                   '--resume', saved['path'], '--attempt', '0002')
        monkeypatch.setattr(_body_shared_training, 'run_recipe_training', original)
        reconciliation = []
        if hard_death:
            # subprocess.run has reaped the stopped owner before reconciliation.
            with pytest.raises(FileExistsError):
                public(recipe, registrations, interrupted, '--case', 'width-8-ratio-50',
                       '--resume', saved['path'], '--attempt', '0002')
            reconciliation = ['--reconcile-interrupted', '0001']
        public(recipe, registrations, interrupted, '--case', 'width-8-ratio-50',
               '--resume', saved['path'], '--attempt', '0002', *reconciliation)
        assert read(attempt/'attempt.json')['status'] == 'interrupted'
        if hard_death:
            from jscc.shared_runs import reconcile_recipe_attempt
            receipt = (attempt/'reconciliation.json').read_bytes()
            with pytest.raises(ValueError, match='unresolved running attempt'):
                reconcile_recipe_attempt(interrupted, 'width-8-ratio-50', '0001')
            assert (attempt/'reconciliation.json').read_bytes() == receipt
            assert not (interrupted/'cases/width-8-ratio-50/.reconcile-active').exists()
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
        tested.append({'role': role, 'step': step, 'resumed_updates': 24-step,
                       'reaped_subprocess_exit': 23 if hard_death else None})
    (root/'test-cost.json').write_text(json.dumps({'seconds': perf_counter()-started,
        'reference_updates': 24, 'boundaries': tested, 'selected_rows': 12, 'batch_size': 16}))


def test_multiple_cases_selection_frozen_yaml_and_record_integrity(joined_assets, monkeypatch):
    recipe, registrations, _ = registered(joined_assets)
    authored = yaml.safe_load(recipe.read_text())
    authored['reconstruction_percentages'] = [50, 75]
    authored['save'].update(reconstruction_fractions=[.5, 1], functional_fractions=[.5, 1])
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
    with pytest.raises(ValueError, match='already completed'):
        public(recipe, registrations, selected)
    assert not (selected/'cases/width-8-ratio-50/attempts').exists()
    assert not list(selected.glob('cases/*/.training-active'))
    assert not (selected/'cases/width-8-ratio-50/training.json').exists()
    case_record = selected/'cases/width-8-ratio-50/counts.json'
    case_record.write_text('{}')
    with pytest.raises(ValueError, match='record changed'):
        public(recipe, registrations, selected, '--case', 'width-8-ratio-50')
    recipe.write_text(yaml.safe_dump(authored))
    from jscc import shared_runs as recipe_runs
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
    from jscc import shared_assets as recipe_assets, asset_identity, shared_inputs, shared_preparation, checkpoint_readers
    hashes = []
    def measured(path):
        if Path(path).suffix in ('.safetensors', '.bin', '.pt', '.arrow'):
            hashes.append(str(path))
        return file_digest(path)
    for module in (recipe_assets, asset_identity, shared_inputs, shared_preparation, checkpoint_readers):
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
    with pytest.raises(ValueError, match=message):
        from request_helpers import invoke
        from jscc.shared_config import read_recipe
        science = read_recipe(recipe)
        science.pop('schema')
        invoke(recipe.parent, 'shared', 'check', {'resolution': {'model_config': str(joined_assets['model']/'config.json')}}, science)


def test_public_ordinary_dispatch_preserved(joined_assets, monkeypatch):
    from request_helpers import invoke
    calls = []
    monkeypatch.setattr('jscc.ordinary_runs.train_task', lambda config, output, **kwargs: calls.append((config, output, kwargs)))
    config = yaml.safe_load(joined_assets['task'].read_text())
    config['training'].pop('mixed_precision', None)
    config['training'].pop('record_coco_presentations', None)
    root = joined_assets['root']/'ordinary'
    invoke(root.parent, 'ordinary', 'train', {'root': str(root), 'attempt': '0001'}, config)
    assert calls[0][0]['task'] == 'hellaswag' and calls[0][1] == str(root)
    assert calls[0][2]['resume'] is None


def test_public_check_rejects_unsupported_task(joined_assets):
    recipe, registrations, _ = registered(joined_assets)
    task = yaml.safe_load(joined_assets['task'].read_text())
    task['task'] = 'unsupported-task'
    joined_assets['task'].write_text(yaml.safe_dump(task))
    with pytest.raises(ValueError, match='HellaSwag and COCO'):
        from request_helpers import invoke
        from jscc.shared_config import read_recipe
        science = read_recipe(recipe)
        science.pop('schema')
        invoke(recipe.parent, 'shared', 'check', {'resolution': {'model_config': str(joined_assets['model']/'config.json')}}, science)


def test_prepared_distinct_case_reservations_can_overlap(joined_assets, monkeypatch):
    """Mock engine proves the reservation graph; real CPU state tests are above."""
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    import threading
    from jscc import shared_runs as recipe_runs
    recipe, registrations, _ = registered(joined_assets)
    authored = yaml.safe_load(recipe.read_text())
    authored['reconstruction_percentages'] = [50, 75]
    authored['save'].update(reconstruction_fractions=[.5, 1], functional_fractions=[.5, 1])
    recipe.write_text(yaml.safe_dump(authored))
    root = joined_assets['root']/'concurrent'
    public(recipe, registrations, root, '--prepare-only')
    rows = read(root/'cases/width-8-ratio-50/data_selection.json')
    config = yaml.safe_load((root/'cases/width-8-ratio-50/config.yaml').read_text())
    monkeypatch.setattr(recipe_runs, 'load_recipe_workflow',
        lambda *a, **k: (None, config, SimpleNamespace(trained_sites=tuple(authored['locations'])),
                        object(), {}, {}, rows))
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
    monkeypatch.setattr(_body_shared_training, 'run_recipe_training', engine)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(recipe_runs.train_recipe_case, root, case)
                   for case in ('width-8-ratio-50', 'width-8-ratio-75')]
        attempts = [future.result(timeout=30) for future in futures]
    assert len(set(attempts)) == 2
    assert all(read(path/'attempt.json')['status'] == 'completed' for path in attempts)
    assert not list(root.glob('cases/*/.training-active'))


def test_terminal_after_save_interruption_explicit_finalization(joined_assets, monkeypatch):
    from jscc.panel_evaluation import open_recipe_terminal
    recipe, registrations, _ = registered(joined_assets)
    root = joined_assets['root']/'terminal-interrupted'
    original = _body_shared_training.run_recipe_training
    def stop_after_terminal(*args, **kwargs):
        publication = kwargs['on_checkpoint']
        def publish(saved, role):
            publication(saved, role)
            if role == 'terminal':
                raise KeyboardInterrupt('durable terminal precedes freeze publication')
        kwargs['on_checkpoint'] = publish
        return original(*args, **kwargs)
    monkeypatch.setattr(_body_shared_training, 'run_recipe_training', stop_after_terminal)
    with pytest.raises(KeyboardInterrupt):
        public(recipe, registrations, root, '--case', 'width-8-ratio-50')
    prior = root/'cases/width-8-ratio-50/attempts/training-0001'
    saved = read(prior/'checkpoints.json')['checkpoints'][-1]
    assert saved['role'] == 'terminal' and saved['step'] == 24
    assert not (root/'cases/width-8-ratio-50/checkpoint.json').exists()
    assert not (prior/'engine/freeze.json').exists()
    before = torch.load(saved['path'], map_location='cpu', weights_only=False)
    monkeypatch.setattr(_body_shared_training, 'run_recipe_training', original)
    with pytest.raises(ValueError, match='terminal'):
        public(recipe, registrations, root, '--case', 'width-8-ratio-50',
               '--resume', saved['path'], '--attempt', '0002')
    # Rejection before attempt reservation must preserve the fresh identifier.
    public(recipe, registrations, root, '--case', 'width-8-ratio-50',
           '--resume', saved['path'], '--attempt', '0002', '--finalize-terminal')
    attempt, campaign = result(root, attempt='0002')
    after = terminal_payload(campaign)
    assert campaign['segment_updates'] == 0 and campaign['completed_updates'] == 24
    assert len(campaign['checkpoints']) == 1 and campaign['checkpoints'][0]['role'] == 'terminal'
    for field in ('model', 'optimizer', 'scheduler', 'scaler', 'rng', 'stream'):
        assert state_exact(after[field], before[field]), field
    state, _ = open_recipe_terminal(attempt/'engine')
    assert state.reference.path == campaign['checkpoints'][0]['path']
    assert state.payload['metadata']['completed_updates'] == 24
    assert file_digest(Path(saved['path'])) == saved['sha256']


def test_bound_request_helper_uses_recorded_root_without_authored_yaml(tmp_path, monkeypatch):
    root = tmp_path/'bound'
    root.mkdir()
    authored = tmp_path/'later-edit.yaml'
    authored.write_text('invalid authored YAML after binding: [')
    calls = []
    monkeypatch.setattr('request_helpers.invoke', lambda directory, workflow, operation, execution, science=None:
        calls.append((workflow, operation, execution, science)))
    expected = {'cases': ['recorded-case']}
    monkeypatch.setattr('jscc.shared_runs.recipe_run_record', lambda value: expected)
    assert public(authored, tmp_path/'registrations', root, '--case', 'recorded-case') == expected
    assert calls == [('shared', 'train', {'root': str(root), 'attempt': '0001',
        'timeout_seconds': 600, 'case': 'recorded-case'}, None)]


@pytest.mark.parametrize('flag_position', ['before', 'after'])
def test_public_helper_preserves_standalone_terminal_finalization(tmp_path, monkeypatch, flag_position):
    root = tmp_path/'bound'
    checkpoint = root/'cases/case/attempts/training-prior/engine/terminal.pt'
    checkpoint.parent.mkdir(parents=True)
    saved = {'path': str(checkpoint), 'sha256': '1'*64, 'size': 1, 'role': 'terminal', 'step': 24}
    (checkpoint.parents[1]/'checkpoints.json').write_text(json.dumps({'checkpoints': [saved]}))
    calls = []
    monkeypatch.setattr('request_helpers.invoke', lambda directory, workflow, operation, execution, science=None:
        calls.append((workflow, operation, execution, science)))
    record = {'cases': ['case']}
    monkeypatch.setattr('jscc.shared_runs.recipe_run_record', lambda value: record)
    arguments = ['--case', 'case', '--resume', str(checkpoint), '--attempt', 'finalize']
    arguments.insert(0 if flag_position == 'before' else len(arguments), '--finalize-terminal')
    assert public(tmp_path/'unused.yaml', tmp_path/'registrations', root, *arguments) == record
    assert calls == [('shared', 'resume', {'root': str(root), 'attempt': 'finalize',
        'timeout_seconds': 600, 'case': 'case', 'continuation': {
            'path': str(checkpoint), 'sha256': saved['sha256'], 'size_bytes': 1,
            'role': 'terminal', 'step': 24, 'parent_attempt': str(checkpoint.parents[1]),
            'finalize_terminal': True}}, None)]
