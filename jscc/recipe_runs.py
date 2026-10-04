"""Public recipe roots, independent cases and explicit execution attempts."""
import json
import math
import os
import re
from pathlib import Path
import time

import torch

from .activation_replay import canonical_digest, file_digest
from .experiment_records import write_json_atomic
from .ordinary_runs import (reserve_root, bind_case, bound_config, reserve_attempt,
                            validate_reserved_attempt, finish_attempt)
from .recipe_assets import prepare_recipe_assets
from .recipe_workflow import (_read_prepared, load_recipe_workflow, validate_recipe_prepared,
                              verify_recipe_source, initialize_recipe)

SCHEMA = 'recipe-run-v1'


def case_id(description):
    return f"width-{description['width']}-ratio-{description['ratio']}"


def _read(path):
    return json.loads(Path(path).read_text())


def _manifest(record):
    reference = record['prepared']
    manifest = _read_prepared(reference['path'], reference['sha256'])
    validate_recipe_prepared(manifest)
    return manifest


def _case_descriptions(manifest, selected):
    descriptions = {case_id(item): item for item in manifest['resolved']['trajectories']}
    if selected is not None and selected not in descriptions:
        raise ValueError(f'unknown recipe case {selected!r}; declared: {list(descriptions)}')
    return descriptions


def prepare_recipe_run(recipe, output, *, registrations):
    """Bind all declared cases once. No model construction or execution."""
    root = reserve_root(output, kind='shared-codec-recipe')
    manifest = prepare_recipe_assets(recipe, registrations=registrations, output=root/'preparation')
    # Select actual rows with the data owner's policy, without tokenizing or
    # constructing Q views. All cases share the same authored data settings.
    from datasets import Dataset
    from .data.hellaswag import recipe_selection_ids
    from .original13_inputs import validate_recipe_cache
    cache = _read(manifest['cache']['path'])
    arrow, _ = validate_recipe_cache(cache)
    raw = Dataset.from_file(str(arrow))
    descriptions = _case_descriptions(manifest, None)
    cases = []
    for name, item in descriptions.items():
        selected = recipe_selection_ids(raw, item['config']['data'], cache['fixed_data_ids'])
        case = bind_case(root, item['config'], authored_config=recipe, case_id=name)
        counts = {**item['counts'], 'selected_rows': len(selected['train_rows']),
                  'processed_presentations': item['counts']['site_presentations']}
        write_json_atomic(case/'recipe.json', item)
        write_json_atomic(case/'counts.json', counts)
        write_json_atomic(case/'data_selection.json', selected)
        cases.append({'case_id': name, 'path': str(case), 'width': item['width'], 'ratio': item['ratio'],
                      'recipe_sha256': file_digest(case/'recipe.json'),
                      'counts_sha256': file_digest(case/'counts.json'),
                      'selection_sha256': file_digest(case/'data_selection.json'), 'counts': counts})
    record = {'schema': SCHEMA, 'root': str(root),
              'prepared': {'path': str(root/'preparation/prepared.json'),
                           'sha256': file_digest(root/'preparation/prepared.json')},
              'cases': cases, 'totals': {'cases': len(cases), **{
                  key: sum(row['counts'][key] for row in cases) for key in
                  ('selected_rows', 'optimizer_updates', 'base_stream_presentations', 'processed_presentations')}}}
    record['identity'] = canonical_digest(record)
    write_json_atomic(root/'recipe-run.json', record)
    return root


def recipe_run_record(root):
    """Validate small bound records; never read the author's mutable YAML."""
    root = Path(root).resolve()
    record = _read(root/'recipe-run.json')
    if (record.get('schema') != SCHEMA or record.get('root') != str(root)
            or record.get('identity') != canonical_digest({k: v for k, v in record.items() if k != 'identity'})):
        raise ValueError('bound recipe root identity differs')
    manifest = _manifest(record)
    descriptions = _case_descriptions(manifest, None)
    if [row['case_id'] for row in record['cases']] != list(descriptions):
        raise ValueError('bound recipe case list differs')
    for row in record['cases']:
        case = root/'cases'/row['case_id']
        if row['path'] != str(case):
            raise ValueError('bound recipe case path differs')
        for name, key in (('recipe.json', 'recipe_sha256'), ('counts.json', 'counts_sha256'),
                          ('data_selection.json', 'selection_sha256')):
            if file_digest(case/name) != row[key]:
                raise ValueError('bound recipe case record changed: '+name)
        if _read(case/'recipe.json') != descriptions[row['case_id']] or bound_config(case) != descriptions[row['case_id']]['config']:
            raise ValueError('bound recipe effective case differs')
    return record


def _guard(config, *, synthetic, timeout_seconds, device_uuid):
    if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError('recipe finite positive execution timeout required')
    device = config['model']['device']
    if device == 'cpu':
        if not synthetic:
            raise ValueError('production recipe requires CUDA; CPU requires registered synthetic assets')
    elif device == 'cuda':
        if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
            raise ValueError('recipe requires one assigned BF16-capable GPU')
        if device_uuid is not None and os.environ.get('CUDA_VISIBLE_DEVICES') != device_uuid:
            raise ValueError('recipe assigned GPU differs from explicit device UUID')
    else:
        raise ValueError('recipe supports CUDA or registered synthetic CPU only')
    deadline = time.monotonic()+timeout_seconds
    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError('recipe execution timeout reached')
        if device_uuid is not None and os.environ.get('CUDA_VISIBLE_DEVICES') != device_uuid:
            raise ValueError('recipe assigned GPU changed during execution')
    return check


def _resume_reference(case, checkpoint, *, finalize_terminal=False):
    """Only an exact durable checkpoint from this case can start a new attempt."""
    from .experiment_state import CheckpointRef
    wanted = str(Path(checkpoint).resolve())
    matches = []
    for path in (case/'attempts').glob('training-*/checkpoints.json'):
        attempt = _read(path.parent/'attempt.json')
        if attempt['status'] not in ('interrupted', 'failed', 'partial'):
            continue
        matches.extend(row for row in _read(path)['checkpoints'] if row['path'] == wanted)
    if len(matches) != 1 or (matches[0]['role'] == 'terminal' and not finalize_terminal):
        raise ValueError('resume requires one recorded nonterminal checkpoint from this case')
    saved = matches[0]
    return CheckpointRef(saved['path'], saved['sha256'], saved['size'])


def reconcile_recipe_attempt(root, selected, attempt_id):
    """Explicit caller assertion that the prior case worker has stopped.

    No PID, remote state or scheduler inference. This is an operational write;
    invoking it asserts the prior worker cannot resume or publish more records.
    Retain its durable checkpoint inventory and original attempt identity.
    """
    root = Path(root).resolve()
    record = recipe_run_record(root)
    if selected not in [row['case_id'] for row in record['cases']]:
        raise ValueError('unknown recipe case')
    if not isinstance(attempt_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', attempt_id):
        raise ValueError('attempt identifier must be a plain nonempty name')
    case = root/'cases'/selected
    attempt = case/'attempts'/f'training-{attempt_id}'
    previous = _read(attempt/'attempt.json')
    if previous['status'] != 'running' or previous['case'] != str(case):
        raise ValueError("reconciliation requires this case's unresolved running attempt")
    # Serialize reconcilers, without acquiring/stealing the active worker lock.
    lock = case/'.reconcile-active'
    lock.mkdir(exist_ok=False)
    try:
        write_json_atomic(attempt/'reconciliation.json', {
            'schema': SCHEMA, 'assertion': 'prior worker stopped; no further writes',
            'previous_attempt': previous, 'time_unix': time.time()})
        finish_attempt(attempt, 'interrupted', details={
            'reason': 'explicit stopped-worker reconciliation',
            'receipt': str(attempt/'reconciliation.json')})
        active = case/'.training-active'
        if active.exists():
            active.rmdir()
    finally:
        lock.rmdir()
    return attempt


def train_recipe_case(root, selected, *, attempt_id='0001', resume=None,
                      timeout_seconds=86400, device_uuid=None, finalize_terminal=False):
    """Execute one pending case or explicitly continue its recorded checkpoint."""
    root = Path(root).resolve()
    record = recipe_run_record(root)
    manifest = _manifest(record)
    descriptions = _case_descriptions(manifest, selected)
    if selected is None:
        raise ValueError('case execution requires a named case')
    case = root/'cases'/selected
    item = descriptions[selected]
    lock = case/'.training-active'
    lock.mkdir(exist_ok=False)
    attempt = None
    started = time.perf_counter()
    try:
        existing = list((case/'attempts').glob('training-*/attempt.json'))
        if any(_read(path)['status'] == 'completed' for path in existing):
            raise ValueError('recipe case already completed; use its recorded checkpoint for evaluation')
        if existing and resume is None:
            raise ValueError('existing recipe attempts require explicit checkpoint continuation; no implicit retry')
        reference = _resume_reference(case, resume, finalize_terminal=finalize_terminal) if resume else None
        if finalize_terminal and reference is None:
            raise ValueError('recipe finalization requires an exact terminal checkpoint')
        if reference is not None and attempt_id == '0001':
            raise ValueError('recipe continuation requires an explicit new attempt identifier')
        attempt = reserve_attempt(case, 'training', attempt_id=attempt_id,
                                  request={'case_id': selected, 'resume': None if reference is None else reference.__dict__})
        validate_reserved_attempt(attempt, 'training')
        engine = attempt/'engine'
        write_json_atomic(case/'training.json', {'path': str(attempt), 'engine_path': str(engine)})
        checkpoints = {'schema': SCHEMA, 'case_id': selected, 'checkpoints': []}
        write_json_atomic(attempt/'checkpoints.json', checkpoints)
        verify_recipe_source(manifest, manifest['source_commit'])
        # The public adapter is also fixed source beyond the legacy inventory.
        from .runtime import source_state
        if source_state().get('dirty'):
            raise ValueError('recipe execution requires clean source')
        guard = _guard(item['config'], synthetic=manifest['synthetic_cpu'],
                       timeout_seconds=timeout_seconds, device_uuid=device_uuid)
        guard()
        _, config, plan, stream, initial, metadata, ids = load_recipe_workflow(
            record['prepared']['path'], checksum=record['prepared']['sha256'], width=item['width'], ratio=item['ratio'])
        assert plan is not None and stream is not None and metadata is not None and ids is not None
        if ids['train_rows'] != _read(case/'data_selection.json')['train_rows']:
            raise ValueError('recipe execution row selection differs from bound case')
        processor, model = initialize_recipe(config, initial, plan.trained_sites, synthetic=manifest['synthetic_cpu'])
        guard()
        from .original13_training import run_recipe_training
        from .encoder_stages_evaluation import make_recipe_observer
        def publish(saved, role):
            row = {'path': saved.reference.path, 'sha256': saved.reference.sha256, 'size': saved.reference.size,
                   'step': saved.payload['metadata']['completed_updates'], 'role': role}
            checkpoints['checkpoints'].append(row)
            write_json_atomic(attempt/'checkpoints.json', checkpoints)
        result = run_recipe_training(model, processor, plan=plan, batches=stream, metadata=metadata,
            output=engine, guard=guard, observe=make_recipe_observer(_read(manifest['observation_template']['path'])),
            resume=reference, data_ids=ids, synthetic=manifest['synthetic_cpu'], on_checkpoint=publish,
            finalize_terminal=finalize_terminal)
        terminal = next(row for row in checkpoints['checkpoints'] if row['role'] == 'terminal')
        write_json_atomic(case/'checkpoint.json', {**terminal, 'training_attempt': str(attempt),
                                                  'engine_path': str(engine), 'policy': 'recipe-terminal'})
        finish_attempt(attempt, 'completed', details={'result': result, 'engine_path': str(engine),
            'counts': _read(case/'counts.json'), 'elapsed_seconds': time.perf_counter()-started})
    except BaseException as error:
        if attempt is not None:
            finish_attempt(attempt, 'interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed',
                details={'error_type': type(error).__name__, 'message': str(error),
                         'elapsed_seconds': time.perf_counter()-started})
        raise
    finally:
        lock.rmdir()
    return attempt


def train_recipe(recipe, output, *, registrations=None, selected=None, attempt_id='0001', resume=None,
                 prepare_only=False, timeout_seconds=86400, device_uuid=None, finalize_terminal=False):
    """Fresh roots run declared cases sequentially; existing roots require selection.

    Preparation freezes all cases so resource-assigned case jobs can consume one
    root concurrently. Selection never grants retry or sweep authorization.
    """
    if finalize_terminal and (resume is None or prepare_only):
        raise ValueError('recipe finalization requires exact terminal continuation')
    root = Path(output).expanduser().resolve()
    if root.exists():
        record = recipe_run_record(root)
        if selected is None or prepare_only:
            raise ValueError('existing recipe root requires one explicit named case')
    else:
        if resume is not None or attempt_id != '0001':
            raise ValueError('continuation requires an existing bound root')
        if registrations is None:
            raise ValueError('recipe requires registered pinned assets; pass --registrations')
        root = prepare_recipe_run(recipe, root, registrations=registrations)
        record = recipe_run_record(root)
    manifest = _manifest(record)
    descriptions = _case_descriptions(manifest, selected)
    if prepare_only:
        return root
    for name in ([selected] if selected else list(descriptions)):
        train_recipe_case(root, name, attempt_id=attempt_id, resume=resume,
                          timeout_seconds=timeout_seconds, device_uuid=device_uuid, finalize_terminal=finalize_terminal)
    return root
