"""Current shared evaluation bodies; saved record meanings preserved."""
from __future__ import annotations

import json
from pathlib import Path
from .shared_runs import recipe_run_record
import yaml
from .shared_config import _validate_scoring_conditions
from .shared_runs import _manifest
from .activation_replay import canonical_digest
from . import panel_evaluation as engine
from .experiment_state import CheckpointRef
from .run_records import runtime_identity, bound_config
from .shared_preparation import load_recipe_workflow, verify_recipe_source, _read_reference
from .runtime import source_state
import time
from .activation_replay import file_digest
from .experiment_records import write_json_atomic
from .experiment_state import isolated_rng
from .run_records import validate_reserved_attempt, finish_attempt
from .shared_runs import _guard
from .shared_preparation import initialize_recipe
from .run_records import _name, reserve_attempt

SCHEMA = 'recipe-run-evaluation-v1'


def _read(path):
    return json.loads(Path(path).read_text())


def is_recipe_run(path):
    path = Path(path).resolve()
    return ((path/'recipe-run.json').is_file() or
            ((path/'recipe.json').is_file() and (path.parent.parent/'recipe-run.json').is_file()))


def _selection(root, selected):
    path = Path(root).resolve()
    if (path/'recipe.json').is_file():
        if selected is not None and selected != path.name:
            raise ValueError('named case differs from selected recipe case path')
        selected, path = path.name, path.parent.parent
    record = recipe_run_record(path)
    choices = selected if isinstance(selected, list) else ([selected] if selected is not None else None)
    if choices is not None and (not choices or len(set(choices)) != len(choices)
            or any(c not in [r['case_id'] for r in record['cases']] for c in choices)):
        raise ValueError('unknown or duplicate recipe case')
    rows = [row for row in record['cases'] if choices is None or row['case_id'] in choices]
    if not rows:
        raise ValueError('unknown recipe case')
    return path, record, rows


def _policies(record, rows, scope, overrides_path):
    if scope not in ('initial', 'supplemental', 'development'):
        raise ValueError('unknown recipe evaluation scope')
    overrides = yaml.safe_load(Path(overrides_path).read_text()) if overrides_path else None
    manifest = _manifest(record)
    descriptions = {row['case_id']: _read(Path(row['path'])/'recipe.json') for row in rows}
    for description in descriptions.values():
        evaluation = description['evaluation']
        for name in ('conditions', 'supplemental_conditions'):
            _validate_scoring_conditions(evaluation[name], name, allow_empty=name == 'supplemental_conditions',
                                         allow_no_noise=name == 'conditions')
        for policy in description['observations']:
            _validate_scoring_conditions(policy['conditions'], 'observation.conditions')
        # Existing recipe science binds its scorer request into the checkpoint.
        # A supported changed condition request is the explicit YAML supplement.
        # Arbitrary overrides cannot rewrite that bound science.
        if overrides is not None:
            if not isinstance(overrides, dict) or set(overrides) != {'evaluation'}:
                raise ValueError('recipe overrides require only the frozen evaluation mapping')
            supplied = overrides['evaluation']
            if isinstance(supplied, dict) and 'conditions' in supplied:
                _validate_scoring_conditions(supplied['conditions'], 'override conditions')
            if supplied != evaluation:
                raise ValueError('recipe scoring overrides differ from frozen policy; use a declared supplemental scope or a new recipe')
        if scope == 'supplemental' and not evaluation['supplemental_conditions']:
            raise ValueError('recipe supplemental conditions were not requested in the frozen policy')
    return manifest, descriptions, overrides


def recipe_evaluation_target(root, selected, *, checkpoint=None, expected_step=None, scope='initial'):
    """Preflight an exact own recorded target; raises before model/output creation.

    Development supports only configured observation checkpoints, with an exact
    path and step. It never repairs or marks the training observation complete.
    """
    root, record, rows = _selection(root, selected)
    if len(rows) != 1:
        raise ValueError('checkpoint eligibility requires one named recipe case')
    manifest, descriptions, _ = _policies(record, rows, scope, None)
    row = rows[0]
    case = Path(row['path'])
    description = descriptions[row['case_id']]
    if scope == 'development':
        if checkpoint is None or type(expected_step) is not int:
            raise ValueError('development evaluation requires an exact checkpoint and expected step')
        wanted = str(Path(checkpoint).resolve())
        matches = []
        for inventory in (case/'attempts').glob('training-*/checkpoints.json'):
            matches.extend((item, inventory.parent) for item in _read(inventory)['checkpoints']
                           if item['path'] == wanted)
        if len(matches) != 1:
            raise ValueError('development requires one unique recorded checkpoint from this case')
        saved, attempt = matches[0]
        target = {**saved, 'training_attempt': str(attempt), 'engine_path': str(attempt/'engine'),
                  'policy': 'recipe-development'}
    else:
        if checkpoint is not None:
            raise ValueError('terminal evaluation selects its recorded target; explicit checkpoints require development scope')
        target = _read(case/'checkpoint.json')
        attempt = Path(target['training_attempt'])
        training = _read(case/'training.json')
        if (training != {'path': str(attempt), 'engine_path': target['engine_path']}
                or target['policy'] != 'recipe-terminal' or target['role'] != 'terminal'):
            raise ValueError('recorded recipe terminal engine/policy differs')
    if (attempt.parent != case/'attempts' or not attempt.name.startswith('training-')
            or Path(target['engine_path']) != attempt/'engine'
            or Path(target['path']).parent != attempt/'engine'):
        raise ValueError('recorded checkpoint attempt/engine containment differs')
    training_attempt = _read(attempt/'attempt.json')
    if (training_attempt['case'] != str(case) or training_attempt['stage'] != 'training'
            or training_attempt['status'] not in ('completed', 'failed', 'interrupted', 'partial')
            or (scope != 'development' and training_attempt['status'] != 'completed')):
        raise ValueError('recorded training attempt is ineligible')
    if (training_attempt['source'].get('revision') != manifest['source_commit']
            or training_attempt['source'].get('dirty') or training_attempt['runtime'] != runtime_identity()):
        raise ValueError('recorded training source/runtime differs')
    inventory = _read(attempt/'checkpoints.json')
    expected = {k: target[k] for k in ('path', 'sha256', 'size', 'step', 'role')}
    matches = [item for item in inventory['checkpoints'] if item['path'] == target['path']]
    if inventory.get('case_id') != row['case_id'] or len(matches) != 1 or matches[0] != expected:
        raise ValueError('recorded checkpoint role/step/hash/size is missing or ambiguous')
    reference = CheckpointRef(**{k: target[k] for k in ('path', 'sha256', 'size')})
    state, plan = engine._reopen_recipe(engine._open_bound_reference(reference))
    if (type(target['step']) is not int or target['step'] != state.payload['metadata']['completed_updates']
            or (expected_step is not None and target['step'] != expected_step)
            or plan.description != description):
        raise ValueError('recorded checkpoint step/effective recipe differs')
    verify_recipe_source(manifest, manifest['source_commit'])
    if source_state().get('dirty'):
        raise ValueError('recipe evaluation requires clean source')
    # Existing preparation/data consumer reconstructs the exact compatibility
    # identities. It loads only registered cached assets, with no download.
    _, config, acquired_plan, _, initial, metadata, _ = load_recipe_workflow(
        record['prepared']['path'], checksum=record['prepared']['sha256'],
        width=row['width'], ratio=row['ratio'])
    assert acquired_plan is not None and metadata is not None
    for key, value in {**metadata, 'plan_identity': acquired_plan.identity}.items():
        if state.payload['metadata'].get(key) != value:
            raise ValueError('recipe evaluation acquired identity differs: '+key)
    if config != bound_config(case):
        raise ValueError('recipe evaluation effective configuration differs')
    if scope == 'development':
        policy = next((p for p in description['observations']
                       if (p['step'], p['state']) == (target['step'], target['role'])), None)
        if policy is None:
            raise ValueError('development supports only configured observation checkpoint roles')
        template = _read(_read_reference(manifest['observation_template']))
        policy_request = {**policy, 'plan_identity': plan.identity, 'checkpoint_sha256': reference.sha256}
        request = engine._recipe_observation_request(state, policy_request, template)
    else:
        terminal, _ = engine.open_recipe_terminal(target['engine_path'])
        if terminal.reference != reference:
            raise ValueError('recorded terminal differs from own frozen checkpoint')
        template = _read(_read_reference(manifest['terminal_template']))
        request = engine.terminal_recipe_request(target['engine_path'], template, scope=scope)
    campaign = _read(attempt/'engine/campaign.json')
    owned = [item for item in campaign['checkpoints'] if item['path'] == reference.path]
    if (campaign['schema'] != 'recipe-training-v1' or campaign['plan_identity'] != plan.identity
            or campaign['status'] not in ('incomplete', 'training_frozen') or len(owned) != 1
            or {key: owned[0][key] for key in ('path', 'sha256', 'size', 'role')} !=
               {key: target[key] for key in ('path', 'sha256', 'size', 'role')}):
        raise ValueError('recipe campaign does not own the selected checkpoint')
    eligibility = {'schema': SCHEMA, 'case_id': row['case_id'], 'eligible': True,
        'final_eligible': scope == 'initial', 'scope': scope, 'checkpoint': target,
        'request_identity': canonical_digest(request), 'source': training_attempt['source'],
        'runtime': training_attempt['runtime'], 'prepared': record['prepared'],
        'observations_status': campaign.get('observations_status'),
        'observations_status_recorded': 'observations_status' in campaign,
        'observations': campaign['observations']}
    return eligibility, request, state, config, initial, manifest


def _counts(request, receipt=None):
    panels = [] if receipt is None else receipt['panels']
    requested = len(request['sites'])*len(request['conditions'])
    completed = sum(panel['status'] == 'complete' for panel in panels)
    failed = sum(panel['status'] == 'failed' for panel in panels)
    return {'requested': requested, 'completed': completed, 'failed': failed,
            'pending': requested-completed-failed,
            'requested_items': requested*request['expected_items'],
            'observed_items': sum(panel['completed_items'] for panel in panels),
            **({} if receipt is None else receipt.get('worker_counts', {}))}


def _case_evaluation(attempt, eligibility, request, state, config, initial, manifest, *,
                     timeout_seconds, device_uuid, parallel=None, root=None, attempt_id=None, checkpoint=None, expected_step=None):
    started = time.perf_counter()
    result = {'schema': SCHEMA, 'case_id': eligibility['case_id'], 'scope': eligibility['scope'],
              'status': 'incomplete', 'final_eligible': False, 'eligibility': eligibility,
              'request': request, 'panels': _counts(request), 'panel_receipt': None}
    try:
        write_json_atomic(attempt/'eligibility.json', eligibility)
        write_json_atomic(attempt/'request.json', request)
        write_json_atomic(attempt/'result.json', result)
        validate_reserved_attempt(attempt, 'evaluation')
        # Recheck the selected bytes after reservation and before model load.
        engine._reopen_recipe(state)
        if parallel is not None:
            from .panel_evaluation import evaluate_parallel_recipe
            receipt = evaluate_parallel_recipe(root=root, eligibility=eligibility, request=request, state=state,
                output=attempt/'panels', controls=parallel, attempt_id=attempt_id, checkpoint=checkpoint,
                expected_step=expected_step, timeout_seconds=timeout_seconds)
            result['status'] = 'complete' if receipt['status'] == 'complete' else 'incomplete'
            result['final_eligible'] = receipt['final_eligible']
            return attempt
        guard = _guard(config, synthetic=manifest['synthetic_cpu'],
                       timeout_seconds=timeout_seconds, device_uuid=device_uuid)
        guard()
        with isolated_rng():
            processor, model = initialize_recipe(config, initial, state.payload['metadata']['recipe_plan']['description']['locations'],
                                                evaluation=True, synthetic=manifest['synthetic_cpu'])
            guard()
            if eligibility['scope'] == 'development':
                engine.observe_recipe(model, processor, state=state, request=request, output=attempt/'panels', guard=guard)
            else:
                engine.run_recipe_evaluation(model, processor, request=request, output=attempt/'panels', guard=guard)
        result['status'] = 'complete'
        result['final_eligible'] = (eligibility['scope'] == 'initial'
                                    and engine.verified_observations(eligibility, state))
    except BaseException as error:
        result.update(status='interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed',
                      error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        receipt_path = attempt/'panels/result.json'
        try:
            if receipt_path.is_file():
                receipt = _read(receipt_path)
                engine._verify_recipe_result(receipt, state, observation=eligibility['scope'] == 'development')
                result.update(panels=_counts(request, receipt),
                              panel_receipt={'path': str(receipt_path), 'sha256': file_digest(receipt_path)})
        except Exception as error:
            result.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
            raise
        finally:
            result['elapsed_seconds'] = time.perf_counter()-started
            write_json_atomic(attempt/'result.json', result)
            finish_attempt(attempt, 'completed' if result['status'] == 'complete' else 'partial' if result['status'] == 'incomplete' else result['status'],
                           details={'result': str(attempt/'result.json'), 'panels': result['panels'],
                                    'elapsed_seconds': result['elapsed_seconds']})
    return attempt


def _planned_counts(description, scope, expected_step):
    if scope == 'development':
        policies = [p for p in description['observations'] if p['step'] == expected_step]
        if len(policies) != 1:
            # An unsupported endpoint has no resolved scientific denominator.
            return False, {key: 0 for key in ('requested', 'completed', 'failed', 'pending',
                                             'requested_items', 'observed_items')}
        policy = policies[0]
        sites, conditions = policy['sites'], policy['conditions']
    else:
        policy = description['evaluation']
        sites = description['locations']
        conditions = policy['conditions' if scope == 'initial' else 'supplemental_conditions']
    return True, _counts({'sites': sites, 'conditions': conditions, 'expected_items': policy['items']})


def evaluate_recipe_run(root, *, selected=None, attempt_id='0001', checkpoint=None,
                        expected_step=None, scope='initial', timeout_seconds=86400,
                        device_uuid=None, overrides_path=None, parallel=None):
    """Evaluate declared cases sequentially, retaining all missing/failed outcomes.

    Returns a fresh aggregate directory even for incomplete outcomes. Callers must
    inspect results.status and gate.final_eligible rather than infer success from
    the returned path. No training checkpoint/observation record is rewritten.
    """
    root, record, rows = _selection(root, selected)
    if parallel is not None:
        from .panel_workers import validate_parallel_controls
        controls = dict(parallel)
        controls.pop('parent_attempt', None)
        validate_parallel_controls(controls)
        if len(rows) != 1 or selected is None or overrides_path is not None or device_uuid is not None:
            raise ValueError('parallel evaluation requires one selected shared case and no overrides/device_uuid')
    _, descriptions, overrides = _policies(record, rows, scope, overrides_path)
    _name(attempt_id)
    if scope == 'development' and (len(rows) != 1 or checkpoint is None or type(expected_step) is not int):
        raise ValueError('development requires one explicit case, checkpoint and expected step')
    if checkpoint is not None and scope != 'development':
        raise ValueError('explicit recipe checkpoint requires development scope')
    if expected_step is not None and scope != 'development':
        raise ValueError('recipe expected step requires development scope')
    output = root/'evaluations'/f'evaluation-{attempt_id}'
    output.mkdir(parents=True, exist_ok=False)
    results = {'schema': SCHEMA, 'root': str(root), 'scope': scope, 'attempt_id': attempt_id,
               'recipe_identity': record['identity'], 'status': 'incomplete', 'cases': [],
               'source': source_state(), 'runtime': runtime_identity(), 'overrides': overrides}
    started = time.perf_counter()
    for row in rows:
        description = descriptions[row['case_id']]
        resolved, counts = _planned_counts(description, scope, expected_step)
        results['cases'].append({'case_id': row['case_id'], 'status': 'pending', 'eligible': False,
            'request_resolved': resolved, 'evaluation': None, 'panels': counts})
    write_json_atomic(output/'results.json', results)
    try:
        for row, summary in zip(rows, results['cases']):
            owned_attempt = None
            try:
                try:
                    eligibility, request, state, config, initial, manifest = recipe_evaluation_target(
                        root, row['case_id'], checkpoint=checkpoint, expected_step=expected_step, scope=scope)
                except (ValueError, FileNotFoundError) as error:
                    summary.update(status='missing' if isinstance(error, FileNotFoundError) else 'failed',
                                   error={'type': type(error).__name__, 'message': str(error)})
                    continue
                summary.update(eligible=True, eligibility=eligibility, panels=_counts(request))
                owned_attempt = reserve_attempt(Path(row['path']), 'evaluation', attempt_id=attempt_id,
                    request={'scope': scope, 'checkpoint': eligibility['checkpoint'],
                             'request_identity': canonical_digest(request)})
                _case_evaluation(owned_attempt, eligibility, request, state, config, initial,
                                 manifest, timeout_seconds=timeout_seconds, device_uuid=device_uuid, parallel=parallel,
                                 root=root, attempt_id=attempt_id, checkpoint=checkpoint, expected_step=expected_step)
                summary['status'] = _read(owned_attempt/'result.json')['status']
            except engine.ObservationScoringError as error:
                summary.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
            except BaseException as error:
                summary.update(status='interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed',
                               error={'type': type(error).__name__, 'message': str(error)})
                raise
            finally:
                if owned_attempt is not None and (owned_attempt/'result.json').is_file():
                    saved = _read(owned_attempt/'result.json')
                    summary.update(evaluation={'path': str(owned_attempt/'result.json'),
                                              'sha256': file_digest(owned_attempt/'result.json')},
                                   panels=saved['panels'], final_eligible=saved.get('final_eligible', False))
                write_json_atomic(output/'results.json', results)
        results['status'] = 'complete' if all(row['status'] == 'complete' for row in results['cases']) else 'incomplete'
    finally:
        results['elapsed_seconds'] = time.perf_counter()-started
        results['counts'] = {'cases': {'requested': len(rows),
            'eligible': sum(row['eligible'] for row in results['cases']),
            **{status: sum(row['status'] == status for row in results['cases'])
               for status in ('complete', 'failed', 'missing', 'pending', 'interrupted')}},
            'panels': {key: sum(row['panels'][key] for row in results['cases']) for key in
                       ('requested', 'completed', 'failed', 'pending', 'requested_items', 'observed_items')}}
        if parallel is not None:
            results['counts']['panels'].update({key: sum(row['panels'].get(key, 0) for row in results['cases'])
                for key in ('selected', 'started', 'missing', 'interrupted', 'running')})
        results['unresolved_requests'] = sum(not row['request_resolved'] for row in results['cases'])
        write_json_atomic(output/'results.json', results)
        write_json_atomic(output/'eligibility.json', {'schema': SCHEMA, 'scope': scope,
            'cases': [{key: row[key] for key in ('case_id', 'eligible', 'status')} |
                      ({'eligibility': row['eligibility']} if 'eligibility' in row else {'error': row.get('error')})
                      for row in results['cases']]})
        write_json_atomic(output/'gate.json', {'schema': SCHEMA, 'scope': scope,
            'final_eligible': scope == 'initial' and results['status'] == 'complete'
                and all(row.get('final_eligible') is True for row in results['cases']),
            'status': results['status'], 'counts': results['counts'],
            'results': {'path': str(output/'results.json'), 'sha256': file_digest(output/'results.json')},
            'observations': 'Retained separately; pending prior obligations are not completed by evaluation.'})
    return output

