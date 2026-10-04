"""Evaluation-owned legacy entry seam and original13 observation contract."""
import copy
import hashlib
import io

import torch

from .activation_replay import canonical_digest
from .encoder_stages_plan import ORIGINAL13_SITES, OBSERVATION_SITES, Original13Plan
from .sharing_qualification_guard import state_exact

from dataclasses import asdict
import json
from pathlib import Path
import time

from .activation_replay import file_digest
from .baseline_protocol import validate_task_receipt, write_manifest
from .encoder_pilot_plan import SITES
from .evaluation import evaluate_checkpoint, verify_observation_artifacts
from .experiment_state import CheckpointRef, open_state

from .original13_scoring import (  # noqa: F401
    _original13_runtime, _original13_scoring_policy, _score_original13_panel,
)

CONDITIONS = ('no_noise', -6, 0, 6, 18)


class ObservationScoringError(RuntimeError):
    """Recoverable scoring/output error; never integrity or restoration failure."""


def open_stages_terminal(training_run, metadata, plan):
    root = Path(training_run).resolve()
    campaign = json.loads((root / 'campaign.json').read_text())
    freeze_path = root / 'freeze.json'
    if (campaign['status'] != 'training_frozen' or campaign['completed_updates'] != plan.horizon
            or campaign['prepared_identity'] != metadata['prepared_identity']
            or campaign['freeze_reference'] != {'path': str(freeze_path), 'sha256': file_digest(freeze_path)}):
        raise ValueError('own training trajectory is not completely frozen')
    freeze = json.loads(freeze_path.read_text())
    if (freeze['status'] != 'complete' or freeze['arm'] != plan.arm or freeze['width'] != plan.width
            or freeze['final_panels'] != [[site, condition] for condition in CONDITIONS for site in SITES]
            or file_digest(root / 'metrics.jsonl') != freeze['metrics']['sha256']):
        raise ValueError('terminal study plan or scalar log differs')
    expected = {k: v for k, v in metadata.items() if k not in {'phase', 'lineage'}}
    if plan.arm != 'A':
        reset = campaign['phase_reset']
        parent_ref = CheckpointRef(**reset['parent'])
        if Path(parent_ref.path).resolve() != root / f'step_{plan.horizon // 2:06d}.pt':
            raise ValueError('phase reset parent path differs from own training run')
        parent = open_state(parent_ref, expected=metadata)
        if parent.payload['metadata']['completed_updates'] != plan.horizon // 2:
            raise ValueError('phase reset parent did not complete stage one')
        expected['parent_identity'] = parent_ref.sha256
    state = open_state(CheckpointRef(**freeze['checkpoint']), expected=expected)
    if (state.payload['metadata']['completed_updates'] != plan.horizon
            or state.payload['metadata']['sharing']['arm'] != plan.arm
            or state.payload['metadata']['sharing']['width'] != plan.width):
        raise ValueError('terminal checkpoint differs from own arm/width')
    return state, campaign['freeze_reference']


def run_stages_evaluation(model, processor, *, state, freeze_reference, template, output, guard):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    details = state.payload['metadata']
    if template['conditions'] != list(CONDITIONS):
        raise ValueError('terminal five-condition template required')
    result = {'schema': 'encoder-stages-evaluation-v1', 'status': 'incomplete',
              'arm': details['sharing']['arm'], 'width': details['sharing']['width'],
              'prepared_identity': details['prepared_identity'],
              'checkpoint': asdict(state.reference), 'freeze_reference': freeze_reference, 'completed_panels': [],
              'interpretation': 'previously inspected validation, six trained encoder sites, one seed',
              'vanilla_execution': 'reuse compatible external evidence after identity matching'}
    begun = time.monotonic()
    write_manifest(output / 'result.json', result)
    try:
        for condition in CONDITIONS:
            for site in SITES:
                guard()
                request = {**template, 'schema': 'codec-observation-v1', 'conditions': [condition],
                    'checkpoint_sha256': state.reference.sha256, 'target_site': details['evaluation_sites'][site]['site'],
                    'target_role': 'trained', 'learner_kind': 'shared_encoder_v2', 'step': details['completed_updates'],
                    'site_step': details['sharing']['completed_per_site'][site], 'parent': details['parent_identity'],
                    'comparison': details['comparison_controls'], 'panel': f'final:{site}:{condition}',
                    'source': details['source_identity'], 'config': details['config_identity'],
                    'data': details['data_identity'], 'protocol': details['protocol_identity']}
                request.update(study_arm=details['sharing']['arm'], study_width=details['sharing']['width'],
                               prepared_identity=details['prepared_identity'])
                request['settings'] = {**request['settings'], 'evidence_mode': 'compact-v1'}
                path = output / f'{site}_{condition}'
                receipt = evaluate_checkpoint(state, request, model=model, processor=processor, output=path)
                validate_task_receipt(receipt, request)
                verify_observation_artifacts(receipt, path)
                guard()
                result['completed_panels'].append({'site': site, 'condition': condition,
                    'checkpoint': state.reference.sha256, 'completed_items': receipt['conditions'][0]['completed'],
                    'metrics': receipt['conditions'][0]['metrics'],
                    'observation': {'path': str(path / 'observation.json'), 'sha256': file_digest(path / 'observation.json')}})
                result['elapsed_seconds'] = time.monotonic() - begun
                write_manifest(output / 'result.json', result)
                print(json.dumps({'phase': 'evaluation', 'completed_codec_panels': len(result['completed_panels']),
                                  'elapsed_seconds': result['elapsed_seconds']}), flush=True)
        result.update(status='complete', elapsed_seconds=time.monotonic() - begun)
        write_manifest(output / 'result.json', result)
        return result
    except Exception as error:
        result.update(error={'type': type(error).__name__, 'message': str(error)}, elapsed_seconds=time.monotonic() - begun)
        write_manifest(output / 'result.json', result)
        raise


# Original13 uses separate request/receipt schemas. Legacy thirty-panel calls above
# retain their original checkpoint contract and accounting.
INITIAL_CONDITIONS = ('no_noise', -6)
SUPPLEMENTAL_CONDITIONS = (0, 6, 18)


def _reopen_original13(state):
    reopened = open_state(state.reference, expected=state.payload['metadata'])
    if reopened.payload['kind'] != 'original13_v1' or not state_exact(reopened.payload, state.payload):
        raise ValueError('original13 validated checkpoint payload differs')
    return reopened, Original13Plan.from_resolved(reopened.payload['metadata']['original13_plan'])


def _open_bound_reference(reference):
    # Discover identities only from the explicitly selected trusted local bytes.
    # Central open_state still validates the full envelope and those identities.
    data = Path(reference.path).read_bytes()
    if len(data) != reference.size or hashlib.sha256(data).hexdigest() != reference.sha256:
        raise ValueError('checkpoint bytes incomplete or corrupt')
    payload = torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)
    return open_state(reference, expected=payload['metadata'])


def open_original13_terminal(training_run):
    """Open the sole terminal in this frozen segment; never search best/latest."""
    root = Path(training_run).resolve()
    campaign = json.loads((root / 'campaign.json').read_text())
    freeze_path = root / 'freeze.json'
    reference = {'path': str(freeze_path), 'sha256': file_digest(freeze_path)}
    if (campaign['schema'] != 'original13-training-v1' or campaign['status'] != 'training_frozen'
            or campaign['freeze_reference'] != reference):
        raise ValueError('original13 own frozen training run required')
    freeze = json.loads(freeze_path.read_text())
    state = _open_bound_reference(CheckpointRef(**freeze['checkpoint']))
    state, plan = _reopen_original13(state)
    terminal = [c for c in campaign['checkpoints'] if c['role'] == 'terminal']
    if (freeze['schema'] != 'original13-freeze-v1' or freeze['status'] != 'training_frozen'
            or campaign['plan_identity'] != plan.identity or freeze['plan_identity'] != plan.identity
            or campaign['completed_updates'] != plan.horizon
            or state.payload['metadata']['completed_updates'] != plan.horizon
            or state.payload['metadata']['phase_start_step'] != plan.reconstruction_stop
            or len(terminal) != 1
            or {k: terminal[0][k] for k in ('path', 'sha256', 'size')} != asdict(state.reference)
            or Path(state.reference.path).resolve().parent != root
            or Path(freeze['metrics']['path']).resolve() != root / 'metrics.jsonl'
            or file_digest(root / 'metrics.jsonl') != freeze['metrics']['sha256']):
        raise ValueError('original13 terminal, plan, endpoint or scalar bytes differ')
    return state, reference


def _validate_template(template, count):
    required = ('task', 'input_ids', 'source_family_ids', 'settings', 'noise',
                'data_settings', 'scorer', 'precision', 'backend', 'reference_corpus',
                'prompt_policy', 'layout')
    if any(key not in template for key in required):
        raise ValueError('original13 scoring template is incomplete')
    settings = template['settings']
    if (template['task'] != 'hellaswag' or template['expected_items'] != count
            or template['input_ids'] != list(range(count))
            or len(template['source_family_ids']) != count
            or any(not isinstance(f, str) or not f for f in template['source_family_ids'])
            or settings['num_samples'] != count or settings['num_fewshot'] != 5
            or settings['batch_size'] != 16 or settings['scoring_policy'] != 'fp32-v1'
            or settings.get('numerical_policy', 'native') != 'native'
            or settings.get('vanilla') is not False
            or template['scorer'] != 'lm-eval-hellaswag-fp32-v1-acc-and-acc_norm'
            or template['reference_corpus']['split'] != 'validation'
            or type(template['noise']['seed']) is not int or not template['noise']['namespace']):
        raise ValueError('original13 fixed full/five-shot/candidate16/FP32 recipe differs')


def terminal_original13_request(training_run, template, *, scope='initial'):
    """Bind an explicit scope; supporting supplements does not request them."""
    if scope not in ('initial', 'supplemental'):
        raise ValueError('unknown original13 terminal request scope')
    state, freeze = open_original13_terminal(training_run)
    _, plan = _reopen_original13(state)
    _validate_template(template, 10042)
    details = state.payload['metadata']
    return {**copy.deepcopy(template), 'schema': 'original13-terminal-request-v1',
            'scope': scope, 'training_run': str(Path(training_run).resolve()),
            'freeze_reference': freeze, 'checkpoint': asdict(state.reference),
            'checkpoint_sha256': state.reference.sha256, 'kind': 'original13_v1',
            'plan_identity': plan.identity, 'input_identity': plan.input_identity,
            'source': details['source_identity'], 'config': details['config_identity'],
            'data': details['data_identity'], 'protocol': details['protocol_identity'],
            'prepared_identity': details['prepared_identity'], 'step': plan.horizon,
            'width': plan.width, 'ratio': plan.ratio, 'sites': list(ORIGINAL13_SITES),
            'conditions': list(INITIAL_CONDITIONS if scope == 'initial' else SUPPLEMENTAL_CONDITIONS)}


def _validate_request(state, request, *, observation):
    state, plan = _reopen_original13(state)
    details = state.payload['metadata']
    if (request['plan_identity'] != plan.identity
            or request['checkpoint_sha256'] != state.reference.sha256
            or request['step'] != details['completed_updates']):
        raise ValueError('original13 request checkpoint/plan/step differs')
    if observation:
        _validate_template(request, 256)
        panel = request['panel']
        if (request['schema'] != 'original13-observation-request-v1'
                or canonical_digest(panel) != plan.observation_identity
                or (request['step'], request['state_role']) not in plan.observations
                or request['sites'] != list(OBSERVATION_SITES)
                or request['conditions'] != list(INITIAL_CONDITIONS)
                or request['input_ids'] != panel['input_ids']
                or request['source_family_ids'] != panel['source_family_ids']
                or request['noise'] != panel['noise']
                or (request['state_role'] == 'pre_reset' and details['phase_start_step'] != 0)
                or (request['state_role'] == 'functional_observation'
                    and details['phase_start_step'] != plan.reconstruction_stop)):
            raise ValueError('original13 fixed observation panel/endpoint differs')
    else:
        rebound = terminal_original13_request(request['training_run'], request, scope=request['scope'])
        if request != rebound or request['kind'] != 'original13_v1':
            raise ValueError('original13 terminal request differs from own frozen run')
    return state, plan


def _run_original13_panels(model, processor, *, state, request, output, guard, observation):
    recipe = state.payload['kind'] == 'recipe_v1'
    validator = _validate_recipe_request if recipe else _validate_request
    state, plan = validator(state, request, observation=observation)
    output = Path(output).resolve()
    # Destination and initial receipt failures are fatal: no durable failure record
    # exists yet. Never overwrite or resume an existing panel destination.
    output.mkdir(parents=True, exist_ok=False)
    panels = [{'site': site, 'condition': condition, 'status': 'pending',
               'requested_items': request['expected_items'], 'completed_items': 0}
              for condition in request['conditions'] for site in request['sites']]
    result = {'schema': 'recipe-evaluation-receipt-v1' if recipe else 'original13-evaluation-receipt-v1', 'request': copy.deepcopy(request),
              'request_identity': canonical_digest(request), 'status': 'incomplete',
              'checkpoint': asdict(state.reference), 'plan_identity': plan.identity,
              'panels': panels, 'elapsed_seconds': 0.0}
    write_manifest(output / 'result.json', result)
    begun = time.monotonic()
    scoring_error = None
    row = None
    try:
        with _original13_runtime(model, state):
            for row in panels:
                guard()  # Resource/stop guard is fatal, never a scoring failure.
                path = output / f"{row['site']}_{row['condition']}"
                path.mkdir(exist_ok=False)
                panel_request = {**request, 'schema': 'recipe-panel-request-v1' if recipe else 'original13-panel-request-v1',
                                 'target_site': row['site'], 'conditions': [row['condition']]}
                write_manifest(path / 'request.json', panel_request)
                started = time.monotonic()
                try:
                    scored = _score_original13_panel(model, processor, state=state, request=request,
                        site_id=row['site'], condition=row['condition'], output=path)
                except (ObservationScoringError, OSError) as error:
                    row.update(status='failed', error={'type': type(error).__name__, 'message': str(error)},
                               elapsed_seconds=time.monotonic() - started)
                    scoring_error = error
                    break
                if (scored['completed_items'] != request['expected_items']
                        or scored['denominator'] != request['expected_items']):
                    raise ValueError('original13 panel denominator differs')
                row.update(scored, status='complete', elapsed_seconds=time.monotonic() - started,
                           panel_identity=canonical_digest(panel_request))
                row['output_hashes'] = {p.name: file_digest(p) for p in sorted(path.iterdir()) if p.is_file()}
                write_manifest(path / 'receipt.json', row)
                row['receipt'] = {'path': str(path / 'receipt.json'), 'sha256': file_digest(path / 'receipt.json')}
                write_manifest(output / 'result.json', result)
                guard()
    except Exception as error:
        if row is not None and row['status'] == 'pending':
            row.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        result.update(error={'type': type(error).__name__, 'message': str(error)},
                      elapsed_seconds=time.monotonic() - begun)
        write_manifest(output / 'result.json', result)
        raise
    result.update(status='failed' if scoring_error else 'complete', elapsed_seconds=time.monotonic() - begun,
                  requested_panels=len(panels), completed_panels=sum(p['status'] == 'complete' for p in panels),
                  failed_panels=sum(p['status'] == 'failed' for p in panels),
                  pending_panels=sum(p['status'] == 'pending' for p in panels))
    write_manifest(output / 'result.json', result)  # Save failure is fatal.
    if scoring_error is not None:
        raise ObservationScoringError(str(scoring_error)) from scoring_error
    return result


def observe_original13(model, processor, *, state, request, output, guard):
    """Eight fixed panels. Only typed isolated scoring/output failures recover.

    Integrity, identity, nonfinite, route/restoration, guards and receipt-save
    failures propagate fatally. No failed or elapsed observation is retried here.
    The training runner verifies optimizer/counters/exposure in addition to this
    consumer's weight/gradient/RNG/runtime restoration.
    """
    return _run_original13_panels(model, processor, state=state, request=request,
                                 output=output, guard=guard, observation=True)


def run_original13_evaluation(model, processor, *, request, output, guard):
    state, _ = open_original13_terminal(request['training_run'])
    return _run_original13_panels(model, processor, state=state, request=request,
                                 output=output, guard=guard, observation=False)


def original13_training_rows(training_run):
    """Preserve raw/weighted stream measures and denominators; never smooth/select.

    A continued segment exposes only its own rows and the prior-segment pointers.
    The verified terminal exposure ledger remains the cumulative authority.
    """
    from .experiment_records import read_events
    root = Path(training_run).resolve()
    state, _ = open_original13_terminal(root)
    campaign = json.loads((root / 'campaign.json').read_text())
    rows = []
    for event in read_events(root / 'metrics.jsonl'):
        if event['event_type'] != 'update':
            continue
        payload = event['payload']
        if payload['nonfinite'] or payload['skipped']:
            raise ValueError('original13 report contains unsuccessful scalar update')
        rows.append({'global_update': payload['completed_step'], 'phase': event['phase_id'],
                     'site': payload['site_id'], 'objective': copy.deepcopy(payload['objective']),
                     'exposure': copy.deepcopy(payload['exposure']), 'lr_used': payload['lr_used'],
                     'cost': {'timing': copy.deepcopy(payload.get('timing', {})),
                              'memory': copy.deepcopy(payload.get('memory', {}))}})
    count = campaign['segment_updates']
    if [r['global_update'] for r in rows] != list(range(campaign['completed_updates']-count+1,
                                                       campaign['completed_updates']+1)):
        raise ValueError('original13 segment scalar cadence differs')
    return {'rows': rows, 'segment_updates': count,
            'segment_ancestry': state.payload['metadata']['segment_ancestry'],
            'cumulative_exposure': state.payload['metadata']['exposure'],
            'metrics_sha256': file_digest(root / 'metrics.jsonl')}


def original13_report(training_run, *, initial=None, supplemental=None, observations=(),
                      initial_request=None, supplemental_request=None):
    """Offline descriptive accounting from requests and durable receipts only."""
    state, _ = open_original13_terminal(training_run)
    _, plan = _reopen_original13(state)
    campaign = json.loads((Path(training_run) / 'campaign.json').read_text())
    report = {'schema': 'original13-report-v1', 'width': plan.width, 'ratio': plan.ratio,
              'checkpoint': asdict(state.reference), 'training': campaign['status'],
              'observations': copy.deepcopy(campaign['observations']), 'terminal': {},
              'collection': 'complete', 'results': [],
              'training_scalars': original13_training_rows(training_run),
              'limits': 'Descriptive single-seed development evidence; common seeds do not prove equal realized AWGN. '
                        'Old six-encoder runs are contextual. No superiority/convergence claim.'}
    for scope, root, requested in (('initial', initial, initial_request),
                                   ('supplemental', supplemental, supplemental_request)):
        if requested is not None:
            _validate_request(state, requested, observation=False)
            if requested['scope'] != scope:
                raise ValueError('original13 report explicit request scope differs')
        if root is None and requested is not None:
            count = len(requested['sites']) * len(requested['conditions'])
            report['terminal'][scope] = {'status': 'pending', 'requested': count, 'completed': 0,
                                        'failed': 0, 'pending': count}
            continue
        if root is None:
            report['terminal'][scope] = {'status': 'not_requested', 'requested': 0, 'completed': 0,
                                        'failed': 0, 'pending': 0}
            continue
        receipt = json.loads((Path(root) / 'result.json').read_text())
        _verify_original13_result(receipt, state, observation=False)
        if (receipt['request']['scope'] != scope
                or (requested is not None and requested != receipt['request'])):
            raise ValueError('original13 report request scope differs')
        report['terminal'][scope] = _panel_counts(receipt)
        report['results'].extend(copy.deepcopy(receipt['panels']))
    endpoints = set()
    for root in observations:
        receipt = json.loads((Path(root) / 'result.json').read_text())
        observed_state = _open_bound_reference(CheckpointRef(**receipt['checkpoint']))
        _verify_original13_result(receipt, observed_state, observation=True)
        endpoint = (receipt['request']['step'], receipt['request']['state_role'])
        if endpoint in endpoints:
            raise ValueError('original13 report duplicate observation endpoint')
        endpoints.add(endpoint)
        if (receipt['plan_identity'] != plan.identity
                or not _own_observation_checkpoint(state, campaign, receipt['checkpoint'], receipt['request'])):
            raise ValueError('original13 report observation trajectory differs')
        report.setdefault('observation_receipts', []).append(_panel_counts(receipt))
    report['observations_status'] = campaign['observations_status']
    return report


def _own_observation_checkpoint(terminal, campaign, reference, request):
    """Endpoint must be saved by this segment or an explicitly carried ancestor.

    Continuation keeps an earlier failed/pending observation pending; a supplied
    receipt is descriptive evidence, not an automatic training-status rewrite.
    """
    details = terminal.payload['metadata']
    campaigns = [(campaign, details['completed_updates'])]
    for parent_reference in details['segment_ancestry']:
        parent = _open_bound_reference(CheckpointRef(**parent_reference))
        fields = ('source_identity', 'config_identity', 'prepared_identity', 'plan_identity',
                  'input_identity', 'initialization_identity', 'stream_identity')
        if any(parent.payload['metadata'][key] != details[key] for key in fields):
            raise ValueError('original13 report ancestor identity differs')
        parent_campaign = json.loads((Path(parent_reference['path']).parent / 'campaign.json').read_text())
        if not any({k: saved[k] for k in ('path', 'sha256', 'size')} == parent_reference
                   for saved in parent_campaign['checkpoints']):
            raise ValueError('original13 ancestor campaign does not own carried checkpoint')
        campaigns.append((parent_campaign, parent.payload['metadata']['completed_updates']))
    for owner, inherited_through in campaigns:
        if request['step'] > inherited_through:
            continue
        matches = [saved for saved in owner['checkpoints'] if saved['role'] == request['state_role']
                   and {k: saved[k] for k in ('path', 'sha256', 'size')} == reference]
        observation = next((row for row in owner['observations']
                            if (row['step'], row['state_role']) == (request['step'], request['state_role'])), None)
        if matches and observation and observation.get('request_identity') == canonical_digest(request):
            return True
    return False


def _panel_counts(receipt):
    return {'status': receipt['status'], 'requested': len(receipt['panels']),
            'completed': sum(p['status'] == 'complete' for p in receipt['panels']),
            'failed': sum(p['status'] == 'failed' for p in receipt['panels']),
            'pending': sum(p['status'] == 'pending' for p in receipt['panels']),
            'requested_items': sum(p['requested_items'] for p in receipt['panels']),
            'observed_items': sum(p['completed_items'] for p in receipt['panels']),
            'elapsed_seconds': receipt['elapsed_seconds']}


def _verify_original13_result(receipt, state, *, observation):
    _validate_request(state, receipt['request'], observation=observation)
    if receipt['schema'] != 'original13-evaluation-receipt-v1':
        raise ValueError('original13 result schema differs')
    _verify_panel_result(receipt, state)


def _verify_panel_result(receipt, state):
    if (receipt['request_identity'] != canonical_digest(receipt['request'])
            or receipt['checkpoint'] != asdict(state.reference)
            or receipt['plan_identity'] != state.payload['metadata']['plan_identity']):
        raise ValueError('original13 result identity differs')
    if receipt['status'] not in ('complete', 'failed', 'incomplete'):
        raise ValueError('unsupported evaluation receipt status')
    expected = [(site, condition) for condition in receipt['request']['conditions']
                for site in receipt['request']['sites']]
    if [(p['site'], p['condition']) for p in receipt['panels']] != expected:
        raise ValueError('original13 result panels missing/duplicated/reordered')
    for panel in receipt['panels']:
        if panel['status'] not in ('complete', 'pending', 'failed'):
            raise ValueError('original13 unsupported panel status')
        if (type(panel['requested_items']) is not int
                or panel['requested_items'] != receipt['request']['expected_items']
                or type(panel['completed_items']) is not int
                or not 0 <= panel['completed_items'] <= panel['requested_items']
                or (panel['status'] != 'complete' and panel['completed_items'] != 0)):
            raise ValueError('evaluation panel requested/completed denominator differs')
        if panel['status'] == 'complete':
            panel_request = {**receipt['request'], 'schema': ('recipe-panel-request-v1'
                if receipt['schema'] == 'recipe-evaluation-receipt-v1' else 'original13-panel-request-v1'),
                             'target_site': panel['site'], 'conditions': [panel['condition']]}
            if (panel['panel_identity'] != canonical_digest(panel_request)
                    or panel['completed_items'] != panel['requested_items']
                    or panel['denominator'] != panel['requested_items']):
                raise ValueError('original13 panel identity/completed denominator differs')
            path = Path(panel['receipt']['path'])
            if file_digest(path) != panel['receipt']['sha256']:
                raise ValueError('original13 panel receipt bytes differ')
            if json.loads((path.parent / 'request.json').read_text()) != panel_request:
                raise ValueError('original13 panel request differs')
            saved = json.loads(path.read_text())
            if saved != {k: v for k, v in panel.items() if k != 'receipt'}:
                raise ValueError('original13 panel summary differs from receipt')
            for name, checksum in saved['output_hashes'].items():
                if Path(name).name != name or file_digest(path.parent / name) != checksum:
                    raise ValueError('original13 panel artifact bytes differ')
    if receipt['status'] == 'failed' and not any(p['status'] == 'failed' for p in receipt['panels']):
        raise ValueError('failed evaluation receipt requires a failed panel')
    if (receipt['status'] == 'complete') != all(p['status'] == 'complete' for p in receipt['panels']):
        raise ValueError('original13 completion differs from actual panel receipts')


def _reopen_recipe(state):
    from .original13_state import _recipe_plan
    reopened = open_state(state.reference, expected=state.payload['metadata'])
    if reopened.payload['kind'] != 'recipe_v1' or not state_exact(reopened.payload, state.payload):
        raise ValueError('recipe validated checkpoint payload differs')
    return reopened, _recipe_plan(reopened.payload)


def open_recipe_terminal(training_run):
    """Select the one durable terminal; preserve the original state file."""
    root = Path(training_run).resolve()
    campaign = json.loads((root/'campaign.json').read_text())
    freeze_path = root/'freeze.json'
    reference = {'path': str(freeze_path), 'sha256': file_digest(freeze_path)}
    if (campaign['schema'] != 'recipe-training-v1' or campaign['status'] != 'training_frozen'
            or campaign['freeze_reference'] != reference):
        raise ValueError('own frozen recipe training segment required')
    freeze = json.loads(freeze_path.read_text())
    state, plan = _reopen_recipe(_open_bound_reference(CheckpointRef(**freeze['checkpoint'])))
    terminal = [c for c in campaign['checkpoints'] if c['role'] == 'terminal']
    if (freeze['schema'] != 'recipe-freeze-v1' or freeze['status'] != 'training_frozen'
            or campaign['plan_identity'] != plan.identity or freeze['plan_identity'] != plan.identity
            or campaign['completed_updates'] != plan.horizon
            or state.payload['metadata']['completed_updates'] != plan.horizon
            or state.payload['metadata']['phase_start_step'] != plan.stages[1].start
            or len(terminal) != 1
            or {k: terminal[0][k] for k in ('path', 'sha256', 'size')} != asdict(state.reference)
            or Path(state.reference.path).resolve().parent != root
            or Path(freeze['metrics']['path']).resolve() != root/'metrics.jsonl'
            or file_digest(root/'metrics.jsonl') != freeze['metrics']['sha256']):
        raise ValueError('recipe terminal, plan, endpoint or scalar bytes differ')
    return state, reference


def terminal_recipe_request(training_run, template, *, scope='initial'):
    from .recipe_observation_policy import scoring_template
    if scope not in ('initial', 'supplemental'):
        raise ValueError('unknown recipe terminal scope')
    state, freeze = open_recipe_terminal(training_run)
    _, plan = _reopen_recipe(state)
    policy = plan.description['evaluation']
    conditions = policy['conditions' if scope == 'initial' else 'supplemental_conditions']
    if not conditions:
        raise ValueError('empty terminal request is not requested')
    value = scoring_template(template, policy, policy)
    details = state.payload['metadata']
    if ('terminal_template_identity' in details
            and canonical_digest(value) != details['terminal_template_identity']):
        raise ValueError('recipe acquired terminal template differs')
    return {**value, 'schema': 'recipe-terminal-request-v1', 'scope': scope,
        'training_run': str(Path(training_run).resolve()), 'freeze_reference': freeze,
        'checkpoint': asdict(state.reference), 'checkpoint_sha256': state.reference.sha256,
        'kind': 'recipe_v1', 'plan_identity': plan.identity, 'step': plan.horizon,
        'sites': list(plan.trained_sites), 'conditions': list(conditions),
        'source': details['source_identity'], 'config': details['config_identity'],
        'data': details['data_identity'], 'prepared_identity': details['prepared_identity']}


def _recipe_observation_request(state, policy_request, template):
    from .recipe_observation_policy import scoring_template, recipe_observation_identity
    state, plan = _reopen_recipe(state)
    policy = next((r for r in plan.description['observations']
                   if (r['step'], r['state']) == (policy_request['step'], policy_request['state'])), None)
    if policy is None or policy_request != {**policy, 'plan_identity': plan.identity,
                                           'checkpoint_sha256': state.reference.sha256}:
        raise ValueError('recipe observation policy/checkpoint differs')
    expected_phase = 0 if policy['state'] == 'pre_reset' else plan.stages[1].start
    if state.payload['metadata']['phase_start_step'] != expected_phase:
        raise ValueError('recipe observation reset phase differs')
    value = scoring_template(template, policy, plan.description['evaluation'])
    if state.payload['metadata']['observation_identity'] != recipe_observation_identity(plan.description, template):
        raise ValueError('recipe fixed observation template identity differs')
    return {**value, **copy.deepcopy(policy_request), 'schema': 'recipe-observation-request-v1',
        'state_role': policy['state'], 'policy_request': copy.deepcopy(policy_request)}


def _validate_recipe_request(state, request, *, observation):
    state, plan = _reopen_recipe(state)
    if (request['plan_identity'] != plan.identity or request['checkpoint_sha256'] != state.reference.sha256
            or request['step'] != state.payload['metadata']['completed_updates']):
        raise ValueError('recipe request checkpoint/plan/step differs')
    expected = (_recipe_observation_request(state, request['policy_request'], request) if observation else
                terminal_recipe_request(request['training_run'], request, scope=request['scope']))
    if request != expected:
        raise ValueError('recipe request differs from resolved policy')
    return state, plan


def make_recipe_observer(template):
    """Bind the acquired fixed panel once; use as run_recipe_training.observe."""
    bound = copy.deepcopy(template)
    def observe(model, processor, *, state, request, output, guard):
        full = _recipe_observation_request(state, request, bound)
        receipt = observe_recipe(model, processor, state=state, request=full, output=output, guard=guard)
        # Runner binds the policy request; durable result binds the full scorer request.
        return {**receipt, 'request_identity': canonical_digest(request)}
    return observe


def observe_recipe(model, processor, *, state, request, output, guard):
    return _run_original13_panels(model, processor, state=state, request=request,
                                 output=output, guard=guard, observation=True)


def run_recipe_evaluation(model, processor, *, request, output, guard):
    state, _ = open_recipe_terminal(request['training_run'])
    return _run_original13_panels(model, processor, state=state, request=request,
                                 output=output, guard=guard, observation=False)


def recipe_training_rows(training_run):
    """Read the recipe event schema without imposing retained loss coefficients."""
    root = Path(training_run).resolve()
    state, _ = open_recipe_terminal(root)
    _, plan = _reopen_recipe(state)
    campaign = json.loads((root/'campaign.json').read_text())
    rows = []
    for line in (root/'metrics.jsonl').read_text().splitlines():
        event = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError('nonfinite recipe scalar')))
        p = event['payload']
        if (event['schema'] != 'recipe-training-event-v1' or event['event_type'] != 'update'
                or p['nonfinite'] or p['skipped']):
            raise ValueError('recipe report contains unsupported/unsuccessful scalar update')
        scheduled = plan.update_at(p['completed_step']-1)
        expected_identity = {'source': state.payload['metadata']['source_identity'],
            'config': state.payload['metadata']['config_identity'],
            'data': state.payload['metadata']['data_identity']}
        if (any(event['identity'][k] != v for k, v in expected_identity.items())
                or p['recipe_identity'] != plan.identity or p['site_id'] != scheduled.site
                or p['site_step'] != scheduled.site_local_index+1
                or p['objective']['resolved_weights'] != plan.objective_at(p['completed_step']-1)
                or p['lr_used'] != [plan.description['config']['training']['lr']*scheduled.lr_factor]):
            raise ValueError('recipe scalar identity/order/objective/LR differs')
        rows.append({'global_update': p['completed_step'], 'phase': event['phase_id'],
            'site': p['site_id'], 'objective': copy.deepcopy(p['objective']),
            'exposure': copy.deepcopy(p['exposure']), 'lr_used': p['lr_used'],
            'cost': {'timing': copy.deepcopy(p.get('timing', {})), 'memory': copy.deepcopy(p.get('memory', {}))}})
    count = campaign['segment_updates']
    if [r['global_update'] for r in rows] != list(range(campaign['completed_updates']-count+1,
                                                       campaign['completed_updates']+1)):
        raise ValueError('recipe segment scalar cadence differs')
    return {'rows': rows, 'segment_updates': count,
        'segment_ancestry': state.payload['metadata']['segment_ancestry'],
        'cumulative_exposure': state.payload['metadata']['exposure'],
        'metrics_sha256': file_digest(root/'metrics.jsonl')}


def _own_recipe_observation(terminal, campaign, receipt):
    details = terminal.payload['metadata']
    owners = [(campaign, details['completed_updates'])]
    for reference in details['segment_ancestry']:
        parent, _ = _reopen_recipe(_open_bound_reference(CheckpointRef(**reference)))
        keys = ('source_identity', 'config_identity', 'data_identity', 'prepared_identity',
                'plan_identity', 'initialization_identity', 'stream_identity', 'observation_identity')
        if any(parent.payload['metadata'][k] != details[k] for k in keys):
            raise ValueError('recipe observation ancestor identity differs')
        owner = json.loads((Path(reference['path']).parent/'campaign.json').read_text())
        if not any({k: c[k] for k in ('path', 'sha256', 'size')} == reference for c in owner['checkpoints']):
            raise ValueError('recipe ancestor does not own carried checkpoint')
        owners.append((owner, parent.payload['metadata']['completed_updates']))
    request = receipt['request']['policy_request']
    for owner, through in owners:
        if request['step'] > through:
            continue
        saved = any(c['role'] == request['state'] and
                    {k: c[k] for k in ('path', 'sha256', 'size')} == receipt['checkpoint']
                    for c in owner['checkpoints'])
        obligation = next((r for r in owner['observations']
                           if (r['step'], r['state']) == (request['step'], request['state'])), None)
        if saved and obligation and obligation.get('request_identity') == canonical_digest(request):
            return True
    return False


def recipe_report(training_run, *, initial=None, supplemental=None, observations=(),
                  initial_request=None, supplemental_request=None):
    """Explicit request accounting; never retry, select, or rewrite obligations."""
    state, _ = open_recipe_terminal(training_run)
    _, plan = _reopen_recipe(state)
    campaign = json.loads((Path(training_run)/'campaign.json').read_text())
    report = {'schema': 'recipe-report-v1', 'checkpoint': asdict(state.reference),
        'training': campaign['status'], 'terminal': {}, 'observations': copy.deepcopy(campaign['observations']),
        'observations_status': campaign['observations_status'], 'results': [], 'observation_receipts': [],
        'training_scalars': recipe_training_rows(training_run)}
    for scope, root, requested in (('initial', initial, initial_request),
                                   ('supplemental', supplemental, supplemental_request)):
        if requested is not None:
            _validate_recipe_request(state, requested, observation=False)
            if requested['scope'] != scope:
                raise ValueError('recipe report explicit scope differs')
        count = 0 if requested is None else len(requested['sites'])*len(requested['conditions'])
        if root is None:
            report['terminal'][scope] = {'status': 'not_requested' if requested is None else 'requested',
                'requested': count, 'completed': 0, 'failed': 0, 'pending': count}
            continue
        receipt = json.loads((Path(root)/'result.json').read_text())
        _verify_recipe_result(receipt, state, observation=False)
        if receipt['request']['scope'] != scope or (requested is not None and requested != receipt['request']):
            raise ValueError('recipe report scope/request differs')
        report['terminal'][scope] = _panel_counts(receipt)
        report['results'].extend(copy.deepcopy(receipt['panels']))
    endpoints = set()
    for root in observations:
        receipt = json.loads((Path(root)/'result.json').read_text())
        observed = _open_bound_reference(CheckpointRef(**receipt['checkpoint']))
        _verify_recipe_result(receipt, observed, observation=True)
        endpoint = (receipt['request']['step'], receipt['request']['state_role'])
        if endpoint in endpoints:
            raise ValueError('recipe report duplicate observation endpoint')
        endpoints.add(endpoint)
        if receipt['plan_identity'] != plan.identity or not _own_recipe_observation(state, campaign, receipt):
            raise ValueError('recipe report observation trajectory differs')
        report['observation_receipts'].append(_panel_counts(receipt))
    return report


def _verify_recipe_result(receipt, state, *, observation):
    _validate_recipe_request(state, receipt['request'], observation=observation)
    if receipt['schema'] != 'recipe-evaluation-receipt-v1':
        raise ValueError('recipe result schema differs')
    # Artifact/denominator validation is identical to the retained receipt format.
    _verify_panel_result(receipt, state)
