"""Recorded shared terminal requests, panel validation and reports."""

from __future__ import annotations
import copy
import math
import os
from . import panel_workers as core
from .activation_replay import canonical_digest
from .qualification import state_exact
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from .activation_replay import file_digest
from .experiment_state import CheckpointRef, open_state
from .checkpoint_readers import _open_bound_reference
from .checkpoint_readers import _verify_panel_result
from .checkpoint_readers import _panel_counts
import time
from .experiment_records import write_manifest, validate_recipe_update
from .checkpoint_readers import _validate_request as _validate_request
from contextlib import contextmanager
from .experiment_state import capture_rng, restore_rng
from .asset_identity import state_dict_identity
import torch
from contextlib import nullcontext
from jscc.experiment_records import write_json_atomic



# Transferred from encoder_stages_evaluation; serialized meanings remain unchanged.
def _reopen_recipe(state):
    from .shared_state import _recipe_plan
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
            or state.payload['metadata']['phase_start_step'] != plan.stages[-1].start
            or len(terminal) != 1
            or {k: terminal[0][k] for k in ('path', 'sha256', 'size')} != asdict(state.reference)
            or Path(state.reference.path).resolve().parent != root
            or Path(freeze['metrics']['path']).resolve() != root/'metrics.jsonl'
            or file_digest(root/'metrics.jsonl') != freeze['metrics']['sha256']):
        raise ValueError('recipe terminal, plan, endpoint or scalar bytes differ')
    return state, reference


def terminal_recipe_request(training_run, template, *, scope='initial'):
    from .observation_policy import scoring_template
    if scope not in ('initial', 'supplemental'):
        raise ValueError('unknown recipe terminal scope')
    state, freeze = open_recipe_terminal(training_run)
    _, plan = _reopen_recipe(state)
    policy = plan.description['evaluation']
    conditions = policy['conditions' if scope == 'initial' else 'supplemental_conditions']
    if not conditions:
        raise ValueError('empty terminal request is not requested')
    value = scoring_template(template, policy, policy)
    if value['task'] != plan.description['config']['task']:
        raise ValueError('recipe scoring task differs from checkpoint task')
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


def _validate_recipe_request(state, request, *, observation):

    state, plan = _reopen_recipe(state)
    if request['task'] != plan.description['config']['task']:
        raise ValueError('recipe scoring task differs from checkpoint task')
    if (request['plan_identity'] != plan.identity or request['checkpoint_sha256'] != state.reference.sha256
            or request['step'] != state.payload['metadata']['completed_updates']):
        raise ValueError('recipe request checkpoint/plan/step differs')
    expected = (_recipe_observation_request(state, request['policy_request'], request) if observation else
                terminal_recipe_request(request['training_run'], request, scope=request['scope']))
    if request != expected:
        raise ValueError('recipe request differs from resolved policy')
    return state, plan


def recipe_training_rows(training_run):
    """Read the recipe event schema without imposing retained loss coefficients."""
    root = Path(training_run).resolve()
    state, _ = open_recipe_terminal(root)
    _, plan = _reopen_recipe(state)
    description = plan.description
    campaign = json.loads((root/'campaign.json').read_text())
    source_prefix = [0]
    for view in plan.views:
        source_prefix.append(source_prefix[-1] + view.source_tokens)
    sites = len(plan.trained_sites)
    rows = []
    for line in (root/'metrics.jsonl').read_text().splitlines():
        event = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError('nonfinite recipe scalar')))
        p = event['payload']
        if (event['schema'] != 'recipe-training-event-v1' or event['event_type'] != 'update'
                or p['nonfinite'] or p['skipped']):
            raise ValueError('recipe report contains unsupported/unsuccessful scalar update')
        scheduled = plan.update_at(p['completed_step']-1)
        validate_recipe_update(p, weights=plan.objective_at(p['completed_step']-1))
        stage = plan.stage_at(p['completed_step']-1)
        completed_views, partial_sites = divmod(p['completed_step'], sites)
        cumulative_source = source_prefix[completed_views] * sites
        if partial_sites:
            cumulative_source += partial_sites * plan.views[completed_views].source_tokens
        expected_identity = {'source': state.payload['metadata']['source_identity'],
            'config': state.payload['metadata']['config_identity'],
            'data': state.payload['metadata']['data_identity']}
        if (any(event['identity'][k] != v for k, v in expected_identity.items())
                or p['recipe_identity'] != plan.identity or p['site_id'] != scheduled.site
                or p['site_step'] != scheduled.site_local_index+1
                or event['phase_id'] != stage.phase_id or p['phase_start_step'] != stage.start
                or p['final_step'] != plan.horizon
                or p['configured_counts'] != description['counts']
                or p['recipe_progress'] != plan.progress(p['completed_step'], stage.start)
                or p['channel'] != description['config']['channel']
                or p['objective']['kind'] != stage.objective_kind
                or any(p['exposure'][key] != getattr(scheduled.view, key)
                       for key in ('source_tokens', 'target_tokens', 'sequences', 'padded_tokens'))
                or p['exposure']['cumulative_valid_tokens'] != cumulative_source
                or p['objective']['resolved_weights'] != plan.objective_at(p['completed_step']-1)
                or p['lr_used'] != [description['config']['training']['lr']*scheduled.lr_factor]):
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





class ObservationScoringError(RuntimeError):
    """Recoverable scoring/output error; never integrity or restoration failure."""


def _run_panels(model, processor, *, state, request, output, guard, observation):
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
        with _panel_runtime(model, state):
            data = None
            if request['task'] == 'coco':
                from .shared_inputs import load_coco_panel_data
                data = load_coco_panel_data(request, processor)
            for row in panels:
                guard()  # Resource/stop guard is fatal, never a scoring failure.
                path = output / f"{row['site']}_{row['condition']}"
                path.mkdir(exist_ok=False)
                panel_request = {**request, 'schema': 'recipe-panel-request-v1' if recipe else 'original13-panel-request-v1',
                                 'target_site': row['site'], 'conditions': [row['condition']]}
                write_manifest(path / 'request.json', panel_request)
                started = time.monotonic()
                try:
                    scored = _score_panel(model, processor, state=state, request=request,
                        site_id=row['site'], condition=row['condition'], output=path,
                        **({'data': data} if request['task'] == 'coco' else {}))
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


def _recipe_observation_request(state, policy_request, template):
    from .observation_policy import scoring_template, recipe_observation_identity
    state, plan = _reopen_recipe(state)
    policy = next((r for r in plan.description['observations']
                   if (r['step'], r['state']) == (policy_request['step'], policy_request['state'])), None)
    if policy is None or policy_request != {**policy, 'plan_identity': plan.identity,
                                           'checkpoint_sha256': state.reference.sha256}:
        raise ValueError('recipe observation policy/checkpoint differs')
    expected_phase = 0 if policy['state'] == 'pre_reset' else plan.stages[-1].start
    if state.payload['metadata']['phase_start_step'] != expected_phase:
        raise ValueError('recipe observation reset phase differs')
    value = scoring_template(template, policy, plan.description['evaluation'])
    if value['task'] != plan.description['config']['task']:
        raise ValueError('recipe scoring task differs from checkpoint task')
    if state.payload['metadata']['observation_identity'] != recipe_observation_identity(plan.description, template):
        raise ValueError('recipe fixed observation template identity differs')
    return {**value, **copy.deepcopy(policy_request), 'schema': 'recipe-observation-request-v1',
        'state_role': policy['state'], 'policy_request': copy.deepcopy(policy_request)}


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
    return _run_panels(model, processor, state=state, request=request,
                                 output=output, guard=guard, observation=True)


def run_recipe_evaluation(model, processor, *, request, output, guard):
    state, _ = open_recipe_terminal(request['training_run'])
    return _run_panels(model, processor, state=state, request=request,
                                 output=output, guard=guard, observation=False)


@contextmanager
def _panel_runtime(model, state, *, checkpoint_payload=None):
    """Restore canonical weights, gradients, global RNG and all idle route state.

    The routing owner provides the model-side idle-state contract. Validation and
    restoration errors are fatal, including when a scorer/output error also occurs.
    """
    from .checkpoint_readers import _reopen_original13
    from . import site_routing as routing
    from .shared_state import recipe_runtime_state, restore_recipe_runtime
    recipe = state.payload['kind'] == 'recipe_v1'
    runtime = recipe_runtime_state if recipe else routing.observation_state
    restore_runtime = restore_recipe_runtime if recipe else routing.restore_runtime_state
    codec = routing.canonical_codec(model)
    previous_weights = copy.deepcopy(routing.export_codec(model))
    previous_route = copy.deepcopy(runtime(model))
    parameters = list(codec.parameters())
    grads = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
    rng = capture_rng()
    try:
        routing.load_codec(model, state.payload['model'])
        model.eval()
        yield
    finally:
        try:
            unchanged = state_dict_identity(routing.export_codec(model)) == state_dict_identity(state.payload['model'])
        finally:
            # Even integrity inspection itself can fail. Always attempt route and
            # RNG cleanup; restoration failure remains fatal and may supersede it.
            try:
                routing.load_codec(model, previous_weights)
            finally:
                try:
                    for parameter, grad in zip(parameters, grads, strict=True):
                        parameter.grad = grad
                    restore_runtime(model, previous_route)
                finally:
                    restore_rng(rng)
        if not unchanged:
            raise ValueError('original13 scorer mutated checkpoint weights')
        if checkpoint_payload is not None:
            # A process worker has already validated and retained this envelope.
            # Recheck unchanged bytes and the resident payload without loading it again.
            path = Path(state.reference.path)
            if path.stat().st_size != state.reference.size or file_digest(path) != state.reference.sha256:
                raise ValueError('worker checkpoint bytes changed')
            if not state_exact(state.payload, checkpoint_payload):
                raise ValueError('worker checkpoint resident payload changed')
            reopened = state
        elif recipe:
            from .experiment_state import open_state
            reopened = open_state(state.reference, expected=state.payload['metadata'])
        else:
            reopened, _ = _reopen_original13(state)
        if (not state_exact(reopened.payload, state.payload)
                or not state_exact(routing.export_codec(model), previous_weights)
                or not state_exact(runtime(model), previous_route)
                or not state_exact([p.grad for p in parameters], grads)
                or not state_exact(capture_rng(), rng)):
            raise ValueError('original13 observation restoration/checkpoint integrity differs')


@contextmanager
def _panel_scoring_policy(settings):
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(settings.get('deterministic_algorithms', enabled))
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _score_panel(model, processor, *, state, request, site_id, condition, output, data=None):
    """Use the real task adapter, with no codec bypass in the no_noise condition."""

    from .evaluation import evaluate_coco, evaluate_hellaswag, _observation_items, TaskScoringError
    from .models import split_model
    from .experiment_state import isolated_rng
    task = request['task']
    if task not in ('coco', 'hellaswag'):
        raise ValueError('unsupported shared panel task')
    if (state.payload['kind'] == 'recipe_v1'
            and task != state.payload['metadata']['recipe_plan']['description']['config']['task']):
        raise ValueError('recipe scoring task differs from checkpoint task')
    if task == 'coco':
        if state.payload['kind'] != 'recipe_v1':
            raise ValueError('shared COCO requires a task-bound recipe checkpoint')
        if data is None:
            from .shared_inputs import load_coco_panel_data
            data = load_coco_panel_data(request, processor)
        corpus = request['reference_corpus']
        ids = data.ids['report_ids']
        if (ids != request['input_ids'] or ids != request['source_family_ids']
                or ids != corpus['image_ids']
                or data.demo_captions != corpus['demo_captions']
                or data.ids['demo_ids'] != corpus['demo_ids']
                or len(ids) != request['expected_items'] or len(data.report) != len(ids)):
            raise ValueError('shared COCO acquired image/reference/demo order differs')
    resolver = split_model.resolve_training_site if state.payload['kind'] == 'recipe_v1' else split_model.resolve_original13_site
    site = resolver(model.base, site_id, state.payload['metadata']['model_revision'])
    settings = {**request['settings'], 'noise_seed': request['noise']['seed'],
                'noise_namespace': request['noise']['namespace'], 'evidence_mode': 'compact-v1'}
    dtype = next(model.base.parameters()).dtype
    dtype_names = {torch.float32: 'fp32', torch.bfloat16: 'bf16', torch.float16: 'fp16'}
    backend = getattr(model.base.config.decoder, '_attn_implementation', None)
    if (request['precision'] not in (str(dtype), dtype_names.get(dtype))
            or request['backend'] != backend
            or getattr(model, 'numerical_policy', 'native') != 'native'):
        raise ValueError('original13 actual model precision/backend differs')
    snr = None if condition == 'no_noise' else float(condition)
    trained_sites = (state.payload['metadata']['recipe_plan']['description']['locations']
                     if state.payload['kind'] == 'recipe_v1' else [site_id])
    site_role = 'trained' if site_id in trained_sites else 'diagnostic'
    from .models.channel import IdentityChannel
    noise_scope = (nullcontext() if type(model.channel) is IdentityChannel else
                   model.channel.replay(settings['noise_seed'], settings['noise_namespace']))
    with _panel_scoring_policy(settings), isolated_rng(settings['noise_seed']), noise_scope, model.at_site(
            site, purpose='evaluate', authorization={'sites': {site_id: site_role}}), model.transmission(snr):
        try:
            if task == 'coco':
                metrics = evaluate_coco(model, processor, data, settings, output, condition, request['data_settings'])
            else:
                metrics = evaluate_hellaswag(model, processor, settings, output, condition, request['data_settings'],
                                             **({'prepared_task': data} if data is not None else {}))
        except TaskScoringError as error:
            raise ObservationScoringError(str(error)) from error
        payload = {'allocated': dict(model.channel_uses_allocated), 'valid': dict(model.valid_payload_counts)}
    items, ids, families = _observation_items(output, task, condition)
    if ids != request['input_ids'] or families != request['source_family_ids']:
        raise ValueError('shared observed item/source-family order differs')
    if task == 'coco':
        import math
        if (not metrics.get('finite_scores') or not math.isfinite(metrics['cider'])
                or metrics['num_samples'] != request['expected_items']
                or len(items) != request['expected_items']
                or any(not math.isfinite(item['cider']) for item in items)
                or [item['references'] for item in items] != request['reference_corpus']['references']):
            raise ValueError('shared COCO finite score/image denominator differs')
        payload['mask_valid'] = dict(payload['valid'])
        payload['generation_valid'] = metrics['channel_uses_generation_valid']
        payload['generation_valid_policy'] = metrics['generation_valid_policy']
        payload['mask_valid_limitation'] = metrics['mask_valid_limitation']
    elif not metrics.get('finite_scores') or metrics['candidate_forward_requests'] != 4 * request['expected_items']:
        raise ValueError('original13 finite score/candidate denominator differs')
    return {'completed_items': len(items), 'denominator': len(items), 'metrics': metrics, 'payload': payload}





def _native_panel_require(value, message):
    if not value:
        raise ValueError(message)


def evaluate_saved_vanilla_panel(model, processor, request, output):
    """Evaluate and retain the complete shared vanilla panel on CPU or CUDA.

    The compact writer still verifies observed payload counters. The condition
    describes the bypass route consistently through scoring, readback and receipt.
    """
    from jscc.evaluation import evaluate_hellaswag, _observation_items

    _native_panel_require(request["expected_items"] == 256, "development panel must contain256")
    _native_panel_require(request["settings"]["num_samples"] == 256, "vanilla sample limit differs")
    directory = Path(output)
    settings = {**request["settings"], "evidence_mode": "compact-v1"}
    model.eval()
    with torch.no_grad(), model.transmission(bypass=True):
        scores = evaluate_hellaswag(
            model, processor, settings, directory, "vanilla", request["data_settings"]
        )
    items, ids, families = _observation_items(directory, "hellaswag", "vanilla")
    _native_panel_require(
        ids == request["input_ids"] and families == request["source_family_ids"],
        "vanilla membership differs",
    )
    _native_panel_require(len(items) == 256, "vanilla panel incomplete")
    receipt = {
        "schema": "vanilla-panel-v1",
        "requested": 256,
        "completed": len(items),
        "failed": 0,
        "condition": "vanilla",
        "codec_bypassed": True,
        "request": request,
        "metrics": scores,
        "items": items,
    }
    write_json_atomic(directory / "receipt.json", receipt)
    model.train()
    return receipt



def worker_checkpoint_reference(target):
    """Translate the recipe target byte count at the immutable core boundary."""
    if not isinstance(target, dict) or type(target.get('size')) is not int or target['size'] < 0:
        raise ValueError('recipe checkpoint size must be a nonnegative integer')
    reference = copy.deepcopy(target)
    if 'bytes' in reference and (type(reference['bytes']) is not int or reference['bytes'] != target['size']):
        raise ValueError('recipe checkpoint size/bytes differ')
    reference['bytes'] = target['size']
    actual = core.file_reference(reference['path'])
    if any(reference.get(key) != actual[key] for key in ('path', 'sha256', 'bytes')):
        raise ValueError('recipe checkpoint bytes/hash differ')
    return reference


@dataclass(frozen=True)
class RecipeRuntimeFactory:
    root: str
    case_id: str
    checkpoint: str | None
    expected_step: int | None
    scope: str
    request: dict
    timeout_seconds: float

    def __call__(self, plan, device):
        # Bind visibility before selector/runtime/model code can initialize CUDA.
        if device != 'cpu':
            index = int(device.split(':')[1])
            visible = os.environ.get('CUDA_VISIBLE_DEVICES')
            devices = visible.split(',') if visible else None
            if devices is not None and index >= len(devices):
                raise ValueError('worker device outside inherited GPU assignment')
            os.environ['CUDA_VISIBLE_DEVICES'] = devices[index] if devices else str(index)
        from .shared_evaluation import recipe_evaluation_target
        from .shared_preparation import initialize_recipe
        from .shared_runs import _guard
        eligibility, request, state, config, initial, manifest = recipe_evaluation_target(
            self.root, self.case_id, checkpoint=self.checkpoint,
            expected_step=self.expected_step, scope=self.scope)
        if request != self.request or worker_checkpoint_reference(eligibility['checkpoint']) != plan['checkpoint']:
            raise ValueError('worker selected request/checkpoint differs from immutable plan')
        _validate_recipe_request(state, request, observation=self.scope == 'development')
        config = copy.deepcopy(config)
        config['model']['device'] = 'cpu' if device == 'cpu' else 'cuda'
        guard = _guard(config, synthetic=manifest['synthetic_cpu'],
                       timeout_seconds=self.timeout_seconds,
                       device_uuid=None if device == 'cpu' else os.environ['CUDA_VISIBLE_DEVICES'])
        guard()
        processor, model = initialize_recipe(config, initial,
            state.payload['metadata']['recipe_plan']['description']['locations'],
            evaluation=True, synthetic=manifest['synthetic_cpu'])
        if str(next(model.base.parameters()).device) != ('cpu' if device == 'cpu' else 'cuda:0'):
            raise ValueError('actual worker model device differs')
        data = None
        if request['task'] == 'coco':
            from .shared_inputs import load_coco_panel_data
            data = load_coco_panel_data(request, processor)
        else:
            from .evaluation import prepare_hellaswag_task
            data = prepare_hellaswag_task(request['data_settings'])
        return {'model': model, 'processor': processor, 'state': state,
                'request': request, 'data': data, 'guard': guard,
                'checkpoint_payload': copy.deepcopy(state.payload)}


def score_recipe_panel(runtime, plan, panel, output):
    """One full corpus/inventory per call; worker identity never enters noise."""
    request, state = runtime['request'], runtime['state']
    output = Path(output)
    panel_request = {**request, 'schema': 'recipe-panel-request-v1',
                     'target_site': panel['site'], 'conditions': [panel['condition']]}
    write_json_atomic(output/'request.json', panel_request)
    runtime['guard']()
    started = time.monotonic()
    try:
        with _panel_runtime(runtime['model'], state, **({'checkpoint_payload': runtime['checkpoint_payload']}
                if 'checkpoint_payload' in runtime else {})):
            scored = _score_panel(runtime['model'], runtime['processor'], state=state,
                request=request, site_id=panel['site'], condition=panel['condition'],
                output=output, **({'data': runtime['data']} if runtime.get('data') is not None else {}))
    except ObservationScoringError as error:
        raise core.PanelScoringError(str(error)) from error
    runtime['guard']()
    if scored['completed_items'] != plan['expected_items'] or scored['denominator'] != plan['expected_items']:
        raise ValueError('worker whole panel denominator differs')
    row: dict = dict(scored, site=panel['site'], condition=panel['condition'], status='complete',
               requested_items=plan['expected_items'], elapsed_seconds=time.monotonic()-started,
               panel_identity=canonical_digest(panel_request))
    row['output_hashes'] = {p.name: file_digest(p) for p in sorted(output.iterdir()) if p.is_file()}
    write_json_atomic(output/'receipt.json', row)
    scores = {k: v for k, v in scored['metrics'].items()
              if type(v) in (int, float) and math.isfinite(v)}
    return {'item_ids': request['input_ids'], 'scores': scores,
            'noise_digest': core.canonical_digest(plan['noise']),
            'noise_evidence': 'policy identity only; sampled noise is not hashed',
            'scientific_receipt': core.file_reference(output/'receipt.json')}


def bridge_receipt(aggregate, plan, request, state, attempt, output):
    rows = []
    for panel in aggregate['rows']:
        status = panel['status']
        if status == 'complete':
            result = json.loads(Path(panel['receipt']['result']['path']).read_text())
            reference = result['scientific_receipt']
            path = core._verify_reference(reference, Path(panel['owner'])/'outputs'/panel['panel_id'])
            if core.file_reference(path) != reference:
                raise ValueError('scientific panel receipt identity differs')
            row = json.loads(path.read_text())
            row['receipt'] = {'path': str(path), 'sha256': file_digest(path)}
        else:
            row = {'site': panel['site'], 'condition': panel['condition'],
                   'status': 'failed' if status == 'failed' else 'pending',
                   'requested_items': plan['expected_items'], 'completed_items': 0,
                   'worker_status': status}
        rows.append(row)
    receipt = {'schema': 'recipe-evaluation-receipt-v1', 'request': request,
        'request_identity': canonical_digest(request), 'checkpoint': asdict(state.reference),
        'plan_identity': state.payload['metadata']['plan_identity'],
        'status': 'complete' if not aggregate['incomplete'] else 'incomplete',
        'final_eligible': aggregate['final_eligible'],
        'worker_counts': aggregate['counts'],
        'panels': rows, 'parallel': {'attempt': str(attempt),
            'plan': core.file_reference(attempt/'plan.json'),
            'aggregate': core.file_reference(attempt/'aggregate.json'),
            'final': core.file_reference(attempt/'final.json')}, 'elapsed_seconds': 0.0}
    _verify_recipe_result(receipt, state, observation=plan['scope'] == 'development')
    write_json_atomic(output/'result.json', receipt)
    return receipt


def verified_observations(eligibility, state):
    if (eligibility.get('observations_status_recorded') is not True
            or eligibility.get('observations_status') != 'complete'
            or any(row.get('status') != 'complete' for row in eligibility['observations'])):
        return False
    campaign = json.loads((Path(eligibility['checkpoint']['engine_path'])/'campaign.json').read_text())
    owners = [Path(eligibility['checkpoint']['engine_path'])]
    owners.extend(Path(ref['path']).parent for ref in state.payload['metadata']['segment_ancestry'])
    for obligation in eligibility['observations']:
        candidates = [owner/f"observation_{obligation['step']:06d}_{obligation['state']}"/'result.json'
                      for owner in owners]
        candidates = [p for p in candidates if p.is_file()]
        if len(candidates) != 1:
            return False
        receipt = json.loads(candidates[0].read_text())
        from .experiment_state import CheckpointRef
        observed = _open_bound_reference(CheckpointRef(**receipt['checkpoint']))
        _verify_recipe_result(receipt, observed, observation=True)
        if receipt['status'] != 'complete' or not _own_recipe_observation(state, campaign, receipt):
            return False
    return True


def evaluate_parallel_recipe(*, root, eligibility, request, state, output, controls,
                             attempt_id, checkpoint, expected_step, timeout_seconds):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    controls = dict(controls)
    parent = controls.pop('parent_attempt', None)
    controls = core.validate_parallel_controls(controls)
    plan = core.make_plan(case_id=eligibility['case_id'], checkpoint=worker_checkpoint_reference(eligibility['checkpoint']),
        scope=eligibility['scope'], sites=request['sites'], conditions=request['conditions'],
        item_ids=request['input_ids'], identities={'source': eligibility['source'],
            'runtime': eligibility['runtime'], 'input': canonical_digest(request),
            'scorer': core.file_reference(Path(__file__)),
            'training_observation_eligible': verified_observations(eligibility, state)}, noise=request['noise'])
    attempt = core.reserve_attempt(output/'worker-attempts', attempt_id, plan, parent=parent)
    inherited = json.loads((attempt/'attempt.json').read_text())['inherited']
    done = {row['panel_id'] for row in inherited}
    assignments = core.assignments_for(plan, devices=controls['devices'], workers=controls['workers'],
                                       panel_ids=[p['id'] for p in plan['panels'] if p['id'] not in done])
    factory = RecipeRuntimeFactory(str(root), eligibility['case_id'], checkpoint, expected_step,
                                   eligibility['scope'], request, timeout_seconds)
    try:
        worker_states = core.run_workers(plan, attempt, assignments, runtime_factory=factory,
            score_panel=score_recipe_panel, timeout_seconds=timeout_seconds, publish_partial=True)
    except BaseException as original:
        # The core has already joined/terminated children and saved terminal evidence.
        # Preserve its original fatal event even if integrity prevents finalization.
        try:
            aggregate = core.collect_results(attempt, plan, final=True)
            bridge_receipt(aggregate, plan, request, state, attempt, output)
        except BaseException as finalization_error:
            original.add_note('Interrupted worker evidence could not be finalized: '
                              + type(finalization_error).__name__ + ': ' + str(finalization_error))
        raise
    aggregate = core.collect_results(attempt, plan, worker_states=worker_states, final=True)
    receipt = bridge_receipt(aggregate, plan, request, state, attempt, output)
    return receipt
