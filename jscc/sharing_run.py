"""Finite ticket17 shared/specialist lifecycle using the existing loss and scorer.

No architecture decision, GPU reservation, retry, or sweep is made here. A prepared
entry names exactly one candidate and retains every mandatory output.
"""
from __future__ import annotations

import copy
import shutil
from dataclasses import asdict
from pathlib import Path
import time

import torch

from .activation_replay import canonical_digest, file_digest
from .baseline_protocol import (objective_record_request, objective_settings,
                                read_prepared_batch, validate_task_receipt, write_manifest, bind_batch_identity)
from .evaluation import (codec_observation_identity, evaluate_checkpoint,
                         observation_event_payload, verify_observation_artifacts)
from .experiment_records import (append_event, artifact_ref, completion_status,
                                 observation_identity, read_events, measure)
from .experiment_state import open_state, restore_state, save_state
from .models.split_model import build_model
from .sharing_protocol import SharingLearner, batch_view
from .sharing_schedule import SharingPlan, TRAINED_SITES
from .sharing_state import build_sharing_state, create_study_freeze, verify_study_freeze


def panel_id(observation):
    return f'{observation.purpose}:{observation.learner}:{observation.site}:{observation.step}'


def _controls(learner, metadata):
    return {
        'architecture': canonical_digest(learner.model.codec.config),
        'initialization': metadata['initialization_identity'],
        'training_data': metadata['stream_identity'],
        'objective': canonical_digest(objective_settings('combined')),
        'exposure': canonical_digest({**({'execution_partition': metadata['execution_partition']}
                                       if 'execution_partition' in metadata else {}),
                                      'q': learner.schedule.q, 'batch': learner.effective_batch,
                                      'sites': list(TRAINED_SITES) if learner.schedule.name == 'shared'
                                      else [learner.schedule.name.removeprefix('specialist_')]}),
        'schedule': canonical_digest({'horizon': learner.schedule.horizon,
                                      'warmup': max(1, learner.schedule.horizon // 20),
                                      'policy': 'sharing-linear-warmup-cosine-v1'}),
    }


def run_sharing(model, processor, *, plan, update_batches, validation_batches,
                metadata, task_template, output, geometry_batches, resource_guard, microbatch_size=None):
    """Execute one frozen four-learner plan; callbacks only bound resources.

    All task observations use the real versioned evaluation entry. CPU tests
    replace only the external task adapter, never this lifecycle or state I/O.
    """
    from .sharing_accumulation import partition_policy, partition_native64
    partition = None if microbatch_size is None else partition_policy(microbatch_size)
    if (partition is not None and plan.batch_size != 64) or (plan.synthetic and plan.batch_size == 64 and partition is None):
        raise ValueError("explicit accumulation requires parent effective64")
    metadata = copy.deepcopy(metadata)
    if partition is not None:
        if metadata.get("execution_partition", partition) != partition:
            raise ValueError("metadata execution partition differs")
        metadata["execution_partition"] = partition
    elif metadata.get("execution_partition") is not None:
        raise ValueError("metadata requires an explicit execution partition")
    if microbatch_size is not None:
        # Bind observation requests to the actual physical layouts used below.
        validation_batches = [micro for batch in validation_batches
            for micro in (partition_native64(batch, microbatch_size=microbatch_size)
                          if len(batch["labels"]) == 64 else [batch])]
        if any(len(batch["labels"]) > microbatch_size for batch in validation_batches):
            raise ValueError("objective input must be parent64 or at most one microbatch")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if metadata.get('prepared_package'):
        package = metadata['prepared_package']
        retained = output / 'inputs'
        retained.mkdir()
        for name, reference in package['retained'].items():
            path = Path(reference['path'])
            if file_digest(path) != reference['sha256']:
                raise ValueError('prepared input changed before lifecycle acquisition')
            shutil.copyfile(path, retained / name)
        write_manifest(retained / 'package-reference.json', package)
    update_bindings = ([bind_batch_identity(batch) for batch in update_batches]
                       if microbatch_size is not None else None)
    initial = copy.deepcopy(model.codec.state_dict())
    learners, roots, manifests, expected, states = {}, {}, {}, {}, {}
    begun = time.monotonic()
    all_observations = []
    costs = []
    input_panel = canonical_digest({key: task_template[key] for key in (
        'input_ids', 'source_family_ids', 'prompt_policy', 'layout', 'scorer', 'data_settings', 'settings')})
    campaign = {'schema': 'sharing-run-v1', 'status': 'incomplete',
                'synthetic': plan.synthetic, 'recipe_identity': metadata['recipe_identity'],
                'architecture_decision': metadata['architecture_decision'],
                'planned_updates': 6 * plan.q, 'planned_checkpoints': 17,
                'planned_panels': plan.condition_panels,
                'obligations': [panel_id(o) for o in plan.observations] + ['vanilla', 'geometry:atlas'],
                'completed_obligations': []}
    write_manifest(output / 'campaign.json', campaign)

    def persist():
        write_manifest(output / 'campaign.json', campaign)

    def cost(learner, category, started, site, *, frozen=False):
        payload = {'category': category, 'seconds': measure(time.monotonic() - started),
                   'attribution': 'first_use', 'site_id': site, 'scope': 'host_wall:' + category,
                   'concurrency': 'single_process', 'device_count': int(next(model.base.parameters()).is_cuda)}
        costs.append({'learner': learner.schedule.name, **payload})
        if not frozen:
            learner._event('cost', 'combined', payload)
        write_manifest(output / 'costs.json', costs)

    def make(name):
        root = output / name
        root.mkdir()
        learner = SharingLearner(model, schedule=plan.learner(name), run_id=name,
            identity={'source': metadata['source_identity'], 'config': metadata['config_identity'],
                      'data': metadata['data_identity'], 'parent': None},
            model_revision=metadata['model_revision'], enc_fn_reuse=not plan.synthetic,
            microbatch_size=microbatch_size)
        def sink(event):
            append_event(root / 'metrics.jsonl', event)
            if event['event_type'] == 'update':
                write_manifest(root / 'draws' / f'{learner.completed:06d}.json', learner.last_draws)
        learner.event_sink = sink
        roots[name], learners[name], states[name] = root, learner, {}
        obs = [o for o in plan.observations if o.stage == 'pre_freeze' and
               (o.learner == name or (name == 'shared' and o.learner == 'common_initialization'))]
        expected[name] = {panel_id(o): 'pending:' + panel_id(o) for o in obs}
        manifests[name] = {'schema': 'experiment-records-v1', 'run_id': name,
            'phases': [{'phase_id': 'combined', 'updates': learner.schedule.horizon,
                        'start_step': 0, 'start_valid_tokens': 0}],
            'expected_observations': list(expected[name].values()),
            'expected_artifacts': [f'step_{step:06d}.pt' for step in
                                   ((0,) if name == 'shared' else ()) + learner.schedule.checkpoint_steps]}
        write_manifest(root / 'manifest.json', manifests[name])
        return learner

    def save(learner):
        resource_guard()
        started = time.monotonic()
        name, step = learner.schedule.name, learner.completed
        sharing, stream = build_sharing_state(synthetic=plan.synthetic, learner=name,
            ordered_view_identities=[v.view_sha256 for v in plan.views], completed_updates=step,
            source_valid_per_view=[v.source_tokens for v in plan.views],
            target_valid_per_view=[v.target_tokens for v in plan.views],
            padded_per_view=[v.padded_tokens for v in plan.views], batch_size=plan.batch_size,
            protocol_id=metadata['protocol_identity'])
        sites = {key: {'site': asdict(value), 'role': 'heldout_after_freeze' if key == 'enc_l14' else 'trained'}
                 for key, value in learner.sites.items() if name == 'shared' or key == name.removeprefix('specialist_')}
        details = {**metadata, 'parent_identity': None, 'lineage': [], 'phase': 'both',
                   'sharing': sharing, 'completed_updates': step, 'evaluation_sites': sites,
                   'snapshot_role': 'initialization' if step == 0 else 'trained',
                   'model_state_contract': 'codec-only-stateless-channel-v1', 'numerical_policy': 'native'}
        if name != 'shared':
            details['site'] = sites[name.removeprefix('specialist_')]['site']
        details['comparison_controls'] = _controls(learner, details)
        reference = save_state(roots[name] / f'step_{step:06d}.pt', model=model.codec,
            optimizer=learner.optimizer, scheduler=learner.scheduler, scaler=learner.scaler,
            metadata=details, stream_state=stream)
        validated = open_state(reference, expected=details)
        # Exercise complete restore at every boundary, not just a read/weight load.
        restored = restore_state(validated, model=model.codec, optimizer=learner.optimizer,
                                 scheduler=learner.scheduler, scaler=learner.scaler)
        if restored != stream:
            raise ValueError('saved sharing stream did not restore exactly')
        cost(learner, 'checkpoint_io', started, name)
        states[name][step] = validated
        return validated

    def observe(learner, observation, state, freeze=None):
        resource_guard()
        started = time.monotonic()
        name, site, tag = learner.schedule.name, observation.site, panel_id(observation)
        details = state.payload['metadata']
        if freeze is not None:
            receipt = verify_study_freeze(freeze, checkpoint_sha256=state.reference.sha256,
                protocol_identity=metadata['protocol_identity'], site_policy=details['evaluation_sites'], obligation=tag)
        else:
            receipt = None
        path = (output / 'heldout' if receipt else roots[name]) / tag.replace(':', '_')
        if observation.purpose == 'task':
            request = {**task_template, 'schema': 'codec-observation-v1',
                'checkpoint_sha256': state.reference.sha256, 'target_site': asdict(learner.sites[site]),
                'target_role': 'heldout_after_freeze' if receipt else 'trained',
                'learner_kind': state.payload['kind'], 'step': observation.step,
                'site_step': observation.step // 3 if name == 'shared' else observation.step,
                'parent': None, 'comparison': details['comparison_controls'], 'panel': input_panel,
                'source': metadata['source_identity'], 'config': metadata['config_identity'],
                'data': metadata['data_identity'], 'protocol': metadata['protocol_identity']}
            if receipt:
                request.update(freeze_receipt=receipt, freeze_reference=freeze)
            identity = codec_observation_identity(request)
        else:
            request = objective_record_request(learner, state, validation_batches, 'combined')
            request.update(site=site, role='heldout' if receipt else 'trained',
                learner_kind='initialization' if observation.step == 0 else
                             'shared' if name == 'shared' else 'specialist',
                step=observation.step, site_step=observation.step // 3 if name == 'shared' else observation.step,
                comparison=details['comparison_controls'])
            identity = observation_identity(request)
        if receipt is None:
            expected[name][tag] = identity
            manifests[name]['expected_observations'] = list(expected[name].values())
            write_manifest(roots[name] / 'manifest.json', manifests[name])
        else:
            write_manifest(output / 'heldout' / (tag.replace(':', '_') + '-request.json'), request)
        if observation.purpose == 'task':
            result = evaluate_checkpoint(state, request, model=model, processor=processor, output=path)
            validate_task_receipt(result, request)
            verify_observation_artifacts(result, path)
            payload = observation_event_payload(result)
        else:
            before = copy.deepcopy(model.codec.state_dict())
            try:
                model.codec.load_state_dict(state.payload['model'])
                authorization = None if receipt is None else {
                    'sites': {site: 'heldout'}, 'heldout_freeze_receipt': canonical_digest(receipt)}
                rows = learner.validate_site(validation_batches, site, authorization=authorization)
            finally:
                model.codec.load_state_dict(before)
            write_manifest(path.with_suffix('.json'), {'request': request, 'conditions': rows})
            items = [{'item_id': str(int(i)), 'source_id': str(f)} for batch in validation_batches
                     for i, f in zip(batch['row_ids'], batch['source_family_ids'])]
            payload = {'request': request, 'identity': identity, 'status': 'complete', 'reuse': None,
                'conditions': [{'condition': row['condition'], 'requested': len(items), 'completed': len(items),
                    'failed': 0, 'failure_reason': None, 'denominator': len(items), 'metrics': {},
                    'items': items, 'objectives': row['objective']['components']} for row in rows]}
        if receipt is None:
            learner._event('observation', 'combined', payload)
        else:
            write_manifest(output / 'heldout' / (tag.replace(':', '_') + '-event.json'), payload)
        cost(learner, observation.purpose + '_evaluation', started, site, frozen=receipt is not None)
        all_observations.append(payload)
        campaign['completed_obligations'].append(tag)
        persist()

    def trained_observations(learner, step, state):
        owner = 'common_initialization' if step == 0 else learner.schedule.name
        for observation in plan.observations:
            if observation.learner == owner and observation.step == step and observation.stage == 'pre_freeze':
                observe(learner, observation, state)

    def finish_run(name):
        root = roots[name]
        learners[name].exposure.finalize()
        checkpoints = sorted(root.glob('step_*.pt'))
        learners[name]._event('footprint', 'combined', {
            'scope': 'shared' if name == 'shared' else 'per_site', 'site_id': name,
            'parameter_count': measure(sum(p.numel() for p in model.codec.parameters())),
            'checkpoint_bytes': measure(sum(p.stat().st_size for p in checkpoints)),
            'checkpoint_refs': [p.name for p in checkpoints]})
        inventory = {'schema': 'experiment-records-v1', 'mode': 'tensor-complete', 'omitted_tensors': [],
            'artifacts': [artifact_ref(p, root, 'tensor' if p.suffix == '.pt' else 'metadata')
                          for p in sorted(root.rglob('*')) if p.is_file()]}
        write_manifest(root / 'inventory.json', inventory)
        status = completion_status(manifests[name], read_events(root / 'metrics.jsonl'), inventory, root)
        write_manifest(root / 'completion.json', status)
        if status['status'] != 'complete':
            raise ValueError(f'incomplete trained run {name}: {status}')
        return {'root': str(root.resolve()), **{label: {'path': str((root / filename).resolve()),
                    'sha256': file_digest(root / filename)} for label, filename in
                    [('manifest', 'manifest.json'), ('inventory', 'inventory.json'), ('events', 'metrics.jsonl')]}}

    try:
        shared = make('shared')
        initial_state = save(shared)
        trained_observations(shared, 0, initial_state)
        # Vanilla belongs to the same input/scorer contract, but transmits no codec.
        from . import evaluation
        vanilla = output / 'vanilla'
        vanilla.mkdir()
        resource_guard()
        vanilla_started = time.monotonic()
        with torch.no_grad(), model.transmission(bypass=True):
            model.eval()
            metrics = evaluation.evaluate_hellaswag(model, processor, task_template['settings'], vanilla,
                                                    'no_noise', task_template['data_settings'])
        rows, ids, families = evaluation._observation_items(vanilla, 'hellaswag', 'no_noise')
        if ids != task_template['input_ids'] or families != task_template['source_family_ids']:
            raise ValueError('vanilla membership mismatch')
        write_manifest(vanilla / 'receipt.json', {'status': 'complete', 'metrics': metrics,
            'requested': task_template['expected_items'], 'completed': len(rows),
            'source': metadata['source_identity'], 'data': metadata['data_identity'],
            'request': task_template, 'output_hashes': {p.name: file_digest(p) for p in vanilla.iterdir() if p.is_file()}})
        cost(shared, 'vanilla_evaluation', vanilla_started, 'bypass')
        campaign['completed_obligations'].append('vanilla')
        trained_runs = {}
        for name in plan.learners:
            model.codec.load_state_dict(initial)
            learner = shared if name == 'shared' else make(name)
            model.train()
            learner.started = time.monotonic()
            for index in range(learner.schedule.horizon):
                resource_guard()
                update = learner.schedule.update_at(index)
                learner.update([update_batches[update.site_local_index]],
                    batch_identities=None if update_bindings is None else [update_bindings[update.site_local_index]])
                if learner.completed in learner.schedule.checkpoint_steps:
                    state = save(learner)
                    trained_observations(learner, learner.completed, state)
            trained_runs[name] = finish_run(name)
        policy = initial_state.payload['metadata']['evaluation_sites']
        freeze = create_study_freeze(output / 'study-freeze.json',
            protocol_identity=metadata['protocol_identity'], recipe_identity=metadata['recipe_identity'],
            architecture_decision=metadata['architecture_decision'], site_policy=policy,
            trained_runs=trained_runs, synthetic=plan.synthetic,
            obligation_plan={**{purpose: [panel_id(o) for o in plan.observations if o.purpose == purpose]
                                for purpose in ('task', 'objective')}, 'geometry': ['geometry:atlas'],
                             'task': [panel_id(o) for o in plan.observations if o.purpose == 'task'] + [input_panel]})
        for observation in plan.observations:
            if observation.stage == 'post_freeze':
                observe(shared, observation, states['shared'][observation.step], freeze)
        from .sharing_geometry_capture import capture_geometry_atlas
        resource_guard()
        geometry_started = time.monotonic()
        geometry = capture_geometry_atlas(model, geometry_batches, model_revision=metadata['model_revision'],
            include_heldout=True, freeze_reference=freeze, obligation='geometry:atlas')
        write_manifest(output / 'geometry.json', geometry)
        cost(shared, 'geometry_capture_and_analysis', geometry_started, 'all', frozen=True)
        campaign['completed_obligations'].append('geometry:atlas')
        if set(campaign['completed_obligations']) != set(campaign['obligations']):
            raise ValueError('campaign mandatory observations incomplete')
        from .sharing_report import write_sharing_report
        write_sharing_report(output, all_observations, plan, event_files={name: root / 'metrics.jsonl' for name, root in roots.items()})
        campaign.update(status='complete', completed_updates=sum(item.completed for item in learners.values()),
            completed_checkpoints=sum(len(s) for s in states.values()), freeze_reference=freeze,
            elapsed_seconds=time.monotonic() - begun)
        write_manifest(output / 'observations.json', all_observations)
        persist()
        inventory = {'schema': 'experiment-records-v1', 'mode': 'tensor-complete', 'omitted_tensors': [],
            'artifacts': [artifact_ref(path, output, 'tensor' if path.suffix == '.pt' else 'metadata')
                          for path in sorted(output.rglob('*')) if path.is_file()]}
        from .experiment_records import verify_inventory
        verify_inventory(output, inventory)
        write_manifest(output / 'inventory.json', inventory)
        return campaign
    except Exception as error:
        campaign.update(status='incomplete', error={'type': type(error).__name__, 'message': str(error)},
                        completed_updates=sum(item.completed for item in learners.values()),
                        elapsed_seconds=time.monotonic() - begun)
        persist()
        raise


def execute_prepared(manifest, root, output):
    """Exact prepared-artifact entry, including actual model/state/lifecycle paths."""
    from .sharing_preparation import state_dict_identity, require_execution_ready
    root = Path(root)
    started = time.monotonic()
    def guard():
        from .sharing_controller import check_worker_resources
        check_worker_resources(manifest)
        if time.monotonic() - started >= manifest['hard_cap_seconds']:
            raise TimeoutError('sharing allocation deadline; partial work retained, no retry')
    config = manifest['resolved_config']
    require_execution_ready(manifest)
    guard()
    from .sharing_controller import check_target_device
    check_target_device(manifest)
    from .runtime import configure_training_determinism, seed_everything
    configure_training_determinism(config['training'])
    seed_everything(config['seed'])
    processor, model = build_model(copy.deepcopy(config))
    if manifest['synthetic_cpu'] and next(model.base.parameters()).device.type != 'cpu':
        raise ValueError('synthetic lifecycle is CPU only')
    model.codec.load_state_dict(manifest['initialization_state'], strict=True)
    if state_dict_identity(model.codec.state_dict()) != manifest['initialization']['state_identity']:
        raise ValueError('actual initialization differs from prepared codec')
    from .sharing_startup import run_startup_if_requested
    run_startup_if_requested(manifest, root, processor, model, guard)
    batches = [read_prepared_batch(ref, root) for ref in manifest['updates']]
    validation = [read_prepared_batch(ref, root) for ref in manifest['validation']]
    plan = SharingPlan(tuple(batch_view(b) for b in batches), manifest['study_pairing_id'],
                       synthetic=manifest['synthetic_cpu'], protocol_id=manifest['protocol_id'])
    geometry = {partition: [read_prepared_batch(ref, root) for ref in refs]
                for partition, refs in manifest['geometry'].items()}
    metadata = {'source_identity': manifest['source_identity'], 'config_identity': manifest['config_identity'],
        'data_identity': canonical_digest(manifest['data_ids']), 'model_revision': manifest['model_revision'],
        'initialization_identity': manifest['initialization']['state_identity'],
        'stream_identity': canonical_digest([asdict(v) for v in plan.views]),
        'protocol_identity': manifest['protocol_id'], 'recipe_identity': manifest['recipe_identity'],
        'architecture_decision': manifest['architecture_decision'],
        'prepared_identity': manifest['prepared_identity'],
        'prepared_package': {'root': str(root.resolve()), 'identity': manifest['prepared_identity'],
            'retained': {**{name: {'path': str((root / manifest[key]['path']).resolve()),
                                 'sha256': manifest[key]['sha256']} for name, key in
                           [('config.yaml', 'config'), ('source.zip', 'source_archive'), ('task.json', 'task_request')]},
                         'prepared.json': {'path': manifest['prepared_path'],
                                           'sha256': file_digest(Path(manifest['prepared_path']))}}}}

    return run_sharing(model, processor, plan=plan, update_batches=batches, validation_batches=validation,
        metadata=metadata, task_template=manifest['task_template'], output=output,
        geometry_batches=geometry, resource_guard=guard,
        microbatch_size=manifest.get('execution_partition', {}).get('microbatch_size'))
