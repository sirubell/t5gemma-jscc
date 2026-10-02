"""One fixed six-site learner and terminal development evaluation; no dispatch."""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from .activation_replay import canonical_digest, file_digest
from .baseline_protocol import (objective_record_request, read_prepared_batch,
                                validate_task_receipt, write_manifest)
from .config import load_config
from .encoder_pilot_plan import EncoderPilotPlan, SITE_SPECS, SITES, build_pilot_state, final_panels
from .evaluation import evaluate_checkpoint, verify_observation_artifacts
from .experiment_records import (AppendOnlyEventWriter, artifact_ref, completion_status,
                                 observation_identity, read_events, verify_inventory)
from .experiment_state import CheckpointRef, isolated_rng, open_state, restore_state, save_state
from .models.split_model import build_model
from .presentation import PresentationSampler
from .runtime import configure_training_determinism, seed_everything
from .sharing_preparation import (_reference, source_inventory, state_dict_identity,
                                  validate_task_template)
from .sharing_protocol import SharingLearner, batch_view
from .sharing_schedule import ENCODER_PILOT_PROTOCOL, NATIVE16_WEIGHT_DECAY


def pilot_source_inventory():
    source = Path(__file__).resolve().parents[1]
    return {**source_inventory(source),
            'scripts/encoder_shared_pilot.py': file_digest(source / 'scripts/encoder_shared_pilot.py')}


def load_pilot(path):
    """Verify the exact source, configuration, immutable inputs and pool prefix."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    raw = {k: v for k, v in manifest.items() if k != 'manifest_identity'}
    if (manifest.get('schema') != 'encoder-original6-prepared-v1'
            or manifest.get('protocol_id') != ENCODER_PILOT_PROTOCOL
            or manifest.get('manifest_identity') != canonical_digest(raw)
            or manifest.get('source_inventory') != pilot_source_inventory()
            or manifest.get('source_identity') != canonical_digest(manifest['source_inventory'])):
        raise ValueError('pilot manifest or execution source identity mismatch')
    root = path.parent
    config = load_config(_reference(root, manifest['config']))
    if canonical_digest(config) != manifest['config_identity']:
        raise ValueError('pilot resolved configuration identity mismatch')
    batches = [read_prepared_batch(ref, root) for ref in manifest['updates']]
    validation = [read_prepared_batch(ref, root) for ref in manifest['validation']]
    plan = EncoderPilotPlan(tuple(batch_view(b) for b in batches), manifest['study_pairing_id'],
                            synthetic=manifest['synthetic_cpu'])
    training, codec = config['training'], config['codec']
    if (config.get('protocol') != ENCODER_PILOT_PROTOCOL or config['task'] != 'hellaswag'
            or config['seed'] != 0 or config['model']['revision'] != manifest['model_revision']
            or config['model'].get('numerical_policy', 'native') != 'native'
            or config['split'] != SITE_SPECS['enc_fn']
            or codec['architecture'] != 'direct_affine' or codec['layernorm'] != 'none'
            or codec['n_res_blocks'] != 0 or codec['snr_film'] or codec['dropout'] != 0
            or training['batch_size'] != 16 or training['gradient_accumulation'] != 1
            or training['lr'] != 2e-4 or training['weight_decay'] != NATIVE16_WEIGHT_DECAY
            or training['schedule_steps'] != plan.horizon or training['max_steps'] != plan.horizon
            or training['kl_weight'] != 1 or training['mse_weight'] != .1
            or training['temperature'] != 1 or training['grad_clip'] != 1
            or training.get('patience') is not None or training.get('warmup_ratio') != .05
            or training.get('min_lr_ratio') != 0 or training.get('valid_only_kl') is not True
            or training.get('loss_weights') is not None):
        raise ValueError('pilot architecture, objective, native batch or fixed horizon differs')
    channel = config['channel']
    if (channel['type'] != 'awgn' or not channel['normalize_power'] or not channel['train_noise']
            or channel['train_snr_range'] != [-6.0, 18.0]):
        raise ValueError('pilot channel differs')
    template = json.loads(_reference(root, manifest['task_request']).read_text())
    validate_task_template(template, config, synthetic=manifest['synthetic_cpu'])
    if template['conditions'] != ['no_noise', -6, 0, 6, 12, 18]:
        raise ValueError('pilot terminal conditions differ')
    if not manifest['synthetic_cpu']:
        rows = [int(i) for batch in batches for i in batch['row_ids']]
        objective = [int(i) for batch in validation for i in batch['row_ids']]
        pool = manifest['data_ids']['train_rows']
        presentations = plan.q * 16
        expected = PresentationSampler(pool, presentations, seed=20260920).actual_ids.tolist()
        if (len(pool) != 39058 or len(set(pool)) != 39058 or rows != expected
                or len(set(rows)) != min(len(pool), presentations)
                or manifest['presentation_stream'] != {
                    'policy': 'epoch-permutations-v1', 'pool_identity': canonical_digest(pool),
                    'seed': 20260920, 'start_presentation': 0, 'total_presentations': presentations}
                or len(objective) != 128 or len(set(objective)) != 128
                or set(objective) & set(pool)
                or set(objective) != set(manifest['data_ids']['objective_rows'])
                or codec['bottleneck_dim'] != 512 or config['model']['dtype'] != 'bfloat16'
                or config['model']['device'] != 'cuda'
                or manifest['model_revision'] != 'dd0a2683227859151b1730ca3a63087df5b5f39b'
                or config['model']['attn_implementation'] != 'sdpa'
                or template['expected_items'] != 256
                or template['input_ids'] != manifest['data_ids']['development']['input_ids']
                or template['source_family_ids'] != manifest['data_ids']['development']['source_family_ids']
                or template['settings'].get('batch_size') != 16
                or template['settings'].get('num_fewshot') != 5
                or template['settings'].get('scoring_policy') != 'fp32-v1'):
            raise ValueError('pilot pool/selection/development or precision contract differs')
    initialization_path = _reference(root, manifest['initialization'])
    initial = torch.load(initialization_path, map_location='cpu', weights_only=True)
    if state_dict_identity(initial) != manifest['initialization']['state_identity']:
        raise ValueError('pilot initialization tensor identity mismatch')
    return manifest, config, plan, batches, validation, template, initial


def metadata_for(manifest):
    return {'source_identity': manifest['source_identity'], 'config_identity': manifest['config_identity'],
            'data_identity': canonical_digest(manifest['data_ids']), 'parent_identity': None,
            'initialization_identity': manifest['initialization']['state_identity'],
            'stream_identity': canonical_digest(manifest['updates']), 'protocol_identity': ENCODER_PILOT_PROTOCOL,
            'prepared_identity': manifest['manifest_identity'], 'model_revision': manifest['model_revision'],
            'final_task_identity': manifest['task_request']['sha256'], 'lineage': [], 'phase': 'both'}


def make_learner(model, plan, metadata):
    return SharingLearner(model, schedule=plan, run_id='shared',
        identity={'source': metadata['source_identity'], 'config': metadata['config_identity'],
                  'data': metadata['data_identity'], 'parent': None},
        model_revision=metadata['model_revision'], site_specs=SITE_SPECS)


def save_pilot(learner, metadata, path):
    plan = learner.schedule
    sharing, stream = build_pilot_state(plan, learner.completed)
    controls = {'architecture': canonical_digest(learner.model.codec.config),
                'initialization': metadata['initialization_identity'], 'training_data': metadata['stream_identity'],
                'objective': canonical_digest({'K': 1, 'R': .1, 'temperature': 1}),
                'exposure': canonical_digest({'sites': SITES, 'q': plan.q, 'batch': 16}),
                'schedule': canonical_digest({'horizon': plan.horizon, 'warmup': max(1, plan.horizon // 20),
                                              'policy': 'sharing-linear-warmup-cosine-v1'})}
    details = {**metadata, 'sharing': sharing, 'completed_updates': learner.completed,
               'evaluation_sites': {site: {'site': asdict(resolved), 'role': 'trained'}
                                    for site, resolved in learner.sites.items()},
               'snapshot_role': 'initialization' if learner.completed == 0 else 'trained',
               'model_state_contract': 'codec-only-stateless-channel-v1', 'numerical_policy': 'native',
               'comparison_controls': controls}
    reference = save_state(path, model=learner.model.codec, optimizer=learner.optimizer,
                           scheduler=learner.scheduler, scaler=learner.scaler,
                           metadata=details, stream_state=stream)
    validated = open_state(reference, expected=details)
    if restore_state(validated, model=learner.model.codec, optimizer=learner.optimizer,
                     scheduler=learner.scheduler, scaler=learner.scaler) != stream:
        raise ValueError('pilot full-state stream restore mismatch')
    return validated


def run_training(model, *, plan, batches, validation, metadata, output, guard):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    learner = make_learner(model, plan, metadata)
    writer = AppendOnlyEventWriter(output / 'metrics.jsonl')
    learner.event_sink = writer.append
    begun = time.monotonic()
    campaign = {'schema': 'encoder-original6-training-v1', 'status': 'incomplete',
                'prepared_identity': metadata['prepared_identity'], 'completed_updates': 0,
                'planned_updates': plan.horizon, 'objective_observations': [], 'checkpoints': []}
    manifest = {'schema': 'experiment-records-v1', 'run_id': 'shared',
                'phases': [{'phase_id': 'combined', 'updates': plan.horizon, 'start_step': 0, 'start_valid_tokens': 0}],
                'expected_observations': [],
                'expected_artifacts': [f'step_{step:06d}.pt' for step in plan.checkpoint_steps]}
    write_manifest(output / 'campaign.json', campaign)
    try:
        for step in range(plan.horizon + 1):
            guard()
            if step in plan.checkpoint_steps:
                state = save_pilot(learner, metadata, output / f'step_{step:06d}.pt')
                campaign['checkpoints'].append(asdict(state.reference))
                if step in plan.objective_steps:
                    for site in SITES:
                        guard()
                        request = objective_record_request(learner, state, validation, 'combined')
                        request.update(site=site, role='trained', learner_kind='initialization' if step == 0 else 'shared',
                                       step=step, site_step=step // 6, conditions=['no_noise'],
                                       comparison=state.payload['metadata']['comparison_controls'])
                        rows = learner.validate_site(validation, site, conditions=('no_noise',))
                        items = [{'item_id': str(int(i)), 'source_id': str(f)} for batch in validation
                                 for i, f in zip(batch['row_ids'], batch['source_family_ids'])]
                        payload = {'request': request, 'identity': observation_identity(request),
                                   'status': 'complete', 'reuse': None,
                                   'conditions': [{'condition': row['condition'], 'requested': len(items),
                                        'completed': len(items), 'failed': 0, 'failure_reason': None,
                                        'denominator': len(items), 'metrics': {}, 'items': items,
                                        'objectives': row['objective']['components']} for row in rows]}
                        learner._event('observation', 'combined', payload)
                        path = output / f'objective_{step:06d}_{site}.json'
                        write_manifest(path, {'request': request, 'conditions': rows})
                        manifest['expected_observations'].append(payload['identity'])
                        campaign['objective_observations'].append({'step': step, 'site': site, 'reference': str(path)})
            if step == plan.horizon:
                break
            learner.update([batches[step // 6]])
            if learner.completed % 60 == 0 or learner.completed == plan.horizon:
                campaign.update(completed_updates=learner.completed, elapsed_seconds=time.monotonic() - begun,
                                per_site={s: c['updates'] for s, c in learner.exposure.per_site.items()})
                write_manifest(output / 'campaign.json', campaign)
                print(json.dumps({'phase': 'training', 'completed_updates': learner.completed,
                                  'elapsed_seconds': campaign['elapsed_seconds']}), flush=True)
        counts = learner.exposure.finalize()
        guard()
        write_manifest(output / 'manifest.json', manifest)
        inventory = {'schema': 'experiment-records-v1', 'mode': 'tensor-complete', 'omitted_tensors': [],
                     'artifacts': [artifact_ref(p, output, 'tensor' if p.suffix == '.pt' else 'metadata')
                                   for p in sorted(output.iterdir()) if p.is_file() and p.name != 'campaign.json']}
        write_manifest(output / 'inventory.json', inventory)
        completion = completion_status(manifest, read_events(output / 'metrics.jsonl'), inventory, output)
        write_manifest(output / 'completion.json', completion)
        if completion['status'] != 'complete':
            raise ValueError('pilot training records incomplete')
        freeze = {'schema': 'encoder-original6-freeze-v1', 'status': 'complete',
                  'prepared_identity': metadata['prepared_identity'], 'checkpoint': campaign['checkpoints'][-1],
                  'final_panels': [list(panel) for panel in final_panels()], 'vanilla_after_codec_panels': 6,
                  'evidence': {name: {'path': str((output / name).resolve()), 'sha256': file_digest(output / name)}
                               for name in ('manifest.json', 'inventory.json', 'completion.json', 'metrics.jsonl')},
                  'per_site_counts': counts}
        write_manifest(output / 'freeze.json', freeze)
        guard()
        campaign.update(status='training_frozen', completed_updates=learner.completed,
                        elapsed_seconds=time.monotonic() - begun,
                        freeze_reference={'path': str((output / 'freeze.json').resolve()),
                                          'sha256': file_digest(output / 'freeze.json')})
        write_manifest(output / 'campaign.json', campaign)
        return campaign
    except Exception as error:
        campaign.update(completed_updates=learner.completed, elapsed_seconds=time.monotonic() - begun,
                        error={'type': type(error).__name__, 'message': str(error)})
        write_manifest(output / 'campaign.json', campaign)
        raise


def open_terminal(training_run, metadata, plan):
    root = Path(training_run).resolve()
    campaign = json.loads((root / 'campaign.json').read_text())
    freeze_path = root / 'freeze.json'
    if (campaign['status'] != 'training_frozen' or campaign['completed_updates'] != plan.horizon
            or campaign['prepared_identity'] != metadata['prepared_identity']
            or campaign['freeze_reference'] != {'path': str(freeze_path), 'sha256': file_digest(freeze_path)}):
        raise ValueError('pilot training is not completely frozen for this prepared package')
    freeze = json.loads(freeze_path.read_text())
    if (freeze['schema'] != 'encoder-original6-freeze-v1' or freeze['status'] != 'complete'
            or freeze['prepared_identity'] != metadata['prepared_identity']
            or freeze['final_panels'] != [list(panel) for panel in final_panels()]
            or freeze['vanilla_after_codec_panels'] != 6):
        raise ValueError('pilot frozen final panel plan differs')
    for reference in freeze['evidence'].values():
        path = Path(reference['path']).resolve()
        if not path.is_relative_to(root) or file_digest(path) != reference['sha256']:
            raise ValueError('pilot freeze evidence bytes changed')
    inventory = json.loads((root / 'inventory.json').read_text())
    verify_inventory(root, inventory)
    manifest = json.loads((root / 'manifest.json').read_text())
    if completion_status(manifest, read_events(root / 'metrics.jsonl'), inventory, root)['status'] != 'complete':
        raise ValueError('pilot training records no longer complete')
    reference = CheckpointRef(**freeze['checkpoint'])
    if Path(reference.path).resolve() != root / f'step_{plan.horizon:06d}.pt':
        raise ValueError('pilot terminal checkpoint path differs')
    state = open_state(reference, expected=metadata)
    if state.payload['metadata']['completed_updates'] != plan.horizon:
        raise ValueError('pilot terminal checkpoint count differs')
    return state, campaign['freeze_reference']


def run_evaluation(model, processor, *, state, freeze_reference, template, output, guard):
    from . import evaluation
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    details = state.payload['metadata']
    result = {'schema': 'encoder-original6-evaluation-v1', 'status': 'incomplete',
              'checkpoint': asdict(state.reference), 'freeze_reference': freeze_reference,
              'interpretation': 'already inspected development; all six sites trained; no untouched-test claim',
              'completed_panels': [], 'vanilla_reference': None}
    begun = time.monotonic()
    write_manifest(output / 'result.json', result)
    try:
        for index, (site, condition) in enumerate(final_panels()):
            guard()
            if index == 6:
                path = output / 'vanilla'
                path.mkdir()
                settings = {**template['settings'], 'evidence_mode': 'compact-v1',
                            'noise_seed': template['noise']['seed'], 'noise_namespace': template['noise']['namespace']}
                with isolated_rng(settings['noise_seed']), torch.no_grad(), model.transmission(bypass=True):
                    model.eval()
                    metrics = evaluation.evaluate_hellaswag(model, processor, settings, path, 'vanilla', template['data_settings'])
                rows, ids, families = evaluation._observation_items(path, 'hellaswag', 'vanilla')
                if (len(rows) != template['expected_items'] or ids != template['input_ids']
                        or families != template['source_family_ids'] or any(model.channel_uses.values())):
                    raise ValueError('pilot vanilla membership or bypass differs')
                receipt = {'status': 'complete', 'metrics': metrics, 'request': template,
                           'completed': len(rows), 'transmission': 'bypass',
                           'output_hashes': {p.name: file_digest(p) for p in path.iterdir() if p.is_file()}}
                write_manifest(path / 'receipt.json', receipt)
                result['vanilla_reference'] = {'path': str((path / 'receipt.json').resolve()),
                                               'sha256': file_digest(path / 'receipt.json'),
                                               'metrics': metrics, 'completed_items': len(rows)}
                write_manifest(output / 'result.json', result)
                guard()
            request = {**template, 'schema': 'codec-observation-v1', 'conditions': [condition],
                'checkpoint_sha256': state.reference.sha256, 'target_site': details['evaluation_sites'][site]['site'],
                'target_role': 'trained', 'learner_kind': 'shared_encoder_v2', 'step': details['completed_updates'],
                'site_step': details['sharing']['completed_per_site'][site], 'parent': None,
                'comparison': details['comparison_controls'], 'panel': f'final:{site}:{condition}',
                'source': details['source_identity'], 'config': details['config_identity'],
                'data': details['data_identity'], 'protocol': details['protocol_identity']}
            request['settings'] = {**request['settings'], 'evidence_mode': 'compact-v1'}
            path = output / f'{site}_{condition}'
            receipt = evaluate_checkpoint(state, request, model=model, processor=processor, output=path)
            validate_task_receipt(receipt, request)
            verify_observation_artifacts(receipt, path)
            guard()
            result['completed_panels'].append({'site': site, 'condition': condition,
                'checkpoint': state.reference.sha256, 'completed_items': receipt['conditions'][0]['completed'],
                'metrics': receipt['conditions'][0]['metrics'],
                'observation': {'path': str((path / 'observation.json').resolve()),
                                'sha256': file_digest(path / 'observation.json')}})
            result['elapsed_seconds'] = time.monotonic() - begun
            write_manifest(output / 'result.json', result)
            print(json.dumps({'phase': 'evaluation', 'completed_codec_panels': len(result['completed_panels']),
                              'site': site, 'condition': condition, 'elapsed_seconds': result['elapsed_seconds']}), flush=True)
        guard()
        if result['vanilla_reference'] is None or len(result['completed_panels']) != 36:
            raise ValueError('pilot final panel obligations incomplete')
        result.update(status='complete', elapsed_seconds=time.monotonic() - begun)
        write_manifest(output / 'result.json', result)
        return result
    except Exception as error:
        result.update(error={'type': type(error).__name__, 'message': str(error)}, elapsed_seconds=time.monotonic() - begun)
        write_manifest(output / 'result.json', result)
        raise


def initialize(config, initial):
    configure_training_determinism(config['training'])
    seed_everything(config['seed'])
    processor, model = build_model(copy.deepcopy(config))
    model.codec.load_state_dict(initial, strict=True)
    if state_dict_identity(model.codec.state_dict()) != state_dict_identity(initial):
        raise ValueError('pilot exact initial codec load mismatch')
    return processor, model


def readiness_view_indices(batches):
    """Test both extreme layouts, taking the larger companion length on ties."""
    lengths = [(batch['input_ids'].shape[1], batch['labels'].shape[1]) for batch in batches]
    return {max(range(len(batches)), key=lambda i: lengths[i]),
            max(range(len(batches)), key=lambda i: tuple(reversed(lengths[i])))}


def readiness(model, *, plan, batches, metadata, output, guard):
    """Isolated six-update cycles on both actual worst layouts; no science updates."""
    from .sharing_qualification_guard import restore, snapshot, state_exact
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    chosen = readiness_view_indices(batches)
    result = {'status': 'incomplete', 'scientific_updates': 0, 'cycles': []}
    begun = time.monotonic()
    write_manifest(output / 'result.json', result)
    try:
        for index in sorted(chosen):
            reordered = [batches[index], *batches[:index], *batches[index + 1:]]
            diagnostic = EncoderPilotPlan(tuple(batch_view(b) for b in reordered),
                                          plan.study_pairing_id + ':readiness', synthetic=plan.synthetic)
            learner = make_learner(model, diagnostic, metadata)
            learner.event_sink = AppendOnlyEventWriter(output / f'view{index}-metrics.jsonl').append
            before = snapshot(learner)
            rows = []
            for site in SITES:
                guard()
                if next(model.base.parameters()).is_cuda:
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                started = time.monotonic()
                learner.update([reordered[0]])
                if next(model.base.parameters()).is_cuda:
                    torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() if next(model.base.parameters()).is_cuda else 0
                rows.append({'site': site, 'seconds': time.monotonic() - started, 'peak_allocated_bytes': peak})
                print(json.dumps({'phase': 'readiness', 'view': index, **rows[-1]}), flush=True)
            guard()
            saved = snapshot(learner)
            state = save_pilot(learner, metadata, output / f'view{index}-state.pt')
            if not state_exact(saved, snapshot(learner)):
                raise ValueError('readiness save/reload changed full learner state')
            restore(learner, before)
            if not state_exact(before, snapshot(learner)):
                raise ValueError('readiness fresh-state restoration failed')
            result['cycles'].append({'view': index, 'source_length': batches[index]['input_ids'].shape[1],
                                     'target_length': batches[index]['labels'].shape[1], 'sites': rows,
                                     'checkpoint': asdict(state.reference), 'fresh_state_restored_exact': True})
            write_manifest(output / 'result.json', result)
        guard()
        result.update(status='pass', elapsed_seconds=time.monotonic() - begun)
        write_manifest(output / 'result.json', result)
        return result
    except Exception as error:
        result.update(error={'type': type(error).__name__, 'message': str(error)}, elapsed_seconds=time.monotonic() - begun)
        write_manifest(output / 'result.json', result)
        raise
