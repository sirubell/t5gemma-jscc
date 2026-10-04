"""Original13 training and continuation orchestration."""
from dataclasses import asdict
import copy
from pathlib import Path

from .activation_replay import canonical_digest, file_digest
from .baseline_protocol import write_manifest
from .experiment_records import AppendOnlyEventWriter, read_events
from .experiment_state import open_state
from .original13_state import Original13TrainingState
from .sharing_protocol import batch_view


def run_original13_training(model, processor, *, plan, batches, metadata, observation_request,
                            output, guard, routing=None, observe=None, resume=None, config=None, data_ids=None):
    """Fresh segment or explicit continuation; no automatic retry or model loading."""
    from . import original13_routing
    from .baseline_protocol import bind_batch_identity
    from .encoder_stages_evaluation import ObservationScoringError, observe_original13
    from .encoder_stages_plan import original13_progress
    from .experiment_state import capture_rng, restore_rng
    routing = original13_routing if routing is None else routing
    observe = observe_original13 if observe is None else observe
    if not plan.synthetic and (config is None or data_ids is None):
        raise ValueError('production original13 requires resolved configuration and data IDs')
    if config is not None and canonical_digest(config) != metadata['config_identity']:
        raise ValueError('original13 run configuration identity differs')
    if data_ids is not None and canonical_digest(data_ids) != metadata['data_identity']:
        raise ValueError('original13 run data membership differs')
    if (len(batches) != plan.q or tuple(batch_view(b) for b in batches) != plan.views
            or canonical_digest(observation_request['panel']) != plan.observation_identity):
        raise ValueError('original13 verified inputs/observation panel differ')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = {'schema': 'original13-training-v1', 'status': 'incomplete', 'ratio': plan.ratio, 'width': plan.width,
              'plan_identity': plan.identity, 'completed_updates': 0, 'checkpoints': [],
              'observations': [{'step': step, 'state_role': role, 'status': 'pending',
                                'requires_explicit_completion': False} for step, role in plan.observations],
              'segment_parent': None if resume is None else asdict(resume)}
    write_manifest(output / 'campaign.json', result)
    learner = None
    try:
        if config is not None:
            from .config import save_config
            save_config(config, output / 'config.yaml')
        if data_ids is not None:
            write_manifest(output / 'data_ids.json', data_ids)
        learner = routing.create_original13_learner(model, plan=plan, metadata=metadata,
                                                   event_sink=AppendOnlyEventWriter(output / 'metrics.jsonl').append)
        state_machine = Original13TrainingState(learner, plan=plan, metadata=metadata, routing=routing)
        if resume is not None:
            restored = state_machine.restore(resume)
            for observation in result['observations']:
                if observation['step'] <= learner.completed:
                    observation.update(status='pending_prior_segment', requires_explicit_completion=True)
            if learner.completed == plan.reconstruction_stop and learner.phase_start_step == 0:
                state_machine.reset(restored)
                reset = state_machine.save(output / f'step_{learner.completed:06d}_post_reset.pt')
                result['checkpoints'].append({'role': 'post_reset', **asdict(reset.reference)})
        initial_completed = learner.completed
        bindings = [bind_batch_identity(b, expected_identity=v.view_sha256) for b, v in zip(batches, plan.views)]
        while True:
            guard()
            step = learner.completed
            for save_step, role in plan.checkpoint_states:
                if save_step != step or (role == 'post_reset') or (resume is not None and step == initial_completed):
                    continue
                state = state_machine.save(output / f'step_{step:06d}_{role}.pt')
                result['checkpoints'].append({'role': role, **asdict(state.reference)})
                if (step, role) in plan.observations:
                    from .sharing_qualification_guard import state_exact
                    saved_rng = capture_rng()
                    def training_state():
                        return copy.deepcopy({'weights': routing.export_codec(model),
                            'optimizer': learner.optimizer.state_dict(), 'scheduler': learner.scheduler.state_dict(),
                            'scaler': learner.scaler.state_dict() if learner.scaler is not None else None,
                            'counters': [learner.completed, learner.attempted, learner.offset, learner.valid_tokens,
                                         learner.phase_start_step, learner.failed],
                            'exposure': [learner.exposure.completed, learner.exposure.failed,
                                         learner.exposure.per_site, learner.exposure.receipts],
                            'grads': [None if p.grad is None else p.grad.detach().clone() for p in learner.parameters],
                            'route': routing.observation_state(model)})
                    saved_training = training_state()
                    request = {**observation_request, 'plan_identity': plan.identity, 'step': step,
                               'state_role': role, 'checkpoint_sha256': state.reference.sha256}
                    error = None
                    try:
                        receipt = observe(model, processor, state=state, request=request,
                                          output=output / f'observation_{step:06d}', guard=guard)
                        if receipt.get('status') != 'complete' or receipt.get('request_identity') != canonical_digest(request):
                            raise ValueError('observation receipt identity/status differs')
                    except ObservationScoringError as exc:
                        error = {'type': type(exc).__name__, 'message': str(exc)}
                    finally:
                        restore_rng(saved_rng)
                    # Consumer restores routes/noise; revalidate checkpoint and full
                    # training state before continuing even an isolated scoring error.
                    open_state(state.reference, expected=state.payload['metadata'])
                    if not state_exact(training_state(), saved_training) or not state_exact(capture_rng(), saved_rng):
                        raise ValueError('observation training state restoration differs')
                    row = next(r for r in result['observations'] if r['step'] == step)
                    row.update(request_identity=canonical_digest(request),
                        status='failed' if error else 'complete', error=error,
                        requires_explicit_completion=bool(error))
                    receipt_path = output / f'observation_{step:06d}' / 'result.json'
                    if receipt_path.is_file():
                        row['receipt'] = {'path': str(receipt_path), 'sha256': file_digest(receipt_path)}
                    write_manifest(output / 'campaign.json', result)
                if role == 'pre_reset':
                    state_machine.reset(state)
                    reset = state_machine.save(output / f'step_{step:06d}_post_reset.pt')
                    result['checkpoints'].append({'role': 'post_reset', **asdict(reset.reference)})
            if step == plan.horizon:
                break
            expected = plan.update_at(step)
            used = [g['lr'] for g in learner.optimizer.param_groups]
            event = learner.update([batches[expected.site_local_index]],
                                   batch_identities=[bindings[expected.site_local_index]])
            if (learner.completed != step + 1 or learner.exposure.completed != learner.completed
                    or any(lr != 2e-4 * expected.lr_factor for lr in used)
                    or event['payload']['lr_used'] != used):
                raise ValueError('original13 update progress or consumed LR differs')
            if learner.completed % (len(plan.trained_sites) * 100) == 0:
                result['completed_updates'] = learner.completed
                write_manifest(output / 'campaign.json', result)
        counts = learner.exposure.finalize()
        events = [e for e in read_events(output / 'metrics.jsonl') if e['event_type'] == 'update']
        if [e['payload']['completed_step'] for e in events] != list(range(initial_completed + 1, plan.horizon + 1)):
            raise ValueError('original13 segment scalar log incomplete')
        terminal = next((c for c in result['checkpoints'] if c['role'] == 'terminal'), None)
        if terminal is None:
            raise ValueError('continuing an already-terminal state requires evaluation, not training')
        freeze = {'schema': 'original13-freeze-v1', 'status': 'training_frozen', 'plan_identity': plan.identity,
                  'checkpoint': {k: terminal[k] for k in ('path', 'sha256', 'size')}, 'counts': counts,
                  'progress': original13_progress(plan, learner.completed, learner.phase_start_step),
                  'segment_ancestry': state_machine.ancestry,
                  'metrics': {'path': str(output / 'metrics.jsonl'), 'sha256': file_digest(output / 'metrics.jsonl')}}
        write_manifest(output / 'freeze.json', freeze)
        result.update(status='training_frozen', completed_updates=learner.completed,
                      segment_updates=len(events), observations_status='pending' if any(
                          r['status'] != 'complete' for r in result['observations']) else 'complete',
                      freeze_reference={'path': str(output / 'freeze.json'), 'sha256': file_digest(output / 'freeze.json')})
        write_manifest(output / 'campaign.json', result)
        return result
    except Exception as exc:
        result.update(completed_updates=0 if learner is None else learner.completed,
                      error={'type': type(exc).__name__, 'message': str(exc)})
        write_manifest(output / 'campaign.json', result)
        raise

