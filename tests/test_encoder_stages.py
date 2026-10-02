"""Actual staged optimizer/checkpoint/data lifecycles on a tiny frozen model."""
from copy import deepcopy

import pytest
import torch

from jscc.encoder_stages import (CONDITIONS, EncoderStagesLearner, open_stages_terminal,
                                run_stages_evaluation, run_stages_training, save_stages)
from jscc.encoder_stages_plan import ENCODER_STAGES_PROTOCOL, EncoderStagesPlan
from jscc.experiment_records import read_events
from jscc.experiment_state import restore_state
from jscc.sharing_preparation import state_dict_identity
from jscc.sharing_protocol import batch_view
from test_encoder_shared_pilot import fixture
from test_shared_lifecycle import install_adapter, task_template


def stages_fixture(arm):
    model, old, batches, metadata = fixture()
    width = int(model.codec.config['bottleneck_dim'])
    plan = EncoderStagesPlan(tuple(batch_view(b) for b in batches), old.study_pairing_id,
                             arm, synthetic=True, width=width)
    metadata.update(protocol_identity=ENCODER_STAGES_PROTOCOL, phase='phase1')
    return model, plan, batches, metadata


@pytest.mark.parametrize('arm', ['A', 'E', 'R'])
def test_actual_full_stages_and_own_terminal_checkpoint(tmp_path, arm):
    model, plan, batches, metadata = stages_fixture(arm)
    output = tmp_path / arm
    result = run_stages_training(model, plan=plan, batches=batches, metadata=metadata,
                                 output=output, guard=lambda: None)
    assert result['status'] == 'training_frozen'
    assert result['scalar_updates'] == 24
    state, _ = open_stages_terminal(output, metadata, plan)
    assert state.payload['metadata']['sharing']['completed_per_site'] == dict.fromkeys(plan.trained_sites, 4)
    assert state.payload['scheduler']['last_epoch'] == (24 if arm == 'A' else 12)
    events = [e for e in read_events(output / 'metrics.jsonl') if e['event_type'] == 'update']
    assert len(events) == 24
    assert [e['payload']['site_id'] for e in events] == list(plan.trained_sites) * 4
    assert all(not e['payload']['nonfinite'] and not e['payload']['skipped'] for e in events)
    if arm == 'R':
        assert all(e['payload']['objective']['kind'] == 'local' for e in events[:12])
        assert all(e['payload']['objective']['components']['K']['raw']['value'] is None for e in events[:12])
    assert all(e['payload']['objective']['kind'] == 'combined' for e in events[12:])
    assert (result['phase_reset'] is None) == (arm == 'A')


def test_reset_retains_global_stream_and_only_resets_optimizer(tmp_path):
    model, plan, batches, metadata = stages_fixture('R')
    learner = EncoderStagesLearner(model, plan=plan, metadata=metadata)
    for index in range(plan.horizon // 2):
        learner.update([batches[index // 6]])
    state = save_stages(learner, metadata, tmp_path / 'phase1.pt')
    codec = state_dict_identity(model.codec.state_dict())
    progress = (learner.completed, learner.attempted, learner.offset, learner.valid_tokens)
    exposure = deepcopy(learner.exposure.per_site)
    rng = torch.get_rng_state().clone()
    next_update = plan.update_at(learner.completed)
    learner.reset_phase(state)
    assert state_dict_identity(model.codec.state_dict()) == codec
    assert (learner.completed, learner.attempted, learner.offset, learner.valid_tokens) == progress
    assert learner.exposure.per_site == exposure and torch.equal(torch.get_rng_state(), rng)
    assert learner.optimizer.state == {} and learner.scheduler.last_epoch == 0
    assert learner.optimizer.param_groups[0]['lr'] == 0
    reset = save_stages(learner, metadata, tmp_path / 'reset.pt')
    assert reset.payload['metadata']['parent_identity'] == state.reference.sha256
    assert reset.payload['metadata']['lineage'][-1]['parent_checkpoint']['sha256'] == state.reference.sha256
    restore_state(reset, model=model.codec, optimizer=learner.optimizer, scheduler=learner.scheduler, scaler=learner.scaler)
    assert plan.update_at(learner.completed) == next_update
    learner.update([batches[2]])
    assert learner.completed == 13 and learner.scheduler.last_epoch == 1
    with pytest.raises(ValueError, match='reset requires'):
        learner.reset_phase(state)


def test_e_r_phase_two_matches_actual_noise_draws():
    draws = []
    for arm in ('E', 'R'):
        model, plan, batches, metadata = stages_fixture(arm)
        learner = EncoderStagesLearner(model, plan=plan, metadata=metadata)
        # Simulate only the independently CPU-verified phase entry to inspect the actual draw path.
        learner.completed = learner.attempted = 12
        learner.offset = 12 * 16
        learner.phase_start_step = 12
        learner._new_optimizer()
        learner.exposure.completed = 12
        learner.update([batches[2]])
        draws.append(learner.last_draws)
    assert draws[0] == draws[1]


def test_terminal_thirty_panels_use_one_frozen_staged_checkpoint(tmp_path, monkeypatch):
    calls = install_adapter(monkeypatch)
    model, plan, batches, metadata = stages_fixture('R')
    run_stages_training(model, plan=plan, batches=batches, metadata=metadata,
                        output=tmp_path / 'train', guard=lambda: None)
    state, freeze = open_stages_terminal(tmp_path / 'train', metadata, plan)
    template = task_template()
    template['conditions'] = list(CONDITIONS)
    template['backend'] = model.base.config.decoder._attn_implementation
    result = run_stages_evaluation(model, object(), state=state, freeze_reference=freeze,
                                   template=template, output=tmp_path / 'evaluate', guard=lambda: None)
    assert result['status'] == 'complete' and len(calls) == 30
    assert [p['condition'] for p in result['completed_panels']] == [c for c in CONDITIONS for _ in range(6)]
    assert {p['checkpoint'] for p in result['completed_panels']} == {state.reference.sha256}
    assert all(p['completed_items'] == 2 for p in result['completed_panels'])
