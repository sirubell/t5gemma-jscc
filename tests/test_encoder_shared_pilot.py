"""Actual six-site update/state/scoring lifecycle, without pretrained weights."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest
import torch

from jscc.encoder_pilot_plan import EncoderPilotPlan, SITE_SPECS, SITES, build_pilot_state, final_panels
from jscc.activation_replay import canonical_digest
from jscc.encoder_shared_pilot import (make_learner, open_terminal, readiness,
                                      readiness_view_indices, run_evaluation, run_training)
from jscc.experiment_records import read_events
from jscc.models.split_model import resolve_encoder_site
from jscc.sharing_protocol import batch_view
from jscc.sharing_schedule import ENCODER_PILOT_PROTOCOL, NATIVE16_WEIGHT_DECAY
from jscc.sharing_state import validate_sharing_state
from sharing_fixtures import sharing_model, sharing_plan
from test_shared_lifecycle import install_adapter, task_template
from test_sharing_native16 import native_batches


def fixture():
    torch.set_num_threads(1)
    batches = native_batches()
    for batch in batches:
        batch['source_family_ids'] = [f'train-{int(i)}' for i in batch['row_ids']]
    plan = EncoderPilotPlan(tuple(batch_view(b) for b in batches), 'six-pilot-cpu', synthetic=True)
    metadata = {'source_identity': 'source', 'config_identity': 'config', 'data_identity': 'data',
                'parent_identity': None, 'initialization_identity': 'init', 'stream_identity': 'stream',
                'protocol_identity': ENCODER_PILOT_PROTOCOL, 'prepared_identity': 'prepared',
                'model_revision': 'tiny-cpu', 'final_task_identity': 'task', 'lineage': [], 'phase': 'both'}
    return sharing_model('D-N'), plan, batches, metadata


def test_fixed_schedule_and_balanced_exposure():
    _, small, _, _ = fixture()
    views = tuple(replace(small.views[i % 4], view_sha256=canonical_digest(i)) for i in range(400))
    plan = EncoderPilotPlan(views, 'production-plan')
    assert plan.horizon == 2400 and plan.checkpoint_steps == (0, 1200, 2400)
    for step in (0, 1, 5, 6, 1200, 2400):
        sharing, stream = build_pilot_state(plan, step)
        assert sum(sharing['completed_per_site'].values()) == step
        assert stream['offset'] == step * 16
        assert max(sharing['completed_per_site'].values()) - min(sharing['completed_per_site'].values()) <= 1
    assert build_pilot_state(plan, 2400)[0]['completed_per_site'] == dict.fromkeys(SITES, 400)
    assert [plan.update_at(i).site for i in range(12)] == list(SITES) * 2
    assert plan.update_at(2399).lr_factor > 0
    with pytest.raises(ValueError, match='horizon'):
        plan.update_at(2400)
    with pytest.raises(ValueError, match='original six'):
        EncoderPilotPlan(views, 'x', trained_sites=tuple(reversed(SITES)))


def test_long_schedule_balances_same_stream_at_all_retained_states():
    _, small, _, _ = fixture()
    views = tuple(replace(small.views[i % 4], view_sha256=canonical_digest(i)) for i in range(6400))
    plan = EncoderPilotPlan(views, 'production-long')
    assert plan.horizon == 38400
    assert plan.checkpoint_steps == (0, 9600, 19200, 28800, 38400)
    assert plan.objective_steps == (0, 19200, 38400)
    assert plan.update_at(1920).lr_factor == 1
    assert plan.update_at(38399).lr_factor > 0
    assert [plan.update_at(i).site for i in range(12)] == list(SITES) * 2
    for step in plan.checkpoint_steps:
        sharing, stream = build_pilot_state(plan, step)
        assert sharing['completed_per_site'] == dict.fromkeys(SITES, step // 6)
        assert stream['offset'] == step * 16
        assert sharing['source_valid_per_site'] == dict.fromkeys(
            SITES, sum(v.source_tokens for v in views[:step // 6]))
    with pytest.raises(ValueError, match='horizon'):
        plan.update_at(38400)
    with pytest.raises(ValueError, match='original six'):
        EncoderPilotPlan(views[:-1], 'unapproved-count')


def test_readiness_ties_use_larger_companion_length():
    batches = [{'input_ids': torch.empty((16, source)), 'labels': torch.empty((16, target))}
               for source, target in [(684, 53), (684, 63), (544, 96), (600, 96)]]
    assert readiness_view_indices(batches) == {1, 3}


def test_embedding_insertion_is_actual_first_layer_input_and_restores():
    model, _, batches, _ = fixture()
    site = resolve_encoder_site(model.base, SITE_SPECS['enc_emb'], 'tiny-cpu')
    assert site.site_id == 'enc_emb' and site.module_path == 'encoder.layers.0.input'
    captured = []
    def capture(module, args, kwargs):
        captured.append((args[0] if args else kwargs['hidden_states']).detach().clone())
    first = model.base.get_encoder().text_model.layers[0]
    handle = first.register_forward_pre_hook(capture, with_kwargs=True, prepend=True)
    original = dict(model.split)
    try:
        with model.at_site(site, purpose='train', authorization={'sites': {'enc_emb': 'trained'}}):
            batch = batches[0]
            with model.transmission(None, encoder_mask=batch['attention_mask']):
                model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            assert model.activation is not None
            assert torch.equal(model.activation, captured[-1])
            raise RuntimeError('exercise finally')
    except RuntimeError as error:
        assert str(error) == 'exercise finally'
    finally:
        handle.remove()
    assert model.split == original and not model._routing_active
    with pytest.raises(ValueError, match='no layer index'):
        resolve_encoder_site(model.base, {**SITE_SPECS['enc_emb'], 'index': 0}, 'tiny-cpu')


@pytest.mark.parametrize('quarter_checkpoints', [False, True])
def test_actual_single_learner_lifecycle_and_terminal_panels(tmp_path, monkeypatch, quarter_checkpoints):
    calls = install_adapter(monkeypatch)
    model, plan, batches, metadata = fixture()
    if quarter_checkpoints:
        monkeypatch.setattr(EncoderPilotPlan, 'checkpoint_steps', property(
            lambda p: tuple(p.horizon * i // 4 for i in range(5))))
    output = tmp_path / 'train'
    result = run_training(model, plan=plan, batches=batches, validation=batches[:1],
                          metadata=metadata, output=output, guard=lambda: None)
    assert result['status'] == 'training_frozen' and result['completed_updates'] == 24
    assert len(result['checkpoints']) == (5 if quarter_checkpoints else 3)
    assert len(result['objective_observations']) == 18
    assert calls == []
    events = read_events(output / 'metrics.jsonl')
    updates = [e['payload'] for e in events if e['event_type'] == 'update']
    assert [u['site_id'] for u in updates] == list(SITES) * 4
    state, freeze = open_terminal(output, metadata, plan)
    validate_sharing_state(state.payload)
    assert state.payload['metadata']['sharing']['completed_per_site'] == dict.fromkeys(SITES, 4)
    assert all(d['role'] == 'trained' for d in state.payload['metadata']['evaluation_sites'].values())
    terminal_objective = json.loads((output / 'objective_000024_enc_emb.json').read_text())
    assert terminal_objective['request']['comparison'] == state.payload['metadata']['comparison_controls']
    assert state.payload['optimizer']['param_groups'][0]['weight_decay'] == NATIVE16_WEIGHT_DECAY
    assert state.payload['optimizer']['param_groups'][0]['lr'] == 0
    template = task_template()
    template['backend'] = model.base.config.decoder._attn_implementation
    result = run_evaluation(model, object(), state=state, freeze_reference=freeze, template=template,
                            output=tmp_path / 'evaluate', guard=lambda: None)
    assert result['status'] == 'complete' and len(result['completed_panels']) == 36
    assert len({p['checkpoint'] for p in result['completed_panels']}) == 1
    assert [condition for _, condition in calls] == ['no_noise'] * 6 + ['vanilla'] + [c for c in (-6, 0, 6, 12, 18) for _ in SITES]
    assert result['vanilla_reference']['completed_items'] == 2
    from jscc.evaluation import observation_record_request
    request = json.loads((tmp_path / 'evaluate/enc_emb_no_noise/observation.json').read_text())['request']
    assert observation_record_request(request)['site'] == 'enc_emb'
    (output / 'metrics.jsonl').write_text('')
    with pytest.raises(ValueError, match='bytes changed'):
        open_terminal(output, metadata, plan)


@pytest.mark.parametrize('mutation', [
    lambda p: p['metadata']['sharing']['completed_per_site'].update(enc_emb=0),
    lambda p: p['metadata']['sharing'].update(next_site='enc_fn'),
    lambda p: p['metadata']['evaluation_sites']['enc_l14'].update(role='heldout_after_freeze'),
    lambda p: p['metadata']['evaluation_sites']['enc_emb']['site'].update(module_path='encoder.embed_tokens'),
    lambda p: p['metadata']['sharing'].update(schema='sharing-state-v1'),
])
def test_saved_pilot_drift_rejected(tmp_path, mutation):
    from jscc.encoder_shared_pilot import save_pilot
    model, plan, batches, metadata = fixture()
    learner = make_learner(model, plan, metadata)
    learner.update([batches[0]])
    saved = save_pilot(learner, metadata, tmp_path / 'state.pt')
    damaged = deepcopy(saved.payload)
    mutation(damaged)
    with pytest.raises(ValueError):
        validate_sharing_state(damaged)


def test_readiness_discards_cycles_and_restores_exact_initial_state(tmp_path):
    model, plan, batches, metadata = fixture()
    initial = deepcopy(model.codec.state_dict())
    result = readiness(model, plan=plan, batches=batches, metadata=metadata,
                       output=tmp_path / 'readiness', guard=lambda: None)
    assert result['status'] == 'pass' and result['scientific_updates'] == 0
    assert all([row['site'] for row in cycle['sites']] == list(SITES) for cycle in result['cycles'])
    assert all(cycle['fresh_state_restored_exact'] for cycle in result['cycles'])
    assert all(torch.equal(model.codec.state_dict()[k], v) for k, v in initial.items())


def test_formal_layer14_still_requires_freeze():
    from jscc.sharing_protocol import SharingLearner
    model, _, batches, metadata = fixture()
    formal = SharingLearner(model, schedule=sharing_plan().learner('shared'), run_id='formal',
                           identity={'source': 's', 'config': 'c', 'data': 'd', 'parent': None},
                           model_revision=metadata['model_revision'])
    with pytest.raises(ValueError, match='freeze'):
        formal.validate_site(batches[:1], 'enc_l14', authorization={'sites': {'enc_l14': 'trained'}})


def test_deadline_cannot_publish_training_or_final_success(tmp_path, monkeypatch):
    model, plan, batches, metadata = fixture()
    def expired():
        raise TimeoutError('test deadline')
    with pytest.raises(TimeoutError):
        run_training(model, plan=plan, batches=batches, validation=batches[:1], metadata=metadata,
                     output=tmp_path / 'late', guard=expired)
    assert json.loads((tmp_path / 'late/campaign.json').read_text())['status'] == 'incomplete'
    assert not (tmp_path / 'late/freeze.json').exists()
    assert not torch.cuda.is_initialized()


def test_cpu_cli_accepts_real_path_argument_without_constructing_model(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    from jscc.activation_replay import file_digest
    source = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('encoder_pilot_entry_test', source / 'scripts/encoder_shared_pilot.py')
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    path = tmp_path / 'prepared.json'
    path.write_text('{}')
    _, plan, _, metadata = fixture()
    manifest = {'source_inventory': {}, 'manifest_identity': metadata['prepared_identity'],
                'source_identity': metadata['source_identity'], 'config_identity': metadata['config_identity'],
                'data_ids': {}, 'initialization': {'state_identity': 'initial'}, 'updates': [],
                'model_revision': 'tiny-cpu', 'task_request': {'sha256': 'task'}}
    monkeypatch.setattr(cli, 'load_pilot', lambda path: (manifest, {}, plan, [], [], {}, {}))
    monkeypatch.setattr(cli.subprocess, 'check_output', lambda argv, **kwargs: 'cpu-test' if 'rev-parse' in argv else '')
    monkeypatch.setattr(cli.subprocess, 'run', lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, 'initialize', lambda *args: pytest.fail('CPU check constructed model'))
    result = cli.main(['check', '--prepared', str(path), '--prepared-sha', file_digest(path),
                       '--approved-source', 'cpu-test'])
    assert result['status'] == 'CPU_verified' and result['GPU_launched'] is False
    command = ['train', '--prepared', str(path), '--prepared-sha', file_digest(path),
               '--approved-source', 'cpu-test', '--output', str(tmp_path / 'train')]
    with pytest.raises(ValueError, match='four-hour'):
        cli.main([*command, '--timeout-seconds', '14401'])
    views = tuple(replace(plan.views[i % 4], view_sha256=canonical_digest(i)) for i in range(6400))
    long_plan = EncoderPilotPlan(views, 'production-long')
    manifest.update(synthetic_cpu=False, device_uuid='approved-UUID')
    monkeypatch.setattr(cli, 'load_pilot', lambda path: (manifest, {}, long_plan, [], [], {}, {}))
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    with pytest.raises(ValueError, match='GPU assignment'):
        cli.main([*command, '--timeout-seconds', '21600'])
    with pytest.raises(ValueError, match='per-stage timeout'):
        cli.main([*command, '--timeout-seconds', '21601'])


def test_full_validation_scope_rejects_subset_or_extra_condition():
    from jscc.sharing_preparation import validate_task_template
    template = task_template()
    template.update(expected_items=10042, input_ids=list(range(10042)),
                    source_family_ids=[f'family-{i}' for i in range(10042)], conditions=['no_noise'])
    template['settings'].update(num_samples=10042, batch_size=16, num_fewshot=5, scoring_policy='fp32-v1')
    config = {'model': {'dtype': 'float32', 'attn_implementation': 'eager'}}
    validate_task_template(template, config, synthetic=False, final=True, no_noise_only=True)
    assert final_panels(6400) == tuple((site, 'no_noise') for site in SITES)
    assert len(final_panels(400)) == 36
    with pytest.raises(ValueError, match='conditions differ'):
        validate_task_template({**template, 'conditions': ['no_noise', 6]}, config,
                               synthetic=False, final=True, no_noise_only=True)
    with pytest.raises(ValueError, match='full ordered'):
        validate_task_template({**template, 'input_ids': list(reversed(template['input_ids']))}, config,
                               synthetic=False, final=True, no_noise_only=True)
    with pytest.raises(ValueError, match='fixed HellaSwag panel'):
        validate_task_template({**template, 'expected_items': 256}, config,
                               synthetic=False, final=True, no_noise_only=True)


def test_no_noise_only_lifecycle_and_evaluation_failure_preserves_freeze(tmp_path, monkeypatch):
    calls = install_adapter(monkeypatch)
    model, plan, batches, metadata = fixture()
    output = tmp_path / 'train'
    run_training(model, plan=plan, batches=batches, validation=batches[:1], metadata=metadata,
                 output=output, guard=lambda: None)
    state, freeze = open_terminal(output, metadata, plan)
    before = (output / 'freeze.json').read_bytes()
    # Exercise the real six-panel evaluator with the CPU q4 checkpoint and scorer adapter.
    monkeypatch.setattr('jscc.encoder_shared_pilot.final_panels', lambda q: final_panels(6400))
    template = task_template()
    template.update(backend=model.base.config.decoder._attn_implementation, conditions=['no_noise'])
    result = run_evaluation(model, object(), state=state, freeze_reference=freeze, template=template,
                            output=tmp_path / 'evaluate', guard=lambda: None)
    assert result['status'] == 'complete' and len(result['completed_panels']) == 6
    assert [condition for _, condition in calls] == ['no_noise'] * 6
    assert result['vanilla_reference'] is None and not (tmp_path / 'evaluate/vanilla').exists()
    def expired():
        raise TimeoutError('evaluation deadline')
    with pytest.raises(TimeoutError):
        run_evaluation(model, object(), state=state, freeze_reference=freeze, template=template,
                       output=tmp_path / 'failed-evaluate', guard=expired)
    assert json.loads((tmp_path / 'failed-evaluate/result.json').read_text())['status'] == 'incomplete'
    assert (output / 'freeze.json').read_bytes() == before
    monkeypatch.setattr('jscc.encoder_shared_pilot.final_panels', final_panels)
    assert open_terminal(output, metadata, plan)[0].reference == state.reference
