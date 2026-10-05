"""Joined YAML CLI CPU acceptance; frozen large-model evidence stays separate."""

import jscc
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, cast
from time import perf_counter

from datasets import Dataset, DatasetDict
from datasets.arrow_writer import ArrowWriter
import pytest
import torch
import yaml

from jscc.activation_replay import file_digest
from jscc.shared_inputs import bind_recipe_data_cache
from jscc.asset_identity import state_dict_identity
from jscc.qualification import state_exact
from model_helpers import tiny_backbone
from shared.test_recipe_data import fixture_cache
from shared.test_shared_case_evaluation import template, score_stub

ROOT = Path(jscc.__file__).resolve().parents[1]


def write_yaml(path, value):
    path.write_text(yaml.safe_dump(value))
    return path


@pytest.fixture
def joined_assets(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(19)
    raw, tokenizer, _, config, _, fixed = fixture_cache(tmp_path)
    # Keep native prompt/demo/mask/row semantics but bound CPU attention cost.
    raw = Dataset.from_list([{**cast(dict[str, Any], row), 'ctx_a': 'Person enters room', 'ctx_b': 'Room',
                              'endings': ['wrong', 'right ending']} for row in raw])
    arrow = tmp_path/'short.arrow'
    with ArrowWriter(path=str(arrow)) as writer:
        writer.write_table(raw.data.table)
    assets = tmp_path/'assets'
    base = tiny_backbone(26)
    base.save_pretrained(assets)
    config['model'].update(name=str(assets), revision='tiny-workflow', device='cpu', dtype='float32',
                          numerical_policy='native', attn_implementation='sdpa')
    config['codec']['hidden_dim'] = 16
    config['training']['mixed_precision'] = False
    task = write_yaml(tmp_path/'task.yaml', config)
    fixed['development']['source_family_ids'] = ['family-0', 'family-1']
    fixed['final_validation']['source_family_ids'] = ['family-0', 'family-1', 'family-2']
    cache = bind_recipe_data_cache(dataset_path=arrow, assets=assets,
        dataset_name=config['data']['name'], dataset_revision=config['data']['revision'],
        model_revision='tiny-workflow', fixed_data_ids=fixed, output=tmp_path/'short-cache.json')
    from jscc import asset_identity as provenance
    from request_helpers import invoke
    original_digest = provenance.file_digest
    weight_hashes = []
    def measured_digest(path):
        path = Path(path)
        if path.suffix in ('.safetensors', '.bin') and path.parent == assets:
            weight_hashes.append(path.name)
        return original_digest(path)
    monkeypatch.setattr(provenance, 'file_digest', measured_digest)
    model_binding_path = tmp_path/'model-binding.json'
    started = perf_counter()
    invoke(tmp_path, 'shared', 'acquire', {'acquisition': {'model_path': str(assets), 'model_revision': 'tiny-workflow',
           'synthetic_cpu': True, 'output': str(model_binding_path)}})
    (tmp_path/'model-acquisition-cost.json').write_text(json.dumps({
        'seconds': perf_counter()-started, 'weight_files_hashed': list(weight_hashes)}))
    model_binding = {'path': str(model_binding_path), 'sha256': file_digest(model_binding_path)}
    from jscc.models.codec import Codec
    codec_config = deepcopy(config['codec'])
    codec_config['bottleneck_dim'] = 8
    torch.manual_seed(0)
    initial = Codec(16, codec_config).state_dict()
    init_path = tmp_path/'initial.pt'
    torch.save(initial, init_path)
    initializer = {'8': {'path': str(init_path), 'sha256': file_digest(init_path),
                         'state_identity': state_dict_identity(initial), 'seed': 0}}
    native = DatasetDict({'train': Dataset.from_list([
        {'ctx_a': 'x', 'ctx_b': 'x', 'activity_label': 'x', 'endings': ['x', 'x x', 'x x x', 'x x x x'],
         'label': str(i % 4), 'source_id': f'family-{i}', 'ind': i} for i in range(8)]),
        'validation': Dataset.from_list([
        {'ctx_a': 'x', 'ctx_b': 'x', 'activity_label': 'x', 'endings': ['x', 'x x', 'x x x', 'x x x x'],
         'label': str(i % 4), 'source_id': f'family-{i}', 'ind': i} for i in range(8)])})
    monkeypatch.setattr('datasets.load_dataset', lambda *args, **kwargs: native)
    from lm_eval.tasks import TaskManager
    manager = TaskManager()
    monkeypatch.setattr('lm_eval.tasks.TaskManager', lambda: manager)
    return dict(root=tmp_path, task=task, model=assets, cache=cache, initializer=initializer,
                tokenizer=tokenizer, fixed=fixed, model_binding=model_binding, weight_hashes=weight_hashes)


def workflow(assets, name, *, retained=False, reverse=False, subset=12, seed=23, optimizer=False):
    from jscc.shared_preparation import bind_recipe_workflow
    root = assets['root']/name
    root.mkdir()
    from jscc.shared_config import read_recipe
    recipe = read_recipe(ROOT/'configs/experiments/shared/shared_hellaswag_reference.yaml')
    recipe.update(task_config=str(assets['task']), widths=[8], reconstruction_percentages=[50],
                  updates_per_location=40 if retained else 8)
    if not retained:
        recipe['locations'] = ['dec_l3', 'enc_l2', 'enc_emb']
        if reverse:
            recipe['locations'].reverse()
        recipe['overrides']['data'] = {'num_train': subset, 'selection_seed': seed}
        recipe['observation'] = {'items': 2, 'sites': ['enc_l2', 'dec_l3'], 'conditions': ['no_noise']}
        recipe['save'].update(functional_observation_fraction=.5)
        if optimizer:
            recipe['overrides']['training'].update(lr=.003, weight_decay=.2, warmup_ratio=.5,
                                                   min_lr_ratio=.6, betas=[.7, .9], eps=1e-6)
            recipe['save'].update(reconstruction_fractions=[.5, 1], functional_fractions=[.5, 1])
    recipe['evaluation'].update(items=3, conditions=['no_noise'], supplemental_conditions=[-6],
                                candidate_batch=2, fewshot=0)
    recipe_path = write_yaml(root/'recipe.yaml', recipe)
    observation = template(recipe['observation']['items'])
    cache = assets['cache']
    if retained:
        cache_raw = json.loads(cache.read_text())
        fixed = deepcopy(cache_raw['fixed_data_ids'])
        fixed['development'].update(input_ids=list(range(256)),
                                    source_family_ids=[f'family-{i}' for i in range(256)])
        cache = bind_recipe_data_cache(dataset_path=cache_raw['dataset']['path'], assets=assets['model'],
            dataset_name=cache_raw['dataset']['name'], dataset_revision=cache_raw['dataset']['revision'],
            model_revision='tiny-workflow', fixed_data_ids=fixed, output=root/'retained-cache.json')
    if not retained:
        observation.update(prompt_policy='zero-shot')
        observation['settings'].update(num_fewshot=0, batch_size=2)
    observation['data_settings'] = {'name': 'fixture/hellaswag', 'revision': 'fixed-data'}
    observation['noise'] = deepcopy(assets['fixed']['development']['noise'])
    terminal = template(3)
    terminal['data_settings'] = {'name': 'fixture/hellaswag', 'revision': 'fixed-data'}
    terminal.update(prompt_policy='zero-shot')
    terminal['settings'].update(num_fewshot=0, batch_size=2)
    for path, value in ((root/'observation.json', observation), (root/'terminal.json', terminal)):
        path.write_text(json.dumps(value))
    prepared = root/'prepared.json'
    started = perf_counter()
    manifest = bind_recipe_workflow(recipe_path, model_config_path=assets['model']/'config.json',
        model_path=assets['model'], model_binding=assets['model_binding'], cache_manifest=cache, cache_sha256=file_digest(cache),
        initializations=assets['initializer'],
        observation_template={'path': str(root/'observation.json'), 'sha256': file_digest(root/'observation.json')},
        terminal_template={'path': str(root/'terminal.json'), 'sha256': file_digest(root/'terminal.json')},
        study_pairing_id='joined-tiny', output=prepared, synthetic=True)
    (root/'costs.json').write_text(json.dumps({'bind_resolve_assets_seconds': perf_counter()-started}))
    return root, recipe, recipe_path, prepared, manifest


def terminal_payload(result):
    terminal = next(row for row in result['checkpoints'] if row['role'] == 'terminal')
    return torch.load(terminal['path'], map_location='cpu', weights_only=False)


def train(prepared, manifest, output, *, mode='train', resume=None, recipe=None):
    """Exercise the shared scientific engine, never the retired CLI producer."""
    from jscc.panel_evaluation import make_recipe_observer
    from jscc.shared_config import resolve_recipe
    from jscc.experiment_state import CheckpointRef
    from jscc.shared_training import run_recipe_training
    from jscc.shared_preparation import initialize_recipe, load_recipe_workflow
    resolved = resolve_recipe(recipe, model_config_path=manifest['model_config']['path']) if recipe else None
    _, config, plan, stream, initial, metadata, ids = load_recipe_workflow(
        prepared, checksum=file_digest(prepared), width=8, ratio=50, resolved=resolved)
    assert plan is not None and stream is not None and metadata is not None
    processor, model = initialize_recipe(config, initial, plan.trained_sites, synthetic=True)
    reference = CheckpointRef(resume['path'], resume['sha256'], resume['size']) if resume else None
    observation = json.loads(Path(manifest['observation_template']['path']).read_text())
    started = perf_counter()
    result = run_recipe_training(model, processor, plan=plan, batches=stream, metadata=metadata,
        output=output, guard=lambda: None, observe=make_recipe_observer(observation), resume=reference,
        data_ids=ids, synthetic=True)
    (output/'wall-cost.json').write_text(json.dumps({'operation': mode,
                                                  'seconds': perf_counter()-started}))
    return result


def test_retained_yaml_real_updates_eleven_states_two_fixed256_templates(joined_assets, monkeypatch):
    from jscc import panel_evaluation as evaluation
    from jscc.shared_preparation import load_recipe_workflow
    root, recipe, _, prepared, manifest = workflow(joined_assets, 'retained', retained=True)
    # Explicit scorer fixture: this proves lifecycle/count parity, not 256-row model quality.
    monkeypatch.setattr(evaluation, '_score_panel', score_stub)
    result = train(prepared, manifest, root/'train')
    _, config, plan, stream, _, _, _ = load_recipe_workflow(
        prepared, checksum=file_digest(prepared), width=8, ratio=50)
    assert plan is not None and stream is not None
    assert plan.horizon == 520 and len(stream) == 40
    assert len(result['checkpoints']) == 11
    assert [r['state'] for r in result['observations']] == ['pre_reset', 'functional_observation']
    assert result['completed_updates'] == result['segment_updates'] == 520
    report = evaluation.recipe_report(root/'train', observations=sorted((root/'train').glob('observation_*')))
    assert [row['observed_items'] for row in report['observation_receipts']] == [2048, 2048]
    assert len(report['training_scalars']['rows']) == 520
    payload = terminal_payload(result)
    assert payload['optimizer']['param_groups'][0]['lr'] == 0
    assert payload['optimizer']['param_groups'][0]['weight_decay'] == config['training']['weight_decay']
    assert payload['metadata']['completed_updates'] == 520


@pytest.mark.parametrize('reverse,subset,seed,optimizer', [(False, 12, 23, False), (True, 20, 2, True)])
def test_changed_yaml_native_training_resume_terminal_and_reporting(joined_assets, reverse, subset, seed, optimizer):
    from jscc import panel_evaluation as evaluation
    from jscc.shared_preparation import load_recipe_workflow
    root, recipe, _, prepared, manifest = workflow(joined_assets, 'changed', reverse=reverse,
                                                  subset=subset, seed=seed, optimizer=optimizer)
    started = perf_counter()
    _, config, plan, stream, initial, metadata, ids = load_recipe_workflow(
        prepared, checksum=file_digest(prepared), width=8, ratio=50)
    costs = json.loads((root/'costs.json').read_text())
    costs['cached_data_views_load_seconds'] = perf_counter()-started
    (root/'costs.json').write_text(json.dumps(costs))
    assert plan is not None and stream is not None and ids is not None
    assert plan.trained_sites == tuple(recipe['locations'])
    assert len(ids['train_rows']) == subset and len(stream) == 8 and plan.horizon == 24
    assert ids['development'] == joined_assets['fixed']['development']
    assert stream[0]['labels'].shape[0] == 16
    assert torch.equal(stream[0]['decoder_attention_mask'], stream[0]['labels'].ne(-100))
    result = train(prepared, manifest, root/'train')
    events = [json.loads(row) for row in (root/'train/metrics.jsonl').read_text().splitlines()]
    assert [row['payload']['site_id'] for row in events] == recipe['locations']*8
    for row in events:
        index = row['payload']['completed_step']-1
        assert row['schema'] == 'recipe-training-event-v1'
        assert row['payload']['lr_used'] == [config['training']['lr']*plan.update_at(index).lr_factor]
    final = terminal_payload(result)
    group = final['optimizer']['param_groups'][0]
    assert group['weight_decay'] == config['training']['weight_decay']
    assert tuple(group['betas']) == tuple(config['training']['betas'])
    assert group['eps'] == config['training']['eps']
    assert final['metadata']['completed_updates'] == 24
    assert {site: row['updates'] for site, row in final['metadata']['exposure'].items()} == {site: 8 for site in plan.trained_sites}
    assert all(row['sequences'] == 128 for row in final['metadata']['exposure'].values())
    assert final['scheduler']['last_epoch'] == 12
    pre_payload = torch.load(next(row['path'] for row in result['checkpoints'] if row['role'] == 'pre_reset'),
                             map_location='cpu', weights_only=False)
    post_payload = torch.load(next(row['path'] for row in result['checkpoints'] if row['role'] == 'post_reset'),
                              map_location='cpu', weights_only=False)
    assert state_exact(pre_payload['model'], post_payload['model'])
    assert state_exact(pre_payload['metadata']['exposure'], post_payload['metadata']['exposure'])
    assert pre_payload['optimizer']['state'] and not post_payload['optimizer']['state']
    for role in ('pre_reset', 'post_reset', 'functional'):
        choices = [row for row in result['checkpoints'] if row['role'] == role]
        if role == 'functional':
            choices = [row for row in result['checkpoints'] if row['role'] not in ('terminal', 'pre_reset', 'post_reset')
                       and torch.load(row['path'], map_location='cpu', weights_only=False)['metadata']['completed_updates'] > plan.stages[1].start]
        saved = choices[0]
        resumed = train(prepared, manifest, root/f'resume-{role}', mode='resume', resume=saved)
        assert state_exact(terminal_payload(resumed)['model'], final['model'])
        saved_step = torch.load(saved['path'], map_location='cpu', weights_only=False)['metadata']['completed_updates']
        assert resumed['segment_updates'] == 24-saved_step
        assert resumed['observations_status'] == 'pending'
    request = evaluation.terminal_recipe_request(root/'train', json.loads((root/'terminal.json').read_text()))
    state, _ = evaluation.open_recipe_terminal(root/'train')
    evaluation._validate_recipe_request(state, request, observation=False)
    started = perf_counter()
    from jscc.shared_preparation import initialize_recipe
    processor, model = initialize_recipe(config, initial, plan.trained_sites, evaluation=True, synthetic=True)
    scored = evaluation.run_recipe_evaluation(model, processor, request=request,
        output=root/'terminal-score', guard=lambda: None)
    (root/'terminal-score/wall-cost.json').write_text(json.dumps({'operation': 'evaluate',
                                                                'seconds': perf_counter()-started}))
    scored = cast(dict[str, Any], scored)
    assert scored['status'] == 'complete'
    assert all(row['completed_items'] == 3 for row in scored['panels'])
    assert all(row['metrics']['candidate_forward_requests'] == 12 for row in scored['panels'])
    observations = sorted((root/'train').glob('observation_*'))
    report = evaluation.recipe_report(root/'train', initial=root/'terminal-score', observations=observations)
    assert report['terminal']['initial']['observed_items'] == 9
    assert [row['observed_items'] for row in report['observation_receipts']] == [4, 4]
    assert len(report['training_scalars']['rows']) == 24
    for field in ('families', 'dataset', 'noise'):
        bad_template = json.loads((root/'terminal.json').read_text())
        bad_request = deepcopy(request)
        for value in (bad_template, bad_request):
            if field == 'families':
                value['source_family_ids'][0] = 'wrong-family'
            elif field == 'dataset':
                value['data_settings']['revision'] = 'wrong-revision'
            else:
                value['noise']['namespace'] = 'wrong-noise'
        with pytest.raises(ValueError, match='terminal template'):
            evaluation.terminal_recipe_request(root/'train', bad_template)
        with pytest.raises(ValueError, match='terminal template'):
            evaluation._validate_recipe_request(state, bad_request, observation=False)
        with pytest.raises(ValueError, match='terminal template'):
            evaluation.recipe_report(root/'train', initial_request=bad_request)


def test_prepared_persisted_resolution_and_changed_resume_rejection(joined_assets):
    from jscc.shared_preparation import load_recipe_workflow
    root, recipe, recipe_path, prepared, manifest = workflow(joined_assets, 'persisted', optimizer=True)
    direct = train(prepared, manifest, root/'direct')
    recipe['overrides']['training']['lr'] = .0015
    write_yaml(recipe_path, recipe)
    persisted = train(prepared, manifest, root/'persisted')
    assert state_exact(terminal_payload(persisted)['model'], terminal_payload(direct)['model'])
    assert state_exact(terminal_payload(persisted)['optimizer'], terminal_payload(direct)['optimizer'])
    pre = next(row for row in direct['checkpoints'] if row['role'] == 'pre_reset')
    with pytest.raises(ValueError, match='identity|compatibility'):
        train(prepared, manifest, root/'bad-resume', mode='resume', resume=pre,
              recipe=recipe_path)
    if (root/'bad-resume/campaign.json').exists():
        assert json.loads((root/'bad-resume/campaign.json').read_text())['status'] == 'incomplete'
    assert not (root/'bad-resume/freeze.json').exists()
    from jscc.shared_config import resolve_recipe
    from jscc.shared_preparation import validate_recipe_prepared
    requested = resolve_recipe(recipe_path, model_config_path=manifest['model_config']['path'])
    checked = validate_recipe_prepared(manifest, resolved=requested)
    assert checked['launch_authorized'] is False
    assert checked['trajectories'][0]['config']['training']['lr'] == .0015
    # Same acquisition accepts changed subset/order/quota through a YAML override.
    recipe['locations'] = ['enc_emb', 'dec_l3']
    recipe['updates_per_location'] = 16
    recipe['overrides']['data'] = {'num_train': 20, 'selection_seed': 2}
    write_yaml(recipe_path, recipe)
    changed = train(prepared, manifest, root/'yaml-override', recipe=recipe_path)
    assert changed['completed_updates'] == 32
    _, _, baseline_plan, _, _, _, baseline_ids = load_recipe_workflow(
        prepared, checksum=file_digest(prepared), width=8, ratio=50)
    from jscc.shared_config import resolve_recipe
    _, _, changed_plan, changed_stream, _, _, changed_ids = load_recipe_workflow(
        prepared, checksum=file_digest(prepared), width=8, ratio=50,
        resolved=resolve_recipe(recipe_path, model_config_path=joined_assets['model']/'config.json'))
    assert changed_plan is not None and changed_stream is not None and changed_ids is not None
    assert baseline_plan is not None and baseline_ids is not None
    assert changed_plan.trained_sites == ('enc_emb', 'dec_l3') and len(changed_stream) == 16
    assert changed_ids['train_rows'] != baseline_ids['train_rows']
    assert len(changed_ids['train_rows']) == 20 and len(baseline_ids['train_rows']) == 12
    assert changed_plan.identity != baseline_plan.identity


@pytest.mark.parametrize('mutation', ['dataset', 'noise'])
def test_acquired_observation_binding_rejects_self_checksummed_wrong_template(joined_assets, mutation):
    from jscc.shared_preparation import bind_recipe_workflow
    root, _, recipe_path, _, manifest = workflow(joined_assets, 'bad-bind')
    path = root/'observation.json'
    value = json.loads(path.read_text())
    if mutation == 'dataset':
        value['data_settings']['revision'] = 'wrong-revision'
    else:
        value['noise']['namespace'] = 'wrong-noise'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='fixed template/cache membership'):
        bind_recipe_workflow(recipe_path, model_config_path=joined_assets['model']/'config.json',
            model_path=joined_assets['model'], model_binding=joined_assets['model_binding'], cache_manifest=manifest['cache']['path'],
            cache_sha256=manifest['cache']['sha256'], initializations=manifest['initializations'],
            observation_template={'path': str(path), 'sha256': file_digest(path)},
            terminal_template=manifest['terminal_template'], study_pairing_id='joined-tiny',
            output=root/'bad-prepared.json', synthetic=True)
    assert not (root/'bad-prepared.json').exists()



def test_one_time_model_acquisition_cli_and_repeated_prepare_never_rehash_weights(joined_assets):
    assert joined_assets['weight_hashes'] == ['model.safetensors']
    joined_assets['weight_hashes'].clear()
    first = workflow(joined_assets, 'prepare-one')
    second = workflow(joined_assets, 'prepare-two', reverse=True, subset=20, seed=2, optimizer=True)
    assert first[4]['model_binding'] == second[4]['model_binding'] == joined_assets['model_binding']
    assert first[4]['model_assets'] == second[4]['model_assets']
    assert joined_assets['weight_hashes'] == []
    (joined_assets['root']/'repeated-prepare-cost.json').write_text(json.dumps({
        'fresh_preparations': 2, 'weight_files_hashed': list(joined_assets['weight_hashes']),
        'bindings': [str(first[3]), str(second[3])]}))






def test_changed_source_requires_fresh_recipe_binding_preserves_saved_bytes(joined_assets):
    from jscc.activation_replay import canonical_digest
    from jscc.shared_preparation import validate_recipe_prepared
    root, _, _, prepared, manifest = workflow(joined_assets, 'fresh-source')
    before = file_digest(prepared)
    validate_recipe_prepared(manifest)
    historical = deepcopy(manifest)
    historical['source_inventory']['jscc/recipe_workflow.py'] = '0'*64
    historical['source_identity'] = canonical_digest(historical['source_inventory'])
    historical['manifest_identity'] = canonical_digest({k: v for k, v in historical.items() if k != 'manifest_identity'})
    path = root/'historical-prepared.json'
    path.write_text(json.dumps(historical))
    old_hash = file_digest(path)
    with pytest.raises(ValueError, match='source inventory'):
        validate_recipe_prepared(historical)
    assert file_digest(path) == old_hash and file_digest(prepared) == before
    # Explicit saved metadata validation does not assert execution compatibility.
    validate_recipe_prepared(historical, check_source=False)
