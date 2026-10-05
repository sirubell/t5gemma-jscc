"""Joined local COCO acquire/register/prepare/load boundaries, without execution."""

import jscc
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from jscc.activation_replay import canonical_digest, file_digest
from jscc.config import load_config
from jscc.shared_inputs import CocoCachedData, bind_coco_data_cache
from jscc.shared_preparation import load_recipe_workflow, validate_recipe_prepared
from model_helpers import tiny_backbone
from request_helpers import invoke
from shared.test_shared_coco_inputs import acquired as acquired

ROOT = Path(jscc.__file__).resolve().parents[1]


def reference(path):
    return {'path':str(path),'sha256':file_digest(path)}


def write_json(path,value):
    path.write_text(json.dumps(value))
    return reference(path)


@pytest.fixture
def coco_preparation(acquired,tmp_path,monkeypatch):
    _, processor, _, acquisition = acquired
    assets = acquisition['assets']
    # Finalize the real four-layer multimodal model config/weights before
    # either model or cache acquisition so their common files have one identity.
    torch.manual_seed(17)
    tiny_backbone(4).save_pretrained(assets)
    revision = acquisition['model_revision']
    model_binding_path = tmp_path/'joined-model.json'
    invoke(tmp_path,'shared','acquire',{'acquisition':{
        'model_path':str(assets),'model_revision':revision,
        'synthetic_cpu':True,'output':str(model_binding_path)}})
    cache_path = tmp_path/'joined-cache.json'
    invoke(tmp_path,'shared','acquire',{'acquisition':{
        'kind':'coco-cache',**{key:str(value) if isinstance(value,Path) else value
                             for key,value in acquisition.items() if key!='output'},
        'output':str(cache_path)}})
    cache = json.loads(cache_path.read_text())
    config = load_config(ROOT/'configs/experiments/shared/coco_base.yaml', validate=False)
    config['model'].update(name=str(assets),revision=revision,device='cpu',dtype='float32',
                           attn_implementation='sdpa',numerical_policy='native')
    config['codec']['hidden_dim'] = 16
    config['codec']['bottleneck_dim'] = 8
    config['data'].update(name=acquisition['dataset_name'],revision=acquisition['dataset_revision'],
        karpathy_name=acquisition['karpathy_name'],karpathy_revision=acquisition['karpathy_revision'],
        num_train=2,num_demos=2,num_validation=2,num_report=2,
        max_prompt_length=3,max_target_length=3,num_workers=0,caption_seed=23)
    config['run']['output_dir'] = str(tmp_path/'unused-runs')
    task = tmp_path/'joined-task.yaml'
    task.write_text(yaml.safe_dump(config))
    science = yaml.safe_load((ROOT/'configs/experiments/shared/shared_coco_example.yaml').read_text())['experiment']
    science.update(task_config=str(task),locations=['enc_l2','dec_l1'],widths=[8],
                   updates_per_location=8,reconstruction_percentages=[50])
    science['overrides']['data'].update(num_train=2)
    science['observation'] = {'items':1,'sites':['enc_emb','dec_l0'],'conditions':['no_noise']}
    science['evaluation'].update(items=1,conditions=['no_noise'],supplemental_conditions=[],
                                  batch_size=1,max_new_tokens=3)
    science['save'].update(reconstruction_fractions=[.5,1],functional_fractions=[.5,1],
                          functional_observation_fraction=.5)
    panels = {}
    for panel in ('development','final_validation'):
        corpus = {**deepcopy(cache['corpora'][panel]),'cache_identity':cache['identity'],
                  'processor_identity':cache['processor_identity']}
        value = {'task':'coco','input_ids':list(corpus['image_ids']),
            'source_family_ids':list(corpus['image_ids']),'cache':reference(cache_path),
            'reference_corpus':corpus,'data_settings':deepcopy(config['data']),
            'scorer':'coco-ptb-cider-v1','precision':'fp32','backend':'sdpa',
            'prompt_policy':'native-demo-caption-v1','layout':'native',
            'noise':{'seed':13,'namespace':'joined-coco'},
            'settings':{'batch_size':1,'max_new_tokens':3,'scoring_policy':'coco-cider-v1',
                        'numerical_policy':'native','vanilla':False}}
        panels[panel] = write_json(tmp_path/(panel+'.json'),value)
    registrations = tmp_path/'registrations'
    registrations.mkdir()
    refs = {'model_binding':reference(model_binding_path),'cache':reference(cache_path),
            'observation_template':panels['development'],
            'terminal_template':panels['final_validation'],'synthetic_cpu':True}
    registration = invoke(tmp_path,'shared','register',{'registration':{
        'model_name':str(assets),'output':str(registrations/'coco.json'),'references':refs}})
    prepared_dir = tmp_path/'prepared'
    manifest = invoke(tmp_path,'shared','prepare',{'preparation':{
        'registrations':str(registrations),'output':str(prepared_dir)}},science)
    # Only the explicit processor fixture is mocked. AutoConfig, source
    # inventory, local Arrow and all acquisition/preparation guards remain real.
    def processor_from_local(path,**kwargs):
        assert Path(path) == assets
        assert kwargs == {'local_files_only':True}
        return processor
    monkeypatch.setattr('transformers.AutoProcessor.from_pretrained',processor_from_local)
    return {'assets':assets,'cache':cache,'manifest':manifest,'directory':prepared_dir,
            'registration':registration,'refs':refs,'science':science,'processor':processor}


def test_public_coco_assets_prepare_load_exact_task_and_prefix(coco_preparation):
    values = coco_preparation
    directory,manifest,cache = values['directory'],values['manifest'],values['cache']
    loaded,config,plan,stream,initial,metadata,ids = load_recipe_workflow(
        directory/'prepared.json',checksum=file_digest(directory/'prepared.json'),width=8,ratio=50)
    assert loaded == manifest
    assert config['task'] == 'coco'
    assert config['model']['asset_binding'] == manifest['model_assets']
    assert config['model']['revision'] == cache['tokenizer']['revision']
    assert config['data']['karpathy_revision'] == cache['karpathy']['revision']
    assert values['registration']['cache'] == manifest['cache']
    assert manifest['model_assets']['files']['preprocessor_config.json'] == cache['tokenizer']['files']['preprocessor_config.json']
    assert cache['scorer_assets']['scorer'] == 'coco-ptb-cider-v1'
    assert set(cache['scorer_assets']['files']) == {'cider','cider_scorer','ptb_tokenizer','stanford_jar'}
    assert isinstance(stream,CocoCachedData) and len(stream)==8
    assert plan is not None and metadata is not None
    assert plan.trained_sites == ('enc_l2','dec_l1') and plan.horizon==16
    assert plan.update_at(0).site == 'enc_l2'
    assert plan.update_at(1).site == 'dec_l1'
    assert stream.position(3,2)['view_index']==1 and stream.position(3,2)['location_index']==1
    assert metadata['prepared_identity']==manifest['manifest_identity']
    assert metadata['data_identity']==canonical_digest(ids)
    assert metadata['config_identity']==canonical_digest(config)
    assert initial and all(value.dtype==torch.float32 for value in initial.values())
    batch = stream[0]
    assert batch['task']=='coco' and batch['processor_identity']==cache['processor_identity']
    assert batch['demo_ids']==[[3,4]]*16
    assert batch['pixel_values'].shape==(48,3,2,2)
    assert batch['labels'].shape==(16,3)
    for key,panel in (('observation_template','development'),('terminal_template','final_validation')):
        template = json.loads(Path(manifest[key]['path']).read_text())
        assert template['input_ids']==cache['corpora'][panel]['image_ids'][:1]
        assert template['reference_corpus']['references']==cache['corpora'][panel]['references'][:1]
        assert template['reference_corpus']['demo_captions']==cache['corpora'][panel]['demo_captions']
        assert template['reference_corpus']['cache_identity']==cache['identity']
        assert template['reference_corpus']['processor_identity']==cache['processor_identity']
        assert template['cache']==manifest['cache']
        assert template['settings']['max_new_tokens']==3


@pytest.mark.parametrize('mutation',['reference','task'])
def test_public_registration_rejects_selfchecksummed_changed_coco_panel(coco_preparation,tmp_path,mutation):
    refs = deepcopy(coco_preparation['refs'])
    value = json.loads(Path(refs['terminal_template']['path']).read_text())
    if mutation=='reference':
        value['reference_corpus']['references'][0][0]='different caption'
    else:
        value['task']='hellaswag'
    refs['terminal_template'] = write_json(tmp_path/('bad-'+mutation+'.json'),value)
    output = tmp_path/('bad-registration-'+mutation+'.json')
    with pytest.raises(ValueError,match='registration'):
        invoke(tmp_path,'shared','register',{'registration':{
            'model_name':str(coco_preparation['assets']),'output':str(output),'references':refs}})
    assert not output.exists()


def test_prepared_selfchecksummed_wrong_task_is_rejected(coco_preparation):
    manifest = deepcopy(coco_preparation['manifest'])
    item = manifest['resolved']['trajectories'][0]
    item['config']['task']='hellaswag'
    item['identity']=canonical_digest({key:value for key,value in item.items() if key!='identity'})
    manifest['resolved']['identity']=canonical_digest({key:value for key,value in manifest['resolved'].items() if key!='identity'})
    manifest['manifest_identity']=canonical_digest({key:value for key,value in manifest.items() if key!='manifest_identity'})
    with pytest.raises(ValueError,match='task differs from acquired cache'):
        validate_recipe_prepared(manifest)


def test_direct_coco_binder_rejects_dangling_output_symlink(acquired,tmp_path):
    *_,acquisition = acquired
    target = tmp_path/'unowned-receipt.json'
    logical = tmp_path/'reserved-output.json'
    logical.symlink_to(target)
    with pytest.raises(FileExistsError,match='symlink'):
        bind_coco_data_cache(**{**acquisition,'output':logical})
    assert logical.is_symlink() and not target.exists()


def bound_root(values, tmp_path):
    return invoke(tmp_path, 'shared', 'prepare', {'root': str(tmp_path/'bound'),
        'preparation': {'registrations': str(tmp_path/'registrations')}}, values['science'])


def test_public_bound_root_coco_selection_counts_match_indexed_provider(coco_preparation, tmp_path):
    from jscc.shared_runs import recipe_run_record
    root = bound_root(coco_preparation, tmp_path)
    record = recipe_run_record(root)
    case = record['cases'][0]
    prepared = record['prepared']
    _, _, plan, stream, _, _, ids = load_recipe_workflow(
        prepared['path'], checksum=prepared['sha256'], width=8, ratio=50)
    assert plan is not None and stream is not None and ids is not None
    selection = json.loads((Path(case['path'])/'data_selection.json').read_text())
    assert selection == ids == stream.data_ids
    assert 'train_rows' not in selection
    assert case['counts']['selected_images'] == case['counts']['selected_rows'] == len(ids['train_ids']) == 2
    assert case['counts']['base_stream_presentations'] == 8*16
    assert case['counts']['processed_presentations'] == 2*8*16
    assert record['totals']['selected_rows'] == 2
    with pytest.raises(FileExistsError):
        bound_root(coco_preparation, tmp_path)


@pytest.mark.parametrize('resume', [False, True])
@pytest.mark.parametrize('mismatch', [False, True])
def test_mocked_admission_train_resume_compare_bound_image_ids(
        coco_preparation, tmp_path, monkeypatch, resume, mismatch):
    """Mock source admission only to reach the selection branch; no model/training.

    This does not establish positive clean-source acceptance. The next test
    exercises actual dirty-source rejection; ticket105 owns clean execution.
    """
    from jscc import shared_runs as runs
    from jscc.experiment_state import CheckpointRef
    root = bound_root(coco_preparation, tmp_path)
    record = runs.recipe_run_record(root)
    selected = record['cases'][0]['case_id']
    prepared = record['prepared']
    loaded: list[Any] = list(load_recipe_workflow(prepared['path'], checksum=prepared['sha256'], width=8, ratio=50))
    if mismatch:
        loaded[-1] = {**loaded[-1], 'train_ids': list(reversed(loaded[-1]['train_ids']))}
    monkeypatch.setattr(runs, 'load_recipe_workflow', lambda *args, **kwargs: tuple(loaded))
    monkeypatch.setattr(runs, 'verify_recipe_source', lambda *args: None)
    monkeypatch.setattr('jscc.runtime.source_state', lambda: {'dirty': False})
    class ReachedInitialization(Exception):
        pass
    def stop_before_model(*args, **kwargs):
        raise ReachedInitialization('image selection verified; model load not run')
    monkeypatch.setattr(runs, 'initialize_recipe', stop_before_model)
    if resume:
        reference = CheckpointRef(str(tmp_path/'fixture-durable.pt'), 'd'*64, 1)
        monkeypatch.setattr(runs, '_resume_reference', lambda *args, **kwargs: reference)
    error, match = (ValueError, 'row/image selection') if mismatch else (ReachedInitialization, 'image selection verified')
    with pytest.raises(error, match=match):
        runs.train_recipe_case(root, selected, attempt_id='fixture1',
                               resume=str(tmp_path/'fixture-durable.pt') if resume else None)
    attempt = Path(record['cases'][0]['path'])/'attempts/training-fixture1'
    assert json.loads((attempt/'attempt.json').read_text())['status'] == 'failed'
    assert not (attempt/'engine').exists()


def test_bound_coco_reported_dirty_source_still_rejects_before_model(coco_preparation, tmp_path, monkeypatch):
    from jscc import shared_runs as runs
    root = bound_root(coco_preparation, tmp_path)
    record = runs.recipe_run_record(root)
    monkeypatch.setattr(runs, 'initialize_recipe',
                        lambda *args, **kwargs: pytest.fail('model before fixed-source admission'))
    # Isolate the unchanged dirty-state gate from the earlier fixed-commit
    # inventory gate. This is negative admission unit coverage, not a positive
    # current-source training receipt (which remains ticket105).
    monkeypatch.setattr(runs, 'verify_recipe_source', lambda *args: None)
    monkeypatch.setattr('jscc.runtime.source_state', lambda: {'dirty': True})
    with pytest.raises(ValueError, match='clean source'):
        runs.train_recipe_case(root, record['cases'][0]['case_id'], attempt_id='dirty1')
    assert not (Path(record['cases'][0]['path'])/'attempts/training-dirty1/engine').exists()
