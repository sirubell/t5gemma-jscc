"""Tiny immutable COCO acquisition and native indexed view contracts."""

import jscc
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import random
from types import SimpleNamespace

from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage
import pytest
import torch

from jscc.activation_replay import canonical_digest, file_digest
from jscc.data.coco import CaptionDataset
from jscc.shared_inputs import (CocoCachedData, batch_identity, bind_coco_data_cache,
                                load_coco_panel_data, validate_recipe_cache)
from jscc.shared_config import resolve_recipe


class Processor:
    def __init__(self, assets):
        self.name_or_path = str(assets)
        self.backend_tokenizer = SimpleNamespace(to_str=lambda: '{}')
        self.image_processor = SimpleNamespace(to_dict=lambda: {'size': 2})

    @property
    def tokenizer(self):
        return self

    def __call__(self, text=None, **kwargs):
        if 'images' in kwargs:
            pixels = torch.stack([torch.full((3,2,2),float(image.getpixel((0,0))[0]))
                                  for image in kwargs['images']])
            return {'input_ids':torch.tensor([[7,8,0]]),'attention_mask':torch.tensor([[1,1,0]]),
                    'pixel_values':pixels}
        assert isinstance(text,str)
        target = torch.tensor([[ord(text[0]),1,0]])
        if not kwargs.get('truncation'):
            return {'input_ids':target[:,:2]}
        return {'input_ids':target,'attention_mask':torch.tensor([[1,1,0]])}


@pytest.fixture
def acquired(tmp_path):
    assets = tmp_path/'assets'
    assets.mkdir()
    for name, content in {'config.json':'{}','tokenizer.json':'{}',
                          'tokenizer_config.json':'{}','preprocessor_config.json':'{"size":2}'}.items():
        (assets/name).write_text(content)
    images = []
    for image_id in range(1,9):
        buffer = BytesIO()
        PILImage.new('RGB',(2,2),(image_id,0,0)).save(buffer,format='PNG')
        images.append({'bytes':buffer.getvalue(),'path':None})
    raw = Dataset.from_dict({'image':images,'file_name':[f'COCO_val2014_{i:012}.jpg' for i in range(1,9)],
        'answer':[['alpha','beta'] for _ in images]},features=Features({
            'image':Image(),'file_name':Value('string'),'answer':Sequence(Value('string'))}))
    folder = tmp_path/'dataset'
    raw.save_to_disk(folder)
    arrow = next(folder.glob('*.arrow'))
    karpathy = tmp_path/'karpathy.json'
    karpathy.write_text(json.dumps({'validation':[5,6,7,8],'test':[],'restval':[1,2,3,4]}))
    fixed = {'train_ids':[1,2],'demo_ids':[3,4],'selection_ids':[5,6],'report_ids':[7,8]}
    output = tmp_path/'receipt.json'
    kwargs = dict(dataset_path=arrow,assets=assets,dataset_name='tiny-coco',dataset_revision='data-pin',
        model_revision='model-pin',karpathy_path=karpathy,karpathy_name='tiny-karpathy',
        karpathy_revision='karpathy-pin',fixed_data_ids=fixed,output=output)
    bind_coco_data_cache(**kwargs)
    receipt = json.loads(output.read_text())
    return raw, Processor(assets), receipt, kwargs


def description():
    config = {'task':'coco','data':{'name':'tiny-coco','revision':'data-pin',
        'karpathy_name':'tiny-karpathy','karpathy_revision':'karpathy-pin','num_train':2,'num_demos':2,
        'caption_policy':'indexed-reference-v1','caption_seed':23,'max_prompt_length':3,'max_target_length':3},
        'training':{'batch_size':16,'gradient_accumulation':1,'presentation_stream':{
            'policy':'epoch-permutations-v1','seed':7,'total_presentations':32,'start_presentation':0}}}
    value = {'config':config,'updates_per_location':2}
    value['identity'] = canonical_digest(value)
    return value


def test_indexed_coco_views_are_complete_fresh_and_rng_independent(acquired):
    raw, processor, receipt, _ = acquired
    model = SimpleNamespace(decoder=SimpleNamespace(bos_token_id=2,pad_token_id=0))
    data = CocoCachedData(description(),raw=raw,processor=processor,model_config=model,receipt=receipt)
    before_random, before_torch = random.getstate(), torch.random.get_rng_state().clone()
    second, first = data[1], data[0]
    assert random.getstate() == before_random
    assert torch.equal(torch.random.get_rng_state(),before_torch)
    assert batch_identity(first) == batch_identity(data[0])
    assert first['pixel_values'].shape == (48,3,2,2)  # Two native demos plus query image per sequence.
    assert first['attention_mask'].shape == (16,3)
    assert first['decoder_attention_mask'].tolist() == first['labels'].ne(-100).tolist()
    assert first['source_family_ids'] == first['coco_image_id'].tolist()
    assert first['demo_ids'] == [[3,4]]*16
    assert first['presentation_offsets'] == list(range(16))
    assert second['presentation_offsets'] == list(range(16,32))
    assert first['coco_target_text'] == [refs['target_text'] for refs in first['source_views']]
    first['pixel_values'].zero_()
    first['source_views'][0]['target_text'] = 'changed'
    assert batch_identity(data[0]) != batch_identity(first)
    assert data.views[0].view_sha256 == batch_identity(data[0])
    assert data.position(3,2) == {'view_index':1,'location_index':1,'complete':False,'stream_identity':data.identity}
    assert data.position(4,2)['complete'] is True


def test_coco_acquisition_preserves_fresh_output_and_processor_trust(acquired):
    _, _, receipt, kwargs = acquired
    assert receipt['schema'] == 'shared-coco-cache-v1'
    assert receipt['corpora']['development']['references'] == [['alpha','beta']]*2
    assert receipt['fixed_data_ids']['development']['source_family_ids'] == [5,6]
    assert len(receipt['images']) == 8
    validate_recipe_cache(receipt)
    with pytest.raises(FileExistsError):
        bind_coco_data_cache(**kwargs)
    (kwargs['assets']/'preprocessor_config.json').write_text('{"size":3}')
    with pytest.raises(ValueError,match='trust signature'):
        validate_recipe_cache(receipt)


def test_panel_loader_uses_exact_local_membership_refs_and_processor(acquired):
    _, processor, receipt, kwargs = acquired
    request = {'task':'coco','input_ids':[7],'source_family_ids':[7],
        'data_settings':{'name':'tiny-coco','revision':'data-pin'},
        'cache':{'path':str(kwargs['output']),'sha256':file_digest(kwargs['output'])},
        'reference_corpus':{**receipt['corpora']['final_validation'],'image_ids':[7],
            'references':[['alpha','beta']],'cache_identity':receipt['identity'],
            'processor_identity':receipt['processor_identity']}}
    task = load_coco_panel_data(request,processor)
    assert task.ids['report_ids'] == [7]
    assert task.demo_captions == ['alpha','alpha']
    assert task.report[0]['file_name'].endswith('000000000007.jpg')
    changed = deepcopy(request)
    changed['reference_corpus']['references'][0][0] = 'replaced'
    with pytest.raises(ValueError,match='references/demos'):
        load_coco_panel_data(changed,processor)
    processor.image_processor = SimpleNamespace(to_dict=lambda:{'size':99})
    with pytest.raises(ValueError,match='preprocessing'):
        load_coco_panel_data(request,processor)


def test_ordinary_caption_default_random_and_explicit_target_are_separate(acquired):
    raw, processor, _, _ = acquired
    config = {'max_prompt_length':3,'max_target_length':3}
    random.seed(6)
    state = random.getstate()
    expected = random.choice(['alpha','beta'])
    random.setstate(state)
    ordinary = CaptionDataset(raw.select([0]),processor,[],[],config,True,record_presentations=True)
    assert ordinary[0]['coco_target_text'] == expected
    state = random.getstate()
    explicit = CaptionDataset(raw.select([0]),processor,[],[],config,True,
                              record_presentations=True,target_indices=[1])
    assert explicit[0]['coco_target_text'] == 'beta'
    assert random.getstate() == state


def test_shared_coco_resolver_uses_generation_schema_and_derived_stream(tmp_path):
    root = Path(jscc.__file__).resolve().parents[1]
    metadata = tmp_path/'config.json'
    metadata.write_text(json.dumps({'encoder':{'num_hidden_layers':26,'hidden_size':1152},
                                   'decoder':{'num_hidden_layers':26,'hidden_size':1152}}))
    resolved = resolve_recipe(root/'configs/experiments/shared/shared_coco_example.yaml',model_config_path=metadata)
    item = resolved['trajectories'][0]
    assert item['config']['task'] == 'coco'
    assert item['evaluation']['scoring_policy'] == 'coco-cider-v1'
    assert set(item['evaluation']) == {'items','conditions','supplemental_conditions','batch_size','max_new_tokens','scoring_policy'}
    assert item['config']['training']['presentation_stream']['total_presentations'] == 80*16


def test_coco_partition_leakage_and_boolean_ids_fail_before_output(acquired):
    _, _, _, kwargs = acquired
    bad = deepcopy(kwargs)
    bad['output'] = kwargs['output'].with_name('bad-receipt.json')
    bad['fixed_data_ids']['train_ids'] = [1,5]
    with pytest.raises(ValueError,match='disjoint'):
        bind_coco_data_cache(**bad)
    assert not bad['output'].exists()
    bad['fixed_data_ids']['train_ids'] = [True,2]
    with pytest.raises(ValueError,match='integer image IDs'):
        bind_coco_data_cache(**bad)
    assert not bad['output'].exists()


def test_coco_selection_seed_is_effective_and_compatibility_bound(acquired):
    raw, processor, receipt, _ = acquired
    model = SimpleNamespace(decoder=SimpleNamespace(bos_token_id=2,pad_token_id=0))
    value = description()
    value['config']['data'].update(num_train=1,selection_seed=0)
    value['identity'] = canonical_digest({k:v for k,v in value.items() if k!='identity'})
    first = CocoCachedData(value,raw=raw,processor=processor,model_config=model,receipt=receipt)
    value['config']['data']['selection_seed'] = 1
    value['identity'] = canonical_digest({k:v for k,v in value.items() if k!='identity'})
    second = CocoCachedData(value,raw=raw,processor=processor,model_config=model,receipt=receipt)
    assert first.data_ids['train_ids'] != second.data_ids['train_ids']
    assert first.compatibility_identity != second.compatibility_identity



def test_acquired_zero_demo_cached_views_are_query_only_and_ordered(acquired):
    raw, _, _, kwargs = acquired
    zero = deepcopy(kwargs)
    zero['fixed_data_ids']['demo_ids'] = []
    zero['output'] = kwargs['output'].with_name('zero-demo-receipt.json')
    bind_coco_data_cache(**zero)
    receipt = json.loads(zero['output'].read_text())
    validate_recipe_cache(receipt)
    assert receipt['fixed_data_ids']['demo_ids'] == []
    assert receipt['training_corpus']['demo_ids'] == []
    assert receipt['training_corpus']['demo_captions'] == []
    seen = []
    class QueryProcessor(Processor):
        def __call__(self, text=None, **settings):
            if 'images' in settings:
                seen.append((text, [image.getpixel((0, 0))[0] for image in settings['images']]))
            return super().__call__(text, **settings)
    processor = QueryProcessor(kwargs['assets'])
    model_config = SimpleNamespace(decoder=SimpleNamespace(bos_token_id=2, pad_token_id=0))
    value = description()
    value['config']['data']['num_demos'] = 0
    value['identity'] = canonical_digest({key: item for key, item in value.items() if key != 'identity'})
    data = CocoCachedData(value, raw=raw, processor=processor, model_config=model_config, receipt=receipt)
    first = data[0]
    assert first['pixel_values'].shape == (16, 3, 2, 2)
    assert first['pixel_counts'] == [1] * 16
    assert first['demo_ids'] == [[] for _ in range(16)]
    assert first['demo_captions'] == []
    assert all(view['demo_ids'] == [] for view in first['source_views'])
    image_ids = first['coco_image_id'].tolist()
    assert seen == [('<start_of_image>', [image_id]) for image_id in image_ids]
    assert first['pixel_values'][:, 0, 0, 0].tolist() == image_ids
    assert first['source_family_ids'] == image_ids
    assert data.data_ids['train_ids'] == [1, 2]
    assert data.data_ids['selection_ids'] == [5, 6]
    assert data.data_ids['report_ids'] == [7, 8]
    assert first['presentation_offsets'] == list(range(16))
    assert batch_identity(first) == batch_identity(data[0])
    assert data[1]['presentation_offsets'] == list(range(16, 32))


@pytest.mark.parametrize('key', ['train_ids', 'selection_ids', 'report_ids'])
def test_zero_demo_does_not_allow_empty_scientific_partitions(acquired, key):
    _, _, _, kwargs = acquired
    invalid = deepcopy(kwargs)
    invalid['fixed_data_ids']['demo_ids'] = []
    invalid['fixed_data_ids'][key] = []
    invalid['output'] = kwargs['output'].with_name(f'empty-{key}.json')
    with pytest.raises(ValueError, match='empty or absent'):
        bind_coco_data_cache(**invalid)
    assert not invalid['output'].exists()


@pytest.mark.parametrize('demos,match', [([True], 'integer image IDs'), ([999], 'absent'),
                                       ([1], 'disjoint'), ([5], 'disjoint')])
def test_demo_membership_guards_remain_with_zero_demo_support(acquired, demos, match):
    _, _, _, kwargs = acquired
    invalid = deepcopy(kwargs)
    invalid['fixed_data_ids']['demo_ids'] = demos
    invalid['output'] = kwargs['output'].with_name('invalid-demo-membership.json')
    with pytest.raises(ValueError, match=match):
        bind_coco_data_cache(**invalid)
    assert not invalid['output'].exists()
