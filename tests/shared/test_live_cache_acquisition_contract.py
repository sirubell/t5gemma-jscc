"""Public local HellaSwag acquisition and real nested COCO processor gates."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoProcessor, GemmaTokenizer
import yaml

from jscc.activation_replay import canonical_digest, file_digest
from jscc.data.hellaswag import recipe_selection_ids
from jscc.experiment import execute, read_request
from jscc.shared_inputs import _validate_coco_processor, validate_recipe_cache
from shared.test_recipe_data import fixture_cache
from shared.test_recipe_workflow import workflow
from shared.test_shared_coco_inputs import acquired as acquired

pytest_plugins = ['shared.test_recipe_workflow']


def request(directory, acquisition, name='acquire.yaml'):
    filename = directory/name
    filename.write_text(yaml.safe_dump({'schema': 'jscc-experiment-v1', 'workflow': 'shared',
        'operation': 'acquire', 'execution': {'acquisition': acquisition}}))
    return filename


@pytest.fixture
def hsw(tmp_path):
    raw, _, _, config, _, fixed = fixture_cache(tmp_path)
    fixed['final_validation']['source_family_ids'] = ['official0', 'official1', 'official2']
    fixed.update(recipe_selection_ids(raw, config['data'], fixed))
    return {'kind': 'hellaswag-cache', 'dataset_path': 'train.arrow', 'assets': 'assets',
        'dataset_name': config['data']['name'], 'dataset_revision': config['data']['revision'],
        'model_revision': config['model']['revision'], 'fixed_data_ids': fixed, 'output': 'public-cache.json'}


def test_public_hsw_acquires_real_local_cache_from_declaring_yaml(hsw, tmp_path, monkeypatch):
    monkeypatch.setattr('huggingface_hub.snapshot_download', lambda *a, **k: pytest.fail('download'))
    monkeypatch.setattr('transformers.AutoModelForSeq2SeqLM.from_pretrained', lambda *a, **k: pytest.fail('weights'))
    filename = request(tmp_path, hsw)
    elsewhere = tmp_path/'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    output = execute(filename)
    receipt = json.loads(output.read_text())
    assert receipt['schema'] == 'recipe-data-cache-v1'
    assert receipt['fixed_data_ids'] == hsw['fixed_data_ids']
    assert receipt['dataset']['revision'] == hsw['dataset_revision']
    assert receipt['tokenizer']['revision'] == hsw['model_revision']
    assert receipt['dataset']['sha256'] == file_digest(tmp_path/'train.arrow')
    for name, record in receipt['tokenizer']['files'].items():
        assert record['sha256'] == file_digest(tmp_path/'assets'/name)
    assert validate_recipe_cache(receipt) == (tmp_path/'train.arrow', tmp_path/'assets')
    with pytest.raises(FileExistsError):
        execute(filename)


@pytest.mark.parametrize('field,value', [('dataset_revision', ''), ('model_revision', ' '),
    ('dataset_name', False), ('fixed_data_ids', []), ('karpathy_path', 'wrong'),
    ('synthetic_cpu', True), ('kind', 'unknown')])
def test_hsw_request_rejects_missing_pins_and_inactive_fields(hsw, tmp_path, field, value):
    with pytest.raises(ValueError):
        execute(request(tmp_path, {**hsw, field: value}))
    assert not (tmp_path/'public-cache.json').exists()


@pytest.mark.parametrize('key', ['dataset_path', 'assets', 'dataset_revision', 'model_revision', 'fixed_data_ids'])
def test_hsw_requires_all_cache_fields(hsw, tmp_path, key):
    del hsw[key]
    with pytest.raises(ValueError):
        read_request(request(tmp_path, hsw))


@pytest.mark.parametrize('key,value', [('train_rows', []), ('train_rows', [True]),
    ('train_rows', [1, 1]), ('validation_rows', [-1]), ('validation_split', 'validation'),
    ('development', {'input_ids': [0], 'source_family_ids': []}),
    ('final_validation', {'input_ids': [0, 0], 'source_family_ids': ['a', 'a']}),
    ('selection_seed', True)])
def test_hsw_requires_typed_fixed_memberships(hsw, tmp_path, key, value):
    hsw['fixed_data_ids'][key] = value
    with pytest.raises(ValueError):
        execute(request(tmp_path, hsw))
    assert not (tmp_path/'public-cache.json').exists()


def test_hsw_rejects_selection_overlap_and_missing_panels(hsw, tmp_path):
    hsw['fixed_data_ids']['train_rows'].append(hsw['fixed_data_ids']['validation_rows'][0])
    with pytest.raises(ValueError, match='disjoint'):
        execute(request(tmp_path, hsw))
    del hsw['fixed_data_ids']['development']
    with pytest.raises(ValueError, match='membership'):
        execute(request(tmp_path, hsw))


@pytest.mark.parametrize('name', ['train.arrow', 'assets/config.json'])
def test_hsw_missing_local_assets_never_publish(hsw, tmp_path, name):
    (tmp_path/name).unlink()
    with pytest.raises(FileNotFoundError):
        execute(request(tmp_path, hsw))
    assert not (tmp_path/'public-cache.json').exists()


@pytest.mark.parametrize('existing', [False, True])
def test_hsw_output_symlink_rejects_before_acquisition(hsw, tmp_path, existing, monkeypatch):
    target = tmp_path/'target.json'
    if existing:
        target.write_text('preserve')
    (tmp_path/'public-cache.json').symlink_to(target)
    monkeypatch.setattr('jscc.shared_inputs.bind_recipe_data_cache', lambda **k: pytest.fail('provider called'))
    with pytest.raises(FileExistsError):
        execute(request(tmp_path, hsw))
    assert target.read_text() == 'preserve' if existing else not target.exists()


@pytest.fixture
def nested_coco(acquired):
    _, _, _, acquisition = acquired
    assets = acquisition['assets']
    (assets/'preprocessor_config.json').unlink()
    impl = Tokenizer(WordLevel({'<unk>': 0, '<image>': 1, '<start_of_image>': 2,
        '<end_of_image>': 3, '<bos>': 4, '<eos>': 5, '<pad>': 6}, unk_token='<unk>'))
    GemmaTokenizer(tokenizer_object=impl, extra_special_tokens={'image_token': '<image>',
        'boi_token': '<start_of_image>', 'eoi_token': '<end_of_image>'}).save_pretrained(assets)
    (assets/'config.json').write_text(json.dumps({'model_type': 't5gemma2', '_commit_hash': 'model-pin'}))
    # Exact nested image fields from ticket107's pinned 718-byte processor config.
    nested = {'do_convert_rgb': True, 'do_normalize': True, 'do_pan_and_scan': None,
        'do_rescale': True, 'do_resize': True, 'image_mean': [0.5]*3,
        'image_processor_type': 'Gemma3ImageProcessor', 'image_seq_length': 256,
        'image_std': [0.5]*3, 'pan_and_scan_max_num_crops': None,
        'pan_and_scan_min_crop_size': None, 'pan_and_scan_min_ratio_to_activate': None,
        'processor_class': 'Gemma3Processor', 'resample': 2, 'rescale_factor': 1/255,
        'size': {'height': 896, 'width': 896}}
    (assets/'processor_config.json').write_text(json.dumps({'image_processor': nested,
        'image_seq_length': 256, 'processor_class': 'Gemma3Processor'}))
    # GemmaTokenizer constructs its native BPE backend on reload; serialize
    # that actual backend before acquisition so the tokenizer byte guard is real.
    AutoProcessor.from_pretrained(assets, local_files_only=True).tokenizer.save_pretrained(assets)
    return {**acquisition, 'kind': 'coco-cache', 'output': assets.parent/'nested-cache.json'}


def coco_request(tmp_path, acquisition):
    return request(tmp_path, {key: str(value) if isinstance(value, Path) else value
                             for key, value in acquisition.items()})


def test_nested_coco_acquisition_and_real_offline_processor_acceptance(nested_coco, tmp_path, monkeypatch):
    monkeypatch.setattr('huggingface_hub.snapshot_download', lambda *a, **k: pytest.fail('download'))
    monkeypatch.setattr('transformers.AutoModelForSeq2SeqLM.from_pretrained', lambda *a, **k: pytest.fail('weights'))
    receipt = json.loads(execute(coco_request(tmp_path, nested_coco)).read_text())
    assert 'processor_config.json' in receipt['tokenizer']['files']
    assert 'preprocessor_config.json' not in receipt['tokenizer']['files']
    assert validate_recipe_cache(receipt) == (nested_coco['dataset_path'], nested_coco['assets'])
    processor = AutoProcessor.from_pretrained(nested_coco['assets'], local_files_only=True)
    _validate_coco_processor(processor, nested_coco['assets'])
    assert type(processor).__name__ == 'Gemma3Processor'
    assert processor.image_processor.size == {'height': 896, 'width': 896}
    with pytest.raises(FileExistsError):
        execute(coco_request(tmp_path, nested_coco))


def test_nested_coco_takes_loader_precedence_over_separate_config(nested_coco, tmp_path):
    (nested_coco['assets']/'preprocessor_config.json').write_text(json.dumps({
        'image_processor_type': 'Gemma3ImageProcessor', 'size': {'height': 8, 'width': 8}}))
    execute(coco_request(tmp_path, nested_coco))
    processor = AutoProcessor.from_pretrained(nested_coco['assets'], local_files_only=True)
    assert processor.image_processor.size == {'height': 896, 'width': 896}
    _validate_coco_processor(processor, nested_coco['assets'])


@pytest.mark.parametrize('payload', [{}, {'processor_class': 'Gemma3Processor'},
    {'image_processor': {}}, {'image_processor': None}, {'image_processor': []},
    {'image_processor': {'size': {'height': 8, 'width': 8}}},
    {'image_processor': {'image_processor_type': 'UnknownImageProcessor'}}])
def test_coco_rejects_absent_or_invalid_nested_image_metadata(nested_coco, tmp_path, payload):
    (nested_coco['assets']/'processor_config.json').write_text(json.dumps(payload))
    with pytest.raises((ValueError, OSError)):
        execute(coco_request(tmp_path, nested_coco))
    assert not nested_coco['output'].exists()


@pytest.mark.parametrize('field,value', [('size', {'height': 8, 'width': 8}),
    ('image_mean', [0.4, 0.5, 0.5]), ('do_pan_and_scan', True), ('do_rescale', False)])
def test_coco_rejects_changed_nested_runtime_semantics(nested_coco, field, value):
    processor = AutoProcessor.from_pretrained(nested_coco['assets'], local_files_only=True)
    actual = processor.image_processor.to_dict()
    actual[field] = value
    processor.image_processor = SimpleNamespace(to_dict=lambda: actual)
    with pytest.raises(ValueError, match='preprocessing differs'):
        _validate_coco_processor(processor, nested_coco['assets'])


def test_nested_coco_asset_mutation_and_receipt_hash_remain_guarded(nested_coco, tmp_path):
    receipt = json.loads(execute(coco_request(tmp_path, nested_coco)).read_text())
    receipt['processor_identity'] = '0'*64
    receipt['identity'] = canonical_digest({k: v for k, v in receipt.items() if k != 'identity'})
    with pytest.raises(ValueError, match='inventory differs'):
        validate_recipe_cache(receipt)
    receipt = json.loads(nested_coco['output'].read_text())
    asset = nested_coco['assets']/'processor_config.json'
    asset.write_text(asset.read_text()+' ')
    with pytest.raises(ValueError, match='size differs|trust signature'):
        validate_recipe_cache(receipt)


@pytest.mark.parametrize('defect', ['membership', 'revision', 'asset'])
def test_public_register_keeps_negative_acquisition_guards(joined_assets, defect):
    root, _, _, _, manifest = workflow(joined_assets, 'registration-guard')
    cache_path = Path(manifest['cache']['path'])
    cached = json.loads(cache_path.read_text())
    if defect == 'membership':
        cached['fixed_data_ids']['development']['source_family_ids'][0] = 'different'
    elif defect == 'revision':
        cached['tokenizer']['revision'] = 'different'
    else:
        asset = joined_assets['model']/'tokenizer.json'
        asset.write_text(asset.read_text()+' ')
    cached['identity'] = canonical_digest({k: v for k, v in cached.items() if k != 'identity'})
    cache_path = root/'changed-cache.json'
    cache_path.write_text(json.dumps(cached))
    refs = {key: deepcopy(manifest[key]) for key in ('model_binding', 'observation_template', 'terminal_template')}
    refs['cache'] = {'path': str(cache_path), 'sha256': file_digest(cache_path)}
    refs['synthetic_cpu'] = True
    output = root/'registration.json'
    document = {'schema': 'jscc-experiment-v1', 'workflow': 'shared', 'operation': 'register',
        'execution': {'registration': {'model_name': str(joined_assets['model']),
            'output': str(output), 'references': refs}}}
    filename = root/'register.yaml'
    filename.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError):
        execute(filename)
    assert not output.exists()
