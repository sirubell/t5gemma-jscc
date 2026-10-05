"""Real local processor loading preserves byte-bound COCO tokenizer semantics."""
from copy import deepcopy
import json

import pytest
from tokenizers import Tokenizer, normalizers
from transformers import AutoProcessor, Gemma3Processor, GemmaTokenizer
from transformers.models.gemma3.image_processing_pil_gemma3 import Gemma3ImageProcessorPil

from jscc.activation_replay import canonical_digest, file_digest
from jscc.asset_paths import asset_trust_signature
from jscc.shared_inputs import _validate_coco_cache, _validate_coco_processor


@pytest.fixture
def loaded(tmp_path):
    assets = tmp_path / 'assets'
    vocab = {token: index for index, token in enumerate([
        '<pad>', '<eos>', '<bos>', '<unk>', '<mask>', '<start_of_image>',
        '<end_of_image>', '<image_soft_token>', 'a', '▁'])}
    tokenizer = GemmaTokenizer(vocab=vocab, padding_side='right',
        model_specific_special_tokens={'boi_token': '<start_of_image>',
            'eoi_token': '<end_of_image>', 'image_token': '<image_soft_token>'})
    processor = Gemma3Processor(Gemma3ImageProcessorPil(size={'height': 8, 'width': 8}),
                               tokenizer, image_seq_length=2)
    processor.save_pretrained(assets)
    (assets / 'config.json').write_text('{"model_type":"t5gemma2"}')
    # Reproduce the sole structural difference in ticket 107's pinned receipt:
    # the official loader promotes the configured mask token to special.
    path = assets / 'tokenizer.json'
    raw = json.loads(path.read_text())
    raw['added_tokens'][4]['special'] = False
    path.write_text(json.dumps(raw))
    processor = AutoProcessor.from_pretrained(assets, local_files_only=True, trust_remote_code=False)
    return assets, processor, raw


def test_local_loader_normalization_passes_without_replacing_acquired_bytes(loaded, monkeypatch):
    assets, processor, raw = loaded
    before = {p.name: (file_digest(p), asset_trust_signature(p)) for p in assets.iterdir()}
    actual = json.loads(processor.tokenizer.backend_tokenizer.to_str())
    assert actual != raw
    normalized = deepcopy(raw)
    normalized['added_tokens'][4]['special'] = True
    assert actual == normalized
    loader = AutoProcessor.from_pretrained
    calls = []

    def local_loader(path, **kwargs):
        calls.append((path, kwargs))
        return loader(path, **kwargs)

    monkeypatch.setattr(AutoProcessor, 'from_pretrained', local_loader)
    _validate_coco_processor(processor, assets)
    assert calls == [(assets, {'local_files_only': True, 'trust_remote_code': False})]
    assert {p.name: (file_digest(p), asset_trust_signature(p)) for p in assets.iterdir()} == before


@pytest.mark.parametrize('mutation', ['vocabulary_ids', 'added_token_flags', 'normalizer', 'decoder'])
def test_changed_backend_semantics_are_rejected(loaded, mutation):
    assets, processor, _ = loaded
    tokenizer = processor.tokenizer
    document = json.loads(tokenizer.backend_tokenizer.to_str())
    if mutation == 'vocabulary_ids':
        vocab = document['model']['vocab']
        vocab['a'], vocab['▁'] = vocab['▁'], vocab['a']
    elif mutation == 'added_token_flags':
        document['added_tokens'][4]['special'] = False
    elif mutation == 'normalizer':
        document['normalizer']['content'] = 'changed'
    else:
        document['decoder'] = {'type': 'Fuse'}
    tokenizer._tokenizer = Tokenizer.from_str(json.dumps(document))
    with pytest.raises(ValueError, match='tokenizer differs'):
        _validate_coco_processor(processor, assets)


@pytest.mark.parametrize(('padding', 'truncation', 'max_length'), [
    ('max_length', True, 3), ('max_length', True, 5), (True, False, None)])
def test_previous_encoding_state_passes_without_mutating_runtime_backend(
        loaded, padding, truncation, max_length):
    assets, processor, _ = loaded
    tokenizer = processor.tokenizer
    tokenizer(['a a a a', 'a'], padding=padding, truncation=truncation, max_length=max_length)
    before = tokenizer.backend_tokenizer.to_str()
    assert json.loads(before)['padding'] is not None
    _validate_coco_processor(processor, assets)
    assert tokenizer.backend_tokenizer.to_str() == before
    tokenizer._tokenizer.normalizer = normalizers.Lowercase()
    with pytest.raises(ValueError, match='tokenizer differs'):
        _validate_coco_processor(processor, assets)


@pytest.mark.parametrize(('field', 'value'), [
    ('padding_side', 'left'), ('truncation_side', 'left'), ('model_max_length', 17),
    ('clean_up_tokenization_spaces', True), ('bos_token', '<eos>'),
    ('image_token', '<mask>'), ('chat_template', 'changed'), ('pad_token', '<unk>'),
    ('model_input_names', ['input_ids'])])
def test_changed_live_tokenizer_configuration_is_rejected(loaded, field, value):
    assets, processor, _ = loaded
    setattr(processor.tokenizer, field, value)
    with pytest.raises(ValueError, match='tokenizer differs'):
        _validate_coco_processor(processor, assets)


def test_changed_stored_tokenizer_configuration_is_rejected(loaded):
    assets, processor, _ = loaded
    processor.tokenizer.init_kwargs['padding_side'] = 'left'
    with pytest.raises(ValueError, match='tokenizer differs'):
        _validate_coco_processor(processor, assets)


@pytest.mark.parametrize(('field', 'value'), [
    ('image_seq_length', 3), ('full_image_sequence', 'changed'),
    ('image_token_id', 4), ('chat_template', 'changed')])
def test_changed_processor_prompt_configuration_is_rejected(loaded, field, value):
    assets, processor, _ = loaded
    setattr(processor, field, value)
    with pytest.raises(ValueError, match='processor configuration differs'):
        _validate_coco_processor(processor, assets)


def test_changed_image_preprocessing_is_rejected(loaded):
    assets, processor, _ = loaded
    processor.image_processor.image_mean = [0.1, 0.2, 0.3]
    with pytest.raises(ValueError, match='image preprocessing differs'):
        _validate_coco_processor(processor, assets)


def test_changed_snapshot_path_is_rejected(loaded, tmp_path):
    assets, processor, _ = loaded
    processor.tokenizer.name_or_path = str(tmp_path / 'other')
    with pytest.raises(ValueError, match='processor differs'):
        _validate_coco_processor(processor, assets)


@pytest.mark.parametrize('name', ['tokenizer.json', 'tokenizer_config.json', 'processor_config.json'])
def test_acquired_asset_changes_still_fail_existing_trust_guard(loaded, tmp_path, monkeypatch, name):
    assets, _, _ = loaded
    arrow, karpathy = tmp_path / 'dataset.arrow', tmp_path / 'karpathy.json'
    arrow.write_bytes(b'bound dataset')
    karpathy.write_text('{}')

    def record(path):
        return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': file_digest(path)}

    files = {p.name: record(p) for p in assets.iterdir()}
    receipt = {'schema': 'shared-coco-cache-v1', 'task': 'coco', 'scorer_assets': {},
        'dataset': record(arrow), 'tokenizer': {'path': str(assets), 'revision': 'pin', 'files': files},
        'karpathy': {**record(karpathy), 'membership': {}}, 'external_images': {},
        'processor_identity': canonical_digest({'revision': 'pin', 'files': files}),
        'trust': {'dataset': asset_trust_signature(arrow), 'karpathy': asset_trust_signature(karpathy),
                  'tokenizer': {p.name: asset_trust_signature(p) for p in assets.iterdir()}}}
    monkeypatch.setattr('jscc.shared_inputs._validate_coco_scorer_assets', lambda _: None)
    receipt['identity'] = canonical_digest(receipt)
    assert _validate_coco_cache(receipt) == (arrow, assets)
    path = assets / name
    # Same size still fails the original stat binding, before processor loading.
    content = path.read_text()
    path.write_text(content.replace('false', 'true ', 1) if 'false' in content else ' ' + content[1:])
    with pytest.raises(ValueError, match='trust signature differs'):
        _validate_coco_cache(receipt)
