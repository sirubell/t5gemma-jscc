"""Complete indexed views, batch identity and local cache primitives."""
from __future__ import annotations

from torch.utils.data import Dataset

from dataclasses import dataclass
from pathlib import Path
import torch
from .activation_replay import canonical_digest, file_digest
from .presentation import tensor_digest
import copy
from .asset_paths import asset_file_path
import json
from types import SimpleNamespace
from typing import Any, cast
from .data.hellaswag import Collator
from .data.hellaswag_prompts import digest
from .presentation import PresentationSampler


# Transferred from baseline_protocol; serialized meanings remain unchanged.
def batch_identity(batch):
    """Bind complete padded inputs and view metadata, including image tensors."""
    return canonical_digest({key: tensor_digest(value) if torch.is_tensor(value) else value
                             for key, value in batch.items() if key not in {"activation", "site_id"}})


def _tensor_binding(value):
    return (id(value), value._version, tuple(value.shape), tuple(value.stride()),
            str(value.dtype), str(value.device), value.data_ptr())


@dataclass(frozen=True)
class BoundBatchIdentity:
    """A complete prepared-batch binding for reuse outside the update timer.

    Keep the bound tensors read-only: ordinary in-place mutation, replacement,
    layout changes, and metadata edits invalidate this token. Unsafe writes
    through ``.data`` or a foreign storage alias are outside this contract.
    The complete digest also binds replay activation/site provenance, while
    the view digest preserves the established paired-noise namespace.
    """
    view_sha256: str
    complete_sha256: str
    _tensor_bindings: tuple
    _metadata_sha256: str
    mask_sha256: str | None = None

    def validate(self, batch):
        tensors = tuple((key, _tensor_binding(value)) for key, value in sorted(batch.items())
                        if torch.is_tensor(value))
        metadata = canonical_digest({key: value for key, value in batch.items()
                                     if not torch.is_tensor(value)})
        if tensors != self._tensor_bindings or metadata != self._metadata_sha256:
            raise ValueError("prepared batch changed after identity binding")
        return self.view_sha256


def bind_batch_identity(batch, *, expected_identity=None):
    """Hash each complete tensor once, before timed updates; verify a known view."""
    contents = {key: tensor_digest(value) if torch.is_tensor(value) else value
                for key, value in batch.items()}
    view = canonical_digest({key: value for key, value in contents.items()
                             if key not in {"activation", "site_id"}})
    if expected_identity is not None and view != expected_identity:
        raise ValueError("prepared input/view identity mismatch")
    return BoundBatchIdentity(view, canonical_digest(contents),
        tuple((key, _tensor_binding(value)) for key, value in sorted(batch.items())
              if torch.is_tensor(value)),
        canonical_digest({key: value for key, value in batch.items() if not torch.is_tensor(value)}),
        contents.get("attention_mask", ""))


def _check_batches(batches, kind, expected_sequences=None):
    if not batches:
        raise ValueError("empty effective batch")
    count = 0
    for batch in batches:
        mask, labels = batch["attention_mask"], batch["labels"]
        if mask.ndim != 2 or labels.ndim != 2 or len(mask) != len(labels):
            raise ValueError("complete source/target sequences required")
        if not bool(((mask == 0) | (mask == 1)).all()) or not bool(mask.bool().any(1).all()):
            raise ValueError("source mask must be binary with no empty real sequence")
        if not bool((labels != -100).any(1).all()):
            raise ValueError("target sequence has no valid native-reference position")
        if batch["input_ids"].shape != mask.shape:
            raise ValueError("source layout differs from mask")
        if kind == "local":
            activation = batch.get("activation")
            if activation is None or activation.ndim != 3 or activation.shape[:2] != mask.shape:
                raise ValueError("local learning requires complete-sequence replay")
        count += len(mask)
    if expected_sequences is not None and count != expected_sequences:
        raise ValueError("effective-batch sequence count differs from frozen plan")
    return count


def read_prepared_batch(reference, root):
    """Load only bytes bound by the preparation inventory; no layout mutation."""
    path = Path(root) / reference["path"]
    if file_digest(path) != reference["sha256"]:
        raise ValueError("prepared batch checksum mismatch")
    batch = torch.load(path, map_location="cpu", weights_only=True)
    if batch_identity(batch) != reference["view_sha256"]:
        raise ValueError("prepared input/view identity mismatch")
    return batch


# Transferred from sharing_schedule; serialized meanings remain unchanged.
@dataclass(frozen=True)
class BatchView:
    view_sha256: str
    mask_sha256: str
    layout_sha256: str
    sequences: int
    source_tokens: int
    target_tokens: int
    padded_tokens: int

    def __post_init__(self):
        for checksum in (self.view_sha256, self.mask_sha256, self.layout_sha256):
            if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
                raise ValueError("view, mask and layout require SHA-256 bindings")
        if any(type(n) is not int or n <= 0 for n in (
            self.sequences, self.source_tokens, self.target_tokens, self.padded_tokens
        )) or self.padded_tokens < self.source_tokens:
            raise ValueError("invalid complete-sequence exposure")


# Transferred from encoder_stages_preparation; serialized meanings remain unchanged.
def _native_rows(rows, raw, builder, tokenizer):
    """Tokenize each source row independently of its eventual padded batch."""
    docs = [raw[row] for row in rows]
    prompts = [builder.build(doc, row) for doc, row in zip(docs, rows)]
    tokens = tokenizer([text for text, _ in prompts], truncation=False, return_offsets_mapping=True)
    labels = tokenizer([doc['endings'][int(doc['label'])] for doc in docs],
                       truncation=True, max_length=512)
    result = []
    for i, (row, doc, (text, demos)) in enumerate(zip(rows, docs, prompts)):
        full_ids = tokens['input_ids'][i]
        kept = full_ids[-2048:]
        query_start = len(text) - len(builder.format(doc)[0])
        full_roles = []
        for position, (start, end) in enumerate(tokens['offset_mapping'][i]):
            if start == end:
                if position != 0 or full_ids[position] != tokenizer.bos_token_id:
                    raise ValueError('unexpected zero-offset token')
                full_roles.append('demo')
            else:
                full_roles.append('query' if end > query_start else 'demo')
        demo_families = [builder.documents[d]['source_id'] for d in demos]
        meta = dict(demo_ids=demos, source_id=doc['source_id'], demo_source_ids=demo_families,
                    source_length=len(full_ids), input_length=len(kept), source_hash=digest(doc),
                    text_hash=digest(text), input_hash=digest(kept), target_hash=digest(labels['input_ids'][i]))
        result.append({'input_ids': kept, 'attention_mask': tokens['attention_mask'][i][-2048:],
            'label_ids': labels['input_ids'][i], 'row_id': row,
            'token_roles': full_roles[-2048:], 'position_start': len(full_ids) - len(kept),
            'view_id': canonical_digest({'row_id': row, 'native_view': meta}),
            'source_family_id': doc['source_id'], 'demo_ids': demos,
            'source_view': {'row_id': row, 'native_prefix': 'reference',
                'demo_source_family_ids': demo_families, 'query_character_start': query_start,
                'native_view': meta}})
    return result


def _collate_native_rows(items, index, tokenizer, config, policy):
    """Retain native padding, metadata and decoder construction for any view."""
    from transformers import T5Gemma2ForConditionalGeneration

    batch: dict = Collator(tokenizer.pad_token_id)(items)
    width = batch['input_ids'].shape[1]
    rows = [item['row_id'] for item in items]
    roles = [item['token_roles'] + ['padding'] * (width - len(item['input_ids'])) for item in items]
    positions = [list(range(item['position_start'], item['position_start'] + width)) for item in items]
    view_ids = [item['view_id'] for item in items]
    batch.update(query_ids=[f'hellaswag/train/{row}' for row in rows], view_ids=view_ids,
                 source_family_ids=[item['source_family_id'] for item in items],
                 demo_ids=[copy.deepcopy(item['demo_ids']) for item in items], token_positions=torch.tensor(positions),
                 token_roles=roles, position_roles=roles,
                 source_views=[copy.deepcopy(item['source_view']) for item in items],
                 prompt_policy=copy.deepcopy(policy),
                 batch_view_id=canonical_digest({'rows': rows, 'view_ids': view_ids, 'native16_index': index}))
    batch['decoder_input_ids'] = T5Gemma2ForConditionalGeneration.prepare_decoder_input_ids_from_labels(
        SimpleNamespace(config=config), labels=batch['labels'])  # pyright: ignore[reportArgumentType]
    batch['decoder_attention_mask'] = batch['labels'].ne(-100)
    return batch


def _native_batch(rows, index, raw, builder, tokenizer, config):
    """Use the established native prompt, target, offset and padding contract."""
    return _collate_native_rows(_native_rows(rows, raw, builder, tokenizer), index,
                               tokenizer, config, builder.policy)


def bind_recipe_data_cache(*, dataset_path, assets, dataset_name, dataset_revision,
                           model_revision, fixed_data_ids, output):
    """Hash local cache bytes once at acquisition, without weights or downloads.

    The resulting small receipt can be reused across widths, sites and budgets.
    Keep the referenced files immutable; ordinary loading checks presence/size,
    not a repeated full-cache audit. Rebind after replacing cached assets.
    """
    from .asset_paths import asset_trust_signature, normalize_asset_path, tokenizer_asset_names
    dataset_path, assets, output = map(normalize_asset_path, (dataset_path, assets, output))
    names = tokenizer_asset_names(assets)
    if not dataset_path.is_file() or 'config.json' not in names or not any(n.startswith('tokenizer') for n in names):
        raise FileNotFoundError('existing Arrow dataset and local tokenizer/config cache required')
    receipt = {'schema': 'recipe-data-cache-v1',
        'dataset': {'name': dataset_name, 'revision': dataset_revision,
                    'path': str(dataset_path), 'bytes': dataset_path.stat().st_size,
                    'sha256': file_digest(dataset_path)},
        'tokenizer': {'path': str(assets), 'revision': model_revision,
                      'files': {name: {'bytes': asset_file_path(assets, name).stat().st_size,
                                       'sha256': file_digest(asset_file_path(assets, name))} for name in names}},
        'fixed_data_ids': copy.deepcopy(fixed_data_ids),
        'trust': {'dataset': asset_trust_signature(dataset_path),
                  'tokenizer': {name: asset_trust_signature(asset_file_path(assets, name)) for name in names}}}
    receipt['identity'] = canonical_digest(receipt)
    with output.open('x') as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return output


# Transferred from original13_inputs; serialized meanings remain unchanged.
class RecipeCachedData:
    """Seekable native16 views from locally cached HellaSwag rows.

    Tokenization occurs once per selected row at setup, not once per update.
    Access returns fresh tensors/metadata and consumes no stream cursor or RNG.
    ``views`` binds all Q lightweight BatchViews for RecipePlan on first use.
    State/scoring consumers own checkpoint and observation restoration.
    """
    def __init__(self, description, *, raw, tokenizer, model_config, cache_identity, fixed_data_ids):
        from .data.hellaswag import recipe_selection_ids
        from .data.hellaswag_prompts import PromptBuilder, validate_policy


        if description.get('identity') != canonical_digest(
                {k: v for k, v in description.items() if k != 'identity'}):
            raise ValueError('changed resolved trajectory identity')
        config = description['config']
        training, data = config['training'], config['data']
        self._q = description['updates_per_location']
        if (config['task'] != 'hellaswag' or type(self._q) is not int or self._q <= 0
                or training['batch_size'] != 16 or training['gradient_accumulation'] != 1):
            raise ValueError('cached recipe data requires HellaSwag native16/accumulation1')
        stream = training['presentation_stream']
        if (stream['policy'] != 'epoch-permutations-v1' or type(stream['seed']) is not int
                or stream['total_presentations'] != self._q * 16
                or stream.get('start_presentation', 0) != 0):
            raise ValueError('recipe stream requires exact Q native16 views starting at zero; seek by view index')
        ids = recipe_selection_ids(raw, data, fixed_data_ids)
        self._data_ids = {**copy.deepcopy(fixed_data_ids), **ids}
        for partition in ('train', 'validation'):
            key = f'{partition}_source_family_ids'
            if key in self._data_ids:
                self._data_ids[key] = [raw[row]['source_id'] for row in ids[f'{partition}_rows']]
        self._policy = copy.deepcopy(data['prompt_policy'])
        validate_policy(self._policy, data['max_length'])
        if not tokenizer.is_fast:
            raise ValueError('cached recipe data requires actual fast-tokenizer offsets')
        builder = PromptBuilder(raw, ids, self._policy)
        self._sampler = PresentationSampler(ids['train_rows'], self._q * 16, seed=stream['seed'])
        self._rows = {}
        for offset in range(0, len(ids['train_rows']), 16):
            selected = ids['train_rows'][offset:offset + 16]
            self._rows.update(zip(selected, _native_rows(selected, raw, builder, tokenizer)))
        self._tokenizer = tokenizer
        self._model_config = copy.deepcopy(model_config)
        self._views = None
        settings = {'schema': 'recipe-native16-data-v1', 'cache_identity': cache_identity,
            'data': {key: data.get(key) for key in ('name', 'revision', 'max_length', 'prompt_policy')},
            'data_ids': self._data_ids, 'stream_seed': stream['seed'],
            'decoder': {'bos_token_id': model_config.decoder.bos_token_id,
                        'pad_token_id': model_config.decoder.pad_token_id},
            'batch_size': 16, 'preprocess_digest': builder.helper_digest}
        self._compatibility_identity = canonical_digest(settings)
        self._identity = canonical_digest({'compatibility': self.compatibility_identity, 'q': self._q})

    @property
    def compatibility_identity(self):
        return self._compatibility_identity

    @property
    def identity(self):
        return self._identity

    @property
    def data_ids(self):
        return copy.deepcopy(self._data_ids)

    def __len__(self):
        return self._q

    def __getitem__(self, index):

        if type(index) is not int or not 0 <= index < self._q:
            raise IndexError('recipe view index outside stream')
        rows = self._sampler.actual_ids[index * 16:(index + 1) * 16].tolist()
        return _collate_native_rows([self._rows[row] for row in rows], index,
                                   self._tokenizer, self._model_config, self._policy)

    @property
    def views(self):
        if self._views is None:
            self._views = tuple(batch_view(self[index]) for index in range(self._q))
        return self._views

    def position(self, completed_updates, locations):
        """Describe a continuation position without advancing the stream."""
        if (type(completed_updates) is not int or type(locations) is not int or locations <= 0
                or not 0 <= completed_updates <= locations * self._q):
            raise ValueError('recipe continuation position outside stream')
        view, location = divmod(completed_updates, locations)
        return {'view_index': view, 'location_index': location,
                'complete': view == self._q, 'stream_identity': self.identity}


def load_recipe_cached_data(description, *, cache_manifest, checksum):
    """Load a small acquisition receipt and immutable local Arrow/tokenizer assets.

    No hub resolution, model weights, cache transfer or per-update files. Byte
    integrity belongs to acquisition/transfer (bind_recipe_data_cache); startup
    validates the receipt, pinned revisions, paths, sizes and acquired stat state.
    Changed acquired assets require an explicit acquisition rebind.
    """
    if description['config']['task'] == 'coco':
        from datasets import Dataset
        from transformers import AutoConfig, AutoProcessor
        receipt = _read_coco_cache({'path':str(Path(cache_manifest).resolve()),'sha256':checksum})
        config = description['config']
        if (receipt['dataset']['name'] != config['data']['name']
                or receipt['dataset']['revision'] != config['data']['revision']
                or receipt['tokenizer']['revision'] != config['model']['revision']
                or receipt['karpathy']['name'] != config['data']['karpathy_name']
                or receipt['karpathy']['revision'] != config['data']['karpathy_revision']):
            raise ValueError('COCO acquired dataset/processor/Karpathy pins differ')
        arrow, assets = _validate_coco_cache(receipt)
        processor = AutoProcessor.from_pretrained(assets,local_files_only=True)
        _validate_coco_processor(processor,assets)
        return CocoCachedData(description,raw=Dataset.from_file(str(arrow)),processor=processor,
            model_config=AutoConfig.from_pretrained(assets,local_files_only=True),receipt=receipt)
    from datasets import Dataset
    from transformers import AutoConfig, AutoTokenizer

    path = Path(cache_manifest).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'recipe data cache receipt missing: {path}')
    if file_digest(path) != checksum:
        raise ValueError('recipe data cache receipt checksum differs')
    receipt = json.loads(path.read_text())
    if (receipt.get('schema') != 'recipe-data-cache-v1' or receipt.get('identity') != canonical_digest(
            {k: v for k, v in receipt.items() if k != 'identity'})):
        raise ValueError('recipe data cache receipt identity differs')
    config = description['config']
    dataset, tokenizer = receipt['dataset'], receipt['tokenizer']
    if (dataset['name'] != config['data']['name'] or dataset['revision'] != config['data']['revision']
            or tokenizer['revision'] != config['model']['revision']):
        raise ValueError('recipe dataset/tokenizer revisions differ from local cache')
    arrow, assets = validate_recipe_cache(receipt)
    raw = Dataset.from_file(str(arrow))
    processor = AutoTokenizer.from_pretrained(assets, local_files_only=True)
    model_config = AutoConfig.from_pretrained(assets, local_files_only=True)
    return RecipeCachedData(description, raw=raw, tokenizer=processor, model_config=model_config,
                            cache_identity=receipt['identity'], fixed_data_ids=receipt['fixed_data_ids'])


def validate_recipe_cache(receipt):
    """Validate local cache receipt and cheap trust state without loading data."""
    if receipt.get('schema') == COCO_CACHE_SCHEMA:
        return _validate_coco_cache(receipt)
    from .asset_paths import asset_file_path, normalize_asset_path, verify_asset_record, tokenizer_asset_names
    if (receipt.get('schema') != 'recipe-data-cache-v1' or receipt.get('identity') != canonical_digest(
            {k: v for k, v in receipt.items() if k != 'identity'})):
        raise ValueError('recipe data cache receipt identity differs')
    dataset, tokenizer = receipt['dataset'], receipt['tokenizer']
    if not Path(dataset['path']).is_absolute() or not Path(tokenizer['path']).is_absolute():
        raise ValueError('recipe cache acquisition receipt requires absolute local paths')
    arrow, assets = normalize_asset_path(dataset['path']), normalize_asset_path(tokenizer['path'])
    files = tokenizer['files']
    if 'config.json' not in files or not any(n.startswith('tokenizer') for n in files):
        raise ValueError('recipe cache receipt requires tokenizer and model config assets')
    trust = receipt.get('trust')
    if trust is not None and (not isinstance(trust, dict) or set(trust) != {'dataset', 'tokenizer'}
                              or not isinstance(trust['tokenizer'], dict) or set(trust['tokenizer']) != set(files)):
        raise ValueError('recipe cache trust inventory differs')
    if trust is not None and set(tokenizer_asset_names(assets)) != set(files):
        raise ValueError('recipe tokenizer asset layout changed; reacquire cache')
    verify_asset_record(arrow, dataset, None if trust is None else trust['dataset'])
    for name, record in files.items():
        if Path(name).name != name:
            raise ValueError('recipe tokenizer asset must be a local filename')
        verify_asset_record(asset_file_path(assets, name), record, None if trust is None else trust['tokenizer'][name])
    return arrow, assets


# Transferred from sharing_protocol; serialized meanings remain unchanged.
def batch_view(batch, *, binding=None):
    """Bind the actual complete CPU/native batch; no geometry token repacking."""
    mask = batch['attention_mask']
    if binding is None:
        view, mask_sha = batch_identity(batch), tensor_digest(mask)
    else:
        if not isinstance(binding, BoundBatchIdentity):
            raise TypeError('batch view requires a bound identity token')
        view = binding.validate(batch)
        mask_sha = binding.mask_sha256 or tensor_digest(mask)
    return BatchView(
        view_sha256=view, mask_sha256=mask_sha,
        layout_sha256=canonical_digest({'shape': list(mask.shape), 'micro': 0}),
        sequences=len(mask), source_tokens=int(mask.sum()),
        target_tokens=int((batch['labels'] != -100).sum()), padded_tokens=mask.numel())


class IndexedRows(Dataset):
    def __init__(self, dataset, row_ids):
        self.dataset, self.row_ids = dataset, row_ids

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index], self.row_ids[index]


class IndexedCollator:
    def __init__(self, collator):
        self.collator = collator

    def __call__(self, rows):
        batch = self.collator([row for row, _ in rows])
        batch["preflight_row_ids"] = [row_id for _, row_id in rows]
        return batch


# New COCO meanings deliberately do not change recipe-data-cache-v1.
COCO_CACHE_SCHEMA = 'shared-coco-cache-v1'


def _coco_image_id(row):
    return int(row['file_name'].rsplit('_', 1)[1].split('.')[0])


def _coco_scorer_paths():
    import importlib.util
    from .asset_paths import normalize_asset_path
    modules = {'cider':'pycocoevalcap.cider.cider',
               'cider_scorer':'pycocoevalcap.cider.cider_scorer',
               'ptb_tokenizer':'pycocoevalcap.tokenizer.ptbtokenizer'}
    paths = {}
    for key, module in modules.items():
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None:
            raise FileNotFoundError('installed COCO scorer asset unavailable: '+module)
        paths[key] = normalize_asset_path(spec.origin)
    jars = sorted(paths['ptb_tokenizer'].parent.glob('stanford-corenlp-*.jar'))
    if len(jars) != 1:
        raise FileNotFoundError('one existing bundled Stanford PTB tokenizer jar is required')
    paths['stanford_jar'] = normalize_asset_path(jars[0])
    return paths


def coco_scorer_assets():
    """Attest the installed CIDEr/PTB implementation and bundled jar, locally."""
    from .asset_paths import asset_trust_signature
    record = {'schema':'shared-coco-scorer-assets-v1','scorer':'coco-ptb-cider-v1',
        'files':{key:{'path':str(path),'bytes':path.stat().st_size,'sha256':file_digest(path),
                      'trust':asset_trust_signature(path)} for key,path in _coco_scorer_paths().items()}}
    record['identity'] = canonical_digest(record)
    return record


def _validate_coco_scorer_assets(record):
    from .asset_paths import verify_asset_record
    if (record.get('schema') != 'shared-coco-scorer-assets-v1' or record.get('scorer') != 'coco-ptb-cider-v1'
            or record.get('identity') != canonical_digest({k:v for k,v in record.items() if k!='identity'})):
        raise ValueError('COCO scorer asset identity differs')
    paths = _coco_scorer_paths()
    if set(record['files']) != set(paths):
        raise ValueError('COCO scorer asset inventory differs')
    for key,path in paths.items():
        expected = record['files'][key]
        if expected['path'] != str(path):
            raise ValueError('COCO installed scorer path differs from acquisition')
        verify_asset_record(path,expected,expected['trust'])


def _processor_asset_names(assets):
    from .asset_paths import tokenizer_asset_names
    return sorted(set(tokenizer_asset_names(assets)) | {p.name for pattern in (
        '*processor*.json', 'chat_template*') for p in Path(assets).glob(pattern) if p.is_file()})


def _coco_membership(raw, fixed, karpathy):
    from .data.coco import validate_ids
    required = {'train_ids','demo_ids','selection_ids','report_ids'}
    if (not isinstance(fixed,dict) or not required <= set(fixed)
            or set(fixed)-required-{'development','final_validation'}):
        raise ValueError('COCO fixed membership fields differ')
    positions = {}
    for index, name in enumerate(raw['file_name']):
        image_id = _coco_image_id({'file_name':name})
        if image_id in positions:
            raise ValueError('COCO cache has duplicate image IDs')
        positions[image_id] = index
    for key in ('train_ids', 'demo_ids', 'selection_ids', 'report_ids'):
        values = fixed[key]
        if not isinstance(values, list) or any(type(value) is not int or value < 0 for value in values):
            raise ValueError('COCO fixed membership requires integer image IDs')
        if not values or not set(values) <= set(positions):
            raise ValueError('COCO fixed membership is empty or absent from acquired rows')
    validate_ids(fixed, set(karpathy['validation']), set(karpathy['test']))
    if not set(fixed['demo_ids']) <= set(karpathy['restval']):
        raise ValueError('COCO demos must belong to acquired Karpathy restval')
    for panel, key in (('development', 'selection_ids'), ('final_validation', 'report_ids')):
        expected = {'input_ids': fixed[key], 'source_family_ids': fixed[key]}
        if fixed.get(panel, expected) != expected:
            raise ValueError('COCO fixed panel differs from ordinary partition')
    return positions


def _coco_corpus(raw, positions, fixed, image_ids):
    references = [list(raw[positions[image_id]]['answer'])
                  for image_id in image_ids]
    demos = [list(raw[positions[image_id]]['answer'])
             for image_id in fixed['demo_ids']]
    if any(not values or any(not isinstance(value, str) or not value.strip() for value in values) for values in references+demos):
        raise ValueError('COCO references and demos require nonempty text')
    return {'image_ids': list(image_ids), 'references': references,
            'demo_ids': list(fixed['demo_ids']), 'demo_captions': [values[0] for values in demos]}


def bind_coco_data_cache(*, dataset_path, assets, dataset_name, dataset_revision,
                         model_revision, karpathy_path, karpathy_name,
                         karpathy_revision, fixed_data_ids, output):
    """Acquire existing local COCO Arrow, processor, membership and image bytes.

    No hub access, model weights or downloads. A fresh output is required.
    Embedded images are bound by Arrow bytes; external images get their own
    immutable byte/stat records. The decoded RGB inventory binds actual pixels.
    """
    import hashlib
    from datasets import Dataset, Image
    from .asset_paths import normalize_asset_path, asset_trust_signature
    if Path(output).expanduser().is_symlink():
        raise FileExistsError('COCO cache output is already reserved by a symlink')
    dataset_path, assets, karpathy_path, output = map(normalize_asset_path,
        (dataset_path, assets, karpathy_path, output))
    for key,value in {'dataset_name':dataset_name,'dataset_revision':dataset_revision,
                      'model_revision':model_revision,'karpathy_name':karpathy_name,
                      'karpathy_revision':karpathy_revision}.items():
        if not isinstance(value,str) or not value.strip():
            raise ValueError('COCO acquisition requires explicit pinned '+key)
    names = _processor_asset_names(assets)
    if ('config.json' not in names or 'preprocessor_config.json' not in names
            or not any(name.startswith('tokenizer') for name in names)):
        raise ValueError('COCO acquisition requires local tokenizer/config/image processor assets')
    metadata = json.loads(asset_file_path(assets,'config.json').read_text())
    if metadata.get('_commit_hash') is not None and metadata['_commit_hash'] != model_revision:
        raise ValueError('COCO model metadata revision differs from acquisition')
    raw = Dataset.from_file(str(dataset_path))
    karpathy = json.loads(karpathy_path.read_text())
    if (set(karpathy) != {'validation', 'test', 'restval'}
            or any(not isinstance(values, list) or any(type(i) is not int or i < 0 for i in values)
                   or len(values) != len(set(values)) for values in karpathy.values())):
        raise ValueError('Karpathy acquisition requires exact validation/test/restval integer IDs')
    memberships = list(karpathy.values())
    flat = [image_id for values in memberships for image_id in values]
    if len(flat) != len(set(flat)):
        raise ValueError('COCO Karpathy validation/test/restval membership must be disjoint')
    fixed = copy.deepcopy(fixed_data_ids)
    positions = _coco_membership(raw, fixed, karpathy)
    for panel, key in (('development', 'selection_ids'), ('final_validation', 'report_ids')):
        fixed[panel] = {'input_ids': list(fixed[key]), 'source_family_ids': list(fixed[key])}
    consumed = list(dict.fromkeys(i for key in ('train_ids','demo_ids','selection_ids','report_ids') for i in fixed[key]))
    encoded = raw.cast_column('image', Image(decode=False))
    images, external = {}, {}
    for image_id in consumed:
        row = raw[positions[image_id]]
        if not row['answer'] or any(not isinstance(c,str) or not c.strip() for c in row['answer']):
            raise ValueError('COCO acquired rows require nonempty caption references')
        rgb = row['image'].convert('RGB')
        images[str(image_id)] = {'size': list(rgb.size), 'mode': 'RGB',
                                'sha256': hashlib.sha256(rgb.tobytes()).hexdigest()}
        backing = encoded[positions[image_id]]['image']
        if backing.get('bytes') is None:
            path = normalize_asset_path(backing['path'])
            if not Path(backing['path']).is_absolute():
                raise ValueError('external COCO image backing must be an absolute local path')
            external[str(path)] = {'bytes': path.stat().st_size, 'sha256': file_digest(path),
                                  'trust': asset_trust_signature(path)}
    files = {name: {'bytes': asset_file_path(assets,name).stat().st_size,
                    'sha256': file_digest(asset_file_path(assets,name))} for name in names}
    processor_identity = canonical_digest({'revision': model_revision, 'files': files})
    receipt = {'schema': COCO_CACHE_SCHEMA, 'task': 'coco',
        'dataset': {'name': dataset_name, 'revision': dataset_revision, 'path': str(dataset_path),
                    'bytes': dataset_path.stat().st_size, 'sha256': file_digest(dataset_path)},
        'tokenizer': {'path': str(assets), 'revision': model_revision, 'files': files},
        'processor_identity': processor_identity, 'scorer_assets':coco_scorer_assets(),
        'karpathy': {'name': karpathy_name, 'revision': karpathy_revision, 'path': str(karpathy_path),
                    'bytes': karpathy_path.stat().st_size, 'sha256': file_digest(karpathy_path),
                    'membership': karpathy},
        'fixed_data_ids': fixed, 'images': images, 'external_images': external,
        'training_corpus': _coco_corpus(raw,positions,fixed,fixed['train_ids']),
        'corpora': {panel: _coco_corpus(raw,positions,fixed,fixed[key]) for panel,key in (
            ('development','selection_ids'),('final_validation','report_ids'))},
        'trust': {'dataset': asset_trust_signature(dataset_path),
                  'karpathy': asset_trust_signature(karpathy_path),
                  'tokenizer': {name: asset_trust_signature(asset_file_path(assets,name)) for name in names}}}
    receipt['identity'] = canonical_digest(receipt)
    with output.open('x') as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return output


def _validate_coco_cache(receipt):
    from .asset_paths import normalize_asset_path, verify_asset_record
    if (receipt.get('schema') != COCO_CACHE_SCHEMA or receipt.get('task') != 'coco'
            or receipt.get('identity') != canonical_digest({k:v for k,v in receipt.items() if k!='identity'})):
        raise ValueError('COCO cache receipt identity differs')
    _validate_coco_scorer_assets(receipt['scorer_assets'])
    dataset, tokenizer, karpathy = receipt['dataset'], receipt['tokenizer'], receipt['karpathy']
    if any(not Path(value['path']).is_absolute() for value in (dataset,tokenizer,karpathy)):
        raise ValueError('COCO cache receipt requires absolute local paths')
    arrow, assets = normalize_asset_path(dataset['path']), normalize_asset_path(tokenizer['path'])
    trust = receipt['trust']
    verify_asset_record(arrow,dataset,trust['dataset'])
    verify_asset_record(normalize_asset_path(karpathy['path']),karpathy,trust['karpathy'])
    files = tokenizer['files']
    if (set(files) != set(_processor_asset_names(assets)) or 'preprocessor_config.json' not in files
            or receipt['processor_identity'] != canonical_digest({'revision':tokenizer['revision'],'files':files})):
        raise ValueError('COCO processor asset inventory differs')
    for name, record in files.items():
        verify_asset_record(asset_file_path(assets,name),record,trust['tokenizer'][name])
    for path, record in receipt['external_images'].items():
        if not Path(path).is_absolute():
            raise ValueError('COCO external image requires absolute local path')
        verify_asset_record(normalize_asset_path(path),record,record['trust'])
    if json.loads(Path(karpathy['path']).read_text()) != karpathy['membership']:
        raise ValueError('COCO Karpathy membership differs')
    return arrow, assets


def _read_coco_cache(reference):
    path = Path(reference['path']).resolve()
    if file_digest(path) != reference['sha256']:
        raise ValueError('COCO cache receipt checksum differs')
    receipt = json.loads(path.read_text())
    _validate_coco_cache(receipt)
    return receipt


def _validate_coco_processor(processor, assets):
    """The runtime processor must come from the same acquired local snapshot."""
    from .asset_paths import normalize_asset_path
    tokenizer = getattr(processor,'tokenizer',None)
    name = getattr(tokenizer,'name_or_path',None)
    if name is None or normalize_asset_path(name) != normalize_asset_path(assets):
        raise ValueError('COCO runtime processor differs from acquired local snapshot')
    expected = json.loads(asset_file_path(assets,'preprocessor_config.json').read_text())
    image_processor = getattr(processor,'image_processor',None)
    actual = image_processor.to_dict() if image_processor is not None else {}
    if any(actual.get(key) != value for key,value in expected.items()):
        raise ValueError('COCO runtime image preprocessing differs from acquired config')
    backend = getattr(tokenizer,'backend_tokenizer',None)
    if backend is None or json.loads(backend.to_str()) != json.loads(asset_file_path(assets,'tokenizer.json').read_text()):
        raise ValueError('COCO runtime tokenizer differs from acquired tokenizer bytes')


def load_coco_panel_data(request, processor):
    """Validate a fixed acquired panel once and construct ordinary TaskData."""
    from datasets import Dataset
    from .data import TaskData
    receipt = _read_coco_cache(request['cache'])
    corpus = request['reference_corpus']
    arrow, assets = _validate_coco_cache(receipt)
    _validate_coco_processor(processor,assets)
    if (corpus['cache_identity'] != receipt['identity']
            or corpus['processor_identity'] != receipt['processor_identity']
            or request['input_ids'] != corpus['image_ids']
            or request['source_family_ids'] != corpus['image_ids']
            or request['data_settings']['name'] != receipt['dataset']['name']
            or request['data_settings']['revision'] != receipt['dataset']['revision']):
        raise ValueError('COCO panel/cache/processor identity differs')
    membership = {k:corpus[k] for k in ('image_ids','references','demo_ids','demo_captions')}
    matched = False
    for acquired in receipt['corpora'].values():
        n = len(corpus['image_ids'])
        expected = {**acquired,'image_ids':acquired['image_ids'][:n],'references':acquired['references'][:n]}
        if membership == expected:
            matched = True
    if not matched or not corpus['image_ids']:
        raise ValueError('COCO panel references/demos differ from acquired fixed membership')
    raw = Dataset.from_file(str(arrow))
    fixed = receipt['fixed_data_ids']
    positions = _coco_membership(raw,fixed,receipt['karpathy']['membership'])
    if _coco_corpus(raw,positions,fixed,corpus['image_ids']) != membership:
        raise ValueError('COCO actual references/demos differ from acquisition')
    demos = [raw[positions[i]] for i in fixed['demo_ids']]
    return TaskData(ids={**copy.deepcopy(fixed),'report_ids':list(corpus['image_ids'])},
        report=raw.select([positions[i] for i in corpus['image_ids']]),
        demo_images=[row['image'].convert('RGB') for row in demos],
        demo_captions=list(corpus['demo_captions']))


def coco_selection_ids(raw, data, receipt):
    """Select ordered acquired image IDs identically for binding and indexed views."""
    fixed = copy.deepcopy(receipt['fixed_data_ids'])
    _coco_membership(raw, fixed, receipt['karpathy']['membership'])
    if 'selection_seed' in data:
        seed = data['selection_seed']
        if type(seed) is not int or seed < 0:
            raise ValueError('COCO selection_seed requires a nonnegative integer')
        indices = torch.randperm(len(fixed['train_ids']), generator=torch.Generator().manual_seed(seed)).tolist()
        fixed['train_ids'] = [fixed['train_ids'][index] for index in indices]
    requested = data.get('num_train')
    if requested is not None:
        if type(requested) is not int or requested < 1:
            raise ValueError('COCO num_train requires a positive integer or null')
        fixed['train_ids'] = fixed['train_ids'][:requested]
    if len(fixed['demo_ids']) != data['num_demos']:
        raise ValueError('COCO acquired demo count differs from recipe')
    return fixed


class CocoCachedData:
    """Seekable native16 multimodal views; caption choice uses indexed identity."""
    def __init__(self, description, *, raw, processor, model_config, receipt):
        if description.get('identity') != canonical_digest({k:v for k,v in description.items() if k!='identity'}):
            raise ValueError('changed resolved COCO trajectory identity')
        config = description['config']
        self._q = description['updates_per_location']
        training, data = config['training'], config['data']
        if (config['task'] != 'coco' or type(self._q) is not int or self._q<=0
                or training['batch_size'] != 16 or training['gradient_accumulation'] != 1
                or data['caption_policy'] != 'indexed-reference-v1' or type(data['caption_seed']) is not int
                or data['caption_seed'] < 0):
            raise ValueError('cached COCO data requires indexed captions and native16/accumulation1')
        stream = training['presentation_stream']
        if (stream['policy'] != 'epoch-permutations-v1' or type(stream['seed']) is not int or stream['seed'] < 0
                or stream['total_presentations'] != self._q*16 or stream.get('start_presentation',0)!=0):
            raise ValueError('cached COCO stream requires exact Q native16 views from zero')
        fixed = copy.deepcopy(receipt['fixed_data_ids'])
        self._positions = _coco_membership(raw,fixed,receipt['karpathy']['membership'])
        if _coco_corpus(raw,self._positions,fixed,fixed['train_ids']) != receipt['training_corpus']:
            raise ValueError('COCO training references/demos differ from acquisition')
        fixed = coco_selection_ids(raw, data, receipt)
        self._data_ids, self._raw, self._receipt = fixed, raw, receipt
        self._processor, self._config = processor, copy.deepcopy(config)
        self._model_config = copy.deepcopy(model_config)
        self._sampler = PresentationSampler(fixed['train_ids'],self._q*16,seed=stream['seed'])
        demos = [raw[self._positions[i]] for i in fixed['demo_ids']]
        self._demo_images = [row['image'].convert('RGB') for row in demos]
        self._demo_captions = [[c for c in row['answer'] if c][0] for row in demos]
        self._views = None
        settings = {'schema':'shared-coco-native16-data-v1','task':'coco','cache_identity':receipt['identity'],
            'processor_identity':receipt['processor_identity'],'data':copy.deepcopy(data), 'data_ids':fixed,
            'demo_captions':self._demo_captions,'stream_seed':stream['seed'],'batch_size':16,
            'caption_algorithm':'sha256(seed:presentation:image_id) modulo nonempty references',
            'decoder':{'bos_token_id':model_config.decoder.bos_token_id,'pad_token_id':model_config.decoder.pad_token_id},
            'preprocess_digest':file_digest(Path(__file__).parent/'data/coco.py')}
        self._compatibility_identity = canonical_digest(settings)
        self._identity = canonical_digest({'compatibility':self._compatibility_identity,'q':self._q})

    @property
    def compatibility_identity(self):
        return self._compatibility_identity

    @property
    def identity(self):
        return self._identity

    @property
    def data_ids(self):
        return copy.deepcopy(self._data_ids)

    def __len__(self):
        return self._q

    def __getitem__(self,index):
        import hashlib
        from torch.utils.data import default_collate
        from transformers import T5Gemma2ForConditionalGeneration
        from .data.coco import CaptionDataset
        if type(index) is not int or not 0<=index<self._q:
            raise IndexError('COCO view index outside stream')
        ids = self._sampler.actual_ids[index*16:(index+1)*16].tolist()
        rows = [self._raw[self._positions[i]] for i in ids]
        captions = [[c for c in row['answer'] if c] for row in rows]
        offsets = list(range(index*16,(index+1)*16))
        choices = [int.from_bytes(hashlib.sha256(f"{self._config['data']['caption_seed']}:{p}:{i}".encode()).digest(),'big')%len(refs)
                   for p,i,refs in zip(offsets,ids,captions,strict=True)]
        dataset = CaptionDataset(rows,self._processor,self._demo_images,self._demo_captions,
            self._config['data'],True,record_presentations=True,target_indices=choices)
        items = [dataset[i] for i in range(16)]
        # Processor owns each image's native crops; concatenate native crops
        # in sequence order rather than assuming every image expands equally.
        pixels = torch.cat([item['pixel_values'] for item in items],dim=0)
        batch = default_collate([{k:v for k,v in item.items() if k != 'pixel_values'} for item in items])
        batch['pixel_values'] = pixels.clone()
        batch['pixel_counts'] = [len(item['pixel_values']) for item in items]
        width = batch['input_ids'].shape[1]
        views = [{'image_id':i,'presentation':p,'caption_index':c,'target_text':refs[c],
                  'image':copy.deepcopy(self._receipt['images'][str(i)]),'demo_ids':list(self._data_ids['demo_ids'])}
                 for i,p,c,refs in zip(ids,offsets,choices,captions,strict=True)]
        view_ids = [canonical_digest(v) for v in views]
        roles = [['source' if valid else 'padding' for valid in mask.tolist()] for mask in batch['attention_mask']]
        batch.update(task='coco',query_ids=[f'coco/image/{i}' for i in ids],source_family_ids=list(ids),
            demo_ids=[list(self._data_ids['demo_ids']) for _ in ids],demo_captions=list(self._demo_captions),
            source_views=views,view_ids=view_ids,caption_indices=choices,presentation_offsets=offsets,
            token_positions=torch.arange(width).expand(16,-1).clone(),token_roles=roles,position_roles=copy.deepcopy(roles),
            caption_policy=self._config['data']['caption_policy'],caption_seed=self._config['data']['caption_seed'],
            processor_identity=self._receipt['processor_identity'],
            batch_view_id=canonical_digest({'image_ids':ids,'view_ids':view_ids,'native16_index':index}))
        batch['decoder_input_ids'] = T5Gemma2ForConditionalGeneration.prepare_decoder_input_ids_from_labels(
            cast(Any, SimpleNamespace(config=self._model_config)),labels=batch['labels'])
        batch['decoder_attention_mask'] = batch['labels'].ne(-100)
        _check_batches([batch],'functional',16)
        return batch

    @property
    def views(self):
        if self._views is None:
            self._views = tuple(batch_view(self[index]) for index in range(self._q))
        return self._views

    def position(self,completed_updates,locations):
        if (type(completed_updates) is not int or type(locations) is not int or locations<=0
                or not 0<=completed_updates<=locations*self._q):
            raise ValueError('COCO continuation position outside stream')
        view,location = divmod(completed_updates,locations)
        return {'view_index':view,'location_index':location,'complete':view==self._q,'stream_identity':self.identity}
