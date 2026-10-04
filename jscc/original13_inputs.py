"""Original13 bound input loading and local model initialization."""
import copy
import json
from pathlib import Path

import torch

from .activation_replay import canonical_digest, file_digest
from .baseline_protocol import read_prepared_batch
from .encoder_pilot_plan import SITE_SPECS
from .original13_provenance import stages_source_inventory
from .presentation import PresentationSampler
from .sharing_preparation import _reference, state_dict_identity
from .sharing_protocol import batch_view


def load_original13_prepared(path, *, checksum, width, ratio, load_batches=True, allow_pending_assets=False):
    """Load only bound local inputs/configs; never select a newer run or checkpoint."""
    from .encoder_stages_plan import Original13Plan, ORIGINAL13_PROTOCOL
    from .encoder_stages_preparation import resolve_original13_plans, validate_original13_config, verify_model_assets
    from .config import validate_config
    path = Path(path).resolve()
    if file_digest(path) != checksum:
        raise ValueError('exact original13 prepared SHA required')
    manifest = json.loads(path.read_text())
    inventory = stages_source_inventory()
    if (manifest.get('schema') != 'original13-prepared-v1'
            or manifest.get('manifest_identity') != canonical_digest(
                {k: v for k, v in manifest.items() if k != 'manifest_identity'})
            or manifest['source_inventory'] != inventory
            or manifest['source_identity'] != canonical_digest(inventory)):
        raise ValueError('original13 prepared source/manifest differs')
    plans = tuple(Original13Plan.from_resolved(row) for row in manifest['plans'])
    selected = [p for p in plans if p.width == width and p.ratio == ratio]
    if len(selected) != 1:
        raise ValueError('exact declared original13 width/ratio required')
    plan = selected[0]
    inputs_path = Path(manifest['inputs_path'])
    if file_digest(inputs_path) != manifest['inputs_sha256'] or plan.input_identity != manifest['inputs_sha256']:
        raise ValueError('original13 input manifest differs')
    inputs = json.loads(inputs_path.read_text())
    if (manifest['data_identity'] != canonical_digest(inputs['data_ids'])
            or canonical_digest(manifest['observation_request']['panel']) != plan.observation_identity):
        raise ValueError('original13 prepared membership differs')
    config = copy.deepcopy(manifest['configs'][str(width)])
    validate_config(config)
    validate_original13_config(config, width)
    assets = verify_model_assets(config, require_complete=not allow_pending_assets, verify_bytes=False)
    if not plan.synthetic and (not assets['production']
            or assets['tokenizer_files'] != inputs['tokenizer_assets']['files']):
        raise ValueError('original13 production input/model assets differ')
    if (config['protocol'] != ORIGINAL13_PROTOCOL or config['split'] != SITE_SPECS['enc_fn']
            or config['training']['max_steps'] != plan.horizon
            or config['training']['schedule_steps'] != plan.horizon):
        raise ValueError('original13 resolved config/horizon differs')
    if not plan.synthetic:
        if (config['model']['revision'] != 'dd0a2683227859151b1730ca3a63087df5b5f39b'
                or config['model']['dtype'] != 'bfloat16' or config['model']['attn_implementation'] != 'sdpa'
                or len(plans) != 9 or set((p.width, p.ratio) for p in plans) != {
                    (w, r) for w in (512, 1152, 2304) for r in (50, 75, 90)}):
            raise ValueError('original13 production model/matrix differs')
        expected = resolve_original13_plans(plan.views, inputs=inputs, input_identity=plan.input_identity)
        if [p.identity for p in expected] != [p.identity for p in plans]:
            raise ValueError('original13 production plan bindings differ')
    initial_ref = manifest['initializations'][str(width)]
    if initial_ref != inputs['initializations'][str(width)] or initial_ref.get('seed') != 0:
        raise ValueError('original13 width initializer origin differs')
    initial = torch.load(_reference(inputs_path.parent, initial_ref), map_location='cpu', weights_only=True)
    if state_dict_identity(initial) != initial_ref['state_identity']:
        raise ValueError('original13 initializer bytes differ')
    batches = None
    if load_batches:
        batches = [read_prepared_batch(ref, inputs_path.parent) for ref in inputs['updates']]
        if tuple(batch_view(b) for b in batches) != plan.views:
            raise ValueError('original13 ordered actual input views differ')
        if not plan.synthetic:
            rows = [int(i) for b in batches for i in b['row_ids']]
            expected_rows = PresentationSampler(inputs['data_ids']['train_rows'], 160000, seed=20260920).actual_ids.tolist()
            if rows != expected_rows:
                raise ValueError('original13 native row stream differs')
    metadata = {'source_identity': manifest['source_identity'], 'source_commit': manifest['source_commit'],
                'model_assets_identity': canonical_digest(assets),
                'config_identity': canonical_digest(config), 'data_identity': manifest['data_identity'],
                'initialization_identity': initial_ref['state_identity'], 'stream_identity': plan.input_identity,
                'protocol_identity': ORIGINAL13_PROTOCOL, 'prepared_identity': manifest['manifest_identity'],
                'model_revision': config['model']['revision']}
    return manifest, config, plan, batches, initial, metadata, inputs['data_ids']


def initialize_original13(config, initial, *, evaluation=False, synthetic=False):
    """Reuse the canonical loader with local cached assets and exact codec bytes."""
    from .models.split_model import build_model, resolve_original13_site
    from .runtime import configure_training_determinism, seed_everything, prepare_trainable_parameters
    from .encoder_stages_plan import ORIGINAL13_SITES
    from . import original13_routing
    from .encoder_stages_preparation import verify_model_assets
    verify_model_assets(config)
    configure_training_determinism(config['training'])
    seed_everything(config['seed'])
    processor, model = build_model(copy.deepcopy(config), local_files_only=True)
    prepare_trainable_parameters(model)
    model.codec.load_state_dict(initial, strict=True)
    if state_dict_identity(model.codec.state_dict()) != state_dict_identity(initial):
        raise ValueError('original13 exact FP32 initializer load differs')
    for site_id in ORIGINAL13_SITES:
        site = resolve_original13_site(model.base, site_id, config['model']['revision'])
        if not synthetic and site.hidden_dim != 1152:
            raise ValueError('original13 production hidden dimension differs')
    if next(model.base.parameters()).dtype != getattr(torch, config['model']['dtype']):
        raise ValueError('original13 actual backbone dtype differs')
    if model.base.config.decoder._attn_implementation != config['model']['attn_implementation']:
        raise ValueError('original13 actual attention backend differs')
    if evaluation:
        model.enable_universal_codec()
        original13_routing.load_codec(model, initial)
    return processor, model



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
        from .encoder_stages_preparation import _native_rows

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
        from .encoder_stages_preparation import _collate_native_rows
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
    validates the receipt, pinned revisions, paths and sizes. Same-size external
    edits violate the immutable-cache contract and require acquisition rebind.
    """
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
    arrow, assets = Path(dataset['path']), Path(tokenizer['path'])
    if not arrow.is_absolute() or not assets.is_absolute():
        raise ValueError('recipe cache acquisition receipt requires absolute local paths')
    required = [(arrow, dataset)]
    if 'config.json' not in tokenizer['files'] or not any(n.startswith('tokenizer') for n in tokenizer['files']):
        raise ValueError('recipe cache receipt requires tokenizer and model config assets')
    for name, record in tokenizer['files'].items():
        if Path(name).name != name:
            raise ValueError('recipe tokenizer asset must be a local filename')
        required.append((assets / name, record))
    for asset_path, record in required:
        if not asset_path.is_file():
            raise FileNotFoundError(f'recipe cached asset missing: {asset_path}')
        if asset_path.stat().st_size != record['bytes']:
            raise ValueError(f'recipe cached asset size differs: {asset_path}')
    raw = Dataset.from_file(str(arrow))
    processor = AutoTokenizer.from_pretrained(assets, local_files_only=True)
    model_config = AutoConfig.from_pretrained(assets, local_files_only=True)
    return RecipeCachedData(description, raw=raw, tokenizer=processor, model_config=model_config,
                            cache_identity=receipt['identity'], fixed_data_ids=receipt['fixed_data_ids'])
