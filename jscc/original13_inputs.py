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

