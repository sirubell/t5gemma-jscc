"""Connect acquired local assets to YAML recipe training and evaluation.

The small prepared receipt binds source, cache, initial codec and fixed templates.
It creates no tensor packages and grants no scheduler or experiment authority.
"""
from copy import deepcopy
import json
from pathlib import Path
import re

import torch

from .activation_replay import canonical_digest, file_digest
from .experiment_recipe import resolve_recipe, resolve_locations
from .original13_inputs import load_recipe_cached_data
from .original13_provenance import (stages_source_inventory,
                                    verify_model_assets, verify_original13_execution_source)
from .recipe_observation_policy import recipe_observation_identity, scoring_template
from .recipe_plan import RecipePlan
from .sharing_preparation import state_dict_identity

SCHEMA = 'recipe-prepared-v1'


def recipe_source_inventory():
    root = Path(__file__).resolve().parents[1]
    return {**stages_source_inventory(), **{name: file_digest(root/name) for name in
        ('scripts/experiment_recipe.py', 'scripts/submit_recipe.py')}}


def _asset_path(path):
    from .asset_paths import normalize_asset_path
    return normalize_asset_path(path)


def _read_reference(reference, *, trusted_stat=False):
    path = Path(reference['path'])
    if not path.is_absolute() or not path.is_file():
        raise ValueError('recipe acquired reference bytes or absolute path differ')
    if trusted_stat and 'stat' in reference:
        from .asset_paths import verify_asset_stat
        verify_asset_stat(path, reference['stat'])
    elif file_digest(path) != reference['sha256']:
        raise ValueError('recipe acquired reference bytes or absolute path differ')
    return path


def _read_prepared(path, checksum):
    path = Path(path).resolve()
    if file_digest(path) != checksum:
        raise ValueError('recipe prepared checksum differs')
    manifest = json.loads(path.read_text())
    if (manifest.get('schema') != SCHEMA or manifest.get('manifest_identity') != canonical_digest(
            {k: v for k, v in manifest.items() if k != 'manifest_identity'})):
        raise ValueError('recipe prepared identity differs')
    return manifest


def workflow_resolution(resolved, manifest):
    """Bind operational asset paths into the persisted complete recipe config."""
    result = deepcopy(resolved)
    assets = manifest['model_assets']
    if json.loads(_read_reference(manifest['model_binding']).read_text()) != assets:
        raise ValueError('recipe acquired model binding reference differs')
    metadata = json.loads(_read_reference(manifest['model_config']).read_text())
    for item in result['trajectories']:
        config = item['config']
        if config['model']['revision'] != assets['revision']:
            raise ValueError('recipe acquired model revision differs')
        config['model']['name'] = assets['path']
        config['model']['asset_binding'] = deepcopy(assets)
        resolve_locations(item['locations'], metadata, config)
        for policy in item['observations']:
            resolve_locations(policy['sites'], metadata, config)
        item['identity'] = canonical_digest({k: v for k, v in item.items() if k != 'identity'})
    result['identity'] = canonical_digest({k: v for k, v in result.items() if k != 'identity'})
    return result


def validate_recipe_prepared(manifest, *, resolved=None, check_source=True):
    """Small receipt/assets preflight, shared by preparation, launch and check."""
    if (manifest.get('schema') != SCHEMA or manifest.get('manifest_identity') != canonical_digest(
            {k: v for k, v in manifest.items() if k != 'manifest_identity'})
            or not re.fullmatch('[0-9a-f]{40}', manifest['source_commit'])
            or type(manifest['synthetic_cpu']) is not bool or not manifest['study_pairing_id'].strip()):
        raise ValueError('recipe prepared identity/source differs')
    if check_source and (manifest['source_inventory'] != recipe_source_inventory()
            or manifest['source_identity'] != canonical_digest(manifest['source_inventory'])):
        raise ValueError('recipe prepared source inventory differs')
    original = manifest['resolved'] if resolved is None else resolved
    if original.get('identity') != canonical_digest({k: v for k, v in original.items() if k != 'identity'}):
        raise ValueError('recipe resolution identity differs')
    bound = workflow_resolution(original, manifest)
    observation = json.loads(_read_reference(manifest['observation_template']).read_text())
    terminal = json.loads(_read_reference(manifest['terminal_template']).read_text())
    cache_path = _read_reference(manifest['cache'])
    cache = json.loads(cache_path.read_text())
    if (cache.get('schema') != 'recipe-data-cache-v1' or cache.get('identity') != canonical_digest(
            {k: v for k, v in cache.items() if k != 'identity'})
            or _asset_path(cache['tokenizer']['path']) != _asset_path(manifest['model_assets']['path'])):
        raise ValueError('recipe cache/model acquisition differs')
    from .original13_inputs import validate_recipe_cache
    validate_recipe_cache(cache)
    for item in bound['trajectories']:
        config = item['config']
        if cache['dataset']['name'] != config['data']['name'] or cache['dataset']['revision'] != config['data']['revision']:
            raise ValueError('recipe dataset acquisition differs')
        verify_model_assets(config)
        if not manifest['synthetic_cpu'] and not manifest['model_assets']['production']:
            raise ValueError('recipe production requires acquired pinned model assets')
        for name, record in cache['tokenizer']['files'].items():
            if manifest['model_assets']['files'].get(name) != record:
                raise ValueError('recipe model/cache tokenizer bytes differ')
        ref = manifest['initializations'].get(str(item['width']))
        if not isinstance(ref, dict) or ref.get('seed') != config['seed']:
            raise ValueError('recipe acquired width initializer missing or seed differs')
        _read_reference(ref, trusted_stat=True)
        recipe_observation_identity(item, observation)
        scoring_template(terminal, item['evaluation'], item['evaluation'])
        fixed = cache['fixed_data_ids']
        for template, key in ((observation, 'development'), (terminal, 'final_validation')):
            panel = fixed.get(key)
            if (not isinstance(panel, dict) or template['input_ids'] != panel['input_ids'][:template['expected_items']]
                    or template['source_family_ids'] != panel['source_family_ids'][:template['expected_items']]
                    or any(template['data_settings'].get(k) != config['data'][k] for k in ('name', 'revision'))
                    or ('noise' in panel and template['noise'] != panel['noise'])):
                raise ValueError('recipe fixed template/cache membership differs')
    return bound


def bind_recipe_workflow(recipe_path, *, model_config_path, model_path, model_binding, cache_manifest, cache_sha256,
                         initializations, observation_template, terminal_template, study_pairing_id,
                         output, synthetic=False):
    """Acquire a new source receipt reusing already acquired immutable assets."""
    from .runtime import source_state
    model_path = _asset_path(model_path)
    config_path = Path(model_config_path).resolve()
    from .asset_paths import asset_file_path
    if config_path != asset_file_path(model_path, 'config.json'):
        raise ValueError('recipe topology must come from acquired model config')
    resolved = resolve_recipe(recipe_path, model_config_path=config_path)
    revision = resolved['trajectories'][0]['config']['model']['revision']
    cache_ref = {'path': str(Path(cache_manifest).resolve()), 'sha256': cache_sha256}
    assets = json.loads(_read_reference(model_binding).read_text())
    if _asset_path(assets['path']) != model_path or assets['revision'] != revision:
        raise ValueError('recipe acquired model binding path/revision differs')
    inventory = recipe_source_inventory()
    manifest = {'schema': SCHEMA, 'source_commit': source_state()['revision'],
        'source_inventory': inventory, 'source_identity': canonical_digest(inventory),
        'synthetic_cpu': synthetic, 'study_pairing_id': study_pairing_id,
        'model_assets': assets, 'model_binding': deepcopy(model_binding), 'model_config': {'path': str(config_path), 'sha256': file_digest(config_path)},
        'cache': cache_ref, 'initializations': deepcopy(initializations),
        'observation_template': deepcopy(observation_template), 'terminal_template': deepcopy(terminal_template),
        'resolved': resolved, 'launch_authorized': False}
    manifest['resolved'] = workflow_resolution(resolved, manifest)
    manifest['manifest_identity'] = canonical_digest(manifest)
    validate_recipe_prepared(manifest)
    # Initializer integrity is a setup boundary; never decode it per update.
    for ref in initializations.values():
        if 'stat' in ref:
            _read_reference(ref, trusted_stat=True)
            continue
        initial = torch.load(_read_reference(ref, trusted_stat=True), map_location='cpu', weights_only=True)
        if state_dict_identity(initial) != ref['state_identity']:
            raise ValueError('recipe acquired initializer tensor identity differs')
    with Path(output).open('x') as handle:
        handle.write(json.dumps(manifest, indent=2)+'\n')
    return manifest


def load_recipe_workflow(path, *, checksum, width, ratio, resolved=None, load_data=True):
    manifest = _read_prepared(path, checksum)
    resolved = validate_recipe_prepared(manifest, resolved=resolved)
    matches = [d for d in resolved['trajectories'] if (d['width'], d['ratio']) == (width, ratio)]
    if len(matches) != 1:
        raise ValueError('recipe requires exact declared width/ratio')
    description = matches[0]
    ref = manifest['initializations'][str(width)]
    initial = torch.load(_read_reference(ref, trusted_stat=True), map_location='cpu', weights_only=True)
    if state_dict_identity(initial) != ref['state_identity']:
        raise ValueError('recipe initializer tensor identity differs')
    if not load_data:
        return manifest, description['config'], None, None, initial, None, None
    stream = load_recipe_cached_data(description, cache_manifest=manifest['cache']['path'], checksum=manifest['cache']['sha256'])
    plan = RecipePlan(description, stream.views, manifest['study_pairing_id'])
    observation = json.loads(_read_reference(manifest['observation_template']).read_text())
    terminal = json.loads(_read_reference(manifest['terminal_template']).read_text())
    metadata = {'source_identity': manifest['source_identity'], 'source_commit': manifest['source_commit'],
        'model_assets_identity': canonical_digest(manifest['model_assets']),
        'config_identity': canonical_digest(description['config']), 'data_identity': canonical_digest(stream.data_ids),
        'prepared_identity': manifest['manifest_identity'], 'initialization_identity': ref['state_identity'],
        'observation_identity': recipe_observation_identity(description, observation),
        'terminal_template_identity': canonical_digest(scoring_template(
            terminal, description['evaluation'], description['evaluation'])),
        'model_revision': description['config']['model']['revision']}
    return manifest, description['config'], plan, stream, initial, metadata, stream.data_ids


def initialize_recipe(config, initial, locations, *, evaluation=False, synthetic=False):
    from .models.split_model import build_model, resolve_training_site
    from .runtime import configure_training_determinism, seed_everything, prepare_trainable_parameters
    verify_model_assets(config)
    configure_training_determinism(config['training'])
    seed_everything(config['seed'])
    processor, model = build_model(deepcopy(config), local_files_only=True)
    prepare_trainable_parameters(model)
    for site in locations:
        resolved = resolve_training_site(model.base, site, config['model']['revision'])
        if resolved.hidden_dim != config['codec']['hidden_dim']:
            raise ValueError('recipe actual loaded hidden dimension differs')
    model.codec.load_state_dict(initial, strict=True)
    if state_dict_identity(model.codec.state_dict()) != state_dict_identity(initial):
        raise ValueError('recipe exact FP32 initializer load differs')
    if (next(model.base.parameters()).dtype != getattr(torch, config['model']['dtype'])
            or model.base.config.decoder._attn_implementation != config['model']['attn_implementation']):
        raise ValueError('recipe actual backbone dtype/backend differs')
    if evaluation:
        model.enable_universal_codec()
    return processor, model


def verify_recipe_source(manifest, approved_source):
    if manifest['source_inventory'] != recipe_source_inventory():
        raise ValueError('recipe source inventory differs')
    verify_original13_execution_source(manifest, approved_source)


def training_cli(args):
    """Generalized branch of the existing source-bound training command."""
    from .original13_provenance import original13_execution_guard
    from .original13_training import run_recipe_training
    from .encoder_stages_evaluation import make_recipe_observer
    from .experiment_state import CheckpointRef
    manifest = _read_prepared(args.prepared, args.prepared_sha256)
    resolved = manifest['resolved']
    if args.recipe:
        resolved = resolve_recipe(args.recipe, model_config_path=_read_reference(manifest['model_config']))
    if args.resolved:
        from .recipe_submission import validate_bundle, normalize_path
        bundle = validate_bundle(json.loads(Path(args.resolved).read_text()))
        if (bundle['identity'] != args.resolved_identity
                or not 0 <= args.trajectory < len(bundle['resolved']['trajectories'])
                or bundle['prepared']['sha256'] != args.prepared_sha256
                or normalize_path(bundle['prepared']['path']) != normalize_path(str(Path(args.prepared).resolve()))
                or bundle['source']['commit'] != manifest['source_commit']
                or normalize_path(bundle['model']['path']) != normalize_path(manifest['model_assets']['path'])):
            raise ValueError('recipe resolved submission binding differs')
        resolved = bundle['resolved']
        selected = resolved['trajectories'][args.trajectory]
        if (args.width, args.ratio) != (selected['width'], selected['ratio']):
            raise ValueError('recipe resolved trajectory differs')
    if args.width is None or args.ratio is None:
        if args.mode != 'check':
            raise ValueError('recipe train/resume require exact width/ratio')
        selected = resolved['trajectories'][0]
        args.width, args.ratio = selected['width'], selected['ratio']
    if args.mode == 'check':
        bound = validate_recipe_prepared(manifest, resolved=resolved)
        if sum((d['width'], d['ratio']) == (args.width, args.ratio) for d in bound['trajectories']) != 1:
            raise ValueError('recipe requires exact declared width/ratio')
        return {'schema': SCHEMA, 'status': 'valid', 'resolved': bound,
                'model_assets_status': manifest['model_assets']['status'], 'launch_authorized': False}
    manifest, config, plan, stream, initial, metadata, data_ids = load_recipe_workflow(
        args.prepared, checksum=args.prepared_sha256, width=args.width, ratio=args.ratio,
        resolved=resolved)
    if not args.output or not args.approved_source:
        raise ValueError('recipe execution requires output and approved source')
    verify_recipe_source(manifest, args.approved_source)
    resume = None
    if args.mode == 'resume':
        if not args.resume_checkpoint or not args.resume_sha256 or not args.resume_size:
            raise ValueError('recipe resume requires exact checkpoint path/hash/size')
        resume = CheckpointRef(args.resume_checkpoint, args.resume_sha256, args.resume_size)
    elif args.resume_checkpoint or args.resume_sha256 or args.resume_size:
        raise ValueError('fresh recipe train cannot take continuation fields')
    guard = original13_execution_guard(config, synthetic=manifest['synthetic_cpu'],
        timeout_seconds=args.timeout_seconds, device_uuid=args.device_uuid)
    guard()
    assert plan is not None and metadata is not None and stream is not None
    processor, model = initialize_recipe(config, initial, plan.trained_sites, synthetic=manifest['synthetic_cpu'])
    if config['model']['device'] == 'cuda' and (torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported()):
        raise ValueError('recipe requires one assigned BF16-capable GPU')
    guard()
    observation = json.loads(_read_reference(manifest['observation_template']).read_text())
    return run_recipe_training(model, processor, plan=plan, batches=stream, metadata=metadata,
        output=args.output, guard=guard, observe=make_recipe_observer(observation), resume=resume,
        data_ids=data_ids, synthetic=manifest['synthetic_cpu'])


def evaluation_cli(args):
    from . import encoder_stages_evaluation as evaluation
    from .original13_state import _recipe_plan
    from .original13_provenance import original13_execution_guard
    from .baseline_protocol import write_manifest

    if args.command in ('run', 'check'):
        request = json.loads(_read_reference({'path': str(Path(args.request).resolve()),
                                             'sha256': args.request_sha256}).read_text())
        state, _ = evaluation.open_recipe_terminal(request['training_run'])
        evaluation._validate_recipe_request(state, request, observation=False)
        if args.command == 'check':
            result = {'status': 'valid', 'checkpoint': state.reference.sha256, 'launch_authorized': False}
        else:
            own_plan = _recipe_plan(state.payload)
            manifest = _read_prepared(args.prepared, args.prepared_sha256)
            resolved = deepcopy(manifest['resolved'])
            resolved['trajectories'] = [own_plan.description]
            resolved['identity'] = canonical_digest({k: v for k, v in resolved.items() if k != 'identity'})
            manifest, config, plan, _, initial, metadata, _ = load_recipe_workflow(args.prepared,
                checksum=args.prepared_sha256, width=own_plan.width, ratio=own_plan.ratio, resolved=resolved)
            verify_recipe_source(manifest, args.approved_source)
            assert plan is not None and metadata is not None
            for key, expected in {**metadata, 'plan_identity': plan.identity}.items():
                if state.payload['metadata'].get(key) != expected:
                    raise ValueError('recipe evaluation acquired identity differs: '+key)
            guard = original13_execution_guard(config, synthetic=manifest['synthetic_cpu'],
                timeout_seconds=args.timeout_seconds, device_uuid=args.device_uuid)
            guard()
            processor, model = initialize_recipe(config, initial, plan.trained_sites,
                evaluation=True, synthetic=manifest['synthetic_cpu'])
            if config['model']['device'] == 'cuda' and (torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported()):
                raise ValueError('recipe evaluation requires one assigned BF16-capable GPU')
            guard()
            result = evaluation.run_recipe_evaluation(model, processor, request=request, output=args.output, guard=guard)
        print(json.dumps(result, indent=2))
        return result
    if args.command == 'request':
        supplied = {'path': str(Path(args.template).resolve()), 'sha256': args.template_sha256}
        value = evaluation.terminal_recipe_request(args.training_run,
            json.loads(_read_reference(supplied).read_text()), scope=args.scope)
    else:
        value = evaluation.recipe_report(args.training_run, initial=args.initial, supplemental=args.supplemental,
            observations=args.observation,
            initial_request=json.loads(Path(args.initial_request).read_text()) if args.initial_request else None,
            supplemental_request=json.loads(Path(args.supplemental_request).read_text()) if args.supplemental_request else None)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x'):
        pass
    write_manifest(path, value)
    result = {'path': str(path.resolve()), 'sha256': file_digest(path)}
    print(json.dumps(result, indent=2))
    return result
