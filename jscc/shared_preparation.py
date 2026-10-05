"""Current shared preparation bodies; saved record meanings preserved."""
from __future__ import annotations

from pathlib import Path
from .activation_replay import file_digest
from .checkpoint_readers import stages_source_inventory
import json
from .activation_replay import canonical_digest
from copy import deepcopy
from .shared_config import resolve_locations
import re
from .asset_identity import verify_model_assets
from .observation_policy import recipe_observation_identity, scoring_template
import torch
from .shared_config import resolve_recipe
from .asset_identity import state_dict_identity
from .shared_inputs import load_recipe_cached_data
from .shared_plan import RecipePlan
from .checkpoint_readers import verify_original13_execution_source

SCHEMA = 'recipe-prepared-v1'


def recipe_source_inventory():
    return stages_source_inventory()


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


def _acquired_initializer(reference):
    """Only a checksum-bound owned acquisition record permits cheap reuse."""
    acquisition = reference.get('acquisition')
    if acquisition is None:
        return False
    record = json.loads(_read_reference(acquisition).read_text())
    if (record.get('schema') != 'recipe-initializer-acquisition-v1'
            or record.get('identity') != canonical_digest({k: v for k, v in record.items() if k != 'identity'})
            or record.get('reference') != {k: v for k, v in reference.items() if k != 'acquisition'}):
        raise ValueError('recipe initializer acquisition evidence differs')
    return True


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
    if (cache.get('schema') not in ('recipe-data-cache-v1', 'shared-coco-cache-v1') or cache.get('identity') != canonical_digest(
            {k: v for k, v in cache.items() if k != 'identity'})
            or _asset_path(cache['tokenizer']['path']) != _asset_path(manifest['model_assets']['path'])):
        raise ValueError('recipe cache/model acquisition differs')
    from .shared_inputs import validate_recipe_cache
    validate_recipe_cache(cache)
    for item in bound['trajectories']:
        config = item['config']
        if cache['dataset']['name'] != config['data']['name'] or cache['dataset']['revision'] != config['data']['revision']:
            raise ValueError('recipe dataset acquisition differs')
        if cache.get('task', 'hellaswag') != config['task']:
            raise ValueError('recipe task differs from acquired cache')
        verify_model_assets(config)
        if not manifest['synthetic_cpu'] and not manifest['model_assets']['production']:
            raise ValueError('recipe production requires acquired pinned model assets')
        for name, record in cache['tokenizer']['files'].items():
            if manifest['model_assets']['files'].get(name) != record:
                raise ValueError('recipe model/cache tokenizer bytes differ')
        ref = manifest['initializations'].get(str(item['width']))
        if not isinstance(ref, dict) or ref.get('seed') != config['seed']:
            raise ValueError('recipe acquired width initializer missing or seed differs')
        _read_reference(ref, trusted_stat=_acquired_initializer(ref))
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
            if config['task'] == 'coco':
                n = template['expected_items']
                corpus = deepcopy(cache['corpora'][key])
                corpus['image_ids'], corpus['references'] = corpus['image_ids'][:n], corpus['references'][:n]
                corpus.update(cache_identity=cache['identity'],processor_identity=cache['processor_identity'])
                if (template.get('cache') != manifest['cache'] or template['reference_corpus'] != corpus
                        or cache['karpathy']['name'] != config['data']['karpathy_name']
                        or cache['karpathy']['revision'] != config['data']['karpathy_revision']):
                    raise ValueError('recipe COCO reference/demo/processor/Karpathy binding differs')
    return bound


def bind_recipe_workflow(recipe_path, *, model_config_path, model_path, model_binding, cache_manifest, cache_sha256,
                         initializations, observation_template, terminal_template, study_pairing_id,
                         output, synthetic=False):
    """Validate acquired assets; output=None admits without publishing a receipt."""
    from .runtime import source_state
    model_path = _asset_path(model_path)
    config_path = Path(model_config_path).resolve()
    from .asset_paths import asset_file_path
    resolved = resolve_recipe(recipe_path, model_config_path=config_path)
    if config_path != asset_file_path(model_path, 'config.json'):
        raise ValueError('recipe topology must come from acquired model config')
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
        acquired = _acquired_initializer(ref)
        path = _read_reference(ref, trusted_stat=acquired)
        if acquired:
            continue
        initial = torch.load(path, map_location='cpu', weights_only=True)
        if state_dict_identity(initial) != ref['state_identity']:
            raise ValueError('recipe acquired initializer tensor identity differs')
    if output is not None:
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
    initial = torch.load(_read_reference(ref, trusted_stat=_acquired_initializer(ref)), map_location='cpu', weights_only=True)
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
    """Check a saved prepared receipt; new execution uses recorded cases."""
    if args.mode != 'check':
        raise ValueError('generalized prepared train/resume retired; use experiment.py --config TRAIN_REQUEST')
    manifest = _read_prepared(args.prepared, args.prepared_sha256)
    resolved = manifest['resolved']
    if args.recipe:
        resolved = resolve_recipe(args.recipe, model_config_path=_read_reference(manifest['model_config']))
    if args.resolved:
        from .submission import validate_bundle, normalize_path
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
        selected = resolved['trajectories'][0]
        args.width, args.ratio = selected['width'], selected['ratio']
    bound = validate_recipe_prepared(manifest, resolved=resolved)
    if sum((d['width'], d['ratio']) == (args.width, args.ratio) for d in bound['trajectories']) != 1:
        raise ValueError('recipe requires exact declared width/ratio')
    # Legacy caller-supplied tensors retain their setup identity check.
    for ref in manifest['initializations'].values():
        if not _acquired_initializer(ref):
            initial = torch.load(_read_reference(ref), map_location='cpu', weights_only=True)
            if state_dict_identity(initial) != ref['state_identity']:
                raise ValueError('recipe initializer tensor identity differs')
    return {'schema': SCHEMA, 'status': 'valid', 'resolved': bound,
            'model_assets_status': manifest['model_assets']['status'], 'launch_authorized': False}


def evaluation_cli(args):
    """Retain saved request/check/report readers without loose execution."""
    if args.command == 'run':
        raise ValueError('generalized prepared evaluation run retired; use experiment.py --config EVALUATION_REQUEST')
    from . import panel_evaluation as evaluation
    from .experiment_records import write_manifest

    if args.command == 'check':
        request = json.loads(_read_reference({'path': str(Path(args.request).resolve()),
                                             'sha256': args.request_sha256}).read_text())
        state, _ = evaluation.open_recipe_terminal(request['training_run'])
        evaluation._validate_recipe_request(state, request, observation=False)
        result = {'status': 'valid', 'checkpoint': state.reference.sha256, 'launch_authorized': False}
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

