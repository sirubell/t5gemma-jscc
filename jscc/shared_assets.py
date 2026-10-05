"""Current shared assets bodies; saved record meanings preserved."""
from __future__ import annotations

import json
from pathlib import Path
from .activation_replay import file_digest
from .activation_replay import canonical_digest
from copy import deepcopy
from .asset_identity import verify_model_assets
from .observation_policy import scoring_template
from .shared_preparation import _read_reference
from .shared_preparation import validate_recipe_prepared
from .config import load_config
import time
import torch
from .models.codec import Codec
from .asset_identity import state_dict_identity
from .shared_config import resolve_recipe, read_recipe
from .shared_preparation import bind_recipe_workflow

SCHEMA = 'recipe-assets-registration-v1'


def _write(path, value):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return {'path': str(Path(path).resolve()), 'sha256': file_digest(path)}


def _read_registration(path):
    record = json.loads(Path(path).read_text())
    if (record.get('schema') != SCHEMA or record.get('identity') != canonical_digest(
            {k: v for k, v in record.items() if k != 'identity'})):
        raise ValueError(f'recipe asset registration identity differs: {path}')
    return record


def register_recipe_assets(*, model_name, model_binding, cache, observation_template,
                           terminal_template, output, synthetic=False):
    """Explicitly register checksum-bound acquisition receipts and fixed panels.

    model_name is the exact task YAML model name (hub ID or local path). Input
    references contain absolute path/sha256. Existing bind_model_assets and
    bind_recipe_data_cache own full-byte acquisition; no manual assets manifest
    or initializer is needed. Output is a fresh registration JSON.
    """
    if not isinstance(model_name, str) or not model_name.strip() or type(synthetic) is not bool:
        raise ValueError('registration requires exact model name and explicit synthetic flag')
    assets = json.loads(_read_reference(model_binding).read_text())
    verify_model_assets({'model': {'name': assets['path'], 'revision': assets['revision'],
                                  'asset_binding': assets}})
    if not synthetic and not assets['production']:
        raise ValueError('production registration requires acquired pinned HF snapshot')
    cached = json.loads(_read_reference(cache).read_text())
    if (cached.get('schema') not in ('recipe-data-cache-v1', 'shared-coco-cache-v1') or cached.get('identity') != canonical_digest(
            {k: v for k, v in cached.items() if k != 'identity'})
            or cached['tokenizer']['revision'] != assets['revision']):
        raise ValueError('registration cache/model revision or path differs')
    from .asset_paths import normalize_asset_path
    from .shared_inputs import validate_recipe_cache
    validate_recipe_cache(cached)
    if normalize_asset_path(cached['tokenizer']['path']) != normalize_asset_path(assets['path']):
        raise ValueError('registration cache/model path differs')
    for name, expected in cached['tokenizer']['files'].items():
        if assets['files'].get(name) != expected:
            raise ValueError('registration tokenizer/config content differs: '+name)
    for ref, key in ((observation_template, 'development'), (terminal_template, 'final_validation')):
        template = json.loads(_read_reference(ref).read_text())
        panel = cached['fixed_data_ids'][key]
        task = cached.get('task', 'hellaswag')
        if (template['task'] != task or template['input_ids'] != panel['input_ids']
                or template['source_family_ids'] != panel['source_family_ids']
                or template['data_settings']['name'] != cached['dataset']['name']
                or template['data_settings']['revision'] != cached['dataset']['revision']
                or ('noise' in panel and panel['noise'] != template['noise'])):
            raise ValueError('registration fixed panel membership/data/noise differs: '+key)
        if task == 'coco':
            corpus = {**cached['corpora'][key], 'cache_identity':cached['identity'],
                      'processor_identity':cached['processor_identity']}
            if template.get('cache') != cache or template['reference_corpus'] != corpus:
                raise ValueError('registration COCO cache/reference/demo/processor differs')
            evaluation = {'batch_size':template['settings']['batch_size'],
                'max_new_tokens':template['settings']['max_new_tokens'],
                'scoring_policy':template['settings']['scoring_policy']}
        else:
            evaluation = {'candidate_batch':template['settings']['batch_size'],
                'fewshot':template['settings']['num_fewshot'],
                'scoring_policy':template['settings']['scoring_policy']}
        scoring_template(template, {'items':len(template['input_ids'])}, evaluation)
    record = {'schema': SCHEMA, 'model_name': model_name, 'model_revision': assets['revision'],
              'dataset_name': cached['dataset']['name'], 'dataset_revision': cached['dataset']['revision'],
              'model_binding': deepcopy(model_binding), 'cache': deepcopy(cache),
              'observation_template': deepcopy(observation_template),
              'terminal_template': deepcopy(terminal_template), 'synthetic_cpu': synthetic}
    record['identity'] = canonical_digest(record)
    _write(output, record)
    return record


def register_prepared_assets(prepared, *, model_name, output):
    """Register a retained prepared receipt without relabeling or changing it."""
    manifest = json.loads(Path(prepared).read_text())
    validate_recipe_prepared(manifest, check_source=False)
    return register_recipe_assets(model_name=model_name, model_binding=manifest['model_binding'],
        cache=manifest['cache'], observation_template=manifest['observation_template'],
        terminal_template=manifest['terminal_template'], output=output, synthetic=manifest['synthetic_cpu'])


def discover_recipe_assets(recipe_path, registrations):
    """Select one exact name/revision/data registration; never choose latest."""
    path = Path(recipe_path).resolve()
    recipe = read_recipe(path)
    config = load_config(path.parent/recipe['task_config'], validate=False)
    # resolve_recipe validates all overrides; selection only needs their data pin.
    data = {**config['data'], **recipe.get('overrides', {}).get('data', {})}
    key = (config['model']['name'], config['model']['revision'], data['name'], data['revision'])
    matches = {}
    for item in sorted(Path(registrations).glob('*.json')):
        record = _read_registration(item)
        if (record['model_name'], record['model_revision'], record['dataset_name'], record['dataset_revision']) == key:
            matches[record['identity']] = record
    if not matches:
        raise FileNotFoundError(f'no registered pinned recipe assets for {key!r}; explicitly acquire/register assets first')
    if len(matches) != 1:
        raise ValueError(f'ambiguous/conflicting registered recipe assets for {key!r}; select a directory containing one exact registration')
    return next(iter(matches.values()))


def _panel(reference, count, evaluation):
    template = json.loads(_read_reference(reference).read_text())
    if count > len(template['input_ids']):
        raise ValueError('recipe panel exceeds acquired fixed panel; explicitly acquire/register the larger panel')
    template['input_ids'] = template['input_ids'][:count]
    template['source_family_ids'] = template['source_family_ids'][:count]
    # Preserve acquired prompt/reference/noise/layout identity. Settings are
    # derived from YAML only when they match the acquired scoring policy.
    if template['task'] == 'coco':
        if template['settings']['scoring_policy'] != evaluation['scoring_policy']:
            raise ValueError('recipe COCO scoring policy differs from acquired panel')
        template['reference_corpus']['image_ids'] = template['reference_corpus']['image_ids'][:count]
        template['reference_corpus']['references'] = template['reference_corpus']['references'][:count]
    elif (template['settings']['num_fewshot'] != evaluation['fewshot']
            or template['settings']['scoring_policy'] != evaluation['scoring_policy']):
        raise ValueError('recipe scoring/prompt policy differs from acquired panel; register matching fixed inputs')
    return scoring_template(template, {'items': count}, evaluation)


def _initializer(config, directory):
    """Create one deterministic codec per design/seed, then reuse stat-bound bytes."""
    from .asset_paths import asset_stat
    identity = canonical_digest({'codec': config['codec'], 'seed': config['seed'],
                                 'implementation': file_digest(Path(__file__).parent/'models/codec.py'),
                                 'torch': torch.__version__, 'acquisition_schema': 'recipe-initializer-acquisition-v1'})
    directory.mkdir(exist_ok=True)
    own = directory/identity
    try:
        own.mkdir()
    except FileExistsError:
        record_path = own/'reference.json'
        deadline = time.monotonic()+10
        while not record_path.is_file() and time.monotonic() < deadline:
            time.sleep(.05)
        if not record_path.is_file():
            raise ValueError(f'initializer acquisition incomplete/in progress: {own}; retain and reconcile before retry') from None
        record = json.loads(record_path.read_text())
        if record.get('identity') != canonical_digest({k: v for k, v in record.items() if k != 'identity'}):
            raise ValueError('initializer acquisition reference identity differs')
        reference = record['reference']
        if (record.get('schema') != 'recipe-initializer-acquisition-v1'
                or record['config_identity'] != identity or reference['seed'] != config['seed']):
            raise ValueError('initializer acquisition config/seed differs')
        _read_reference(reference, trusted_stat=True)
        return {**reference, 'acquisition': {'path': str(record_path.resolve()), 'sha256': file_digest(record_path)}}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config['seed'])
        initial = Codec(config['codec']['hidden_dim'], deepcopy(config['codec'])).state_dict()
    initial_path = own/'initial.pt'
    torch.save(initial, initial_path)
    reference = {'path': str(initial_path.resolve()), 'sha256': file_digest(initial_path),
                 'state_identity': state_dict_identity(initial), 'seed': config['seed'],
                 'stat': asset_stat(initial_path)}
    record = {'schema': 'recipe-initializer-acquisition-v1', 'config_identity': identity, 'reference': reference}
    record['identity'] = canonical_digest(record)
    acquisition = _write(own/'reference.pending.json', record)
    (own/'reference.pending.json').rename(own/'reference.json')
    acquisition['path'] = str((own/'reference.json').resolve())
    return {**reference, 'acquisition': acquisition}


def prepare_recipe_assets(recipe_path, *, registrations, output, study_pairing_id=None):
    """Prepare recipe-prepared-v1 in a fresh owned directory, cache-only.

    Returns the existing manifest. Creates deterministic width/seed codec states,
    fixed-panel prefixes and complete YAML plans, with no backbone load and no
    per-update tensors. Consumer reads output/prepared.json and its checksum.
    A failure retains partial outputs and never overwrites/retries them.
    """
    started = time.perf_counter()
    record = discover_recipe_assets(recipe_path, registrations)
    assets = json.loads(_read_reference(record['model_binding']).read_text())
    from .asset_paths import asset_file_path, normalize_asset_path
    model_path = normalize_asset_path(assets['path'])
    config_path = asset_file_path(model_path, 'config.json')
    resolved = resolve_recipe(recipe_path, model_config_path=config_path)
    first = resolved['trajectories'][0]
    observation = _panel(record['observation_template'], first['observations'][0]['items'], first['evaluation'])
    terminal = _panel(record['terminal_template'], first['evaluation']['items'], first['evaluation'])
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    observation_ref = _write(output/'observation.json', observation)
    terminal_ref = _write(output/'terminal.json', terminal)
    initializations = {}
    for description in resolved['trajectories']:
        width = str(description['width'])
        if width in initializations:
            continue
        config = description['config']
        seed = config['seed']
        if type(seed) is not int or seed < 0:
            raise ValueError('recipe initialization requires nonnegative integer seed')
        initializations[width] = _initializer(config, Path(registrations)/'.initializers')
    manifest = bind_recipe_workflow(recipe_path, model_config_path=model_path/'config.json',
        model_path=model_path, model_binding=record['model_binding'], cache_manifest=record['cache']['path'],
        cache_sha256=record['cache']['sha256'], initializations=initializations,
        observation_template=observation_ref, terminal_template=terminal_ref,
        study_pairing_id=study_pairing_id or record['identity'], output=output/'prepared.json',
        synthetic=record['synthetic_cpu'])
    _write(output/'preparation.json', {'schema': 'recipe-preparation-cost-v1',
        'seconds': time.perf_counter()-started, 'registration_identity': record['identity'],
        'trajectories': len(resolved['trajectories']), 'initializers': len(initializations),
        'optimizer_updates': resolved['matrix_counts']['optimizer_updates'],
        'launch_authorized': False})
    return manifest

