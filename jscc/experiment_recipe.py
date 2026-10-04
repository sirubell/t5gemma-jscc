"""Resolve YAML into complete per-trajectory configs, counts and consumer requests."""
from copy import deepcopy
import json
import math
from pathlib import Path
import re

import yaml

from .activation_replay import canonical_digest
from .config import load_config, validate_config
from .recipe_observation_policy import checkpoint_policy

RECIPE_SCHEMA = 'shared-codec-recipe-v1'


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} requires a positive integer')
    return value


def _number(value, name, *, minimum: float = 0, maximum: float | None = None):
    if (type(value) not in (int, float) or not math.isfinite(value) or value < minimum
            or (maximum is not None and value > maximum)):
        raise ValueError(f'{name} requires a finite number in the supported range')
    return value


def _validate_scoring_conditions(conditions, name, *, allow_empty=False, allow_no_noise=True):
    if not isinstance(conditions, list) or (not conditions and not allow_empty):
        requirement = 'a list' if allow_empty else 'a nonempty list'
        raise ValueError(f'{name} requires {requirement}')
    for condition in conditions:
        if allow_no_noise and condition == 'no_noise':
            continue
        _number(condition, 'SNR' if allow_no_noise else 'supplemental SNR', minimum=-1000, maximum=1000)
    if len(set(conditions)) != len(conditions):
        raise ValueError(f'{name} must not contain duplicates')


def _merge(base, changes):
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = deepcopy(value)


def resolve_locations(locations, model_metadata, config):
    """Validate canonical names against local HF backbone metadata, not a cohort."""
    if (not isinstance(locations, list) or not locations or any(not isinstance(s, str) for s in locations)
            or len(set(locations)) != len(locations)):
        raise ValueError('locations must be a nonempty ordered list without duplicates')
    revision = model_metadata.get('_commit_hash')
    if revision is not None and revision != config['model']['revision']:
        raise ValueError('model metadata revision differs from recipe')
    topology = {}
    for stack in ('encoder', 'decoder'):
        # T5Gemma2 serializes the multimodal encoder's text topology inside
        # text_config; decoder and text-only metadata retain the flat layout.
        metadata = model_metadata[stack].get('text_config', model_metadata[stack])
        count = _positive_int(metadata['num_hidden_layers'], f'{stack} layers')
        hidden = _positive_int(metadata['hidden_size'], f'{stack} hidden size')
        if hidden != config['codec']['hidden_dim']:
            raise ValueError('model/codec hidden dimension differs')
        topology[stack] = {'layers': count, 'hidden_dim': hidden}
    for site in locations:
        if site in ('enc_emb', 'enc_fn'):
            continue
        match = re.fullmatch(r'(enc|dec)_l(0|[1-9][0-9]*)', site) if isinstance(site, str) else None
        if match is None:
            raise ValueError(f'invalid model location: {site!r}')
        stack, index = match[1], int(match[2])
        limit = topology['encoder' if stack == 'enc' else 'decoder']['layers'] - (stack == 'dec')
        if index >= limit:
            raise ValueError(f'location outside actual model topology or receiver suffix: {site}')
    return topology



def resolve_recipe(path, *, model_config_path):
    """Resolve without downloading assets, loading tensors or authorizing a run.

    model_config_path is the config.json from the locally cached selected
    backbone. Actual runtime loaders must recheck its topology on model load.
    """
    path = Path(path).resolve()
    recipe = yaml.safe_load(path.read_text())
    required = {'schema', 'task_config', 'locations', 'updates_per_location', 'widths',
                'reconstruction_percentages', 'overrides', 'objectives', 'save', 'observation', 'evaluation'}
    if not isinstance(recipe, dict) or set(recipe) != required or recipe['schema'] != RECIPE_SCHEMA:
        raise ValueError('unsupported recipe schema or fields (derived budgets must not be declared)')
    config = load_config(path.parent / recipe['task_config'])
    if config['task'] != 'hellaswag':
        raise ValueError('shared-codec recipes currently support HellaSwag; legacy COCO is unchanged')
    overrides = recipe['overrides']
    if not isinstance(overrides, dict) or set(overrides) - {'data', 'training', 'channel', 'run'}:
        raise ValueError('recipe overrides support data/training/channel/run only')
    conflicts = {'max_steps', 'schedule_steps', 'save_steps', 'presentation_stream',
                 'kl_weight', 'mse_weight', 'loss_weights'}
    if conflicts & set(overrides.get('training', {})):
        raise ValueError('conflicting independently declared budget, save or objective settings')
    _merge(config, overrides)
    channel = config['channel']
    if channel['type'] not in ('awgn', 'identity'):
        raise ValueError('recipe channel currently supports awgn or identity')
    if not isinstance(channel.get('kwargs', {}), dict) or channel.get('kwargs', {}):
        raise ValueError('awgn/identity channel kwargs must be empty')
    for key in ('train_noise', 'normalize_power'):
        if type(channel[key]) is not bool:
            raise ValueError(f'channel.{key} requires a boolean')
    snrs = channel['train_snr_range']
    if not isinstance(snrs, list) or len(snrs) != 2:
        raise ValueError('channel.train_snr_range requires two finite ordered endpoints')
    for snr in snrs:
        _number(snr, 'channel SNR', minimum=-1000, maximum=1000)
    if snrs[0] > snrs[1]:
        raise ValueError('channel.train_snr_range must be ordered')
    _number(channel['clean_film_snr'], 'channel clean SNR', minimum=-1000, maximum=1000)
    q = _positive_int(recipe['updates_per_location'], 'updates_per_location')
    if (type(config['training']['batch_size']) is not int or config['training']['batch_size'] != 16
            or type(config['training']['gradient_accumulation']) is not int
            or config['training']['gradient_accumulation'] != 1):
        raise ValueError('shared-codec recipes support native16/accumulation1 only')
    data = config['data']
    if data.get('num_train') is not None:
        _positive_int(data['num_train'], 'data.num_train')
    if 'row_ids' in data and (not isinstance(data['row_ids'], list) or not data['row_ids']
            or any(type(row) is not int or row < 0 for row in data['row_ids'])):
        raise ValueError('data.row_ids must select a nonempty list of nonnegative rows')
    if 'row_ids' in data and data.get('num_train') is not None:
        raise ValueError('select data with row_ids or num_train, not both')
    model_metadata = json.loads(Path(model_config_path).read_text())
    topology = resolve_locations(recipe['locations'], model_metadata, config)
    n = len(recipe['locations'])
    widths, ratios = recipe['widths'], recipe['reconstruction_percentages']
    for name, values in [('widths', widths), ('reconstruction_percentages', ratios)]:
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f'{name} requires nonempty unique choices')
        for value in values:
            _positive_int(value, name)
    if any(r >= 100 or q * r % 100 for r in ratios):
        raise ValueError('phase ratios require 0 < percent < 100 and integral per-location phase quotas')
    training = config['training']
    training.setdefault('betas', [0.9, 0.999])
    training.setdefault('eps', 1e-8)
    training.setdefault('optimizer', 'adamw')
    if training['optimizer'] != 'adamw':
        raise ValueError('recipe optimizer currently supports adamw only')
    if (not isinstance(training['betas'], list) or len(training['betas']) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v < 1
                   for v in training['betas'])):
        raise ValueError('Adam betas require two finite values in [0, 1)')
    _number(training['eps'], 'Adam eps', minimum=1e-30)
    for key in ('lr', 'weight_decay', 'grad_clip', 'temperature'):
        _number(training[key], f'training.{key}', minimum=0 if key == 'weight_decay' else 1e-30)
    _number(training['warmup_ratio'], 'training.warmup_ratio', maximum=1)
    _number(training['min_lr_ratio'], 'training.min_lr_ratio', maximum=1)
    if training['warmup_ratio'] >= 1:
        raise ValueError('warmup_ratio must be less than 1')
    objectives = recipe['objectives']
    if not isinstance(objectives, dict) or set(objectives) != {'encoder', 'decoder', 'functional_kl'}:
        raise ValueError('objectives require encoder, decoder and functional_kl')
    for role in ('encoder', 'decoder'):
        if not isinstance(objectives[role], dict) or set(objectives[role]) != {'hidden', 'memory'}:
            raise ValueError('role objective requires hidden and memory weights')
        for value in objectives[role].values():
            _number(value, 'objective weight')
    _number(objectives['functional_kl'], 'functional KL')
    if objectives['encoder']['memory'] != 0:
        raise ValueError('encoder objective has no receiver memory role')
    observation = recipe['observation']
    if not isinstance(observation, dict) or set(observation) != {'items', 'sites', 'conditions'}:
        raise ValueError('observation requires items, sites and conditions')
    _positive_int(observation['items'], 'observation items')
    resolve_locations(observation['sites'], model_metadata, config)
    for name, request in (('observation', observation), ('evaluation', recipe['evaluation'])):
        if not isinstance(request, dict):
            raise ValueError('scoring requests require conditions')
        _validate_scoring_conditions(request.get('conditions'), f'{name}.conditions')
    evaluation = recipe['evaluation']
    if set(evaluation) != {'items', 'conditions', 'supplemental_conditions', 'candidate_batch', 'fewshot', 'scoring_policy'}:
        raise ValueError('evaluation request fields differ')
    _positive_int(evaluation['items'], 'evaluation items')
    _positive_int(evaluation['candidate_batch'], 'candidate_batch')
    if type(evaluation['fewshot']) is not int or evaluation['fewshot'] < 0 or evaluation['scoring_policy'] != 'fp32-v1':
        raise ValueError('invalid fewshot/scoring policy')
    _validate_scoring_conditions(evaluation['supplemental_conditions'], 'evaluation.supplemental_conditions',
                                 allow_empty=True, allow_no_noise=False)
    counts = {'locations': n, 'optimizer_updates': n * q,
              'base_stream_presentations': q * 16, 'presentations_per_location': q * 16,
              'site_presentations': n * q * 16}
    trajectories = []
    for width in widths:
        for ratio in ratios:
            resolved = deepcopy(config)
            resolved['codec']['bottleneck_dim'] = width
            if 'bottleneck_dim' in resolved['codec'].get('memory', {}):
                resolved['codec']['memory']['bottleneck_dim'] = width
            resolved['training'].update(max_steps=n*q, schedule_steps=n*q,
                                        kl_weight=objectives['functional_kl'], mse_weight=objectives['encoder']['hidden'])
            resolved['training']['presentation_stream'] = {
                **config['training'].get('presentation_stream', {}), 'total_presentations': q*16}
            resolved['training'].pop('save_steps', None)
            resolved['protocol'] = RECIPE_SCHEMA
            validate_config(resolved)
            stop = n * (q * ratio // 100)
            checkpoints, observations = checkpoint_policy(q, ratio, n, recipe['save'])
            item = {'locations': deepcopy(recipe['locations']), 'updates_per_location': q,
                    'width': width, 'ratio': ratio, 'counts': counts, 'config': resolved,
                    'objectives': deepcopy(objectives), 'checkpoint_states': checkpoints,
                    'observations': [{**deepcopy(observation), 'step': step, 'state': state}
                                     for step, state in observations], 'evaluation': deepcopy(evaluation),
                    'phases': [{'phase_id': 'reconstruction', 'objective_kind': 'reconstruction',
                                'start': 0, 'stop': stop, 'reset_optimizer': False},
                               {'phase_id': 'functional', 'objective_kind': 'combined',
                                'start': stop, 'stop': n*q, 'reset_optimizer': True}]}
            item['identity'] = canonical_digest(item)
            trajectories.append(item)
    result = {'schema': RECIPE_SCHEMA, 'model_topology': topology, 'trajectories': trajectories,
              'matrix_counts': {key: value * len(trajectories) for key, value in counts.items() if key != 'locations'},
              'launch_authorized': False}
    result['identity'] = canonical_digest(result)
    return result


def validate_retained_recipe(description, plan, config, observation_request):
    """Bind a YAML description to the retained execution, rejecting new semantics.

    Generalized consumers are delivered separately. This bridge lets the real
    original13 CLI verify a retained YAML recipe without rewriting its learner.
    """
    expected = {k: plan.resolved()['objectives'][k] for k in ('encoder', 'decoder', 'functional_kl')}
    if (description['objectives'] != expected
            or description['locations'] != list(plan.trained_sites)
            or description['updates_per_location'] != plan.q
            or description['width'] != plan.width or description['ratio'] != plan.ratio
            or description['checkpoint_states'] != [list(s) for s in plan.checkpoint_states]
            or [(o['step'], o['state']) for o in description['observations']] != list(plan.observations)):
        raise ValueError('recipe differs from retained execution; requires generalized workflow integration')
    expected_config = deepcopy(config)
    expected_config['protocol'] = RECIPE_SCHEMA
    expected_training = expected_config['training']
    for key, value in {'optimizer': 'adamw', 'betas': [0.9, 0.999], 'eps': 1e-8}.items():
        expected_training.setdefault(key, value)
    expected_training.update(max_steps=plan.horizon, schedule_steps=plan.horizon,
                             kl_weight=expected['functional_kl'], mse_weight=expected['encoder']['hidden'])
    expected_training['presentation_stream'] = {
        **config['training'].get('presentation_stream', {}), 'total_presentations': plan.q * 16}
    expected_training.pop('save_steps', None)
    if description['config'] != expected_config:
        raise ValueError('recipe complete config differs from retained prepared execution')
    for request in description['observations']:
        if (request['items'] != observation_request['panel']['expected_items']
                or request['sites'] != observation_request['sites']
                or request['conditions'] != observation_request['conditions']):
            raise ValueError('recipe observation differs from retained execution')
    evaluation = description['evaluation']
    legacy = plan.resolved()['terminal_requests']
    if (evaluation['items'] != legacy['items'] or evaluation['conditions'] != legacy['initial']
            or evaluation['supplemental_conditions'] != legacy['supplemental_supported']
            or any(evaluation[k] != legacy[k] for k in ('candidate_batch', 'fewshot', 'scoring_policy'))):
        raise ValueError('recipe terminal request differs from retained execution')
