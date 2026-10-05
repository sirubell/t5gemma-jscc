"""One explicit YAML operation over the concrete ordinary and shared engines."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import re
from typing import Any

import yaml

SCHEMA = 'jscc-experiment-v1'
OPERATIONS = {'check', 'acquire', 'register', 'prepare', 'train', 'resume', 'evaluate', 'collect', 'preview', 'submit'}
# These are authored controls, not arbitrary keys accepted as unused provenance.
FIELDS = {
    'model': {'name', 'revision', 'device', 'dtype', 'local_files_only', 'attn_implementation', 'sdpa_backend_policy', 'numerical_policy'},
    'split': {'stack', 'where', 'index'},
    'codec': {'architecture', 'hidden_dim', 'bottleneck_dim', 'n_res_blocks', 'activation', 'dropout', 'layernorm', 'snr_film', 'film_hidden', 'memory', 'rms_norm_eps'},
    'channel': {'type', 'kwargs', 'normalize_power', 'train_noise', 'train_snr_range', 'clean_film_snr'},
    'run': {'name', 'output_dir', 'wandb_project'},
    'runtime': {'device', 'dtype', 'sdpa_backend_policy', 'numerical_policy'},
    'data': {'name', 'revision', 'karpathy_name', 'karpathy_revision', 'num_train', 'num_validation', 'num_report', 'num_demos', 'max_prompt_length', 'max_target_length', 'max_length', 'num_workers', 'selection_seed', 'prompt_policy', 'row_ids'},
    'training': {'max_steps', 'schedule_steps', 'batch_size', 'gradient_accumulation', 'lr', 'weight_decay', 'warmup_ratio', 'min_lr_ratio', 'grad_clip', 'temperature', 'kl_weight', 'mse_weight', 'loss_weights', 'eval_every', 'validation_snrs', 'validation_batches', 'monitor', 'patience', 'min_steps', 'min_delta', 'log_every', 'save_steps', 'selection_steps', 'streamed_backward', 'valid_only_kl', 'low_sync_logging', 'deterministic_algorithms', 'expected_attention_backend', 'presentation_stream', 'paired_randomness', 'feature_summary', 'record_coco_presentations', 'max_minutes'},
    'evaluation': {'snrs', 'batch_size', 'num_samples', 'max_new_tokens', 'num_fewshot', 'vanilla', 'mode', 'scoring_policy', 'evidence_mode', 'noise_seed', 'deterministic_algorithms', 'attention_backend'},
}
EXECUTION = {
    'check': {'root', 'case', 'resolution'},
    'acquire': {'acquisition'}, 'register': {'registration'},
    'prepare': {'root', 'preparation'},
    'train': {'root', 'attempt', 'case', 'cases', 'prepared', 'timeout_seconds', 'device_uuid', 'success_link'},
    'resume': {'root', 'attempt', 'case', 'continuation', 'timeout_seconds', 'device_uuid', 'success_link'},
    'evaluate': {'root', 'attempt', 'case', 'cases', 'evaluation', 'timeout_seconds', 'device_uuid', 'success_link'},
    'collect': {'root', 'collection'},
    'preview': {'plan', 'submission'}, 'submit': {'submission'},
}


class RequestLoader(yaml.SafeLoader):
    """Reject ambiguous mapping ownership before selecting an operation."""
    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError(f'duplicate YAML mapping key: {key!r}')
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def strict_yaml(filename):
    return yaml.load(Path(filename).read_text(), Loader=RequestLoader)


def fields(value, allowed, label, required=()):
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        raise ValueError(f'{label} requires {sorted(required)} and accepts only {sorted(allowed)}')
    return value


def path(value, declaring):
    if not isinstance(value, str) or not value or any(ord(c) < 32 for c in value):
        raise ValueError('paths require nonempty clean strings')
    declared = Path(declaring).parent / Path(value).expanduser()
    if declared.is_symlink() and not declared.exists():
        raise FileExistsError('dangling path is already occupied: '+str(declared))
    return declared.resolve()


def positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name+' requires a positive integer')


def ordinary_fields(science):
    fields(science, {'task', 'seed', 'protocol', 'model_config', *FIELDS}, 'ordinary experiment', ('task', 'seed', 'run', 'data', 'training', 'evaluation'))
    if science['task'] not in ('coco', 'hellaswag') or type(science['seed']) is not int or science['seed'] < 0:
        raise ValueError('ordinary task/seed differs')
    for section in FIELDS:
        if section in science:
            fields(science[section], FIELDS[section], section)
    if 'codec' in science and 'memory' in science['codec']:
        fields(science['codec']['memory'], FIELDS['codec']-{'memory'}, 'codec.memory')
    data, training, evaluation = science['data'], science['training'], science['evaluation']
    inactive = ({'karpathy_name', 'karpathy_revision', 'num_report', 'num_demos', 'max_prompt_length', 'max_target_length'}
                if science['task'] == 'hellaswag' else {'prompt_policy', 'max_length', 'row_ids'})
    inactive.add('row_ids')
    if set(data) & inactive:
        raise ValueError('inactive task data fields: '+str(sorted(set(data) & inactive)))
    if science['task'] == 'hellaswag' and 'record_coco_presentations' in training:
        raise ValueError('HellaSwag does not consume record_coco_presentations')
    if science['task'] == 'hellaswag' and 'max_new_tokens' in evaluation:
        raise ValueError('HellaSwag does not consume max_new_tokens')
    if science['task'] == 'coco' and set(evaluation) & {'num_fewshot', 'scoring_policy'}:
        raise ValueError('COCO does not consume candidate scoring controls')
    for key in ('max_steps', 'schedule_steps', 'batch_size', 'gradient_accumulation', 'eval_every', 'log_every'):
        if key in training:
            positive(training[key], 'training.'+key)
    for key in ('num_train', 'num_validation', 'num_report', 'num_demos', 'max_length', 'max_prompt_length', 'max_target_length'):
        if key in data and data[key] is not None:
            if key == 'num_demos' and data[key] == 0 and type(data[key]) is int:
                continue
            positive(data[key], 'data.'+key)
    for key in ('presentation_stream', 'paired_randomness'):
        if training.get(key) is not None:
            allowed = {'seed', 'total_presentations', 'policy'} if key == 'presentation_stream' else {'seed', 'policy', 'audit_steps'}
            fields(training[key], allowed, 'training.'+key)
    if data.get('prompt_policy'):
        fields(data['prompt_policy'], {'version', 'mode', 'seed', 'source_max_length'}, 'data.prompt_policy')
    for key in ('save_steps', 'selection_steps'):
        if key in training:
            values = training[key]
            if not isinstance(values, list) or any(type(v) is not int or v < 1 for v in values) or len(set(values)) != len(values):
                raise ValueError('training.'+key+' requires unique positive steps')
    for key in ('lr', 'weight_decay', 'grad_clip', 'temperature', 'kl_weight', 'mse_weight', 'min_delta', 'max_minutes', 'warmup_ratio', 'min_lr_ratio'):
        if key in training and training[key] is not None:
            value = training[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('training.'+key+' requires a finite nonnegative number')
            if key in ('temperature', 'max_minutes') and value <= 0:
                raise ValueError('training.'+key+' must be positive')
            if key in ('warmup_ratio', 'min_lr_ratio') and value > 1:
                raise ValueError('training.'+key+' must be at most1')
    for key in ('vanilla', 'deterministic_algorithms'):
        if key in evaluation and type(evaluation[key]) is not bool:
            raise ValueError('evaluation.'+key+' requires a boolean')
    for key in ('streamed_backward', 'valid_only_kl', 'low_sync_logging', 'deterministic_algorithms', 'record_coco_presentations'):
        if key in training and type(training[key]) is not bool:
            raise ValueError('training.'+key+' requires a boolean')
    if 'num_workers' in data and (type(data['num_workers']) is not int or data['num_workers'] < 0):
        raise ValueError('data.num_workers requires a nonnegative integer')
    if 'channel' in science:
        channel = science['channel']
        for key in ('normalize_power', 'train_noise'):
            if key in channel and type(channel[key]) is not bool:
                raise ValueError('channel.'+key+' requires boolean')
        if not isinstance(channel.get('kwargs', {}), dict):
            raise ValueError('channel.kwargs requires a mapping')
        if channel.get('type') in ('awgn', 'identity') and channel.get('kwargs'):
            raise ValueError('built-in channel kwargs must be empty')
    if training.get('feature_summary') is not None:
        fields(training['feature_summary'], {'steps', 'selection_count'}, 'training.feature_summary', ('steps', 'selection_count'))
    if 'split' in science:
        split = science['split']
        if split.get('stack') not in ('enc', 'dec') or split.get('where') not in ('after_embed', 'before_first_layer', 'after_layer', 'after_final_norm'):
            raise ValueError('unsupported split boundary')
        if split['where'] == 'after_layer':
            if type(split.get('index')) is not int or split['index'] < 0:
                raise ValueError('split.index requires a nonnegative integer')
        elif 'index' in split:
            raise ValueError('split.index is inactive at this boundary')
    if training.get('loss_weights') is not None and set(training) & {'kl_weight', 'mse_weight'}:
        raise ValueError('loss_weights conflicts with inactive kl_weight/mse_weight')


def read_request(filename) -> tuple[Path, dict[str, Any]]:
    filename = Path(filename).resolve()
    request = strict_yaml(filename)
    fields(request, {'schema', 'workflow', 'operation', 'execution', 'experiment'}, 'request', ('schema', 'workflow', 'operation', 'execution'))
    if request['schema'] != SCHEMA or request['workflow'] not in ('ordinary', 'shared') or request['operation'] not in OPERATIONS:
        raise ValueError('unsupported request schema/workflow/operation')
    operation, workflow = request['operation'], request['workflow']
    execution = fields(request['execution'], EXECUTION[operation], 'execution')
    if operation in ('acquire', 'register', 'collect', 'preview', 'submit') and 'experiment' in request:
        raise ValueError(operation+' does not accept authored science')
    if operation in ('check', 'prepare') and 'experiment' not in request and not (operation == 'check' and 'root' in execution):
        raise ValueError(operation+' requires authored experiment')
    if operation in ('prepare', 'train', 'resume', 'evaluate', 'collect') and 'root' not in execution:
        if not (operation == 'prepare' and execution.get('preparation', {}).get('output')):
            raise ValueError(operation+' requires explicit root')
    if operation in ('train', 'resume', 'evaluate'):
        if not isinstance(execution.get('attempt'), str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', execution['attempt']):
            raise ValueError('explicit plain new attempt required')
    if 'case' in execution and 'cases' in execution:
        raise ValueError('choose case or cases')
    if workflow == 'ordinary' and set(execution) & {'case', 'cases', 'device_uuid', 'timeout_seconds', 'prepared'}:
        raise ValueError('shared execution fields require shared workflow')
    if 'cases' in execution:
        cases = execution['cases']
        if not isinstance(cases, list) or not cases or any(not isinstance(c, str) or not c for c in cases) or len(set(cases)) != len(cases):
            raise ValueError('cases requires a nonempty unique list')
    if 'case' in execution and (not isinstance(execution['case'], str) or not execution['case']):
        raise ValueError('case requires an explicit name')
    if 'timeout_seconds' in execution and (type(execution['timeout_seconds']) not in (int, float) or not math.isfinite(execution['timeout_seconds']) or execution['timeout_seconds'] <= 0):
        raise ValueError('timeout_seconds must be finite and positive')
    if 'experiment' in request:
        if workflow == 'ordinary':
            ordinary_fields(request['experiment'])
        else:
            from .shared_config import validate_authored_recipe
            validate_authored_recipe(request['experiment'])
    # Resolve paths once against the declaring request, never the caller cwd.
    result = deepcopy(request)
    ex = result['execution']
    for key in ('root',):
        if key in ex:
            ex[key] = str(path(ex[key], filename))
    if 'success_link' in ex:
        path(ex['success_link'], filename)
        ex['success_link'] = str((filename.parent/Path(ex['success_link']).expanduser()).absolute())
    if operation == 'check':
        resolution = fields(ex.get('resolution', {}), {'model_config', 'output'}, 'resolution')
        for key in resolution:
            resolution[key] = str(path(resolution[key], filename))
        ex['resolution'] = resolution
    elif operation == 'acquire':
        acquisition = ex.get('acquisition')
        if isinstance(acquisition, dict) and acquisition.get('kind') == 'coco-cache':
            keys = {'kind', 'dataset_path', 'assets', 'dataset_name', 'dataset_revision',
                    'model_revision', 'karpathy_path', 'karpathy_name', 'karpathy_revision',
                    'fixed_data_ids', 'output'}
            a = fields(acquisition, keys, 'COCO cache acquisition', keys)
            for key in ('dataset_name', 'dataset_revision', 'model_revision', 'karpathy_name', 'karpathy_revision'):
                if not isinstance(a[key], str) or not a[key].strip():
                    raise ValueError('COCO acquisition requires pinned '+key)
            if not isinstance(a['fixed_data_ids'], dict):
                raise ValueError('COCO acquisition requires explicit fixed image partitions')
            if isinstance(a['output'], str) and (filename.parent/Path(a['output']).expanduser()).is_symlink():
                raise FileExistsError('COCO acquisition output is already reserved by a symlink')
            for key in ('dataset_path', 'assets', 'karpathy_path', 'output'):
                a[key] = str(path(a[key], filename))
        else:
            a = fields(acquisition, {'model_path', 'model_revision', 'output', 'synthetic_cpu'}, 'acquisition', ('model_path', 'model_revision', 'output', 'synthetic_cpu'))
            if type(a['synthetic_cpu']) is not bool or not isinstance(a['model_revision'], str) or not a['model_revision']:
                raise ValueError('acquisition requires revision and boolean synthetic_cpu')
            for key in ('model_path', 'output'):
                a[key] = str(path(a[key], filename))
    elif operation == 'register':
        a = fields(ex.get('registration'), {'model_name', 'output', 'references', 'from_prepared'}, 'registration', ('model_name', 'output'))
        if ('references' in a) == ('from_prepared' in a):
            raise ValueError('registration requires references or from_prepared')
        a['output'] = str(path(a['output'], filename))
        if 'from_prepared' in a:
            a['from_prepared'] = str(path(a['from_prepared'], filename))
        else:
            refs = fields(a['references'], {'model_binding', 'cache', 'observation_template', 'terminal_template', 'synthetic_cpu'}, 'references', ('model_binding', 'cache', 'observation_template', 'terminal_template', 'synthetic_cpu'))
            if type(refs['synthetic_cpu']) is not bool:
                raise ValueError('references.synthetic_cpu requires boolean')
            for key in set(refs)-{'synthetic_cpu'}:
                ref = fields(refs[key], {'path', 'sha256'}, 'reference', ('path', 'sha256'))
                ref['path'] = str(path(ref['path'], filename))
    elif operation == 'prepare':
        a = fields(ex.get('preparation'), {'registrations', 'assets', 'model_config', 'output'}, 'preparation')
        if workflow == 'ordinary':
            if a:
                raise ValueError('ordinary preparation accepts empty preparation and explicit root')
        elif ('registrations' in a) == ('assets' in a) or ('assets' in a) != ('model_config' in a):
            raise ValueError('prepare requires registrations or assets plus model_config')
        if ('root' in ex) == ('output' in a):
            raise ValueError('prepare requires exactly one root or preparation.output')
        for key in a:
            a[key] = str(path(a[key], filename))
    elif operation == 'resume':
        a = fields(ex.get('continuation'), {'path', 'sha256', 'size_bytes', 'role', 'step', 'parent_attempt', 'finalize_terminal', 'reconcile_interrupted'}, 'continuation', ('path', 'sha256', 'size_bytes', 'role', 'step', 'parent_attempt'))
        a['path'] = str(path(a['path'], filename))
        a['parent_attempt'] = str(path(a['parent_attempt'], filename))
        positive(a['step'], 'continuation.step')
        positive(a['size_bytes'], 'continuation.size_bytes')
        if 'finalize_terminal' in a and type(a['finalize_terminal']) is not bool:
            raise ValueError('finalize_terminal requires boolean')
        if workflow == 'ordinary' and set(a) & {'finalize_terminal', 'reconcile_interrupted'}:
            raise ValueError('shared continuation options require shared workflow')
    elif operation == 'evaluate':
        a = fields(ex.get('evaluation', {}), {'checkpoint', 'expected_step', 'scope', 'overrides', 'parallel'}, 'evaluation')
        if 'parallel' in a:
            if workflow != 'shared' or 'overrides' in a or not ex.get('case') or 'cases' in ex or 'device_uuid' in ex:
                raise ValueError('parallel evaluation requires one shared case and no overrides/device_uuid')
            from .panel_workers import validate_parallel_controls
            controls = dict(fields(a['parallel'], {'backend', 'workers', 'devices', 'parent_attempt'},
                                   'evaluation.parallel', ('backend', 'workers', 'devices')))
            parent = controls.pop('parent_attempt', None)
            a['parallel'] = validate_parallel_controls(controls)
            if parent is not None:
                a['parallel']['parent_attempt'] = str(path(parent, filename))
        if 'expected_step' in a:
            positive(a['expected_step'], 'evaluation.expected_step')
            if not a.get('checkpoint'):
                raise ValueError('expected_step requires checkpoint')
        if workflow == 'ordinary' and 'scope' in a:
            raise ValueError('ordinary does not consume shared scope')
        if workflow == 'shared' and 'checkpoint' in a:
            a['checkpoint'] = str(path(a['checkpoint'], filename))
        if 'overrides' in a:
            a['overrides'] = str(path(a['overrides'], filename))
        ex['evaluation'] = a
    elif operation == 'collect':
        a = fields(ex.get('collection'), {'destination', 'snapshot', 'select', 'source_uri'}, 'collection', ('destination', 'snapshot'))
        if 'select' in a and (not isinstance(a['select'], list) or any(not isinstance(v, str) for v in a['select'])):
            raise ValueError('collection.select requires an explicit string list')
        for key in ('destination', 'snapshot'):
            a[key] = str(path(a[key], filename))
    elif operation == 'preview':
        a = fields(ex.get('plan'), {'kind', 'path', 'output', 'task'}, 'plan', ('kind', 'path'))
        if a['kind'] not in ('study', 'submission'):
            raise ValueError('preview plan.kind must be study or submission')
        if a['kind'] == 'study' and 'submission' in ex:
            raise ValueError('study preview does not accept submission')
        if a['kind'] == 'submission' and set(a) & {'task', 'output'}:
            raise ValueError('submission preview output belongs to submission bundle')
        for key in ('path', 'output'):
            if key in a:
                a[key] = str(path(a[key], filename))
    if operation == 'submit' or (operation == 'preview' and ex['plan']['kind'] == 'submission'):
        a = fields(ex.get('submission'), {'plan', 'limits', 'bundle_output', 'receipt_dir', 'authorization'}, 'submission', ('limits', 'bundle_output'))
        if operation == 'preview' and set(a) & {'plan', 'receipt_dir', 'authorization'}:
            raise ValueError('preview does not accept submit effects')
        if operation == 'submit' and not {'plan', 'receipt_dir', 'authorization'} <= set(a):
            raise ValueError('submit requires explicit receipt_dir and authorization')
        for key in ('plan', 'limits', 'bundle_output', 'receipt_dir'):
            if key in a:
                a[key] = str(path(a[key], filename))
    if operation == 'submit' and Path(ex['submission']['bundle_output']) != Path(ex['submission']['receipt_dir'])/'bundle.json':
        raise ValueError('submit bundle_output must equal receipt_dir/bundle.json')
    if 'prepared' in ex:
        ref = fields(ex['prepared'], {'path', 'sha256'}, 'prepared', ('path', 'sha256'))
        ref['path'] = str(path(ref['path'], filename))
    return filename, result


def _fresh(filename):
    p = Path(filename)
    if p.exists() or p.is_symlink():
        raise FileExistsError('occupied output; choose a fresh destination: '+str(p))
    return p


def _json_output(filename, value):
    with _fresh(filename).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def _ordinary(filename):
    from .config import load_config
    return load_config(filename)


def _bound(request, filename):
    ex = request['execution']
    root = Path(ex['root'])
    from .ordinary_runs import is_recorded_root
    from .shared_runs import recipe_run_record
    from .run_records import bound_config
    shared = (root/'recipe-run.json').is_file()
    if shared != (request['workflow'] == 'shared'):
        raise ValueError('request workflow differs from bound root')
    if shared:
        record = recipe_run_record(root)
        if 'prepared' in ex and ex['prepared'] != record['prepared']:
            raise ValueError('train prepared reference differs from bound root')
        if 'experiment' in request:
            from .shared_preparation import _read_reference, workflow_resolution
            from .shared_config import resolve_recipe
            manifest = json.loads(Path(record['prepared']['path']).read_text())
            resolved = resolve_recipe(filename, model_config_path=_read_reference(manifest['model_config']))
            if workflow_resolution(resolved, manifest) != manifest['resolved']:
                raise ValueError('authored science differs from bound experiment')
        return record
    if not is_recorded_root(root):
        raise ValueError('root is not a recorded ordinary experiment')
    config = bound_config(root)
    if 'experiment' in request:
        authored = _ordinary(filename)
        authored['run']['output_dir'] = str(root)
        if authored != config:
            raise ValueError('authored science differs from bound experiment')
    return config


def execute(filename) -> Any:
    filename, request = read_request(filename)
    operation, workflow, ex = request['operation'], request['workflow'], request['execution']
    # Validate optional success receipt before any model, root or attempt effect.
    link = None
    if 'success_link' in ex:
        link = _fresh(ex['success_link'])
        root = Path(ex['root'])
        resolved_link = link.resolve()
        if resolved_link == root or root in resolved_link.parents or resolved_link in root.parents:
            raise ValueError('success_link and root must be disjoint in both ancestry directions')
    if operation == 'check':
        if 'root' in ex:
            if ex['resolution']:
                raise ValueError('bound check does not accept resolution overrides')
            value = _bound(request, filename)
        elif workflow == 'ordinary':
            if ex['resolution'].get('model_config'):
                raise ValueError('ordinary model_config belongs to experiment')
            value = _ordinary(filename)
        else:
            from .shared_config import resolve_recipe
            if not ex['resolution'].get('model_config'):
                raise ValueError('shared check requires local resolution.model_config')
            value = resolve_recipe(filename, model_config_path=ex['resolution']['model_config'])
        if 'case' in ex:
            from .shared_runs import case_id
            rows = value.get('cases', value.get('trajectories', []))
            if ex['case'] not in [r.get('case_id', case_id(r) if 'width' in r else None) for r in rows]:
                raise ValueError('unknown declared case')
        if ex['resolution'].get('output'):
            _json_output(ex['resolution']['output'], value)
    elif operation == 'acquire':
        a = ex['acquisition']
        _fresh(a['output'])
        if a.get('kind') == 'coco-cache':
            from .shared_inputs import bind_coco_data_cache
            value = bind_coco_data_cache(**{k: v for k, v in a.items() if k != 'kind'})
        else:
            from .asset_identity import bind_model_assets
            value = bind_model_assets(a['model_path'], a['model_revision'], production=not a['synthetic_cpu'])
            if value['status'] != 'complete':
                raise ValueError('acquire requires complete existing local assets')
            _json_output(a['output'], value)
    elif operation == 'register':
        from .shared_assets import register_recipe_assets, register_prepared_assets
        a = ex['registration']
        _fresh(a['output'])
        if 'from_prepared' in a:
            value = register_prepared_assets(a['from_prepared'], model_name=a['model_name'], output=a['output'])
        else:
            refs = dict(a['references'])
            synthetic = refs.pop('synthetic_cpu')
            value = register_recipe_assets(model_name=a['model_name'], output=a['output'], synthetic=synthetic, **refs)
    elif operation == 'prepare':
        a = ex['preparation']
        destination = ex.get('root', a.get('output'))
        _fresh(destination)
        if workflow == 'ordinary':
            from .ordinary_runs import prepare_task
            value = prepare_task(_ordinary(filename), destination, authored_config=filename)
        elif 'root' in ex and 'registrations' in a:
            from .shared_runs import prepare_recipe_run
            value = prepare_recipe_run(filename, destination, registrations=a['registrations'])
        else:
            if 'registrations' in a:
                from .shared_assets import prepare_recipe_assets
                value = prepare_recipe_assets(filename, registrations=a['registrations'], output=destination)
            else:
                from .shared_preparation import bind_recipe_workflow
                assets = json.loads(Path(a['assets']).read_text())
                # Assets document has its own declaring paths and strict concrete binder contract.
                allowed = {'model_path', 'model_binding', 'cache_manifest', 'cache_sha256', 'initializations', 'observation_template', 'terminal_template', 'study_pairing_id', 'synthetic'}
                fields(assets, allowed, 'assets', allowed-{'synthetic'})
                for key in ('model_path', 'cache_manifest'):
                    assets[key] = str(path(assets[key], a['assets']))
                for key in ('model_binding', 'observation_template', 'terminal_template'):
                    assets[key]['path'] = str(path(assets[key]['path'], a['assets']))
                for ref in assets['initializations'].values():
                    ref['path'] = str(path(ref['path'], a['assets']))
                    if 'acquisition' in ref:
                        ref['acquisition']['path'] = str(path(ref['acquisition']['path'], a['assets']))
                if 'root' in ex:
                    from .shared_runs import prepare_recipe_run
                    value = prepare_recipe_run(filename, destination, assets=assets, model_config=a['model_config'])
                else:
                    value = bind_recipe_workflow(filename, model_config_path=a['model_config'], output=destination, **assets)
    elif operation in ('train', 'resume'):
        resume = None
        if operation == 'resume':
            a = ex['continuation']
            from .activation_replay import file_digest
            cp = Path(a['path'])
            parent = Path(a['parent_attempt'])
            if (parent not in cp.parents or (workflow == 'ordinary' and cp.parent != parent)
                    or cp.stat().st_size != a['size_bytes'] or file_digest(cp) != a['sha256']):
                raise ValueError('continuation checkpoint/parent bytes differ')
            resume = str(cp)
        bound = Path(ex['root']).exists()
        if bound:
            config = _bound(request, filename)
        elif operation == 'resume' or 'experiment' not in request:
            raise ValueError('fresh training requires authored experiment; resume requires bound root')
        elif workflow == 'ordinary':
            config = _ordinary(filename)
        else:
            if 'prepared' not in ex:
                raise ValueError('fresh shared training requires exact prepared reference; prepare separately')
            from .shared_runs import prepare_recipe_run
            prepare_recipe_run(filename, ex['root'], prepared=ex['prepared'])
            config = _bound(request, filename)
        if workflow == 'ordinary':
            from .ordinary_runs import train_task
            if operation == 'resume':
                from .ordinary_continuation import load_checkpoint
                state, target = load_checkpoint(resume)
                if target['role'] != ex['continuation']['role'] or state['step'] != ex['continuation']['step']:
                    raise ValueError('continuation checkpoint role/step differs')
            if link is not None:
                link.parent.mkdir(parents=True, exist_ok=True)
            value = train_task(config, ex['root'], authored_config=filename, resume=resume, attempt_id=ex['attempt'])
        else:
            from .shared_runs import train_recipe_case, recipe_run_record, reconcile_recipe_attempt
            record = recipe_run_record(ex['root'])
            choices = ex.get('cases', [ex['case']] if 'case' in ex else [r['case_id'] for r in record['cases']])
            if operation == 'resume' and len(choices) != 1:
                raise ValueError('resume requires one explicit case')
            if any(c not in [r['case_id'] for r in record['cases']] for c in choices):
                raise ValueError('unknown declared case')
            if operation == 'resume':
                case_path = Path(ex['root'])/'cases'/choices[0]
                checkpoint_index = json.loads((Path(ex['continuation']['parent_attempt'])/'checkpoints.json').read_text())
                matches = [row for row in checkpoint_index['checkpoints'] if row['path'] == resume]
                if (len(matches) != 1 or matches[0]['step'] != ex['continuation']['step']
                        or matches[0]['role'] != ex['continuation']['role']
                        or Path(ex['continuation']['parent_attempt']).parent != case_path/'attempts'):
                    raise ValueError('continuation checkpoint role/step/attempt differs')
                if ex['continuation'].get('reconcile_interrupted'):
                    reconcile_recipe_attempt(ex['root'], choices[0], ex['continuation']['reconcile_interrupted'])
            if link is not None:
                link.parent.mkdir(parents=True, exist_ok=True)
            for case in choices:
                train_recipe_case(ex['root'], case, attempt_id=ex['attempt'], resume=resume,
                    timeout_seconds=ex.get('timeout_seconds', 86400), device_uuid=ex.get('device_uuid'),
                    finalize_terminal=ex.get('continuation', {}).get('finalize_terminal', False))
            value = Path(ex['root'])
    elif operation == 'evaluate':
        _bound(request, filename)
        a = ex['evaluation']
        if link is not None:
            link.parent.mkdir(parents=True, exist_ok=True)
        if workflow == 'ordinary':
            from .ordinary_runs import evaluate_run
            value = evaluate_run(ex['root'], a.get('checkpoint'), a.get('overrides'), expected_step=a.get('expected_step'), attempt_id=ex['attempt'])
        else:
            from .shared_evaluation import evaluate_recipe_run
            value = evaluate_recipe_run(ex['root'], selected=ex.get('cases', ex.get('case')), attempt_id=ex['attempt'], checkpoint=a.get('checkpoint'),
                expected_step=a.get('expected_step'), scope=a.get('scope', 'initial'), overrides_path=a.get('overrides'),
                timeout_seconds=ex.get('timeout_seconds', 86400), device_uuid=ex.get('device_uuid'), parallel=a.get('parallel'))
    elif operation == 'collect':
        root_record = json.loads((Path(ex['root'])/'experiment.json').read_text())
        recorded_workflow = {'ordinary-task': 'ordinary', 'shared-codec-recipe': 'shared'}.get(root_record.get('kind'))
        if root_record.get('schema') != 'experiment-run-v1' or recorded_workflow != workflow:
            raise ValueError('collection workflow differs from recorded root')
        from .result_collection import collect_results
        a = ex['collection']
        value = collect_results(Path(ex['root']), Path(a['destination']), snapshot=Path(a['snapshot']), selected=a.get('select', []), source_uri=a.get('source_uri'))
    elif operation == 'preview' and ex['plan']['kind'] == 'study':
        from .studies import expand_study, export_study
        a = ex['plan']
        plan = expand_study(a['path'], a.get('task'))
        value = export_study(plan, a['output']) if 'output' in a else {'study': plan.name, 'runs': len(plan.runs), 'launch_authorized': False}
    else:
        from .submission import prepare_job, preview_job, submit_job
        a = ex['submission']
        _fresh(a['bundle_output'])
        plan_path = ex['plan']['path'] if operation == 'preview' else a['plan']
        value = prepare_job(plan_path, a['limits'])
        if operation == 'preview':
            _json_output(a['bundle_output'], value)
            value = preview_job(value, a['bundle_output'])
        else:
            if Path(a['bundle_output']) != Path(a['receipt_dir'])/'bundle.json':
                raise ValueError('submit bundle_output must equal receipt_dir/bundle.json')
            value = submit_job(value, a['receipt_dir'], authorization=a['authorization'])
    if link is not None:
        link.parent.mkdir(parents=True, exist_ok=True)
        with link.open('x') as handle:
            handle.write(str(value)+'\n')
    return value


def main(argv=None) -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='one jscc-experiment-v1 YAML request')
    args = parser.parse_args(argv)
    value = execute(args.config)
    print(json.dumps(value, default=str, indent=2, allow_nan=False))
    if isinstance(value, dict) and value.get('status') in ('incomplete', 'failed'):
        raise SystemExit(1)
    if isinstance(value, Path) and (value/'results.json').is_file():
        result = json.loads((value/'results.json').read_text())
        if result.get('status') == 'incomplete':
            raise SystemExit(1)
    return value
