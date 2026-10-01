"""Bind reviewed production inputs to one explicit candidate; CPU preparation only."""
from pathlib import Path
import json
import shutil
import zipfile

import torch
import yaml

from .activation_replay import canonical_digest, file_digest
from .config import load_config
from .models.codec import Codec
from .sharing_accumulation import partition_policy
from .sharing_schedule import DEPTH_PROTOCOL
from .sharing_preparation import (_config_contract, _reference, source_inventory,
                                  state_dict_identity, write_prepared)


def bind_production(*, inputs_path, inputs_sha256, config_path, cell, bottleneck_dim,
                    microbatch_size, architecture_decision, protocol_id, study_pairing_id,
                    recipe_identity, hard_cap_seconds, output):
    """Create a fresh package after selection; never creates acceptance or allocation.

    The supplied resolved/model-composed config must already match the explicit
    choice. Initialization is freshly seeded on CPU, never imported from a
    baseline checkpoint. The final prepared validator rechecks every copied byte.
    """
    inputs_path = Path(inputs_path).resolve()
    if file_digest(inputs_path) != inputs_sha256:
        raise ValueError('reviewed input manifest checksum mismatch')
    inputs = json.loads(inputs_path.read_text())
    quota = 400 if protocol_id == DEPTH_PROTOCOL else 200
    if (inputs.get('schema') != 'sharing-production-inputs-v1'
            or inputs.get('status') != 'cpu_contracts_verified'
            or (inputs.get('q'), inputs.get('batch_size')) != (quota, 64)):
        raise ValueError('reviewed complete production inputs required')
    config = load_config(config_path)
    manifest = dict(schema='sharing-prepared-v1', synthetic_cpu=False, q=quota, batch_size=64,
        cell=cell, selected_bottleneck_dim=bottleneck_dim,
        execution_partition=partition_policy(microbatch_size),
        model_revision=inputs['model_revision'], architecture_decision=architecture_decision,
        protocol_id=protocol_id, study_pairing_id=study_pairing_id, recipe_identity=recipe_identity,
        hard_cap_seconds=hard_cap_seconds, config_identity=canonical_digest(config),
        **{key: inputs[key] for key in ('updates', 'validation', 'geometry', 'producers', 'data_ids')},
        task_request=inputs['task_template'],
        production_input_origin={'sha256': inputs_sha256, 'path': str(inputs_path)},
        initialization_policy={'seed': 0, 'fresh_cpu_codec': True, 'baseline_state_reused': False})
    _config_contract(config, manifest)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    references = [*manifest['updates'], *manifest['validation'], manifest['task_request'],
                  *manifest['producers'].values(),
                  *(ref for group in manifest['geometry'].values() for ref in group)]
    copied = {}
    for reference in references:
        source = _reference(inputs_path.parent, reference)
        target = output / reference['path']
        if reference['path'] in copied:
            if copied[reference['path']] != reference['sha256']:
                raise ValueError('conflicting input reference')
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied[reference['path']] = reference['sha256']
    def ref(name):
        return {'path': name, 'sha256': file_digest(output / name)}
    # Preserve the original panel, then bind only sharing execution settings.
    template = json.loads((output / manifest['task_request']['path']).read_text())
    template['settings']['numerical_policy'] = 'native'
    template['settings']['batch_size'] = microbatch_size
    template['layout'] = {**template['layout'], 'batch_size': microbatch_size}
    (output / 'sharing-task.json').write_text(json.dumps(template, indent=2) + '\n')
    manifest['original_task_template'] = manifest['task_request']
    manifest['task_request'] = ref('sharing-task.json')
    (output / 'config.yaml').write_text(yaml.safe_dump(config))
    manifest['config'] = ref('config.yaml')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        codec = Codec(1152, config['codec'])
    torch.save(codec.state_dict(), output / 'initial.pt')
    manifest['initialization'] = {**ref('initial.pt'), 'state_identity': state_dict_identity(codec.state_dict())}
    inventory = source_inventory()
    source_root = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(output / 'source.zip', 'w') as archive:
        for name in inventory:
            archive.write(source_root / name, name)
    manifest.update(source_inventory=inventory, source_identity=canonical_digest(inventory),
                    source_archive=ref('source.zip'))
    return write_prepared(output / 'prepared.json', manifest)
