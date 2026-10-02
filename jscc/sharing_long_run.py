"""Formal native16 training and frozen, no-noise-first full evaluation.

Uses the existing four-learner lifecycle and versioned scorer. No scheduler,
reservation, cumulative budget, automatic retry or checkpoint selection is added.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import time

import torch

from .activation_replay import canonical_digest, file_digest
from .baseline_protocol import write_manifest, validate_task_receipt
from .evaluation import evaluate_checkpoint, verify_observation_artifacts
from .experiment_state import CheckpointRef, open_state
from .models.split_model import build_model
from .runtime import configure_training_determinism, seed_everything
from .sharing_preparation import _reference
from .sharing_run import final_panel_id
from .sharing_schedule import NATIVE16_LONG_PROTOCOL, long_final_panels
from .sharing_state import verify_study_freeze


def configure_long_entry(manifest, *, approved_source, timeout_seconds):
    """Bind an explicit command to a pinned checkout and its per-job timeout."""
    if manifest['protocol_id'] != NATIVE16_LONG_PROTOCOL:
        raise ValueError('long entry requires its formal native16 protocol')
    if type(timeout_seconds) is not int or not 0 < timeout_seconds <= manifest['hard_cap_seconds']:
        raise ValueError('per-job timeout must be within the prepared timeout')
    if not manifest['synthetic_cpu']:
        source = Path(__file__).resolve().parents[1]
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
        if not approved_source or head != approved_source:
            raise ValueError('exact approved source commit required')
        subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--'], cwd=source,
                       check=True, stdout=subprocess.DEVNULL)
        tracked = set(subprocess.check_output(['git', 'ls-files'], cwd=source, text=True).splitlines())
        if not set(manifest['source_inventory']) <= tracked:
            raise ValueError('prepared execution source must all belong to the approved commit')
        if not os.environ.get('SLURM_JOB_ID'):
            raise ValueError('formal GPU training/evaluation requires a Slurm allocation')
    os.environ['SHARING_LONG_ENTRY_ID'] = manifest['prepared_identity']
    os.environ['SHARING_LONG_DEADLINE_MONOTONIC'] = str(time.monotonic() + timeout_seconds)


def require_long_entry(manifest):
    if os.environ.get('SHARING_LONG_ENTRY_ID') != manifest['prepared_identity']:
        raise ValueError('use sharing_long_study.py with exact source and per-job timeout')
    check_long_deadline()


def check_long_deadline():
    deadline = os.environ.get('SHARING_LONG_DEADLINE_MONOTONIC')
    if deadline is None or time.monotonic() >= float(deadline):
        raise TimeoutError('native16 per-job timeout; partial artifacts retained')


def evaluate_long(manifest, root, training_run, output):
    """Evaluate seven terminal targets, all no-noise first, then approved SNRs."""
    require_long_entry(manifest)
    root, training_run, output = map(Path, (root, training_run, output))
    campaign = json.loads((training_run / 'campaign.json').read_text())
    if campaign['status'] != 'training_frozen':
        raise ValueError('final evaluation requires a completely frozen training run')
    panels = long_final_panels(manifest['q'])
    expected_plan = [asdict(o) | {'panel_id': final_panel_id(o)} for o in panels]
    # JSON represents condition tuples as lists.
    if campaign['final_evaluation_plan'] != json.loads(json.dumps(expected_plan)):
        raise ValueError('final evaluation plan differs from frozen training')
    freeze_reference = campaign['freeze_reference']
    freeze = verify_study_freeze(freeze_reference, protocol_identity=manifest['protocol_id'])
    if freeze['obligation_plan']['task'] != [final_panel_id(o) for o in panels]:
        raise ValueError('final panels differ from the trained-study freeze')
    template_reference = manifest['final_task_request']
    template = json.loads(_reference(root, template_reference).read_text())
    checkpoints = {}
    for observation in panels[:7]:
        path = training_run / observation.learner / f'step_{observation.step:06d}.pt'
        payload = torch.load(path, map_location='cpu', weights_only=False)
        state = open_state(CheckpointRef(str(path.resolve()), file_digest(path), path.stat().st_size),
                           expected=payload['metadata'])
        metadata = state.payload['metadata']
        if (metadata['prepared_identity'] != manifest['prepared_identity']
                or metadata['final_task_identity'] != template_reference['sha256']
                or metadata['source_identity'] != manifest['source_identity']
                or metadata['config_identity'] != manifest['config_identity']
                or metadata['data_identity'] != canonical_digest(manifest['data_ids'])
                or metadata['initialization_identity'] != manifest['initialization']['state_identity']
                or metadata['model_revision'] != manifest['model_revision']
                or state.reference.sha256 not in freeze['checkpoint_hashes']):
            raise ValueError('terminal checkpoint differs from exact prepared/frozen study')
        checkpoints[observation.learner] = state
    # Check every artifact and contract above before constructing a model.
    check_long_deadline()
    config = manifest['resolved_config']
    configure_training_determinism(config['training'])
    seed_everything(config['seed'])
    processor, model = build_model(copy.deepcopy(config))
    output.mkdir(parents=True, exist_ok=False)
    result = {'schema': 'sharing-long-final-evaluation-v1', 'status': 'incomplete',
              'prepared_identity': manifest['prepared_identity'], 'freeze_reference': freeze_reference,
              'data_interpretation': 'already inspected official validation; not an untouched test',
              'planned_panels': len(panels), 'completed_panels': [], 'expected_items_per_panel': template['expected_items']}
    started = time.monotonic()
    write_manifest(output / 'result.json', result)
    try:
        for observation in panels:
            check_long_deadline()
            state = checkpoints[observation.learner]
            details = state.payload['metadata']
            tag = final_panel_id(observation)
            declaration = details['evaluation_sites'][observation.site]
            request = {**template, 'conditions': list(observation.conditions),
                'schema': 'codec-observation-v1', 'checkpoint_sha256': state.reference.sha256,
                'target_site': declaration['site'], 'target_role': declaration['role'],
                'learner_kind': state.payload['kind'], 'step': observation.step,
                'site_step': details['sharing']['completed_per_site'].get(observation.site, 0),
                'parent': None, 'comparison': details['comparison_controls'], 'panel': tag,
                'source': details['source_identity'], 'config': details['config_identity'],
                'data': details['data_identity'], 'protocol': details['protocol_identity']}
            if observation.site == 'enc_l14':
                request.update(freeze_reference=freeze_reference, freeze_receipt=freeze)
            path = output / tag.replace(':', '_')
            receipt = evaluate_checkpoint(state, request, model=model, processor=processor, output=path)
            validate_task_receipt(receipt, request)
            verify_observation_artifacts(receipt, path)
            result['completed_panels'].append({'panel_id': tag, 'learner': observation.learner,
                'site': observation.site, 'condition': observation.conditions[0],
                'checkpoint': state.reference.sha256, 'observation': {'path': str((path / 'observation.json').resolve()),
                'sha256': file_digest(path / 'observation.json')},
                'metrics': receipt['conditions'][0]['metrics'],
                'completed_items': receipt['conditions'][0]['completed']})
            result['elapsed_seconds'] = time.monotonic() - started
            write_manifest(output / 'result.json', result)
        result.update(status='complete', elapsed_seconds=time.monotonic() - started)
        write_manifest(output / 'result.json', result)
        return result
    except Exception as error:
        result.update(error={'type': type(error).__name__, 'message': str(error)},
                      elapsed_seconds=time.monotonic() - started)
        write_manifest(output / 'result.json', result)
        raise
