"""Evaluation-owned legacy entry seam and original13 observation contract."""
from dataclasses import asdict
import json
from pathlib import Path
import time

from .activation_replay import file_digest
from .baseline_protocol import validate_task_receipt, write_manifest
from .encoder_pilot_plan import SITES
from .evaluation import evaluate_checkpoint, verify_observation_artifacts
from .experiment_state import CheckpointRef, open_state

CONDITIONS = ('no_noise', -6, 0, 6, 18)


class ObservationScoringError(RuntimeError):
    """Recoverable scoring/output error; never integrity or restoration failure."""


def observe_original13(model, processor, *, state, request, output, guard):
    """Consumer replaces this stub; restore RNG/routes and verify state on error.

    Return {status: 'complete', request_identity: str, ...}. Raise only
    ObservationScoringError for isolated scoring/output failure AFTER verified
    restoration. Every other exception stops training. No retry is implicit.
    """
    raise NotImplementedError('original13 observation consumer not implemented')


def open_stages_terminal(training_run, metadata, plan):
    root = Path(training_run).resolve()
    campaign = json.loads((root / 'campaign.json').read_text())
    freeze_path = root / 'freeze.json'
    if (campaign['status'] != 'training_frozen' or campaign['completed_updates'] != plan.horizon
            or campaign['prepared_identity'] != metadata['prepared_identity']
            or campaign['freeze_reference'] != {'path': str(freeze_path), 'sha256': file_digest(freeze_path)}):
        raise ValueError('own training trajectory is not completely frozen')
    freeze = json.loads(freeze_path.read_text())
    if (freeze['status'] != 'complete' or freeze['arm'] != plan.arm or freeze['width'] != plan.width
            or freeze['final_panels'] != [[site, condition] for condition in CONDITIONS for site in SITES]
            or file_digest(root / 'metrics.jsonl') != freeze['metrics']['sha256']):
        raise ValueError('terminal study plan or scalar log differs')
    expected = {k: v for k, v in metadata.items() if k not in {'phase', 'lineage'}}
    if plan.arm != 'A':
        reset = campaign['phase_reset']
        parent_ref = CheckpointRef(**reset['parent'])
        if Path(parent_ref.path).resolve() != root / f'step_{plan.horizon // 2:06d}.pt':
            raise ValueError('phase reset parent path differs from own training run')
        parent = open_state(parent_ref, expected=metadata)
        if parent.payload['metadata']['completed_updates'] != plan.horizon // 2:
            raise ValueError('phase reset parent did not complete stage one')
        expected['parent_identity'] = parent_ref.sha256
    state = open_state(CheckpointRef(**freeze['checkpoint']), expected=expected)
    if (state.payload['metadata']['completed_updates'] != plan.horizon
            or state.payload['metadata']['sharing']['arm'] != plan.arm
            or state.payload['metadata']['sharing']['width'] != plan.width):
        raise ValueError('terminal checkpoint differs from own arm/width')
    return state, campaign['freeze_reference']


def run_stages_evaluation(model, processor, *, state, freeze_reference, template, output, guard):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    details = state.payload['metadata']
    if template['conditions'] != list(CONDITIONS):
        raise ValueError('terminal five-condition template required')
    result = {'schema': 'encoder-stages-evaluation-v1', 'status': 'incomplete',
              'arm': details['sharing']['arm'], 'width': details['sharing']['width'],
              'prepared_identity': details['prepared_identity'],
              'checkpoint': asdict(state.reference), 'freeze_reference': freeze_reference, 'completed_panels': [],
              'interpretation': 'previously inspected validation, six trained encoder sites, one seed',
              'vanilla_execution': 'reuse compatible external evidence after identity matching'}
    begun = time.monotonic()
    write_manifest(output / 'result.json', result)
    try:
        for condition in CONDITIONS:
            for site in SITES:
                guard()
                request = {**template, 'schema': 'codec-observation-v1', 'conditions': [condition],
                    'checkpoint_sha256': state.reference.sha256, 'target_site': details['evaluation_sites'][site]['site'],
                    'target_role': 'trained', 'learner_kind': 'shared_encoder_v2', 'step': details['completed_updates'],
                    'site_step': details['sharing']['completed_per_site'][site], 'parent': details['parent_identity'],
                    'comparison': details['comparison_controls'], 'panel': f'final:{site}:{condition}',
                    'source': details['source_identity'], 'config': details['config_identity'],
                    'data': details['data_identity'], 'protocol': details['protocol_identity']}
                request.update(study_arm=details['sharing']['arm'], study_width=details['sharing']['width'],
                               prepared_identity=details['prepared_identity'])
                request['settings'] = {**request['settings'], 'evidence_mode': 'compact-v1'}
                path = output / f'{site}_{condition}'
                receipt = evaluate_checkpoint(state, request, model=model, processor=processor, output=path)
                validate_task_receipt(receipt, request)
                verify_observation_artifacts(receipt, path)
                guard()
                result['completed_panels'].append({'site': site, 'condition': condition,
                    'checkpoint': state.reference.sha256, 'completed_items': receipt['conditions'][0]['completed'],
                    'metrics': receipt['conditions'][0]['metrics'],
                    'observation': {'path': str(path / 'observation.json'), 'sha256': file_digest(path / 'observation.json')}})
                result['elapsed_seconds'] = time.monotonic() - begun
                write_manifest(output / 'result.json', result)
                print(json.dumps({'phase': 'evaluation', 'completed_codec_panels': len(result['completed_panels']),
                                  'elapsed_seconds': result['elapsed_seconds']}), flush=True)
        result.update(status='complete', elapsed_seconds=time.monotonic() - begun)
        write_manifest(output / 'result.json', result)
        return result
    except Exception as error:
        result.update(error={'type': type(error).__name__, 'message': str(error)}, elapsed_seconds=time.monotonic() - begun)
        write_manifest(output / 'result.json', result)
        raise
