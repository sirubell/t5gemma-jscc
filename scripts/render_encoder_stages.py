#!/usr/bin/env python3
"""Offline complete scalar curves and predeclared paired R minus E reporting."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jscc.activation_replay import file_digest
from jscc.encoder_stages import CONDITIONS
from jscc.encoder_pilot_plan import SITES
from jscc.experiment_records import read_events
from render_experiment_report import _plot, _write_csv, paired_group_bootstrap


def training_curves(root, output):
    campaign = json.loads((root / 'campaign.json').read_text())
    freeze = json.loads((root / 'freeze.json').read_text())
    if (campaign['status'] != 'training_frozen' or campaign['scalar_updates'] != campaign['planned_updates']
            or file_digest(root / 'freeze.json') != campaign['freeze_reference']['sha256']
            or file_digest(root / 'metrics.jsonl') != freeze['metrics']['sha256']):
        raise ValueError('frozen training and complete original scalar log required')
    events = [e for e in read_events(root / 'metrics.jsonl') if e['event_type'] == 'update']
    if len(events) != campaign['planned_updates']:
        raise ValueError('scalar row count differs from frozen horizon')
    rows = []
    for i, event in enumerate(events):
        payload = event['payload']
        if payload['completed_step'] != i + 1 or payload['nonfinite'] or payload['skipped']:
            raise ValueError('successful scalar log cadence differs')
        objective = payload['objective']
        rows.append({'arm': campaign['arm'], 'width': campaign['width'], 'phase': event['phase_id'],
                     'global_update': i + 1, 'phase_update': i + 1 - payload['phase_start_step'],
                     'site': payload['site_id'], 'site_update': payload['site_step'],
                     'loss': objective['total']['value'], 'kl': objective['components']['K']['raw']['value'],
                     'kl_status': objective['components']['K']['raw']['reason'],
                     'nmse': objective['components']['R']['raw']['value'],
                     'weighted_nmse': objective['components']['R']['weighted']['value'],
                     'lr_used': payload['lr_used'][0], 'sequences': payload['exposure']['sequences'],
                     'source_valid': payload['exposure']['source_tokens'], 'target_valid': payload['exposure']['target_tokens']})
    prefix = f"b{campaign['width']}_{campaign['arm']}"
    _write_csv(output / f'{prefix}-updates.csv', rows)
    for site in SITES:
        selected = [row for row in rows if row['site'] == site]
        for metric in ('loss', 'kl', 'nmse', 'lr_used'):
            points = [{**row, 'x': row['global_update'], 'value': row[metric],
                       'series': f'{prefix} {site} {metric}'} for row in selected]
            _plot(output / f'{prefix}-{site}-{metric}.png', f'{prefix} {site}: raw per-update {metric}',
                  points, x_label='Completed global optimizer updates')
    return {'arm': campaign['arm'], 'width': campaign['width'], 'scalar_updates': len(rows),
            'log_sha256': freeze['metrics']['sha256'], 'smoothing': 'none', 'selection': 'no intermediate selection'}


def evaluation_panels(root, *, expected_arm):
    result = json.loads((root / 'result.json').read_text())
    expected = [(site, condition) for condition in CONDITIONS for site in SITES]
    if (result['status'] != 'complete' or result.get('arm') != expected_arm
            or [(p['site'], p['condition']) for p in result['completed_panels']] != expected):
        raise ValueError('complete thirty-panel terminal evaluation required')
    panels = {}
    for panel in result['completed_panels']:
        path = root / f"{panel['site']}_{panel['condition']}" / 'observation.json'
        if file_digest(path) != panel['observation']['sha256'] or panel['checkpoint'] != result['checkpoint']['sha256']:
            raise ValueError('own terminal checkpoint or observation bytes differ')
        receipt = json.loads(path.read_text())
        if (receipt['request'].get('study_arm') != expected_arm
                or receipt['request'].get('study_width') != result['width']
                or receipt['request'].get('prepared_identity') != result['prepared_identity']
                or receipt['request']['checkpoint_sha256'] != result['checkpoint']['sha256']):
            raise ValueError('evaluation arm, width, preparation or checkpoint provenance differs')
        if receipt['status'] != 'complete' or receipt['conditions'][0]['completed'] != receipt['request']['expected_items']:
            raise ValueError('terminal condition incomplete')
        panels[(panel['site'], panel['condition'])] = receipt
    return panels


def paired_primary(reconstruction, reset_control):
    contrasts, averages = [], {metric: {'R': {}, 'E': {}} for metric in ('raw_correct', 'normalized_correct')}
    for site in SITES:
        left, right = reconstruction[(site, 'no_noise')], reset_control[(site, 'no_noise')]
        lq, rq = left['request'], right['request']
        if (lq.get('study_arm') != 'R' or rq.get('study_arm') != 'E'
                or lq['checkpoint_sha256'] == rq['checkpoint_sha256']):
            raise ValueError('primary contrast requires distinct own R/E terminal checkpoints')
        common = ('source', 'config', 'data', 'protocol', 'input_ids', 'source_family_ids', 'noise',
                  'prompt_policy', 'layout', 'backend', 'precision', 'scorer', 'settings', 'comparison',
                  'study_width', 'prepared_identity')
        if any(lq[key] != rq[key] for key in common) or lq['target_site'] != rq['target_site']:
            raise ValueError('predeclared R/E input, initialization, LR, exposure or scoring controls differ')
        for metric in averages:
            items = {}
            for arm, receipt in (('R', left), ('E', right)):
                rows = [{'item_id': str(row.get('sample_id', row.get('item_id'))),
                         'source_id': str(row['source_id']), metric: row[metric]}
                        for row in receipt['conditions'][0]['items']]
                items[arm] = rows
                for row in rows:
                    existing = averages[metric][arm].setdefault(row['item_id'],
                        {'item_id': row['item_id'], 'source_id': row['source_id'], metric: 0.0, 'sites': 0})
                    if existing['source_id'] != row['source_id']:
                        raise ValueError('source-family identity differs across sites')
                    existing[metric] += row[metric] / len(SITES)
                    existing['sites'] += 1
            contrasts.append({'site': site, 'condition': 'no_noise', 'metric': metric,
                              **paired_group_bootstrap(items['R'], items['E'], metric)})
    for metric, arms in averages.items():
        if any(row['sites'] != len(SITES) for rows in arms.values() for row in rows.values()):
            raise ValueError('equally weighted six-site mean requires common complete membership')
        contrasts.append({'site': 'equal_mean_six_sites', 'condition': 'no_noise', 'metric': metric,
                          **paired_group_bootstrap(list(arms['R'].values()), list(arms['E'].values()), metric)})
    return contrasts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', action='append', default=[])
    parser.add_argument('--reconstruction-evaluation')
    parser.add_argument('--reset-control-evaluation')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    result: dict = {'training_curves': [training_curves(Path(root), output) for root in args.training_run],
              'primary_contrast': 'R minus E, no_noise, per-site and equally weighted six-site mean',
              'limits': 'One seed; previously inspected validation; descriptive paired source-family intervals; no acceptance threshold'}
    if bool(args.reconstruction_evaluation) != bool(args.reset_control_evaluation):
        raise ValueError('both own R/E terminal evaluations required for primary contrast')
    if args.reconstruction_evaluation:
        reconstruction = evaluation_panels(Path(args.reconstruction_evaluation), expected_arm='R')
        control = evaluation_panels(Path(args.reset_control_evaluation), expected_arm='E')
        result['width'] = reconstruction[(SITES[0], 'no_noise')]['request']['study_width']
        result['paired_primary'] = paired_primary(reconstruction, control)
    (output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
