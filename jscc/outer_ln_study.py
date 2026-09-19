"""Prepare the finite 22-run outer-LN comparison. Never execute or submit jobs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from .config import load_config, save_config, validate_config


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_plan(plan):
    runs = {r["run_id"]: r for r in plan["runs"]}
    ref, none = plan["reference_13"], plan["all_outer_none_13_with_reuse"]
    assert not plan["execution_authorized"]
    assert len(runs) == len(plan["runs"]) == 22
    assert len(ref) == len(none) == 13 and len(set(ref) & set(none)) == 4
    assert set(ref) | set(none) == set(runs)
    assert len(plan["pairs"]) == 9
    assert len(plan["first_wave"]) == 6 and len(plan["second_wave"]) == 16
    assert len(set(plan["first_wave"] + plan["second_wave"])) == 22
    assert set(plan["first_wave"] + plan["second_wave"]) == set(runs)
    for a, b in zip(ref, none):
        assert runs[a]["route"] == runs[b]["route"]
        assert runs[b]["main_outer_norm"] == "none"
        assert runs[b]["memory_outer_norm"] in (None, "none")
    for pair in plan["pairs"]:
        a, b = runs[pair["reference"]], runs[pair["candidate"]]
        assert a["split"] == b["split"] and a["memory_enabled"] == b["memory_enabled"]
    assert plan["counts"]["coarse_panels"] == 89
    assert plan["counts"]["coarse_candidate_requests"] == 89 * 10042 * 4
    return runs


def resolve(plan, task):
    runs = validate_plan(plan)
    result = {}
    for key in plan["first_wave"] + plan["second_wave"]:
        entry = runs[key]
        c = copy.deepcopy(task)
        c['run'].update(name=key, output_dir='../runs')
        c['split'] = copy.deepcopy(entry['split'])
        c['codec']['layernorm'] = entry['main_outer_norm']
        c['codec']['snr_film'] = False
        c['codec']['memory'] = {'layernorm': entry['memory_outer_norm'] or 'both', 'snr_film': False}
        validate_config(c)
        t, e = c['training'], c['evaluation']
        assert (t['batch_size'], t['gradient_accumulation'], t['max_steps'], t['schedule_steps']) == (128, 1, 5000, 5000)
        assert t['presentation_stream']['total_presentations'] == 640000
        assert t['save_steps'] == [1000, 2500, 5000] and t['eval_every'] == 250
        assert t['patience'] is None and t['streamed_backward'] and t['valid_only_kl'] and not t['low_sync_logging']
        assert e['batch_size'] == 64 and e['scoring_policy'] == 'fp32-v1' and e['num_samples'] == 10042
        assert e['snrs'] == ['no_noise', -6, 6, 18] and not e['vanilla'] and e['mode'] == 'codec_only'
        result[key] = c
    for pair in plan['pairs']:
        a, b = (copy.deepcopy(result[pair[k]]) for k in ['reference', 'candidate'])
        a.pop('run')
        b.pop('run')
        if pair['changed_component'] == 'main_codec_outer_norm':
            a['codec']['layernorm'] = b['codec']['layernorm']
        else:
            a['codec']['memory']['layernorm'] = b['codec']['memory']['layernorm']
        assert a == b, f'unapproved pair difference: {pair["pair_id"]}'
    return result


def export(plan_path, task_path, output):
    plan = json.loads(Path(plan_path).read_text())
    configs = resolve(plan, load_config(task_path))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'configs').mkdir()
    (output/'links').mkdir()
    entries = []
    for index, (key, config) in enumerate(configs.items()):
        relative = f'configs/{index:02d}-{key}.yaml'
        save_config(config, output/relative)
        entries.append({'index': index, 'run_id': key, 'experiment': key, 'task': 'hellaswag', 'seed': 0,
                        'config': relative, 'config_sha256': hashlib.sha256((output/relative).read_bytes()).hexdigest(),
                        'run_link': f'links/{index:02d}.txt', 'evaluation_link': f'links/{index:02d}-evaluation.txt', 'fixed_checkpoint': 'step_005000.pt',
                        'expected_step': 5000, 'wave': 1 if index < 6 else 2})
    manifest = {'version': 1, 'study': plan['plan_id'], 'execution_authorized': False, 'runs': entries,
                'pairs': plan['pairs'], 'reference_13': plan['reference_13'],
                'all_outer_none_13_with_reuse': plan['all_outer_none_13_with_reuse'], 'counts': plan['counts']}
    resources = {'single_gpu': True, 'concurrency_enforcement': 'existing Slurm lab/account GPU cap4; no extra array throttle', 'max_concurrent_gpu_jobs_including_eval': 4,
                 'train_hours_per_task': 2, 'evaluate_hours_per_task': 1, 'vanilla_hours': 0.5,
                 'capacity_jobs': 4, 'capacity_hours_each': 0.5, 'scoring_check_hours': 0.5,
                 'max_allocation_gpu_hours': 69, 'max_capacity_updates': 88,
                 'max_preflight_candidate_requests': 4096,
                 'site': {'partition': 'h200q', 'account': 'shaoyulien', 'qos': 'h200q_1g',
                          'cpus_per_task': 16, 'memory_mb_site_assigned_last_observation': 174080},
                 'scope': 'proposed limits; site settings from recorded environment, no fresh remote query'}
    policy = {'execution_authorized': False, 'scientific_policy': plan['proposed_policy'],
              'resources': resources, 'sampler': configs[next(iter(configs))]['training']['presentation_stream'],
              'remaining_gpu_gates': ['length inventory from full tokenized pool', 'uncovered route capacity',
                                      'production FP32 scoring fixture', 'shared vanilla bypass identity']}
    manifest['plan_hash'] = digest({'manifest': manifest, 'policy': policy})
    policy['plan_hash'] = manifest['plan_hash']
    for name, data in [('study-manifest.proposed.json', manifest), ('LOCKED-POLICY.proposed.json', policy)]:
        (output/name).write_text(json.dumps(data, indent=2)+'\n')
    vanilla_manifest = copy.deepcopy(manifest)
    vanilla_manifest['runs'] = [dict(entries[0], evaluation_link='links/vanilla.txt')]
    (output/'vanilla-manifest.proposed.json').write_text(json.dumps(vanilla_manifest, indent=2)+'\n')
    save_config({'mode': 'vanilla_only', 'snrs': [], 'vanilla': True}, output/'vanilla.yaml')
    # Per-entry evaluation dependencies preserve pairing; the verified lab cap enforces concurrency.
    phases = [
        {'id': 'capacity', 'jobs': 4, 'concurrency': 4, 'hours_each': 0.5, 'depends_on': 'owner approval'},
        {'id': 'scoring-check', 'jobs': 1, 'concurrency': 1, 'hours_each': 0.5, 'depends_on': 'capacity + G2/G3'},
        {'id': 'wave1-train', 'indices': '0-5', 'depends_on': 'scoring-check + approved gate'},
        {'id': 'wave1-evaluate', 'indices': '0-5', 'depends_on': 'aftercorr:W1_TRAIN', 'checkpoint': 'step_005000.pt', 'expected_step': 5000},
        {'id': 'vanilla', 'indices': '0', 'depends_on': 'afterok:W1_EVAL', 'overrides': 'vanilla.yaml'},
        {'id': 'wave1-review', 'cpu_only': True, 'depends_on': 'vanilla; check health/provenance not score gain'},
        {'id': 'wave2-train', 'indices': '6-21', 'depends_on': 'all22 authorization + wave1 health gate'},
        {'id': 'wave2-evaluate', 'indices': '6-21', 'depends_on': 'aftercorr:W2_TRAIN', 'checkpoint': 'step_005000.pt', 'expected_step': 5000},
    ]
    (output/'wave-plan.json').write_text(json.dumps(phases, indent=2)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default='configs/studies/hellaswag_outer_ln_plan.json')
    parser.add_argument('--task', default='configs/tasks/hellaswag_outer_ln.yaml')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = export(args.plan, args.task, args.output)
    print(json.dumps({'runs': len(result['runs']), 'counts': result['counts'], 'plan_hash': result['plan_hash'],
                      'execution_authorized': False}, indent=2))


if __name__ == '__main__':
    main()
