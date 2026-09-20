"""Run a preapproved fixed-budget pair in fresh subprocesses, then its frozen eval."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--split', choices=['enc_l9', 'enc_l19', 'enc_fn'], required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    if plan['status'] != 'LOCKED_AFTER_BENCHMARK' or not plan['training_authorized']:
        raise ValueError('Training policy not activated after capacity/throughput gate')
    a.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    records = []
    for mode in ['raw', 'five_shot']:
        path = a.plan.parent / 'configs' / f'{a.split}-{mode}.yaml'
        if hashlib.sha256(path.read_bytes()).hexdigest() != plan['config_sha256'][path.name]:
            raise ValueError('Resolved config changed after budget lock')
        link = a.output / f'{mode}-run.txt'
        subprocess.run([sys.executable, str(root/'train.py'), '--config', str(path),
                        '--run-path-file', str(link)], check=True, cwd=root)
        run = Path(link.read_text().strip())
        c = json.loads((run/'completion.json').read_text())
        n = plan['updates']
        if not (c['status'] == 'FULL_BUDGET_COMPLETED' and c['step'] == c['optimizer_updates'] == n
                and c['presentations'] == n*128 and c['source_verified']):
            raise RuntimeError('Partial training cannot substitute for matched fixed-step endpoint')
        records.append({'run_id': f'{a.split}-{mode}', 'run_path': str(run),
                        'checkpoint': f'step_{n:06d}.pt', 'expected_step': n,
                        'checkpoint_sha256': c['final_checkpoint']['sha256']})
    (a.output/'runs.json').write_text(json.dumps(records, indent=2)+'\n')
    subprocess.run([sys.executable, str(root/'scripts/evaluate_prompt_alignment.py'),
                    '--runs-json', str(a.output/'runs.json'), '--prepared', str(a.prepared),
                    '--output', str(a.output/'evaluation')], check=True, cwd=root)
    (a.output/'completion.json').write_text(json.dumps({'status':'COMPLETE_TRAIN_AND_EVAL_PAIR',
        'split':a.split, 'updates_per_arm':plan['updates'], 'presentations_per_arm':plan['updates']*128,
        'training_runs':records, 'evaluation':str(a.output/'evaluation')}, indent=2)+'\n')


if __name__ == '__main__':
    main()
