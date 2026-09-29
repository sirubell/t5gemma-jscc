"""Run a bounded family with fresh processes and one OOM-only capacity fallback."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--family', choices=['enc','dec'], required=True)
    a = p.parse_args()
    destination = a.root/'results'/a.family
    destination.mkdir(parents=True, exist_ok=False)
    expected_names = ({'enc_emb', 'enc_l4', 'enc_l9', 'enc_l14', 'enc_l19', 'enc_fn'}
                      if a.family == 'enc' else {'dec_l0', 'dec_l4', 'dec_l8', 'dec_l12', 'dec_l16', 'dec_l20', 'dec_l24'})
    configs = sorted((a.root/'configs').glob(a.family+'_*.yaml'))
    if {path.stem for path in configs} != expected_names:
        raise ValueError('Incomplete or unexpected authorized split roster')
    outcomes = []
    for config in configs:
        for microbatch in [128,64]:
            output = destination/f'{config.stem}-b{microbatch}'
            command = [sys.executable, '-m', 'scripts.five_shot_preflight', '--config',str(config),
                       '--prepared',str(a.root/'fixture'), '--output',str(output),'--microbatch',str(microbatch)]
            with (destination/f'{config.stem}-b{microbatch}.log').open('w') as log:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
            record = {'split':config.stem,'microbatch':microbatch,'returncode':code,'output':str(output)}
            failure = output/'failure.json'
            if failure.exists():
                record['failure'] = json.loads(failure.read_text())
            outcomes.append(record)
            (destination/'progress.json').write_text(json.dumps(outcomes,indent=2)+'\n')
            print(json.dumps(record),flush=True)
            if code == 0:
                break
            if not (microbatch == 128 and record.get('failure',{}).get('error_type') == 'OutOfMemoryError'
                    and record.get('failure',{}).get('stage') in ('stress_update', 'timed_updates')):
                break
    completed = len({x['split'] for x in outcomes if x['returncode']==0})
    expected = len(expected_names)
    (destination/'completion.json').write_text(json.dumps({'status':'PASS' if completed==expected else 'INCOMPLETE',
        'completed_splits':completed,'expected_splits':expected,'outcomes':outcomes},indent=2)+'\n')
    if completed != expected:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
