"""One-shot bounded screen lane; scheduling and resource caps are explicit."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ARMS = {'W005': .05, 'W05': .5, 'W5': 5.}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def reserve_probe(root, name):
    with (root/'vjp-reservations.jsonl').open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        old = [json.loads(line) for line in stream]
        if any(row['name'] == name for row in old) or sum(row['reserved'] for row in old)+12 > 96:
            raise RuntimeError('Duplicate probe or global VJP cap')
        stream.write(json.dumps({'name': name, 'reserved': 12, 'unix': time.time()})+'\n')
        stream.flush()


def run(root, lane):
    auth = json.loads((root/'AUTHORIZATION.json').read_text())
    if auth.get('plan_id') != 'MAIN_WEIGHT_EARLY_SCREEN_V1' or not auth.get('execution_authorized'):
        raise ValueError('Missing current screen authorization')
    deadline = auth['gpu_work_stop_at_unix']
    cooperative_deadline = deadline - 45  # Leave checkpoint/receipt flush time before hard cutoff.
    if time.time() >= auth['stop_new_gpu_at_unix']:
        raise TimeoutError('No new work after authorization dispatch cutoff')
    release = Path(__file__).resolve().parents[2]
    for name, h in json.loads((root/'source/execution-source.json').read_text()).items():
        if sha(release/name) != h:
            raise ValueError('Executed source mismatch: '+name)
    for name, h in json.loads((root/'source/input-manifest.json').read_text()).items():
        if sha(root/name) != h:
            raise ValueError('Frozen input mismatch: '+name)
    (root/'logs').mkdir(exist_ok=True)
    (root/f'{lane}.started').open('x').write(str(time.time()))
    records = []
    started = time.time()

    def execute(name, module, args, seconds):
        if time.time() >= min(auth['stop_new_gpu_at_unix'], cooperative_deadline):
            raise TimeoutError('Dispatch cutoff reached')
        command = [sys.executable, '-m', module, *map(str, args)]
        row = {'name': name, 'command': command, 'started_unix': time.time(), 'exit_code': None}
        with (root/f'commands-{lane}.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        try:
            with (root/'logs'/f'{name}.log').open('w') as f:
                proc = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT,
                                      timeout=min(seconds, max(0.001, deadline-time.time())), check=False)
            row['exit_code'] = proc.returncode
            if proc.returncode:
                raise RuntimeError(name+' failed; no retry')
        except subprocess.TimeoutExpired:
            row['hard_timeout'] = True
            row['checkpoint_flush_confirmed'] = False
            raise
        finally:
            row['elapsed_seconds'] = time.time()-row['started_unix']
            records.append(row)
            (root/f'progress-{lane}.json').write_text(json.dumps(records, indent=2))

    def probe(arm, step, checkpoint):
        name = f'probe-{arm}-{step}'
        reserve_probe(root, name)
        execute(name, 'scripts.experiments.main_weight_screen_gradients', [
            '--checkpoint', checkpoint, '--fixture', root/'fixtures/e3-selection.json',
            '--output', root/'gradients'/f'{arm}-{step}', '--arm', arm, '--step', step,
            '--hidden-weight', .05 if arm == 'shared_init' else ARMS[arm],
            '--seconds', 90, '--deadline-epoch', cooperative_deadline], 100)

    result = {'status': 'FAILED_OR_PARTIAL', 'lane': lane}
    try:
        if lane == 'preflight':
            for arm in ARMS:
                out = root/'preflight'/arm
                execute('preflight-'+arm, 'scripts.experiments.main_weight_screen_train', [
                    '--config', root/'configs/base-dec_l20.yaml', '--selection-fixture', root/'fixtures/e3-selection.json', '--output', out,
                    '--arm', arm, '--preflight', '--seconds', 120, '--deadline-epoch', cooperative_deadline], 130)
                receipt = json.loads((out/'report.json').read_text())
                if receipt['status'] != 'MEASURED' or receipt['optimizer_calls'] != 2:
                    raise ValueError('Preflight receipt incomplete')
            initials = [json.loads((root/'preflight'/arm/'report.json').read_text())['initial_state_fp32'] for arm in ARMS]
            if not all(v == initials[0] for v in initials):
                raise ValueError('Preflight initial states differ')
            probe('shared_init', 0, root/'preflight/W005/init.pt')
        else:
            required = 'preflight' if lane == 'W005' else 'W005'
            previous = json.loads((root/f'completion-{required}.json').read_text())
            if previous['status'] != 'COMPLETE':
                raise ValueError('Required predecessor not complete')
            out = root/'training'/lane
            execute('train-'+lane, 'scripts.experiments.main_weight_screen_train', [
                '--config', root/'configs/base-dec_l20.yaml', '--selection-fixture', root/'fixtures/e3-selection.json', '--output', out, '--arm', lane,
                '--seconds', 2400, '--deadline-epoch', cooperative_deadline], 2420)
            receipt = json.loads((out/'report.json').read_text())
            pre = json.loads((root/'preflight/W005/report.json').read_text())
            if receipt['status'] != 'MEASURED' or receipt['optimizer_calls'] != 1000 or receipt['initial_state_fp32'] != pre['initial_state_fp32']:
                raise ValueError('Training count or clean initialization mismatch')
            training_run = Path(receipt['run'])
            for step in [500, 1000]:
                ckpt = training_run/f'step_{step:06d}.pt'
                probe(lane, step, ckpt)
                args = ['--root', root, '--route', 'dec_l20', '--checkpoint', ckpt,
                        '--checkpoint-sha256', sha(ckpt), '--step', step, '--arm', lane]
                if lane == 'W005' and step == 1000:
                    args += ['--vanilla']
                execute(f'eval-{lane}-{step}', 'scripts.experiments.main_weight_screen_eval', args, 90)
        result['status'] = 'COMPLETE'
    except BaseException as exc:
        result['error'] = repr(exc)
        raise
    finally:
        result.update(records=records, elapsed_seconds=time.time()-started)
        (root/f'completion-{lane}.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--lane', choices=['preflight', *ARMS], required=True)
    a = p.parse_args()
    run(a.root, a.lane)
