"""Verify completed first-wave artifacts before releasing the queued second wave."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def check(manifest_path, output):
    path = Path(manifest_path)
    manifest = read(path)
    root = path.parent
    entries = [e for e in manifest['runs'] if e['wave'] == 1]
    evidence = {}
    identities = set()
    for e in entries:
        run = Path((root/e['run_link']).read_text().strip())
        completion = read(run/'completion.json')
        assert completion['status'] == 'FULL_BUDGET_COMPLETED'
        assert completion['step'] == completion['optimizer_updates'] == 5000
        assert completion['presentations'] == 640000 and completion['reason'] == 'max_steps'
        assert completion['source_verified'] and completion['presentation_budget_verified']
        ckpt = completion['final_checkpoint']
        assert ckpt['file'] == 'step_005000.pt' and ckpt['step'] == 5000
        assert sha(run/ckpt['file']) == ckpt['sha256']
        evaluation = Path((root/e['evaluation_link']).read_text().strip())
        results = read(evaluation/'results.json')
        assert results['optimizer_step'] == 5000
        assert str(Path(results['checkpoint']).resolve()) == str((run/'step_005000.pt').resolve())
        conditions = results['conditions']
        assert [x['condition'] for x in conditions] == ['no_noise', -6, 6, 18]
        for c in conditions:
            n = c['num_samples']
            assert (n.get('effective') if isinstance(n, dict) else n) == 10042
            assert 0 <= c['acc'] <= 1 and 0 <= c['acc_norm'] <= 1
            assert c['compact_num_samples'] == 10042 and c['finite_scores']
            identities.add(c['evaluation_identity'])
            assert c['channel_uses_valid']['hidden'] > 0
            assert (c['channel_uses_valid']['memory'] > 0) == ('dec_' in e['run_id'])
        evidence[e['run_id']] = {'completion': completion, 'initial': read(run/'initialization.json'),
                                 'presentations': read(run/'presentations.json'),
                                 'noise': [json.loads(s) for s in (run/'randomness_audit.jsonl').read_text().splitlines()],
                                 'run': str(run), 'evaluation': str(evaluation)}
    assert len(identities) == 1, 'evaluation identities differ across first-wave routes/conditions'
    for record in evidence.values():
        assert record['completion']['consumed_ids_bytes_sha256'] == record['presentations']['actual_ids_bytes_sha256']
        assert {x['step'] for x in record['noise']} == {1, 1000, 2500, 5000}
    for pair in manifest['pairs']:
        if pair['reference'] not in evidence:
            continue
        a, b = evidence[pair['reference']], evidence[pair['candidate']]
        assert a['presentations']['actual_ids_sha256'] == b['presentations']['actual_ids_sha256']
        assert a['noise'] == b['noise'], pair['pair_id']
        assert a['initial'].keys() == b['initial'].keys()
        for stream in a['initial']:
            assert a['initial'][stream]['core_sha256'] == b['initial'][stream]['core_sha256']
            assert a['initial'][stream]['internal_layernorm_count'] == b['initial'][stream]['internal_layernorm_count'] == 4
    vanilla = Path((root/'links/vanilla.txt').read_text().strip())
    v = read(vanilla/'results.json')['conditions']
    assert len(v) == 1 and v[0]['condition'] == 'vanilla'
    assert v[0]['evaluation_identity'] in identities
    assert v[0]['compact_num_samples'] == 10042 and v[0]['finite_scores']
    assert v[0]['channel_uses_valid'] == {'hidden': 0, 'memory': 0}
    Path(output).write_text(json.dumps({'status': 'PASS_FIRST_WAVE_HEALTH_NOT_SCORE_GAIN',
                                       'plan_hash': manifest['plan_hash'], 'runs': list(evidence)}, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    check(a.manifest, a.output)


if __name__ == '__main__':
    main()
