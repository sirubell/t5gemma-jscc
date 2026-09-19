import json
import subprocess
from pathlib import Path

import pytest

from jscc.config import load_config
from jscc.outer_ln_study import export, resolve
from jscc.study_task import run_task


PLAN = Path('configs/studies/hellaswag_outer_ln_plan.json')
TASK = Path('configs/tasks/hellaswag_outer_ln.yaml')


def test_full_plan_and_pair_whitelist(tmp_path):
    manifest = export(PLAN, TASK, tmp_path/'prepared')
    assert len(manifest['runs']) == 22
    assert len(manifest['pairs']) == 9
    assert manifest['counts']['coarse_candidate_requests'] == 3574952
    assert sum(x['wave'] == 1 for x in manifest['runs']) == 6
    configs = [load_config(tmp_path/'prepared'/x['config']) for x in manifest['runs']]
    assert all(c['training']['max_steps'] == c['training']['schedule_steps'] == 5000 for c in configs)
    assert all(c['training']['batch_size'] * c['training']['gradient_accumulation'] * c['training']['max_steps'] == 640000 for c in configs)
    assert all(c['codec']['snr_film'] is False and c['codec']['memory']['snr_film'] is False for c in configs)
    with pytest.raises(FileExistsError):
        export(PLAN, TASK, tmp_path/'prepared')


def test_accidental_accumulation_and_schedule_rejected():
    plan = json.loads(PLAN.read_text())
    task = load_config(TASK)
    task['training']['gradient_accumulation'] = 2
    with pytest.raises(AssertionError):
        resolve(plan, task)


def test_eval_override_reaches_actual_entrypoint(tmp_path, monkeypatch):
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'version': 1, 'runs': [{'task': 'hellaswag', 'experiment': 'ref', 'seed': 0, 'run_link': 'run.txt'}]}))
    (tmp_path/'run.txt').write_text(str(tmp_path/'run'))
    calls = []
    monkeypatch.setattr(subprocess, 'run', lambda c, **kw: calls.append(c))
    override = tmp_path/'vanilla.yaml'
    run_task(manifest, 0, 'evaluate', 'step_005000.pt', 5000, str(override))
    assert calls[0][-6:] == ['--checkpoint', 'step_005000.pt', '--expected-step', '5000', '--config', str(override)]


def test_wrapper_vanilla_options(tmp_path):
    from test_fixed_step import invoke_launcher
    result, args = invoke_launcher(tmp_path, 'evaluate', ['--fixed-step', '--checkpoint', 'step_005000.pt', '--expected-step', '5000', '--evaluation-config', 'vanilla.yaml'])
    assert result.returncode == 0, result.stderr
    assert args is not None
    assert args[-6:] == ['--checkpoint', 'step_005000.pt', '--expected-step', '5000', '--evaluation-config', 'vanilla.yaml']


def test_first_wave_gate_rejects_partial_or_changed_evidence(tmp_path):
    import hashlib
    from jscc.outer_ln_gate import check
    (tmp_path/'links').mkdir()
    entries = []
    pairs = []
    for route in ['enc_fn', 'dec_l8', 'enc_emb']:
        for arm in ['ref', 'outer_none']:
            name = arm+'_'+route
            run = tmp_path/name
            run.mkdir()
            ckpt = run/'step_005000.pt'
            ckpt.write_bytes(b'CPU gate fixture, not a model')
            completion = {'status':'FULL_BUDGET_COMPLETED', 'step':5000, 'optimizer_updates':5000,
                          'presentations':640000, 'reason':'max_steps', 'source_verified':True,
                          'presentation_budget_verified':True, 'consumed_ids_bytes_sha256':'ids',
                          'final_checkpoint':{'file':ckpt.name, 'step':5000, 'sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest()}}
            (run/'completion.json').write_text(json.dumps(completion))
            (run/'presentations.json').write_text(json.dumps({'actual_ids_sha256':'ids', 'actual_ids_bytes_sha256':'ids'}))
            (run/'initialization.json').write_text(json.dumps({'hidden':{'core_sha256':'core', 'internal_layernorm_count':4}}))
            (run/'randomness_audit.jsonl').write_text(''.join(json.dumps({'step':i})+'\n' for i in [1,1000,2500,5000]))
            ev = run/'eval'
            ev.mkdir()
            conditions = [{'condition':c,'num_samples':{'effective':10042},'compact_num_samples':10042,
                           'finite_scores':True,'acc':0.,'acc_norm':0.,'evaluation_identity':'same',
                           'channel_uses_valid':{'hidden':10,'memory':10 if route.startswith('dec') else 0}}
                          for c in ['no_noise',-6,6,18]]
            (ev/'results.json').write_text(json.dumps({'optimizer_step':5000,'checkpoint':str(ckpt),'conditions':conditions}))
            (tmp_path/'links'/f'{name}.txt').write_text(str(run))
            (tmp_path/'links'/f'{name}-eval.txt').write_text(str(ev))
            entries.append({'wave':1,'run_id':name,'run_link':f'links/{name}.txt','evaluation_link':f'links/{name}-eval.txt'})
        pairs.append({'reference':'ref_'+route,'candidate':'outer_none_'+route,'pair_id':route})
    vanilla = tmp_path/'vanilla'
    vanilla.mkdir()
    (vanilla/'results.json').write_text(json.dumps({'conditions':[{'condition':'vanilla','evaluation_identity':'same',
        'compact_num_samples':10042,'finite_scores':True,'channel_uses_valid':{'hidden':0,'memory':0}}]}))
    (tmp_path/'links/vanilla.txt').write_text(str(vanilla))
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'runs':entries,'pairs':pairs,'plan_hash':'plan'}))
    check(manifest,tmp_path/'gate.json')  # low scores do not block a healthy result
    assert json.loads((tmp_path/'gate.json').read_text())['status'].startswith('PASS')
    bad = tmp_path/'ref_enc_fn/completion.json'
    value = json.loads(bad.read_text())
    value['step'] = 4999
    bad.write_text(json.dumps(value))
    with pytest.raises(AssertionError):
        check(manifest,tmp_path/'gate2.json')
