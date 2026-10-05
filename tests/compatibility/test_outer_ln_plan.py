import json
import subprocess
from pathlib import Path

import pytest

from jscc.config import load_config
from jscc.checkpoint_readers import resolve_saved_outer_ln_plan as resolve
from jscc.study_task import run_task


# Resolve saved fixtures from the package source rather than the working directory.
import jscc

SOURCE = Path(jscc.__file__).resolve().parents[1]
PLAN = SOURCE / 'configs/studies/hellaswag_outer_ln_plan.json'
TASK = SOURCE / 'configs/experiments/ordinary/hellaswag_outer_ln.yaml'


def test_full_plan_and_pair_whitelist(tmp_path):
    plan = json.loads(PLAN.read_text())
    configs = list(resolve(plan, load_config(TASK)).values())
    assert len(configs) == len(plan['runs']) == 22
    assert len(plan['pairs']) == 9
    assert plan['counts']['coarse_candidate_requests'] == 3574952
    assert len(plan['first_wave']) == 6
    assert all(c['training']['max_steps'] == c['training']['schedule_steps'] == 5000 for c in configs)
    assert all(c['training']['batch_size'] * c['training']['gradient_accumulation'] * c['training']['max_steps'] == 640000 for c in configs)
    assert all(c['codec']['snr_film'] is False and c['codec']['memory']['snr_film'] is False for c in configs)


@pytest.mark.parametrize('field,value', [('gradient_accumulation', 2), ('schedule_steps', 5001)])
def test_accidental_accumulation_and_schedule_rejected(field, value):
    plan = json.loads(PLAN.read_text())
    task = load_config(TASK)
    task['training'][field] = value
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
    import yaml
    command = calls[0]
    assert command[1] == str(SOURCE / 'experiment.py')
    assert command[-2] == '--config'
    request = yaml.safe_load(Path(command[-1]).read_text())
    assert request['schema'] == 'jscc-experiment-v1'
    assert request['workflow'] == 'ordinary' and request['operation'] == 'evaluate'
    assert request['execution'] == {'root': str(tmp_path / 'run'), 'attempt': '0001',
        'evaluation': {'checkpoint': 'step_005000.pt', 'expected_step': 5000,
                       'overrides': str(override.resolve())}}
    with pytest.raises(FileExistsError):
        run_task(manifest, 0, 'evaluate', 'step_005000.pt', 5000, str(override))
    assert len(calls) == 1


def test_wrapper_vanilla_options(tmp_path):
    from contracts.test_fixed_step import invoke_launcher
    result, args = invoke_launcher(tmp_path, 'evaluate', ['--fixed-step', '--checkpoint', 'step_005000.pt', '--expected-step', '5000', '--evaluation-config', 'vanilla.yaml'])
    assert result.returncode == 0, result.stderr
    assert args is not None
    assert args[-6:] == ['--checkpoint', 'step_005000.pt', '--expected-step', '5000', '--evaluation-config', 'vanilla.yaml']


@pytest.mark.parametrize("recorded", [False, True])
def test_first_wave_gate_rejects_partial_or_changed_evidence(tmp_path, recorded):
    import hashlib
    from jscc.recorded_health import check
    from jscc.identity_compatibility import identity_digest
    identity = {"task_config": {"hellaswag": {
        "fewshot_config": {"process_docs": "<function process_docs at 0x1234>"},
        "process_docs": "def process_docs(dataset):\n    return dataset\n"}}}
    identity_hash = identity_digest(identity)
    (tmp_path/'links').mkdir()
    entries = []
    pairs = []
    for route in ['enc_fn', 'dec_l8', 'enc_emb']:
        for arm in ['ref', 'outer_none']:
            name = arm+'_'+route
            logical = tmp_path/name
            run = logical / 'attempts/training-0001' if recorded else logical
            run.mkdir(parents=True)
            if recorded:
                import yaml
                config = load_config(TASK)
                config['run']['output_dir'] = str(logical)
                encoded = yaml.safe_dump(config)
                (logical/'config.yaml').write_text(encoded)
                (run/'config.yaml').write_text(encoded)
                (logical/'experiment.json').write_text(json.dumps({'schema':'experiment-run-v1','root':str(logical)}))
                (logical/'case.json').write_text(json.dumps({'schema':'experiment-run-v1','experiment':str(logical),
                    'config_sha256':hashlib.sha256(encoded.encode()).hexdigest(),
                    'effective_identity':hashlib.sha256(json.dumps(config,sort_keys=True,allow_nan=False).encode()).hexdigest()}))
                (logical/'training.json').write_text(json.dumps({'path':str(run)}))
                (run/'attempt.json').write_text(json.dumps({'schema':'experiment-run-v1','case':str(logical),
                    'stage':'training','status':'completed','attempt_id':'0001'}))
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
                           'finite_scores':True,'acc':0.,'acc_norm':0.,'evaluation_identity':identity_hash,
                           'channel_uses_valid':{'hidden':10,'memory':10 if route.startswith('dec') else 0}}
                          for c in ['no_noise',-6,6,18]]
            for condition in conditions:
                (ev / f"identity_{condition['condition']}.json").write_text(json.dumps(identity))
            (ev/'results.json').write_text(json.dumps({'optimizer_step':5000,'checkpoint':str(ckpt),'conditions':conditions}))
            (tmp_path/'links'/f'{name}.txt').write_text(str(logical))
            (tmp_path/'links'/f'{name}-eval.txt').write_text(str(ev))
            entries.append({'wave':1,'run_id':name,'run_link':f'links/{name}.txt','evaluation_link':f'links/{name}-eval.txt'})
        pairs.append({'reference':'ref_'+route,'candidate':'outer_none_'+route,'pair_id':route})
    vanilla = tmp_path/'vanilla'
    vanilla.mkdir()
    (vanilla/'results.json').write_text(json.dumps({'conditions':[{'condition':'vanilla','evaluation_identity':identity_hash,
        'compact_num_samples':10042,'finite_scores':True,'channel_uses_valid':{'hidden':0,'memory':0}}]}))
    (vanilla/'identity_vanilla.json').write_text(json.dumps(identity))
    (tmp_path/'links/vanilla.txt').write_text(str(vanilla))
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'runs':entries,'pairs':pairs,'plan_hash':'plan'}))
    check(manifest,tmp_path/'gate.json')  # low scores do not block a healthy result
    assert json.loads((tmp_path/'gate.json').read_text())['status'].startswith('PASS')
    engine = tmp_path/'ref_enc_fn'
    if recorded:
        engine = engine/'attempts/training-0001'
        report = json.loads((tmp_path/'gate.json').read_text())
        assert report['training_runs']['ref_enc_fn'] == {'run':str(engine),'logical_root':str(tmp_path/'ref_enc_fn')}
    identity_file = engine/'eval/identity_no_noise.json'
    identity_file.write_text(json.dumps({'tampered': True}))
    with pytest.raises(ValueError, match='identity hash mismatch'):
        check(manifest, tmp_path/'tampered-gate.json')
    identity_file.write_text(json.dumps(identity))
    bad = engine/'completion.json'
    value = json.loads(bad.read_text())
    value['step'] = 4999
    bad.write_text(json.dumps(value))
    with pytest.raises(AssertionError):
        check(manifest,tmp_path/'gate2.json')


def recorded_training_fixture(tmp_path):
    import hashlib
    import yaml
    from jscc.recorded_health import resolve_training_run
    logical = tmp_path / 'recorded'
    run = logical / 'attempts/training-0001'
    run.mkdir(parents=True)
    config = load_config(TASK)
    config['run']['output_dir'] = str(logical)
    encoded = yaml.safe_dump(config)
    (logical / 'config.yaml').write_text(encoded)
    (run / 'config.yaml').write_text(encoded)
    (logical / 'experiment.json').write_text(json.dumps({'schema': 'experiment-run-v1', 'root': str(logical)}))
    (logical / 'case.json').write_text(json.dumps({'schema': 'experiment-run-v1',
        'experiment': str(logical), 'config_sha256': hashlib.sha256(encoded.encode()).hexdigest(),
        'effective_identity': hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()}))
    (logical / 'training.json').write_text(json.dumps({'path': str(run)}))
    (run / 'attempt.json').write_text(json.dumps({'schema': 'experiment-run-v1',
        'case': str(logical), 'stage': 'training', 'status': 'completed', 'attempt_id': '0001'}))
    return logical, run, config, resolve_training_run


def test_recorded_reader_exact_attempt_and_path_map(tmp_path):
    logical, run, config, resolve_run = recorded_training_fixture(tmp_path)
    assert resolve_run(logical, expected_config=TASK) == (logical, run, config)
    foreign_prefix = tmp_path / 'compute'
    for file, field in [(logical / 'case.json', 'experiment'),
                        (logical / 'training.json', 'path'), (run / 'attempt.json', 'case'),
                        (logical / 'experiment.json', 'root')]:
        value = json.loads(file.read_text())
        value[field] = str(foreign_prefix / Path(value[field]).relative_to(tmp_path))
        file.write_text(json.dumps(value))
    assert resolve_run(foreign_prefix / 'recorded', (str(foreign_prefix / '.'), str(tmp_path))) == (logical, run, config)


@pytest.mark.parametrize('mutation', ['foreign', 'stage', 'partial', 'malformed', 'symlink', 'config', 'case_id', 'root', 'missing_training', 'config_symlink'])
def test_recorded_reader_rejects_foreign_or_invalid_without_legacy_fallback(tmp_path, mutation):
    logical, run, _, resolve_run = recorded_training_fixture(tmp_path)
    # Plausible legacy evidence must not turn a damaged recorded root into a legacy input.
    (logical / 'completion.json').write_text('{}')
    if mutation == 'case_id':
        value = json.loads((logical / 'case.json').read_text())
        value['case_id'] = 'foreign'
        (logical / 'case.json').write_text(json.dumps(value))
    elif mutation == 'root':
        value = json.loads((logical / 'experiment.json').read_text())
        value['root'] = str(tmp_path / 'other')
        (logical / 'experiment.json').write_text(json.dumps(value))
    elif mutation == 'missing_training':
        (logical / 'training.json').unlink()
    elif mutation == 'config_symlink':
        target = tmp_path / 'config.yaml'
        (logical / 'config.yaml').rename(target)
        (logical / 'config.yaml').symlink_to(target)
    elif mutation == 'malformed':
        (logical / 'training.json').write_text('{')
    elif mutation == 'foreign':
        value = json.loads((run / 'attempt.json').read_text())
        value['case'] = str(tmp_path / 'other')
        (run / 'attempt.json').write_text(json.dumps(value))
    elif mutation in ('stage', 'partial'):
        value = json.loads((run / 'attempt.json').read_text())
        value['stage' if mutation == 'stage' else 'status'] = 'evaluation' if mutation == 'stage' else 'partial'
        (run / 'attempt.json').write_text(json.dumps(value))
    elif mutation == 'symlink':
        outside = tmp_path / 'outside'
        run.rename(outside)
        run.symlink_to(outside, target_is_directory=True)
    else:
        (run / 'config.yaml').write_text('{}')
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        resolve_run(logical)


def test_recorded_marker_dangling_symlink_cannot_fall_back(tmp_path):
    from jscc.recorded_health import resolve_training_run
    legacy = tmp_path / 'legacy'
    legacy.mkdir()
    assert resolve_training_run(legacy) == (legacy, legacy, None)
    (legacy / 'training.json').symlink_to(tmp_path / 'missing')
    with pytest.raises((FileNotFoundError, ValueError)):
        resolve_training_run(legacy)


@pytest.mark.parametrize("named_case", [False, True])
def test_recorded_experiment_marker_cannot_escape_root(tmp_path, named_case):
    logical, run, _, resolve_run = recorded_training_fixture(tmp_path)
    experiment = logical
    if named_case:
        child = experiment / "cases" / "named"
        child.parent.mkdir()
        child.mkdir()
        for path in tuple(experiment.iterdir()):
            if path.name not in ("experiment.json", "cases"):
                path.rename(child / path.name)
        logical, run = child, child / "attempts" / run.name
        case = json.loads((logical / "case.json").read_text())
        case["case_id"] = "named"
        (logical / "case.json").write_text(json.dumps(case))
        (logical / "training.json").write_text(json.dumps({"path": str(run)}))
        attempt = json.loads((run / "attempt.json").read_text())
        attempt["case"] = str(logical)
        (run / "attempt.json").write_text(json.dumps(attempt))
    resolve_run(logical)
    outside = tmp_path / "foreign-experiment.json"
    (experiment / "experiment.json").rename(outside)
    (experiment / "experiment.json").symlink_to(outside)
    with pytest.raises(ValueError, match="experiment file outside"):
        resolve_run(logical)
