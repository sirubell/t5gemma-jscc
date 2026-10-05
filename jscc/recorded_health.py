"""Transferred concrete recorded health responsibility."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from .identity_compatibility import compatible_identity

# Transferred from outer_ln_gate; serialized meanings remain unchanged.
def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def resolve_training_run(linked, path_map=None, expected_config=None):
    """Resolve only the finalized training attempt recorded by this bound case."""
    from .config import load_config
    from .run_records import SCHEMA, bound_config

    def storage_path(value):
        path = Path(value).expanduser()
        if path_map:
            prefix, replacement = (Path(part).expanduser().resolve() for part in path_map)
            normalized = path.resolve()
            if normalized.is_relative_to(prefix):
                path = replacement / normalized.relative_to(prefix)
        return path.resolve()

    if not str(linked).strip():
        raise ValueError("Completed training link is empty")
    logical = storage_path(linked)
    markers = ("experiment.json", "case.json", "training.json", "attempt.json", "checkpoint.json")
    if not any((logical / name).exists() or (logical / name).is_symlink() for name in markers):
        return logical, logical, None
    for name in ("case.json", "config.yaml", "training.json"):
        if (logical / name).resolve().parent != logical:
            raise ValueError("recorded case file outside its own case")
    case = read(logical / "case.json")
    experiment = storage_path(case["experiment"])
    experiment_record_path = experiment / "experiment.json"
    if experiment_record_path.resolve().parent != experiment:
        raise ValueError("recorded experiment file outside its own root")
    experiment_record = read(experiment_record_path)
    if (case.get("schema") != SCHEMA
            or (logical != experiment and logical.parent != experiment / "cases")
            or (logical == experiment and case.get("case_id") is not None)
            or (logical != experiment and case.get("case_id") != logical.name)
            or (experiment / "cases").resolve() != experiment / "cases"
            or experiment_record.get("schema") != SCHEMA
            or storage_path(experiment_record["root"]) != experiment):
        raise ValueError("incompatible recorded training case ancestry/schema")
    config = bound_config(logical)
    identity = hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if case.get("effective_identity") != identity:
        raise ValueError("recorded effective configuration identity changed")
    training = read(logical / "training.json")
    if not isinstance(training.get("path"), str) or not training["path"]:
        raise ValueError("invalid recorded training path")
    run = storage_path(training["path"])
    attempts = logical / "attempts"
    if attempts.resolve() != attempts or run.parent != attempts:
        raise ValueError("recorded training attempt outside its own case")
    attempt = read(run / "attempt.json")
    if (attempt.get("schema") != SCHEMA or attempt.get("stage") != "training"
            or attempt.get("status") != "completed" or storage_path(attempt["case"]) != logical
            or run.name != f"training-{attempt.get('attempt_id')}"):
        raise ValueError("recorded training requires its own finalized completed attempt")
    import yaml
    for name in ("attempt.json", "config.yaml", "completion.json"):
        if (run / name).resolve().parent != run:
            raise ValueError("recorded training evidence outside its own attempt")
    if yaml.safe_load((run / "config.yaml").read_text()) != config:
        raise ValueError("training engine configuration differs from its bound case")
    if expected_config is not None:
        expected = load_config(expected_config)
        expected["run"]["output_dir"] = config["run"]["output_dir"]
        if expected != config:
            raise ValueError("recorded configuration differs from prepared study")
    return logical, run, config


def check(manifest_path, output, path_map=None):
    def storage_path(value):
        path = Path(value).expanduser().resolve()
        if path_map:
            prefix, replacement = (Path(part).expanduser().resolve() for part in path_map)
            if path.is_relative_to(prefix):
                path = replacement / path.relative_to(prefix)
        return path.resolve()

    identity_map = {}

    def compatibility(evaluation, condition):
        raw_id = condition["evaluation_identity"]
        identity = read(evaluation / f"identity_{condition['condition']}.json")
        key = compatible_identity(identity, raw_id)
        identity_map[raw_id] = key
        return key

    path = Path(manifest_path)
    manifest = read(path)
    root = path.parent
    entries = [e for e in manifest['runs'] if e['wave'] == 1]
    evidence = {}
    identities = set()
    for e in entries:
        logical, run, _ = resolve_training_run((root/e['run_link']).read_text().strip(), path_map,
                                               root/e['config'] if 'config' in e else None)
        completion = read(run/'completion.json')
        assert completion['status'] == 'FULL_BUDGET_COMPLETED'
        assert completion['step'] == completion['optimizer_updates'] == 5000
        assert completion['presentations'] == 640000 and completion['reason'] == 'max_steps'
        assert completion['source_verified'] and completion['presentation_budget_verified']
        ckpt = completion['final_checkpoint']
        assert ckpt['file'] == 'step_005000.pt' and ckpt['step'] == 5000
        assert (run/ckpt['file']).resolve().parent == run
        assert sha(run/ckpt['file']) == ckpt['sha256']
        evaluation = storage_path((root/e['evaluation_link']).read_text().strip())
        results = read(evaluation/'results.json')
        assert results['optimizer_step'] == 5000
        assert str(storage_path(results['checkpoint']).resolve()) == str((run/'step_005000.pt').resolve())
        conditions = results['conditions']
        assert [x['condition'] for x in conditions] == ['no_noise', -6, 6, 18]
        for c in conditions:
            n = c['num_samples']
            assert (n.get('effective') if isinstance(n, dict) else n) == 10042
            assert 0 <= c['acc'] <= 1 and 0 <= c['acc_norm'] <= 1
            assert c['compact_num_samples'] == 10042 and c['finite_scores']
            identities.add(compatibility(evaluation, c))
            assert c['channel_uses_valid']['hidden'] > 0
            assert (c['channel_uses_valid']['memory'] > 0) == ('dec_' in e['run_id'])
        evidence[e['run_id']] = {'completion': completion, 'initial': read(run/'initialization.json'),
                                 'presentations': read(run/'presentations.json'),
                                 'noise': [json.loads(s) for s in (run/'randomness_audit.jsonl').read_text().splitlines()],
                                 'run': str(run), 'logical_root': str(logical), 'evaluation': str(evaluation)}
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
    vanilla = storage_path((root/'links/vanilla.txt').read_text().strip())
    v = read(vanilla/'results.json')['conditions']
    assert len(v) == 1 and v[0]['condition'] == 'vanilla'
    assert compatibility(vanilla, v[0]) in identities
    assert v[0]['compact_num_samples'] == 10042 and v[0]['finite_scores']
    assert v[0]['channel_uses_valid'] == {'hidden': 0, 'memory': 0}
    Path(output).write_text(json.dumps({'status': 'PASS_FIRST_WAVE_HEALTH_NOT_SCORE_GAIN',
                                       'plan_hash': manifest['plan_hash'], 'runs': list(evidence),
                                       'training_runs': {key: {field: value[field] for field in ('run', 'logical_root')}
                                                         for key, value in evidence.items()},
                                       'identity_compatibility_policy': 'process-docs-address-only-v1',
                                       'raw_to_compatible_identity': identity_map}, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument("--path-map", nargs=2, metavar=("COMPUTE_PREFIX", "LOGIN_PREFIX"))
    a = p.parse_args()
    check(a.manifest, a.output, a.path_map)

