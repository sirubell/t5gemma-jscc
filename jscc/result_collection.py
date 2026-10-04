"""Model-free compact exports and verified, recoverable local collection.

A source directory may be a separately authorized mounted/staged remote root.
This module performs no network access and never changes scientific records.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable
from urllib.parse import quote
import uuid

import yaml

from .experiment_records import write_json_atomic

COMPANIONS = {
    'experiment.json', 'recipe-run.json', 'case.json', 'config.yaml', 'recipe.json',
    'counts.json', 'training.json', 'checkpoint.json', 'attempt.json', 'completion.json',
    'run.json', 'campaign.json', 'freeze.json', 'checkpoints.json', 'publication.json',
    'eligibility.json', 'gate.json', 'request.json', 'result.json', 'results.json',
    'overrides.yaml', 'checkpoint_timing.jsonl', 'reconciliation.json',
}
# These bulk fields have exact original-file references, never an inferred value.
REMOTE_DIRECTORIES = {'cache', 'data', 'model', 'models', 'assets', 'debug'}
BULK_FIELDS = {'items', 'predictions', 'samples', 'documents', 'prompts', 'audit'}


def _digest(path: Path) -> tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _stamp(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _reference(path: Path, root: Path, source_uri: str | None) -> str:
    if source_uri is None:
        return path.as_uri()
    return source_uri.rstrip('/') + '/' + quote(path.relative_to(root).as_posix(), safe='/')


def _compact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _compact(v) for k, v in value.items() if k not in BULK_FIELDS}
    if isinstance(value, list):
        return [_compact(v) for v in value]
    return value


def _relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or '..' in path.parts:
        raise ValueError('selected artifact must be a relative path inside the experiment')
    return path


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f'artifact escapes experiment: {path}')
    return resolved


def _read_stable(path: Path) -> tuple[Any, dict]:
    before = _stamp(path)
    with path.open('rb') as stream:
        raw = stream.read()
    if before != _stamp(path):
        raise ValueError(f'source changed during snapshot: {path}')
    value = json.loads(raw)
    return value, {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def _context(path: Path, root: Path, cases: list[Path], records: dict[Path, Any]) -> dict:
    case = next((c for c in cases if path.is_relative_to(c)), None)
    case_record = records.get(case / 'case.json', {}) if case else {}
    attempts = [p for p in path.parents if p != root and
                p.name.startswith(('training-', 'evaluation-'))]
    attempt = next((p for p in attempts if (p / 'attempt.json') in records), None)
    record = records.get(attempt / 'attempt.json', {}) if attempt else {}
    return {'case_id': case_record.get('case_id'), 'case_path': str(case) if case else None,
            'task': records.get(case / 'config.yaml', {}).get('task') if case else None, 'attempt': str(attempt) if attempt else None,
            'attempt_record': {k: record.get(k) for k in ('schema', 'stage', 'attempt_id', 'status', 'source', 'runtime', 'request')}, 'source_record': str(path)}


def prepare_snapshot(root: Path, snapshot: Path, *, selected: Iterable[str] = (),
                     source_uri: str | None = None) -> dict:
    """Freeze an identified complete scalar/record snapshot; reject growing reads.

    No directory-wide scientific success is inferred. Missing explicit selections
    are retained as unknown-size entries. A later snapshot is needed to bind them.
    """
    begun = time.perf_counter()
    root, snapshot = Path(root).resolve(), Path(snapshot).absolute()
    if snapshot.resolve().is_relative_to(root):
        raise ValueError('snapshot must be outside the scientific source root')
    if snapshot.exists():
        raise FileExistsError(snapshot)
    if not (root / 'experiment.json').is_file():
        raise ValueError('collection requires a recorded experiment root')
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='.collection-snapshot-', dir=snapshot.parent))
    try:
        files = sorted(p for p in root.rglob('*') if p.is_file() and not any(
            part.startswith('.') for part in p.relative_to(root).parts))
        choices = {_relative(s).as_posix() for s in selected}
        records: dict[Path, Any] = {}
        artifacts, remote, errors, result_rows, status_rows = [], [], [], [], []
        for path in files:
            relative = path.relative_to(root).as_posix()
            ref = _reference(path, root, source_uri)
            default_companion = (path.name in COMPANIONS or path.name.endswith('.pt.json')) and not REMOTE_DIRECTORIES.intersection(path.relative_to(root).parts[:-1])
            if default_companion or relative in choices:
                _inside(path, root)
                before = _stamp(path)
                sha, size = _digest(path)
                if before != _stamp(path):
                    raise ValueError(f'source changed during snapshot: {path}')
                artifacts.append({'path': 'files/' + relative, 'source': str(path),
                                  'sha256': sha, 'bytes': size, 'kind': 'companion' if default_companion else 'explicit',
                                  'retrieval_ref': ref})
                if path.suffix == '.json' and default_companion:
                    value, identity = _read_stable(path)
                    if identity['sha256'] != sha:
                        raise ValueError(f'source changed during snapshot: {path}')
                    records[path] = value
                elif path.name == 'config.yaml':
                    before = _stamp(path)
                    raw = path.read_bytes()
                    if before != _stamp(path) or hashlib.sha256(raw).hexdigest() != sha:
                        raise ValueError(f'source changed during snapshot: {path}')
                    records[path] = yaml.safe_load(raw)
            else:
                remote.append({'path': relative, 'bytes': path.stat().st_size,
                               'retrieval_ref': ref, 'verification': 'metadata only; bytes not copied or verified'})
        for relative in sorted(choices - {p.relative_to(root).as_posix() for p in files}):
            source = _inside(root / relative, root)
            artifacts.append({'path': 'files/' + relative, 'source': str(source), 'sha256': None,
                              'bytes': None, 'kind': 'explicit', 'retrieval_ref': _reference(source, root, source_uri)})
        # Preserve exact recorded retrieval identities without reading tensor or
        # repeated-input bytes. Missing small result companions remain selected.
        known = {row['source']: row for row in artifacts}
        remote_by_path = {str(root / row['path']): row for row in remote}
        def references(value):
            if isinstance(value, dict):
                if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
                    yield value
                for child in value.values():
                    yield from references(child)
            elif isinstance(value, list):
                for child in value:
                    yield from references(child)
        for value in records.values():
            for ref in references(value):
                path = Path(ref['path'])
                if not path.is_absolute():
                    continue  # Relative artifact maps remain in their exact record.
                path = path.resolve()
                if not path.is_relative_to(root):
                    continue  # External assets are referenced by the bound records.
                absolute = str(path)
                if absolute in known:
                    if known[absolute]['sha256'] != ref['sha256']:
                        raise ValueError(f'recorded selected identity differs: {path}')
                else:
                    row = remote_by_path.get(absolute)
                    if row is None:
                        row = {'path': path.relative_to(root).as_posix(), 'bytes': ref.get('size', ref.get('size_bytes')),
                               'retrieval_ref': _reference(path, root, source_uri), 'present': path.is_file(),
                               'verification': 'recorded metadata only; bytes not copied or verified'}
                        remote.append(row)
                        remote_by_path[absolute] = row
                    row['recorded_sha256'] = ref['sha256']
                    row['recorded_bytes'] = ref.get('size', ref.get('size_bytes'))
                    if path.name in COMPANIONS and not path.exists():
                        entry = {'path': 'files/' + path.relative_to(root).as_posix(), 'source': absolute,
                                 'sha256': ref['sha256'], 'bytes': ref.get('size', ref.get('size_bytes')),
                                 'kind': 'companion', 'retrieval_ref': row['retrieval_ref']}
                        artifacts.append(entry)
                        known[absolute] = entry
        experiment = records[root / 'experiment.json']
        if experiment.get('schema') != 'experiment-run-v1':
            raise ValueError('unsupported experiment root schema')
        cases = [root] if (root / 'case.json') in records else sorted(
            p.parent for p in records if p.name == 'case.json' and p.parent.parent == root / 'cases')
        if not cases:
            raise ValueError('no recorded cases')
        for path, value in records.items():
            if path.suffix != '.json':
                continue
            context = _context(path, root, cases, records)
            if path.name == 'case.json':
                config = path.parent / 'config.yaml'
                if not config.is_file() or _digest(config)[0] != value.get('config_sha256'):
                    raise ValueError(f'bound case config missing or changed: {config}')
            if path.name in {'attempt.json', 'campaign.json', 'completion.json', 'gate.json', 'training.json', 'checkpoint.json', 'publication.json'}:
                status_rows.append({**context, 'record': _compact(value)})
            if path.name in {'result.json', 'results.json'}:
                # Actual panel/condition rows, plus aggregate accounting with all
                # missing/failed/pending cases. No fabricated metric rows.
                for key in ('panels', 'conditions', 'cases'):
                    if isinstance(value.get(key), list):
                        for row in value[key]:
                            result_rows.append({**context, 'table': key, 'scope': value.get('scope', value.get('request', {}).get('scope')),
                                'status': value.get('status'), 'checkpoint': value.get('checkpoint'),
                                'policy': value.get('request'), 'counts': value.get('counts'),
                                'result_identity': next(a for a in artifacts if a['source'] == str(path)),
                                'result_record': _compact({k: v for k, v in value.items() if k not in ('panels', 'conditions', 'cases')}),
                                'row': _compact(row)})
                if 'error' in value:
                    errors.append({**context, 'error': value['error']})
            if path.name == 'attempt.json' and value.get('details', {}).get('error_type'):
                errors.append({**context, 'error': value['details']})
            if path.name == 'campaign.json' and value.get('error'):
                errors.append({**context, 'error': value['error']})
        exports = work / 'exports'
        exports.mkdir()
        scalar_count = 0
        scalar_sources = []
        with (exports / 'scalars.jsonl').open('w') as output:
            for path in files:
                if path.name != 'metrics.jsonl' or REMOTE_DIRECTORIES.intersection(path.relative_to(root).parts[:-1]):
                    continue
                _inside(path, root)
                before = _stamp(path)
                context = _context(path, root, cases, records)
                # Read the large raw stream once. Only compact rows are spooled;
                # their exact raw-stream identity is attached after its digest.
                with tempfile.TemporaryFile(mode='w+') as spool, path.open('rb') as stream:
                    digest, size = hashlib.sha256(), 0
                    for sequence, line in enumerate(stream):
                        digest.update(line)
                        size += len(line)
                        if not line.endswith(b'\n'):
                            raise ValueError(f'unterminated scalar record in snapshot: {path}:{sequence+1}')
                        event = json.loads(line)
                        if not isinstance(event, dict):
                            raise ValueError('metrics rows must be objects')
                        spool.write(json.dumps({'sequence': sequence, 'event': _compact(event)}, allow_nan=False) + '\n')
                        if event.get('event_type') == 'failure':
                            errors.append({**context, 'sequence': sequence, 'error': _compact(event)})
                    if before != _stamp(path):
                        raise ValueError(f'source changed during scalar snapshot: {path}')
                    raw_identity = {'path': str(path), 'sha256': digest.hexdigest(), 'bytes': size,
                                    'retrieval_ref': _reference(path, root, source_uri)}
                    scalar_sources.append(raw_identity)
                    freeze = records.get(path.parent / 'freeze.json', {})
                    if freeze.get('metrics') and freeze['metrics'].get('sha256') != raw_identity['sha256']:
                        raise ValueError(f'raw metrics differ from frozen identity: {path}')
                    spool.seek(0)
                    for line in spool:
                        output.write(json.dumps({**context, 'raw_metrics': raw_identity, **json.loads(line)}, allow_nan=False) + '\n')
                        scalar_count += 1
        with (exports / 'results.jsonl').open('w') as output:
            for row in result_rows:
                output.write(json.dumps(row, allow_nan=False) + '\n')
        write_json_atomic(exports / 'status.json', {'experiment': experiment, 'cases': [records[c/'case.json'] for c in cases],
                           'records': status_rows, 'science_acceptance': 'not inferred from collection'})
        write_json_atomic(exports / 'errors.json', errors)
        inputs = []
        for path in files:
            if path.name in {'data_ids.json', 'data_selection.json'}:
                _inside(path, root)
                before = _stamp(path)
                sha, size = _digest(path)
                if before != _stamp(path):
                    raise ValueError(f'input identity changed during snapshot: {path}')
                inputs.append({**_context(path, root, cases, records), 'path': str(path), 'sha256': sha,
                               'bytes': size, 'retrieval_ref': _reference(path, root, source_uri), 'copied': False})
        write_json_atomic(exports / 'inputs.json', inputs)
        for path in sorted(exports.iterdir()):
            sha, size = _digest(path)
            artifacts.append({'path': path.relative_to(work).as_posix(), 'source': str(snapshot / path.relative_to(work)),
                              'sha256': sha, 'bytes': size, 'kind': 'export'})
        manifest = {'schema': 'result-collection-snapshot-v1', 'root': str(root), 'source_uri': source_uri,
            'semantics': 'stable recorded bytes; growing/unterminated reads reject; not whole-campaign atomicity',
            'artifacts': artifacts, 'remote': remote, 'errors': errors, 'scalar_sources': scalar_sources, 'selected': sorted(choices),
            'scalar_rows': scalar_count, 'result_rows': len(result_rows), 'case_count': len(cases),
            'export_seconds': time.perf_counter() - begun,
            'checkpoint_verification': 'recorded hashes are metadata; no tensor verification or backup'}
        manifest['snapshot_id'] = hashlib.sha256(json.dumps(manifest, sort_keys=True, allow_nan=False).encode()).hexdigest()
        write_json_atomic(work / 'manifest.json', manifest)
        os.rename(work, snapshot)
        return manifest
    except BaseException as error:
        write_json_atomic(work / 'preparation-failure.json', {'error_type': type(error).__name__, 'message': str(error),
                          'root': str(root), 'requested_snapshot': str(snapshot), 'elapsed_seconds': time.perf_counter()-begun})
        raise


def _matches(identity: tuple[str, int], expected: dict) -> bool:
    return identity[0] == expected['sha256'] and (expected['bytes'] is None or identity[1] == expected['bytes'])


def _copy_verified(source: Path, target: Path, expected: dict, row: dict) -> None:
    """One descriptor, optional known size, atomic publication and readback hash."""
    before = _stamp(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.copy-', dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as out, source.open('rb') as incoming:
            digest = hashlib.sha256()
            row['copied'] = True
            for block in iter(lambda: incoming.read(1024 * 1024), b''):
                out.write(block)
                digest.update(block)
                row['transferred_bytes'] += len(block)
            out.flush()
            os.fsync(out.fileno())
        if before != _stamp(source) or not _matches((digest.hexdigest(), row['transferred_bytes']), expected):
            raise ValueError(f'selected source differs from snapshot: {source}')
        os.replace(temporary, target)
        identity = _digest(target)
        if not _matches(identity, expected):
            raise ValueError(f'destination verification failed: {target}')
        row['verified_bytes'] = identity[1]
    finally:
        temporary.unlink(missing_ok=True)


def collect_snapshot(snapshot: Path, destination: Path) -> dict:
    """Copy exactly selected snapshot bytes; retain every pass and retry failures."""
    begun = time.perf_counter()
    snapshot, destination = Path(snapshot).resolve(), Path(destination).resolve()
    manifest, manifest_ref = _read_stable(snapshot / 'manifest.json')
    expected_identity = hashlib.sha256(json.dumps({k: v for k, v in manifest.items() if k != 'snapshot_id'}, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if expected_identity != manifest.get('snapshot_id'):
        raise ValueError('snapshot manifest identity changed')
    if manifest.get('schema') != 'result-collection-snapshot-v1':
        raise ValueError('unsupported collection snapshot')
    if destination.is_relative_to(Path(manifest['root'])) or destination.is_relative_to(snapshot) or snapshot.is_relative_to(destination):
        raise ValueError('destination must be separate from source and snapshot')
    destination.mkdir(parents=True, exist_ok=True)
    # Receipt parents are also write boundaries; reject escapes before copying.
    _inside(destination / 'receipts', destination)
    _inside(destination / 'collection.json', destination)
    lock = destination / '.collection-lock'
    lock.mkdir()
    receipt: dict[str, Any] = {'schema': 'result-collection-receipt-v1', 'snapshot_id': manifest['snapshot_id'],
        'snapshot': manifest_ref, 'destination': str(destination), 'pass_id': uuid.uuid4().hex,
        'status': 'incomplete', 'artifacts': [], 'counts': {}, 'remote': manifest['remote'],
        'limits': 'Collection is separate from training/evaluation/scientific success; unselected tensor hashes are metadata only.'}
    try:
        for entry in manifest['artifacts']:
            row = {**entry, 'status': 'failed', 'copied': False, 'reused': False, 'transferred_bytes': 0, 'verified_bytes': None}
            receipt['artifacts'].append(row)
            try:
                relative = _relative(entry['path'])
                target = _inside(destination / relative, destination)
                if entry['sha256'] is None:
                    row.update(status='missing', error='No selected snapshot bytes available; bind a new snapshot after acquisition')
                elif target.is_file() and _matches(identity := _digest(target), entry):
                    row.update(status='verified', reused=True, verified_bytes=identity[1])
                else:
                    _copy_verified(Path(entry['source']), target, entry, row)
                    row.update(status='verified', copied=True)
            except FileNotFoundError as error:
                row.update(status='missing', error=str(error))
            except (OSError, ValueError) as error:
                row.update(status='failed', error=str(error))
        for category in ('selected', 'copied', 'verified', 'reused', 'missing', 'failed'):
            rows = [r for r in receipt['artifacts'] if category == 'selected' or
                    (r['status'] == category if category in ('verified', 'missing', 'failed') else r[category])]
            field = 'transferred_bytes' if category == 'copied' else 'verified_bytes' if category in ('verified', 'reused') else 'bytes'
            receipt['counts'][category] = {'count': len(rows), 'bytes': sum(r[field] or 0 for r in rows),
                                           'unknown_bytes': sum(r[field] is None for r in rows)}
        receipt['status'] = 'complete' if not any(r['status'] != 'verified' for r in receipt['artifacts']) else 'incomplete'
        receipt['elapsed_seconds'] = time.perf_counter() - begun
        receipt_path = _inside(destination / 'receipts' / f"{receipt['pass_id']}.json", destination)
        receipt['receipt'] = str(receipt_path)
        write_json_atomic(receipt_path, receipt)
        write_json_atomic(_inside(destination / 'collection.json', destination), receipt)
        return receipt
    finally:
        lock.rmdir()


def collect_results(root: Path, destination: Path, *, snapshot: Path,
                    selected: Iterable[str] = (), source_uri: str | None = None) -> dict:
    """One action to export once and copy/retry an explicitly identified snapshot."""
    root, snapshot = Path(root).resolve(), Path(snapshot).absolute()
    if not snapshot.exists():
        prepare_snapshot(root, snapshot, selected=selected, source_uri=source_uri)
    else:
        saved = json.loads((snapshot / 'manifest.json').read_text())
        requested = {_relative(s).as_posix() for s in selected}
        existing = set(saved['selected'])
        if saved['root'] != str(root) or saved['source_uri'] != source_uri or requested != existing:
            raise ValueError('retry must retain the same source, explicit selection and retrieval identity')
    return collect_snapshot(snapshot, destination)
