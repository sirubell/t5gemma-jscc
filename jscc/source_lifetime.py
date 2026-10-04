"""Fixed checkout/runtime bindings and durable consumer ownership.

No scheduler lookup or checkout mutation is performed here. Terminal evidence
comes from the consumer owner; missing/uncertain evidence keeps the checkout held.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from .activation_replay import canonical_digest, file_digest
from .experiment_records import write_json_atomic

TERMINAL_STATES = {'completed', 'failed', 'cancelled', 'rejected', 'preflight-failed'}
ACTIVE_STATES = {'reserved', 'queued', 'running', 'unresolved'}


def _git(source, *arguments):
    # Attribute this checkout, not caller-supplied repository/index redirects.
    env = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    return subprocess.check_output(['git', '-C', str(source), *arguments], text=True, env=env).strip()


def _runtime(python):
    executable = Path(python).resolve(strict=True)
    # Metadata only: no torch import, tensors, data, network or weight hashing.
    probe = """import importlib.metadata,json,platform,sys
print(json.dumps({'version':sys.version,'implementation':platform.python_implementation(),
 'executable':sys.executable,'prefix':sys.prefix,'platform':platform.platform(),
 'packages':sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions())}))
"""
    info = json.loads(subprocess.check_output([str(python), '-I', '-c', probe], text=True))
    stat = executable.stat()
    return {'python': str(Path(python).absolute()), 'executable': str(executable),
            'executable_metadata': {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                                    'device': stat.st_dev, 'inode': stat.st_ino},
            'environment': info}


def capture_source_binding(source, commit, python):
    """Bind a clean full-SHA checkout, tracked lock and installed Python runtime."""
    source = Path(source).resolve(strict=True)
    if _git(source, 'rev-parse', '--show-toplevel') != str(source):
        raise ValueError('source must be the checkout root')
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('full source SHA required')
    if _git(source, 'rev-parse', 'HEAD') != commit:
        raise ValueError('source checkout differs from approved full SHA')
    if _git(source, 'status', '--porcelain', '--untracked-files=all'):
        raise ValueError('source checkout has tracked changes or untracked files')
    if not _git(source, 'ls-files', '--error-unmatch', 'uv.lock'):
        raise ValueError('tracked uv.lock required')
    binding = {'schema': 'fixed-source-binding-v1', 'path': str(source), 'commit': commit,
               'lock_sha256': file_digest(source/'uv.lock'), 'runtime': _runtime(python)}
    binding['identity'] = canonical_digest(binding)
    return binding


def check_source_binding(binding):
    """Recheck at submission/launch, before data/model execution."""
    current = capture_source_binding(binding['path'], binding['commit'], binding['runtime']['python'])
    if current != binding:
        raise ValueError('source lock or runtime identity differs from approved binding')
    return current


class SourceConsumers:
    """One registry per host root, shared by every submitting owner on that host.

    Records remain after release. POSIX advisory locking serializes concurrent
    workers and releases automatically on process exit. Never guess outcomes.
    """

    def __init__(self, directory):
        self.directory = Path(directory).resolve()

    @contextmanager
    def _locked(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        import fcntl
        with (self.directory/'.update.lock').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _path(self, source):
        return self.directory/(canonical_digest(str(Path(source).resolve()))+'.json')

    def _read(self, source):
        path = self._path(source)
        return json.loads(path.read_text()) if path.exists() else {'source': str(Path(source).resolve()), 'consumers': {}}

    def reserve(self, binding, consumer, *, owner, reference):
        """Reserve before a potentially accepted scheduler call; no auto release."""
        if not all(isinstance(v, str) and v.strip() for v in (consumer, owner, reference)):
            raise ValueError('consumer, owner and durable reference required')
        with self._locked():
            check_source_binding(binding)
            record = self._read(binding['path'])
            held = [c for c in record['consumers'].values() if not c['released']]
            if any(c['binding'] != binding for c in held):
                raise ValueError('checkout/runtime held by a different binding; use another pinned checkout')
            if consumer in record['consumers']:
                raise ValueError('consumer already reserved; recover its existing attempt')
            record['consumers'][consumer] = {'binding': binding, 'owner': owner, 'reference': reference,
                                             'state': 'reserved', 'released': False, 'events': []}
            self._event(record['consumers'][consumer], 'reserved', reference)
            write_json_atomic(self._path(binding['path']), record)
        return record['consumers'][consumer]

    @staticmethod
    def _event(consumer, state, evidence):
        consumer['state'] = state
        consumer['events'].append({'time': datetime.now(timezone.utc).isoformat(),
                                   'state': state, 'evidence': evidence})

    def record_state(self, source, consumer, state, *, owner, evidence):
        """Terminal state means ALL bound tasks have durable terminal evidence."""
        if state not in ACTIVE_STATES | TERMINAL_STATES or not isinstance(evidence, str) or not evidence.strip():
            raise ValueError('supported state and durable evidence required')
        with self._locked():
            record = self._read(source)
            row = record['consumers'][consumer]
            if row['owner'] != owner or row['released'] or row['state'] in TERMINAL_STATES:
                raise ValueError('consumer owner/state differs; released or terminal consumer is immutable')
            # A fast launch can race the submitter's final queued update.
            # Preserve running/uncertain holds rather than regress their state.
            if state in ACTIVE_STATES:
                rank = {'reserved': 0, 'queued': 1, 'running': 2, 'unresolved': 3}
                state = max((row['state'], state), key=rank.__getitem__)
            self._event(row, state, evidence)
            write_json_atomic(self._path(source), record)

    def release(self, source, consumer, *, owner):
        """Release only an explicitly reconciled terminal consumer, retaining history."""
        with self._locked():
            record = self._read(source)
            row = record['consumers'][consumer]
            if row['owner'] != owner or row['released'] or row['state'] not in TERMINAL_STATES:
                raise ValueError('active/unresolved consumer cannot be released or owner differs')
            row['released'] = True
            write_json_atomic(self._path(source), record)

    def require_released(self, source):
        """Read-only hold diagnostic. Use maintenance() for an atomic mutation gate."""
        with self._locked():
            record = self._read(source)
            held = {key: row['state'] for key, row in record['consumers'].items() if not row['released']}
            if held:
                raise ValueError(f'checkout/runtime still held by consumers: {held}')
        return record


    @contextmanager
    def maintenance(self, source, *, python=None):
        """Hold the registry lock through checkout and optional runtime mutation.

        All submitters and maintenance owners must use this same host registry.
        Passing python also protects shared interpreter/environment consumers
        bound to other checkouts. No checkout/runtime bytes are changed here.
        """
        with self._locked():
            record = self._read(source)
            if any(not row['released'] for row in record['consumers'].values()):
                raise ValueError('checkout still held by consumers')
            if python is not None:
                runtime = _runtime(python)
                for path in self.directory.glob('*.json'):
                    for row in json.loads(path.read_text())['consumers'].values():
                        bound = row['binding']['runtime']
                        shares_binary = bound['executable'] == runtime['executable']
                        shares_environment = (Path(bound['environment']['prefix']).resolve()
                                              == Path(runtime['environment']['prefix']).resolve())
                        if not row['released'] and (shares_binary or shares_environment):
                            raise ValueError('runtime still held by consumers on this or another checkout')
            yield record
