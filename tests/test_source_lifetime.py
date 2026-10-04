"""Real local Git fixtures: fixed source/runtime ownership without a scheduler."""
from copy import deepcopy
import json
from concurrent.futures import ThreadPoolExecutor
import subprocess
import selectors
import sys

import pytest

from jscc.source_lifetime import SourceConsumers, capture_source_binding, check_source_binding


def git(source, *args):
    return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()


def ready(child, expected):
    assert child.stdout is not None
    with selectors.DefaultSelector() as selector:
        selector.register(child.stdout, selectors.EVENT_READ)
        assert selector.select(timeout=30), 'child readiness timed out'
        assert child.stdout.readline().strip() == expected


@pytest.fixture
def source(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    git(source, 'init', '-q')
    (source/'uv.lock').write_text('fixture lock')
    (source/'worker.py').write_text('print("fixture")')
    git(source, 'add', '.')
    git(source, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture')
    return source


def bind(source):
    return capture_source_binding(source, git(source, 'rev-parse', 'HEAD'), sys.executable)


def test_shared_matching_source_held_until_every_consumer_released(source, tmp_path):
    binding = bind(source)
    registry = SourceConsumers(tmp_path/'ownership')
    for consumer in ('case-a', 'case-b'):
        registry.reserve(binding, consumer, owner=consumer, reference=consumer+'/SUBMISSION.json')
    registry.record_state(source, 'case-a', 'completed', owner='case-a', evidence='all a tasks terminal')
    registry.release(source, 'case-a', owner='case-a')
    with pytest.raises(ValueError, match='held'):
        registry.require_released(source)
    registry.record_state(source, 'case-b', 'cancelled', owner='case-b', evidence='all b tasks cancelled')
    registry.release(source, 'case-b', owner='case-b')
    record = registry.require_released(source)
    assert record['consumers']['case-a']['binding'] == binding
    assert record['consumers']['case-b']['events'][-1]['evidence'] == 'all b tasks cancelled'


@pytest.mark.parametrize('state', ['reserved', 'queued', 'running', 'unresolved'])
def test_uncertain_active_consumer_never_released(source, tmp_path, state):
    registry = SourceConsumers(tmp_path/'ownership')
    registry.reserve(bind(source), 'case', owner='session', reference='submission.json')
    registry.record_state(source, 'case', state, owner='session', evidence='current attempt evidence')
    # A fresh object must recover the persisted hold, not just in-memory state.
    recovered = SourceConsumers(tmp_path/'ownership')
    with pytest.raises(ValueError, match='cannot be released'):
        recovered.release(source, 'case', owner='session')
    with pytest.raises(ValueError, match='held'):
        recovered.require_released(source)


def test_different_revisions_require_different_checkouts(source, tmp_path):
    original = bind(source)
    registry = SourceConsumers(tmp_path/'ownership')
    registry.reserve(original, 'first', owner='first', reference='first.json')
    other = tmp_path/'other'
    git(source, 'worktree', 'add', '--detach', str(other), original['commit'])
    (other/'worker.py').write_text('print("second")')
    git(other, 'add', '.')
    git(other, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'second')
    second = bind(other)
    registry.reserve(second, 'second', owner='second', reference='second.json')
    assert second['commit'] != original['commit']
    assert check_source_binding(original) == original
    # Simulate forbidden switching of the busy checkout; reservation detects
    # the old lifetime hold even when the new source itself is clean.
    git(source, 'checkout', '--detach', second['commit'])
    changed = bind(source)
    with pytest.raises(ValueError, match='different binding'):
        registry.reserve(changed, 'third', owner='third', reference='third.json')


def test_runtime_and_lock_change_rejected(source):
    binding = bind(source)
    changed = deepcopy(binding)
    changed['runtime']['environment']['version'] = 'another version'
    with pytest.raises(ValueError, match='identity differs'):
        check_source_binding(changed)
    (source/'uv.lock').write_text('changed lock')
    with pytest.raises(ValueError, match='tracked changes'):
        check_source_binding(binding)


def test_claim_ownership_duplicate_and_serialization(source, tmp_path):
    registry = SourceConsumers(tmp_path/'ownership')
    binding = bind(source)
    registry.reserve(binding, 'case', owner='session', reference='submission.json')
    with pytest.raises(ValueError, match='already reserved'):
        registry.reserve(binding, 'case', owner='other', reference='other.json')
    with pytest.raises(ValueError, match='owner/state'):
        registry.record_state(source, 'case', 'completed', owner='other', evidence='terminal')
    with pytest.raises(ValueError, match='evidence required'):
        registry.record_state(source, 'case', 'completed', owner='session', evidence='')
    registry.record_state(source, 'case', 'failed', owner='session', evidence='all tasks failed')
    with pytest.raises(ValueError, match='immutable'):
        registry.record_state(source, 'case', 'queued', owner='session', evidence='retry')
    registry.release(source, 'case', owner='session')
    files = list(registry.directory.glob('*.json'))
    assert len(files) == 1 and json.loads(files[0].read_text())['consumers']['case']['released']


def test_git_subdirectory_and_short_sha_rejected(source):
    (source/'nested').mkdir()
    with pytest.raises(ValueError, match='checkout root'):
        capture_source_binding(source/'nested', git(source, 'rev-parse', 'HEAD'), sys.executable)
    with pytest.raises(ValueError, match='full source SHA'):
        capture_source_binding(source, 'short', sys.executable)


def test_concurrent_reservations_preserve_both_consumers(source, tmp_path):
    binding = bind(source)
    directory = tmp_path/'ownership'

    def reserve(consumer):
        return SourceConsumers(directory).reserve(binding, consumer, owner=consumer, reference=consumer+'.json')

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert len(list(pool.map(reserve, ['a', 'b']))) == 2
    record = json.loads(next(directory.glob('*.json')).read_text())
    assert set(record['consumers']) == {'a', 'b'}


def test_submit_launch_race_cannot_regress_uncertainty(source, tmp_path):
    registry = SourceConsumers(tmp_path/'ownership')
    registry.reserve(bind(source), 'case', owner='owner', reference='attempt')
    for state in ('running', 'queued', 'unresolved', 'running'):
        registry.record_state(source, 'case', state, owner='owner', evidence=state+' receipt')
    record = json.loads(next(registry.directory.glob('*.json')).read_text())
    assert [event['state'] for event in record['consumers']['case']['events']] == [
        'reserved', 'running', 'running', 'unresolved', 'unresolved']


def test_shared_runtime_maintenance_checks_other_checkouts(source, tmp_path):
    registry = SourceConsumers(tmp_path/'ownership')
    other = tmp_path/'other'
    git(source, 'worktree', 'add', '--detach', str(other), git(source, 'rev-parse', 'HEAD'))
    registry.reserve(bind(other), 'other', owner='owner', reference='other attempt')
    # An idle source may change independently, but not its shared runtime.
    with registry.maintenance(source):
        assert git(source, 'rev-parse', 'HEAD')
    with pytest.raises(ValueError, match='runtime still held'):
        with registry.maintenance(source, python=sys.executable):
            pytest.fail('shared runtime mutation allowed')
    registry.record_state(other, 'other', 'completed', owner='owner', evidence='all other tasks terminal')
    registry.release(other, 'other', owner='owner')
    with registry.maintenance(source, python=sys.executable):
        assert git(source, 'rev-parse', 'HEAD')


def test_maintenance_excludes_process_reservation_through_mutation(source, tmp_path):
    registry = SourceConsumers(tmp_path/'ownership')
    binding = bind(source)
    request = tmp_path/'binding.json'
    request.write_text(json.dumps(binding))
    script = '''import json,sys
from jscc.source_lifetime import SourceConsumers
print('begin',flush=True)
try:
 SourceConsumers(sys.argv[1]).reserve(json.load(open(sys.argv[2])), 'child', owner='child', reference='child receipt')
except ValueError as error:
 print(str(error),flush=True)
 sys.exit(3)
'''
    with registry.maintenance(source):
        child = subprocess.Popen([sys.executable, '-c', script, str(registry.directory), str(request)],
                                 text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            ready(child, 'begin')
        except BaseException:
            child.kill()
            child.communicate(timeout=30)
            raise
        assert child.poll() is None
        # The child cannot reserve the previous binding during this mutation.
        (source/'worker.py').write_text('new source')
        git(source, 'add', '.')
        git(source, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'changed')
    stdout, stderr = child.communicate(timeout=30)
    assert child.returncode == 3, stderr
    assert 'source checkout differs' in stdout
    assert registry.require_released(source)['consumers'] == {}


def test_process_exit_releases_maintenance_lock(source, tmp_path):
    registry = SourceConsumers(tmp_path/'ownership')
    script = '''import sys,time
from jscc.source_lifetime import SourceConsumers
with SourceConsumers(sys.argv[1]).maintenance(sys.argv[2]):
 print('locked',flush=True)
 time.sleep(60)
'''
    child = subprocess.Popen([sys.executable, '-c', script, str(registry.directory), str(source)],
                             text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        ready(child, 'locked')
    finally:
        child.kill()
        child.communicate(timeout=30)
    registry.reserve(bind(source), 'after-crash', owner='owner', reference='after crash attempt')
    with pytest.raises(ValueError, match='checkout still held'):
        with registry.maintenance(source):
            pytest.fail('active source mutation allowed')


def test_binding_ignores_inherited_git_repository_redirects(source, tmp_path, monkeypatch):
    expected = bind(source)
    other = tmp_path/'other'
    other.mkdir()
    git(other, 'init', '-q')
    monkeypatch.setenv('GIT_DIR', str(other/'.git'))
    monkeypatch.setenv('GIT_WORK_TREE', str(other))
    monkeypatch.setenv('GIT_INDEX_FILE', str(tmp_path/'foreign-index'))
    assert capture_source_binding(source, expected['commit'], sys.executable) == expected
