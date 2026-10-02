"""Durable append parity, bounded per-event work and fail-stop IO."""
import pytest

from jscc import experiment_records as records
from test_experiment_records import update


def test_append_bytes_schema_order_and_final_inventory_match(tmp_path):
    old, new = tmp_path / 'old.jsonl', tmp_path / 'new.jsonl'
    writer = records.AppendOnlyEventWriter(new)
    events = [update(step=index, final=20) for index in range(1, 21)]
    for event in events:
        records.append_event(old, event)
        writer.append(event)
    assert new.read_bytes() == old.read_bytes()
    assert records.read_events(new) == events
    inventory = {'schema': records.SCHEMA, 'mode': 'metadata-only', 'omitted_tensors': [],
        'artifacts': [records.artifact_ref(new, tmp_path, 'metadata')]}
    records.verify_inventory(tmp_path, inventory)
    manifest = {'schema': records.SCHEMA, 'run_id': 'r',
        'phases': [{'phase_id': 'local', 'updates': 20, 'start_step': 0, 'start_valid_tokens': 0}],
        'expected_observations': [], 'expected_artifacts': ['new.jsonl']}
    assert records.completion_status(manifest, events, inventory, tmp_path)['status'] == 'complete'


@pytest.mark.parametrize('history', [0, 6400, 19200])
def test_long_history_appends_read_and_write_no_prior_records(tmp_path, monkeypatch, history):
    path = tmp_path / 'metrics.jsonl'
    if history:
        path.write_bytes(b''.join(records.canonical_bytes(update(index, final=20000)) + b'\n'
            for index in range(1, history + 1)))
    writer = records.AppendOnlyEventWriter(path)
    def forbidden(*args, **kwargs):
        raise AssertionError('append revisited whole history')
    monkeypatch.setattr(records, 'read_events', forbidden)
    monkeypatch.setattr(records, 'artifact_ref', forbidden)
    monkeypatch.setattr(records, '_atomic_bytes', forbidden)
    real_write = records.os.write
    writes = []
    def counted(fd, data):
        writes.append(len(data))
        return real_write(fd, data)
    monkeypatch.setattr(records.os, 'write', counted)
    events = [update(index, final=20000) for index in range(history + 1, history + 4)]
    for event in events:
        writer.append(event)
    assert writes == [len(records.canonical_bytes(event)) + 1 for event in events]


@pytest.mark.parametrize('data', [b'{"broken":true}\n', b'partial'])
def test_invalid_existing_log_rejected_at_acquisition(tmp_path, data):
    path = tmp_path / 'metrics.jsonl'
    path.write_bytes(data)
    with pytest.raises(records.RecordError):
        records.AppendOnlyEventWriter(path)
    assert path.read_bytes() == data


def test_partial_append_preserves_prefix_and_terminal_tail(tmp_path, monkeypatch):
    path = tmp_path / 'metrics.jsonl'
    writer = records.AppendOnlyEventWriter(path)
    writer.append(update())
    before = path.read_bytes()
    real_write = records.os.write
    monkeypatch.setattr(records.os, 'write', lambda fd, data: real_write(fd, data[:7]))
    with pytest.raises(OSError, match='partial event append'):
        writer.append(update())
    assert path.read_bytes().startswith(before)
    assert len(path.read_bytes()) == len(before) + 7
    with pytest.raises(records.RecordError, match='truncated'):
        records.read_events(path)
    with pytest.raises(RuntimeError, match='no retry'):
        writer.append(update())


def test_fsync_error_is_terminal_even_if_complete_line_exists(tmp_path, monkeypatch):
    path = tmp_path / 'metrics.jsonl'
    writer = records.AppendOnlyEventWriter(path)
    def fail(fd):
        raise OSError('fsync failed')
    monkeypatch.setattr(records.os, 'fsync', fail)
    with pytest.raises(OSError, match='fsync failed'):
        writer.append(update())
    assert writer.failed
    assert len(records.read_events(path)) == 1  # Durability unacknowledged; caller failed.
    with pytest.raises(RuntimeError, match='no retry'):
        writer.append(update())


def test_external_change_stops_single_writer(tmp_path):
    path = tmp_path / 'metrics.jsonl'
    writer = records.AppendOnlyEventWriter(path)
    writer.append(update())
    with path.open('ab') as handle:
        handle.write(b'tampered\n')
    before = path.read_bytes()
    with pytest.raises(records.RecordError, match='outside its single writer'):
        writer.append(update())
    assert path.read_bytes() == before
    assert writer.failed


def test_directory_failure_is_terminal(tmp_path, monkeypatch):
    writer = records.AppendOnlyEventWriter(tmp_path / 'new/metrics.jsonl')
    def fail(*args, **kwargs):
        raise OSError('mkdir failed')
    monkeypatch.setattr(type(tmp_path), 'mkdir', fail)
    with pytest.raises(OSError, match='mkdir failed'):
        writer.append(update())
    assert writer.failed
