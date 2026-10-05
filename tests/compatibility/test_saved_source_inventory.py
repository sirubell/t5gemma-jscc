"""Literal historical inventory semantics against explicit retained source bytes."""

import hashlib
from collections.abc import Callable
from typing import cast
import pytest
from jscc.asset_identity import saved_source_inventory


MANDATORY = {'jscc/retained.py', 'scripts/shared_codec.py', 'scripts/sharing_production.py',
             'scripts/sharing_qualification.py', 'scripts/render_experiment_report.py',
             'uv.lock', 'pyproject.toml'}


def fixture_source(root, names):
    expected = {}
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        value = ('retained fixture bytes: ' + name).encode()
        path.write_bytes(value)
        expected[name] = hashlib.sha256(value).hexdigest()
    return expected


@pytest.mark.parametrize('optional', [set(), {'scripts/prepare_sharing_native16.py'},
    {'scripts/sharing_long_study.py'}, {'scripts/prepare_sharing_native16.py', 'scripts/sharing_long_study.py'}])
def test_saved_inventory_keeps_exact_names_and_bytes(tmp_path, optional):
    expected = fixture_source(tmp_path, MANDATORY | optional)
    fixture_source(tmp_path, {'experiment.py', 'scripts/current_only.py'})
    assert saved_source_inventory(tmp_path) == expected
    (tmp_path / 'jscc/retained.py').write_bytes(b'changed saved source')
    changed = saved_source_inventory(tmp_path)
    assert changed != expected
    assert set(changed) == set(expected)
    assert changed['jscc/retained.py'] == hashlib.sha256(b'changed saved source').hexdigest()


def test_saved_inventory_requires_root_and_missing_mandatory_rejects(tmp_path):
    with pytest.raises(TypeError):
        cast(Callable[..., object], saved_source_inventory)()
    fixture_source(tmp_path, MANDATORY - {'scripts/sharing_qualification.py'})
    with pytest.raises(FileNotFoundError):
        saved_source_inventory(tmp_path)
