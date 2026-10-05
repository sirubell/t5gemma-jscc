"""Current entry source identity retains required-provider and stale-source guards."""

import jscc
from pathlib import Path

import pytest

from jscc import presentation


@pytest.fixture
def source_root(tmp_path, monkeypatch):
    package = tmp_path/'jscc'
    package.mkdir()
    for name in ('presentation.py', 'provider.py'):
        (package/name).write_text('# bound provider\n')
    for name in ('experiment.py', 'uv.lock', 'pyproject.toml'):
        (tmp_path/name).write_text('bound '+name+'\n')
    monkeypatch.setattr(presentation, '__file__', str(package/'presentation.py'))
    return tmp_path


@pytest.mark.parametrize('provider', ['experiment.py', 'jscc/provider.py', 'uv.lock', 'pyproject.toml'])
def test_current_required_providers_change_digest(source_root, provider):
    before = presentation.training_source_digest()
    path = source_root/provider
    path.write_text(path.read_text()+'changed\n')
    assert presentation.training_source_digest() != before


def test_retired_front_is_not_a_current_digest_provider(source_root):
    before = presentation.training_source_digest()
    (source_root/'train.py').write_text('# historical front bytes\n')
    assert presentation.training_source_digest() == before


def test_missing_current_entry_fails(source_root):
    (source_root/'experiment.py').unlink()
    with pytest.raises(FileNotFoundError):
        presentation.training_source_digest()


def test_current_tree_and_pyright_cover_only_current_entry():
    import tomllib
    root = Path(jscc.__file__).resolve().parents[1]
    assert (root/'experiment.py').is_file()
    assert all(not (root/name).exists() for name in ('train.py', 'evaluate.py', 'study.py'))
    config = tomllib.loads((root/'pyproject.toml').read_text())
    assert config['tool']['pyright']['include'] == ['jscc', 'examples', 'tests', 'experiment.py']
