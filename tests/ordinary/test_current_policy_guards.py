"""Current ordinary output, truthful completion and deterministic setup guards."""
import json
import os
from pathlib import Path

import pytest
import torch

from jscc import ordinary_runs, training
from jscc.config import load_config

import jscc

SOURCE = Path(jscc.__file__).resolve().parents[1]


def config():
    return load_config(SOURCE / 'configs/experiments/ordinary/hellaswag.yaml')


def tree_bytes(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob('*') if path.is_file()}


def test_existing_output_recursive_bytes_preserved_before_engine(tmp_path, monkeypatch):
    root = tmp_path / 'occupied'
    (root / 'nested').mkdir(parents=True)
    (root / 'nested/evidence.bin').write_bytes(bytes(range(256)))
    (root / 'completion.json').write_bytes(b'preserved evidence\n')
    before = tree_bytes(root)
    monkeypatch.setattr(training, 'train', lambda *a, **k: pytest.fail('occupied root reached science'))
    with pytest.raises(FileExistsError):
        ordinary_runs.train_task(config(), root)
    assert tree_bytes(root) == before


def test_output_created_during_preparation_is_untouched(tmp_path, monkeypatch):
    root = tmp_path / 'race'
    original = ordinary_runs.reserve_root
    before = {}
    def other_actor(output, **kwargs):
        root.mkdir()
        (root / 'nested').mkdir()
        (root / 'nested/evidence.bin').write_bytes(b'other actor exact bytes\x00')
        before.update(tree_bytes(root))
        return original(output, **kwargs)
    monkeypatch.setattr(ordinary_runs, 'reserve_root', other_actor)
    monkeypatch.setattr(training, 'train', lambda *a, **k: pytest.fail('lost reservation reached science'))
    with pytest.raises(FileExistsError):
        ordinary_runs.train_task(config(), root)
    assert tree_bytes(root) == before


def test_empty_successful_engine_return_cannot_complete(tmp_path, monkeypatch):
    root = tmp_path / 'empty'
    monkeypatch.setattr(training, 'train', lambda _, **kw: kw['run_directory'])
    with pytest.raises(FileNotFoundError, match='completion.json'):
        ordinary_runs.train_task(config(), root)
    attempt = root / 'attempts/training-0001'
    assert json.loads((attempt / 'attempt.json').read_text())['status'] == 'failed'
    assert not (root / 'checkpoint.json').exists()
    assert not (attempt / 'completion.json').exists()
    with pytest.raises(FileExistsError):
        ordinary_runs.train_task(config(), root)


@pytest.mark.parametrize('workspace', [':4096:8', ':16:8'])
def test_determinism_and_cublas_are_applied_before_seed(tmp_path, monkeypatch, workspace):
    settings = config()
    settings['training']['deterministic_algorithms'] = True
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', workspace)
    previous = torch.are_deterministic_algorithms_enabled()
    seen = []
    class BeforeModel(Exception):
        pass
    def seed(value):
        assert torch.are_deterministic_algorithms_enabled()
        assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == workspace
        assert value == settings['seed']
        seen.append(value)
        raise BeforeModel
    monkeypatch.setattr(training, 'seed_everything', seed)
    monkeypatch.setattr(training, 'build_model', lambda _: pytest.fail('seed sentinel reached model'))
    try:
        torch.use_deterministic_algorithms(False)
        with pytest.raises(BeforeModel):
            training.train(settings)
        assert seen == [settings['seed']]
    finally:
        torch.use_deterministic_algorithms(previous)
    assert not list(tmp_path.iterdir())
