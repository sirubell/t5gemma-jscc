"""The public local COCO acquisition variant owns paths and fresh outputs."""
from pathlib import Path

import pytest
import yaml

from jscc.experiment import execute, read_request


def request(tmp_path, **changes):
    acquisition = {'kind': 'coco-cache', 'dataset_path': 'images.arrow', 'assets': 'model',
        'dataset_name': 'coco', 'dataset_revision': 'dataset-pin', 'model_revision': 'model-pin',
        'karpathy_path': 'membership.json', 'karpathy_name': 'karpathy',
        'karpathy_revision': 'karpathy-pin', 'fixed_data_ids': {'train_ids': [1]}, 'output': 'cache.json'}
    acquisition.update(changes)
    filename = tmp_path/'acquire.yaml'
    filename.write_text(yaml.safe_dump({'schema': 'jscc-experiment-v1', 'workflow': 'shared',
        'operation': 'acquire', 'execution': {'acquisition': acquisition}}))
    return filename


def test_local_coco_acquisition_reaches_provider_with_declaring_paths(tmp_path, monkeypatch):
    filename = request(tmp_path)
    seen = []
    def acquire(**kwargs):
        seen.append(kwargs)
        assert 'kind' not in kwargs
        for key in ('dataset_path', 'assets', 'karpathy_path', 'output'):
            assert Path(kwargs[key]).parent == tmp_path
        Path(kwargs['output']).write_text('bound')
        return {'task': 'coco', 'identity': 'acquired'}
    monkeypatch.setattr('jscc.shared_inputs.bind_coco_data_cache', acquire)
    elsewhere = tmp_path/'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert execute(filename) == {'task': 'coco', 'identity': 'acquired'}
    assert len(seen) == 1
    with pytest.raises(FileExistsError):
        execute(filename)
    assert len(seen) == 1


@pytest.mark.parametrize('changes', [
    {'kind': 'unknown'}, {'synthetic_cpu': True}, {'dataset_revision': ''},
    {'fixed_data_ids': []}, {'model_path': 'other-model'}, {'max_new_tokens': 10},
])
def test_cache_acquisition_rejects_unconsumed_or_conflicting_fields(tmp_path, changes):
    with pytest.raises(ValueError):
        read_request(request(tmp_path, **changes))


def test_cache_acquisition_requires_all_declared_assets(tmp_path):
    filename = request(tmp_path)
    document = yaml.safe_load(filename.read_text())
    del document['execution']['acquisition']['karpathy_revision']
    filename.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError):
        read_request(filename)


def test_cache_acquisition_dangling_output_rejects_before_provider(tmp_path, monkeypatch):
    filename = request(tmp_path)
    destination = tmp_path/'cache.json'
    target = tmp_path/'uncreated.json'
    destination.symlink_to(target)
    monkeypatch.setattr('jscc.shared_inputs.bind_coco_data_cache',
                        lambda **kwargs: pytest.fail('acquisition before output ownership check'))
    with pytest.raises(FileExistsError):
        execute(filename)
    assert destination.is_symlink() and not target.exists()
