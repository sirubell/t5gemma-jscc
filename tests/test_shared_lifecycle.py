"""The actual finite lifecycle; only the external task scorer is a fixture."""
import json
from pathlib import Path

import pytest
import torch

from sharing_fixtures import sharing_batches, sharing_model, sharing_plan
from jscc import evaluation
from jscc.sharing_run import run_sharing
from jscc.sharing_state import verify_study_freeze


def task_template():
    return {'task': 'hellaswag', 'input_ids': [80, 81], 'source_family_ids': ['dev80', 'dev81'],
            'prompt_policy': 'synthetic-native', 'noise': {'seed': 0, 'namespace': 'same-eval'},
            'layout': 'tiny-2x3', 'backend': 'eager', 'precision': 'fp32', 'scorer': 'fixture',
            'reference_corpus': 'fixture-dev', 'expected_items': 2,
            'settings': {'num_samples': 2, 'numerical_policy': 'native'}, 'data_settings': {},
            'conditions': ['no_noise', -6, 0, 6, 12, 18]}


def install_adapter(monkeypatch):
    calls = []
    def adapter(model, processor, settings, output, condition, data_settings):
        calls.append((dict(model.split), condition))
        rows = [{'sample_id': i, 'source_id': f'dev{i}', 'tokens': [2, 1],
                 'raw_correct': 1, 'normalized_correct': 1, 'normalized_prediction': 0,
                 'normalized_scores': [-.1, -.2]} for i in [80, 81]]
        (Path(output) / f'compact_{condition}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return {'acc': 1., 'acc_norm': 1.}
    monkeypatch.setattr(evaluation, 'evaluate_hellaswag', adapter)
    return calls


def geometry_batches():
    result = {}
    for partition in ['train', 'selection']:
        batch: dict = sharing_batches()[0]
        batch.update(query_ids=[partition+'0', partition+'1'], view_ids=['v0', 'v1'],
                     token_positions=torch.tensor([[0, 1, 2], [0, 1, 2]]),
                     token_roles=[['demo', 'query', 'query'], ['demo', 'query', 'query']])
        result[partition] = [batch]
    return result


@pytest.mark.parametrize('cell', ['D-LN', 'R-LN'])
def test_complete_actual_lifecycle(tmp_path, monkeypatch, cell):
    calls = install_adapter(monkeypatch)
    model = sharing_model(cell)
    template = task_template()
    template['backend'] = model.base.config.decoder._attn_implementation
    result = run_sharing(model, object(), plan=sharing_plan(), update_batches=sharing_batches(),
        validation_batches=sharing_batches()[:1], metadata={
            'source_identity': 'source', 'config_identity': 'config', 'data_identity': 'data',
            'model_revision': 'tiny-cpu', 'initialization_identity': 'same-init',
            'stream_identity': 'same-stream', 'protocol_identity': 'tiny-sharing',
            'recipe_identity': 'K+.1R', 'architecture_decision': cell},
        task_template=template, output=tmp_path/'run', geometry_batches=geometry_batches(), resource_guard=lambda: None)
    assert result['status'] == 'complete'
    assert result['completed_updates'] == 24
    assert result['completed_checkpoints'] == 17
    assert len(list((tmp_path/'run').rglob('*.pt'))) == 17
    assert len(calls) == 109
    assert verify_study_freeze(result['freeze_reference'])['synthetic']
    observations = json.loads((tmp_path/'run'/'observations.json').read_text())
    assert sum(len(o['conditions']) for o in observations if o['request']['purpose'] == 'task') == 108
    assert sum(len(o['conditions']) for o in observations if o['request']['purpose'] == 'objective') == 192
    assert json.loads((tmp_path/'run'/'geometry.json').read_text())


def test_real_prepared_entry_and_partial_failure(tmp_path, monkeypatch):
    import importlib.util
    import zipfile
    import yaml
    from jscc.activation_replay import canonical_digest, file_digest
    from jscc.baseline_protocol import batch_identity
    from jscc.config import load_config
    from jscc.sharing_preparation import source_inventory, state_dict_identity, write_prepared
    from jscc import sharing_run

    source = Path(__file__).resolve().parents[1]
    root = tmp_path/'prepared'
    root.mkdir()
    model = sharing_model('D-LN')
    config = load_config(source/'configs/tasks/hellaswag.yaml')
    config['model'].update(device='cpu', dtype='float32', revision='tiny-cpu', numerical_policy='native')
    config['codec'] = dict(model.codec.config)
    config['split'] = dict(model.split)
    config['training'].update(batch_size=2, gradient_accumulation=1, lr=2e-4, weight_decay=.01,
                              grad_clip=1, temperature=1, kl_weight=1, mse_weight=.1)
    config['seed'] = 0
    (root/'config.yaml').write_text(yaml.safe_dump(config))
    config = load_config(root/'config.yaml')
    def ref(name):
        return {'path': name, 'sha256': file_digest(root/name)}
    def batch_ref(name, batch):
        torch.save(batch, root/name)
        return {**ref(name), 'view_sha256': batch_identity(batch)}
    updates = [batch_ref(f'update{i}.pt', batch) for i, batch in enumerate(sharing_batches())]
    geometry = {partition: [batch_ref(f'geometry-{partition}.pt', batch)]
                for partition, batches in geometry_batches().items() for batch in batches}
    template = task_template()
    template['backend'] = model.base.config.decoder._attn_implementation
    (root/'task.json').write_text(json.dumps(template))
    torch.save(model.codec.state_dict(), root/'initial.pt')
    inventory = source_inventory(source)
    with zipfile.ZipFile(root/'source.zip', 'w') as bundle:
        for name in inventory:
            bundle.write(source/name, name)
    prepared = write_prepared(root/'prepared.json', dict(
        schema='sharing-prepared-v1', cell='D-LN', synthetic_cpu=True, q=4, batch_size=2,
        protocol_id='sharing-synthetic-test-v1', model_revision='tiny-cpu', study_pairing_id='cpu-pairing',
        recipe_identity='fixed-K+.1R', architecture_decision='D-LN-fixture', hard_cap_seconds=120,
        config=ref('config.yaml'), config_identity=canonical_digest(config), updates=updates,
        validation=[updates[0]], task_request=ref('task.json'), geometry=geometry,
        source_inventory=inventory, source_identity=canonical_digest(inventory), source_archive=ref('source.zip'),
        initialization={**ref('initial.pt'), 'state_identity': state_dict_identity(model.codec.state_dict())},
        data_ids={'synthetic': True}))
    calls = install_adapter(monkeypatch)
    def tiny_builder(cfg):
        assert torch.initial_seed() == 0
        assert torch.are_deterministic_algorithms_enabled() == bool(cfg['training'].get('deterministic_algorithms', False))
        return object(), sharing_model('D-LN')
    monkeypatch.setattr(sharing_run, 'build_model', tiny_builder)
    spec = importlib.util.spec_from_file_location('sharing_actual_entry', source/'scripts/shared_codec.py')
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    result = cli.main(['--execute', str(prepared), '--output', str(tmp_path/'actual-entry')])
    assert result['status'] == 'complete'
    assert result['completed_updates'] == 24
    assert len(calls) == 109
    assert (tmp_path/'actual-entry'/'inputs'/'source.zip').is_file()
    assert (tmp_path/'actual-entry'/'inventory.json').is_file()
    assert len(list((tmp_path/'actual-entry').glob('sharing-*.png'))) == 4
    # A second invocation cannot overwrite/repeat a completed campaign.
    with pytest.raises(ValueError, match='fresh'):
        cli.main(['--execute', str(prepared), '--output', str(tmp_path/'actual-entry')])


def test_failure_is_terminal_and_preserves_partial(tmp_path, monkeypatch):
    install_adapter(monkeypatch)
    count = 0
    def guard():
        nonlocal count
        count += 1
        if count == 15:
            raise TimeoutError('fixture hard deadline')
    model = sharing_model()
    template = task_template()
    template['backend'] = model.base.config.decoder._attn_implementation
    with pytest.raises(TimeoutError, match='deadline'):
        run_sharing(model, object(), plan=sharing_plan(), update_batches=sharing_batches(),
            validation_batches=sharing_batches()[:1], metadata={
                'source_identity': 'source', 'config_identity': 'config', 'data_identity': 'data',
                'model_revision': 'tiny-cpu', 'initialization_identity': 'same-init',
                'stream_identity': 'same-stream', 'protocol_identity': 'tiny-sharing',
                'recipe_identity': 'K+.1R', 'architecture_decision': 'D-LN'},
            task_template=template, output=tmp_path/'partial', geometry_batches=geometry_batches(), resource_guard=guard)
    result = json.loads((tmp_path/'partial'/'campaign.json').read_text())
    assert result['status'] == 'incomplete'
    assert 0 < result['completed_updates'] < 24
    assert list((tmp_path/'partial').rglob('*.pt'))
    assert not (tmp_path/'partial'/'study-freeze.json').exists()
