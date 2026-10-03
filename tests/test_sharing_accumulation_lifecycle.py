"""Actual q4 lifecycle with explicit64 exposure; only task scorer is a fixture."""
import json

import pytest
import torch

from test_shared_lifecycle import geometry_batches, install_adapter, task_template
from test_sharing_accumulation import parent64, plan64
from sharing_fixtures import sharing_batches, sharing_model
from jscc.sharing_run import run_sharing
from jscc.sharing_state import verify_study_freeze


@pytest.mark.parametrize('micro', [32, 16])
def test_effective64_actual_q4_lifecycle(tmp_path, monkeypatch, micro):
    torch.set_num_threads(1)
    calls = install_adapter(monkeypatch)
    model = sharing_model('D-LN')
    template = task_template()
    template['backend'] = model.base.config.decoder._attn_implementation
    root = tmp_path / f'actual-{micro}'
    kwargs = dict(plan=plan64(),
        update_batches=[parent64(i) for i in range(4)], validation_batches=sharing_batches()[:1],
        metadata={'source_identity': 'source', 'config_identity': 'config', 'data_identity': 'data',
            'model_revision': 'tiny-cpu', 'initialization_identity': 'same-init',
            'stream_identity': 'same-stream', 'protocol_identity': 'tiny-sharing-accum64',
            'recipe_identity': 'K+.1R', 'architecture_decision': 'D-LN-test-fixture'},
        task_template=template, output=root, geometry_batches=geometry_batches(),
        resource_guard=lambda: None, microbatch_size=micro)
    if micro == 32:
        result = execute_prepared_tiny(tmp_path, monkeypatch, model, template, root, micro)
    else:
        result = run_sharing(model, object(), **kwargs)
    assert result['status'] == 'complete'
    assert result['completed_updates'] == 24
    assert result['completed_checkpoints'] == 17
    assert len(list(root.rglob('*.pt'))) == 17
    assert len(calls) == 109
    assert verify_study_freeze(result['freeze_reference'])['synthetic']
    observations = json.loads((root / 'observations.json').read_text())
    assert sum(len(o['conditions']) for o in observations if o['request']['purpose'] == 'task') == 108
    assert sum(len(o['conditions']) for o in observations if o['request']['purpose'] == 'objective') == 192
    assert json.loads((root / 'geometry.json').read_text())
    total = 0
    for name in plan64().learners:
        events = [json.loads(line) for line in (root / name / 'metrics.jsonl').read_text().splitlines()]
        updates = [event['payload'] for event in events if event['event_type'] == 'update']
        total += len(updates)
        assert all(row['exposure']['sequences'] == 64 for row in updates)
        assert all(row['exposure']['padded_tokens'] == 64 * 5 for row in updates)
        states = sorted((root / name).glob('step_*.pt'))
        final = torch.load(states[-1], map_location='cpu', weights_only=False)
        assert final['metadata']['sharing']['batch_size'] == 64
        assert final['stream']['offset'] == len(updates) * 64
    assert total == 24
    assert not torch.cuda.is_initialized()


def execute_prepared_tiny(tmp_path, monkeypatch, model, template, output, micro, *,
                          cell="D-LN", model_factory=None):
    """Construct a real immutable prepared package and enter the actual CLI."""
    import importlib.util
    import zipfile
    from pathlib import Path
    import yaml
    from jscc.activation_replay import canonical_digest, file_digest
    from jscc.baseline_protocol import batch_identity
    from jscc.config import load_config
    from jscc.sharing_accumulation import partition_policy
    from jscc.sharing_preparation import source_inventory, state_dict_identity, write_prepared
    from jscc import sharing_run

    source = Path(__file__).resolve().parents[1]
    root = tmp_path / 'prepared'
    root.mkdir()
    config = load_config(source / 'configs/tasks/hellaswag.yaml')
    config['model'].update(device='cpu', dtype='float32', revision='tiny-cpu', numerical_policy='native')
    config['codec'] = dict(model.codec.config)
    config['split'] = dict(model.split)
    config['training'].update(batch_size=micro, gradient_accumulation=64 // micro, lr=2e-4,
                              weight_decay=.01, grad_clip=1, temperature=1, kl_weight=1, mse_weight=.1)
    config['seed'] = 0
    (root / 'config.yaml').write_text(yaml.safe_dump(config))
    config = load_config(root / 'config.yaml')
    def ref(name):
        return {'path': name, 'sha256': file_digest(root / name)}
    def batch_ref(name, batch):
        torch.save(batch, root / name)
        return {**ref(name), 'view_sha256': batch_identity(batch)}
    updates = [batch_ref(f'update{i}.pt', parent64(i)) for i in range(4)]
    validation = [batch_ref('validation.pt', sharing_batches()[0])]
    geometry = {partition: [batch_ref(f'geometry-{partition}.pt', batches[0])]
                for partition, batches in geometry_batches().items()}
    (root / 'task.json').write_text(json.dumps(template))
    torch.save(model.codec.state_dict(), root / 'initial.pt')
    inventory = source_inventory(source)
    with zipfile.ZipFile(root / 'source.zip', 'w') as bundle:
        for name in inventory:
            bundle.write(source / name, name)
    prepared = write_prepared(root / 'prepared.json', dict(
        schema='sharing-prepared-v1', cell=cell, synthetic_cpu=True, q=4, batch_size=64,
        execution_partition=partition_policy(micro), selected_bottleneck_dim=config['codec']['bottleneck_dim'],
        protocol_id='sharing-synthetic-test-v1', model_revision='tiny-cpu', study_pairing_id='cpu-pairing',
        recipe_identity='fixed-K+.1R', architecture_decision=cell+'-test-fixture', hard_cap_seconds=240,
        config=ref('config.yaml'), config_identity=canonical_digest(config), updates=updates,
        validation=validation, task_request=ref('task.json'), geometry=geometry,
        source_inventory=inventory, source_identity=canonical_digest(inventory), source_archive=ref('source.zip'),
        initialization={**ref('initial.pt'), 'state_identity': state_dict_identity(model.codec.state_dict())},
        data_ids={'synthetic': True}))
    # Supply the actual tiny backbone in place of pretrained weight loading only.
    monkeypatch.setattr(sharing_run, 'build_model', lambda config: (object(), model_factory() if model_factory else sharing_model(cell)))
    spec = importlib.util.spec_from_file_location('sharing_accumulation_cli_test', source / 'scripts/shared_codec.py')
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    result = cli.main(['--execute', str(prepared), '--output', str(output)])
    assert (output / 'inputs/source.zip').is_file()
    assert (output / 'inputs/prepared.json').is_file()
    with pytest.raises(ValueError, match='fresh'):
        cli.main(['--execute', str(prepared), '--output', str(output)])
    return result
