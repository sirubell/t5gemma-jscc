"""Explicit no-norm sharing uses the retained single biased Linear architecture."""
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from jscc.config import load_config
from jscc.models.codec import Codec
from jscc.sharing_accumulation import partition_policy
from jscc.sharing_preparation import _config_contract
from jscc.sharing_state import verify_study_freeze
from test_shared_lifecycle import install_adapter, task_template
from test_sharing_accumulation_lifecycle import execute_prepared_tiny
from test_sharing_candidate_binding import candidate_config, t_model


@pytest.mark.parametrize('width', [512, 1152, 2304])
def test_no_norm_matches_literal_affine_and_gradients(width):
    torch.set_num_threads(1)
    torch.manual_seed(19)
    codec = Codec(1152, candidate_config('D-N', width))
    reference = nn.Sequential(nn.Linear(1152, width), nn.Linear(width, 1152))
    reference[0].load_state_dict(codec.encoder[0].state_dict())
    reference[1].load_state_dict(codec.decoder[0].state_dict())
    x = torch.randn(2, 3, 1152, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    actual = codec.decode(codec.encode(x), None)
    expected = reference(y)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
    for a, b in zip(codec.parameters(), reference.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    assert isinstance(codec.input_norm, nn.Identity)
    assert isinstance(codec.output_norm, nn.Identity)
    assert sum(isinstance(m, nn.Linear) for m in codec.modules()) == 2
    assert not any(isinstance(m, (nn.LayerNorm, nn.GELU)) for m in codec.modules())
    assert sum(p.numel() for p in codec.parameters()) == 2 * 1152 * width + width + 1152


@pytest.mark.parametrize('width', [512, 1152, 2304])
def test_named_no_norm_config_is_effective64(width):
    config = load_config(Path(__file__).resolve().parents[1] /
                         f'configs/tasks/sharing_direct_no_norm_b{width}.yaml')
    manifest = dict(synthetic_cpu=False, cell='D-N', selected_bottleneck_dim=width,
                    execution_partition=partition_policy(16), batch_size=64,
                    model_revision=config['model']['revision'])
    _config_contract(config, manifest)
    assert config['codec']['layernorm'] == 'none'
    assert config['training']['batch_size'] * config['training']['gradient_accumulation'] == 64
    with pytest.raises(ValueError, match='topology mismatch'):
        _config_contract(config, {**manifest, 'cell': 'D-LN'})
    config['training']['gradient_accumulation'] = 1
    with pytest.raises(ValueError, match='optimizer/objective'):
        _config_contract(config, manifest)


@pytest.mark.parametrize('width', [512, 1152, 2304])
def test_no_norm_actual_prepared_lifecycle(tmp_path, monkeypatch, width):
    torch.set_num_threads(1)
    calls = install_adapter(monkeypatch)

    def factory():
        model = t_model()
        torch.manual_seed(29)
        model.codec = Codec(1152, candidate_config('D-N', width))
        return model

    model = factory()
    template = task_template()
    template['backend'] = model.base.config.decoder._attn_implementation
    output = tmp_path / f'D-N-B{width}-16x4'
    result = execute_prepared_tiny(tmp_path, monkeypatch, model, template, output, 16,
                                   cell='D-N', model_factory=factory)
    assert result['status'] == 'complete'
    assert result['completed_updates'] == 24
    assert result['completed_checkpoints'] == 17
    assert len(list(output.rglob('*.pt'))) == 17
    assert len(calls) == 109
    assert verify_study_freeze(result['freeze_reference'])['synthetic']
    observations = json.loads((output / 'observations.json').read_text())
    assert sum(len(o['conditions']) for o in observations
               if o['request']['purpose'] == 'task') == 108
    assert sum(len(o['conditions']) for o in observations
               if o['request']['purpose'] == 'objective') == 192
    assert (output / 'geometry.json').is_file()
    for learner in ('specialist_enc_l9', 'specialist_enc_l19', 'specialist_enc_fn', 'shared'):
        events = [json.loads(line) for line in (output / learner / 'metrics.jsonl').read_text().splitlines()]
        updates = [event['payload'] for event in events if event['event_type'] == 'update']
        assert len(updates) == (12 if learner == 'shared' else 4)
        assert all(row['exposure']['sequences'] == 64 for row in updates)
    assert not torch.cuda.is_initialized()
