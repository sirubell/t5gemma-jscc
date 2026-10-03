"""Production-horizon checkpoint guard with real tiny-model native64 updates."""
from dataclasses import asdict

import pytest
import torch

from jscc.activation_replay import canonical_digest
from jscc.experiment_state import CheckpointRef, open_state
from jscc.sharing_protocol import SharingLearner, batch_view
from jscc.sharing_qualification_guard import same_shape_guard, snapshot, state_exact
from jscc.sharing_schedule import DEPTH_PROTOCOL, SharingPlan
from jscc.sharing_run import _controls
from sharing_fixtures import sharing_model
from test_sharing_accumulation import parent64


@pytest.fixture(scope='module')
def parents():
    rows = [parent64(i) for i in range(400)]
    for row in rows:
        tokens = row['input_ids']
        assert isinstance(tokens, torch.Tensor)
        tokens.remainder_(16)
    return rows


@pytest.mark.parametrize('q', [200, 400])
@pytest.mark.parametrize('site', ['enc_l9', 'enc_l19', 'enc_fn'])
def test_non_synthetic_guard_checkpoint_protocol(tmp_path, parents, q, site):
    torch.set_num_threads(1)
    protocol = DEPTH_PROTOCOL if q == 400 else None
    plan = SharingPlan(tuple(batch_view(row) for row in parents[:q]),
                       'production-horizon-cpu-guard', synthetic=False, protocol_id=protocol)
    model = sharing_model('D-N')
    kwargs = dict(schedule=plan.learner('specialist_' + site), run_id='guard-' + site,
                  identity={'source':'source','config':'config','data':'data','parent':None},
                  model_revision='tiny-cpu', microbatch_size=16)
    reference = SharingLearner(model, enc_fn_reuse=False, **kwargs)
    candidate = SharingLearner(model, enc_fn_reuse=site == 'enc_fn', **kwargs)
    zero = snapshot(candidate)
    metadata = dict(source_identity='source', config_identity='config', model_revision='tiny-cpu',
                    recipe_identity='K+.1R', architecture_decision='D-N-tiny-cpu',
                    initialization_identity='fresh-tiny',
                    stream_identity=canonical_digest([asdict(v) for v in plan.views]),
                    protocol_identity=protocol or 'sharing-effective64-v1')
    result = same_shape_guard(reference, candidate, parents[:2],
                              checkpoint_path=tmp_path/'guard.pt', metadata=metadata)
    assert result['status'] == 'passed' and result['physical_updates'] == 5
    assert result['checkpoint']['exact'] and result['restored_second_update_exact']
    assert result['nonzero_lr_update_changed_weights']
    assert state_exact(zero, snapshot(candidate))
    checkpoint = result['checkpoint']
    opened = open_state(CheckpointRef(checkpoint['path'], checkpoint['sha256'], checkpoint['size']),
                        expected={**metadata, 'parent_identity': None})
    details = opened.payload['metadata']
    sharing = details['sharing']
    assert sharing['synthetic'] is False
    assert sharing['q'] == sharing['horizon'] == q
    assert len(set(sharing['ordered_view_identities'])) == q
    assert sharing['batch_size'] == 64 and details['completed_updates'] == 2
    assert details['protocol_identity'] == metadata['protocol_identity']
    assert sharing.get('protocol_id') == protocol
    assert details['comparison_controls'] == _controls(candidate, details)
    assert opened.payload['stream']['offset'] == 128
    assert not torch.cuda.is_initialized()
