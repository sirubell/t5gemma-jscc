"""Production quota state/freeze on real tiny CPU optimizer/checkpoint tensors.

Observation and update records are explicit schema fixtures, not model quality evidence.
"""
from copy import deepcopy
import json

import pytest
import torch

from jscc import experiment_records as records
from jscc.activation_replay import file_digest
from jscc.experiment_state import save_state, open_state, restore_state
from jscc.sharing_schedule import DEPTH_PROTOCOL
from jscc.sharing_state import (LEARNERS, SITES, build_sharing_state, create_study_freeze,
                                verify_study_freeze, validate_sharing_state)
from test_experiment_records import update, observation, objective_observation
from test_sharing_state import policy, fixture_state, complete_metadata, state_components, advance_components


def production_state(learner, step):
    sharing, stream = build_sharing_state(synthetic=False, learner=learner,
        ordered_view_identities=[f'original-parent-{i}' for i in range(400)],
        source_valid_per_view=[100]*400, target_valid_per_view=[90]*400,
        padded_per_view=[128]*400, completed_updates=step, batch_size=64,
        protocol_id=DEPTH_PROTOCOL)
    metadata = complete_metadata(fixture_state(0, learner))
    metadata.update(sharing=sharing, completed_updates=step, protocol_identity=DEPTH_PROTOCOL,
                    recipe_identity='K+0.1R-native-effective64-q400', architecture_decision='D-N/B512-depth')
    return metadata, stream


def test_real_production400_1200_states_and_inventory_freeze(tmp_path):
    runs = {}
    for learner in LEARNERS:
        root = tmp_path / learner
        root.mkdir()
        horizon = 1200 if learner == 'shared' else 400
        steps = ([0] if learner == 'shared' else []) + [horizon*i//4 for i in (1,2,3,4)]
        torch.manual_seed(0)
        components = state_components(horizon)
        artifacts, checkpoint_hashes, events = [], {}, []
        previous = 0
        for step in steps:
            advance_components(components, step-previous)
            previous = step
            metadata, stream = production_state(learner, step)
            model, optimizer, scheduler = components
            target = root / f'step-{step}.pt'
            reference = save_state(target, model=model, optimizer=optimizer, scheduler=scheduler,
                scaler=None, metadata=metadata, stream_state=stream)
            opened = open_state(reference, expected=metadata)
            assert restore_state(opened, model=model, optimizer=optimizer,
                                 scheduler=scheduler, scaler=None) == stream
            artifacts.append(records.artifact_ref(target, root, 'tensor'))
            checkpoint_hashes[step] = reference.sha256
        local = dict.fromkeys(SITES, 0)
        for i in range(horizon):
            site = SITES[((i//3)%3+i%3)%3] if learner == 'shared' else learner.removeprefix('specialist_')
            local[site] += 1
            event = update(i+1, horizon)
            event.update(run_id=learner, phase_id='combined')
            p = event['payload']
            p.update(site_id=site, site_step=local[site], sweep=local[site])
            p['exposure'].update(sequences=64, source_tokens=100, target_tokens=90,
                                 padded_tokens=128, valid_tokens=100, cumulative_valid_tokens=(i+1)*100)
            p['objective'] = deepcopy(objective_observation('combined')['payload']['conditions'][0]['objectives'])
            p['objective'] = {'kind':'combined', 'total':records.measure(1.2), 'components':p['objective']}
            records.validate_event(event)
            events.append(event)
        observed = []
        sites = SITES if learner == 'shared' else (learner.removeprefix('specialist_'),)
        for purpose, points in [('task', [horizon//2,horizon]), ('objective', [horizon*i//4 for i in (1,2,3,4)])]:
            if learner == 'shared':
                points.insert(0,0)
            for site in sites:
                for step in points:
                    event = observation() if purpose == 'task' else objective_observation('combined')
                    event.update(run_id=learner, phase_id='combined')
                    p = event['payload']
                    p['request'].update(checkpoint=checkpoint_hashes[step], site=site, step=step,
                        site_step=step//3 if learner=='shared' else step, protocol=DEPTH_PROTOCOL,
                        learner_kind='shared' if learner=='shared' else 'specialist')
                    p['identity'] = records.observation_identity(p['request'])
                    records.validate_event(event)
                    observed.append(p['identity'])
                    events.append(event)
        manifest = {'schema':records.SCHEMA, 'run_id':learner,
                    'phases':[{'phase_id':'combined','updates':horizon,'start_step':0,'start_valid_tokens':0}],
                    'expected_observations':observed, 'expected_artifacts':[a['path'] for a in artifacts]}
        inventory = {'schema':records.SCHEMA, 'mode':'tensor-complete','artifacts':artifacts,'omitted_tensors':[]}
        (root/'events.jsonl').write_text(''.join(json.dumps(event)+'\n' for event in events))
        (root/'manifest.json').write_text(json.dumps(manifest))
        (root/'inventory.json').write_text(json.dumps(inventory))
        runs[learner] = {'root':str(root), **{key:{'path':str(root/name),'sha256':file_digest(root/name)}
            for key,name in [('events','events.jsonl'),('manifest','manifest.json'),('inventory','inventory.json')]}}
    kwargs = dict(protocol_identity=DEPTH_PROTOCOL, recipe_identity='K+0.1R-native-effective64-q400',
                  architecture_decision='D-N/B512-depth', site_policy=policy(), trained_runs=runs,
                  obligation_plan={'task':['heldout-task'],'objective':['heldout-objective'],'geometry':['atlas']},
                  synthetic=False)
    reference = create_study_freeze(tmp_path/'freeze.json', **kwargs)
    receipt = verify_study_freeze(reference, protocol_identity=DEPTH_PROTOCOL, obligation='atlas')
    assert receipt['synthetic'] is False and len(receipt['checkpoint_hashes']) == 17
    with pytest.raises(ValueError, match='freeze horizon mismatch'):
        create_study_freeze(tmp_path/'wrong-protocol.json', **{**kwargs, 'protocol_identity':'q200'})
    metadata, stream = production_state('shared',1200)
    payload = {'schema':'experiment-state-v2','kind':'shared_encoder_v2','metadata':metadata,
               'stream':stream,'scheduler':{'last_epoch':1200}}
    validate_sharing_state(payload)
    metadata['protocol_identity'] = 'q200'
    with pytest.raises(ValueError):
        validate_sharing_state(payload)
