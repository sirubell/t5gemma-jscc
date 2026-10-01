"""Actual subprocess guard + full tiny lifecycle, with explicit negative gates."""
import json
import os
import subprocess
import sys
import time

import pytest

from jscc import sharing_controller as controller, sharing_startup as startup
from jscc.sharing_preparation import load_prepared
from sharing_fixtures import sharing_model
from test_sharing_controller import ref, SOURCE
from test_shared_lifecycle import task_template
from test_sharing_qualification import stress_parent
import test_sharing_accumulation_lifecycle as fixture


@pytest.fixture(params=[startup.ROUTE, startup.Q400_ROUTE])
def startup_contract(tmp_path, monkeypatch, request):
    route = request.param
    from jscc import sharing_run
    class PackageReady(Exception):
        pass
    def stop_before_execution(*args):
        raise PackageReady
    with monkeypatch.context() as patch:
        patch.setattr(sharing_run, 'execute_prepared', stop_before_execution)
        patch.setattr(fixture, 'parent64', stress_parent)
        from jscc import sharing_preparation
        original_write = sharing_preparation.write_prepared
        def write_with_test_cap(path, manifest):
            return original_write(path, {**manifest, 'hard_cap_seconds': 1000})
        patch.setattr(sharing_preparation, 'write_prepared', write_with_test_cap)
        tiny_model = sharing_model('D-N')
        template = task_template()
        template['backend'] = tiny_model.base.config.decoder._attn_implementation
        with pytest.raises(PackageReady):
            fixture.execute_prepared_tiny(tmp_path, patch, tiny_model, template,
                                          tmp_path/'unused', 16, cell='D-N')
    prepared = tmp_path/'prepared/prepared.json'
    manifest = load_prepared(prepared)
    allocation = tmp_path/'allocation'
    allocation.mkdir()
    panel = {'documents':[{'sample_id':80},{'sample_id':81}],
             'prompts':[{'prompt_id':'a'},{'prompt_id':'b'}], 'fewshots':[{'fewshot_id':'demo'}]}
    (allocation/'panel.json').write_text(json.dumps(panel))
    (allocation/'cpu-proof.json').write_text(json.dumps({'test_fixture':True}))
    contract = dict(schema='sharing-production-controller-v1', prepared=ref(prepared),
        binding=controller.binding_identity(manifest), run_id='startup-cpu-test',
        hard_cap_seconds=1000, output_byte_cap=100000000, automatic_retry=False,
        allocation_root=str(allocation), entry_route=route,
        finish_before_epoch=int(time.time())+2000, target_assets={}, runtime_versions={},
        expected_panel=ref(allocation/'panel.json'))
    request = dict(schema=startup.schema(contract, '-target-startup-request-v1'), entry_route=route,
        run_id=contract['run_id'], prepared=contract['prepared'], binding=contract['binding'],
        initialization_identity=manifest['initialization']['state_identity'], guard_sha256=startup.GUARD_SHA256,
        guard_workload=startup.DIAGNOSTIC_WORKLOAD, production_workload=startup.workload(contract),
        hard_cap_seconds=1000, finish_before_epoch=contract['finish_before_epoch'], automatic_retry=False,
        target_assets={}, runtime_versions={}, expected_panel=contract['expected_panel'],
        cpu_lifecycle_proof=ref(allocation/'cpu-proof.json'), target_class=startup.target_class(contract),
        fit_policy={'safety_multiplier':1, 'unmeasured_seconds':dict.fromkeys(startup.UNMEASURED, .01),
                    'cleanup_reserve_seconds':10})
    (allocation/'startup-request.json').write_text(json.dumps(request))
    contract['target_startup_request'] = ref(allocation/'startup-request.json')
    path = allocation/'contract.json'
    path.write_text(json.dumps(contract))
    hooks = tmp_path/'hooks'
    hooks.mkdir()
    code = (
        'import torch, pytest, json\n'
        'from jscc import sharing_run, evaluation\n'
        'from sharing_fixtures import sharing_model\n'
        'from test_shared_lifecycle import install_adapter\n'
        'torch.set_num_threads(1)\n'
        'sharing_run.build_model=lambda cfg:(object(),sharing_model("D-N"))\n'
        'patch=pytest.MonkeyPatch()\n'
        'install_adapter(patch)\n'
        'old=evaluation.evaluate_hellaswag\n'
        f'panel={panel!r}\n'
        'def scorer(model,processor,settings,output,condition,data):\n'
        '    result=old(model,processor,settings,output,condition,data)\n'
        '    for key,rows in (panel.items() if "startup" in output.parts else []):\n'
        '        (output/(key+".jsonl")).write_text("".join(json.dumps(r)+"\\n" for r in rows))\n'
        '    result["candidate_forward_requests"]=8\n'
        '    return result\n'
        'evaluation.evaluate_hellaswag=scorer\n')
    (hooks/'sitecustomize.py').write_text(code)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(hooks),str(SOURCE),str(SOURCE/'tests')]))
    return path, contract, request, env, hooks


def run_entry(path, env, mode):
    return subprocess.run([sys.executable,str(SOURCE/'scripts/sharing_production.py'),mode,str(path)],
                          env=env,capture_output=True,text=True,timeout=250)


@pytest.mark.parametrize('failure', [None, 'guard', 'fit'])
def test_combined_real_cpu_startup_and_lifecycle(startup_contract, failure):
    path, contract, request, env, hooks = startup_contract
    if failure=='fit':
        request['fit_policy']['unmeasured_seconds']['geometry']=1e9
        (path.parent/'startup-request.json').write_text(json.dumps(request))
        contract['target_startup_request']=ref(path.parent/'startup-request.json')
        path.write_text(json.dumps(contract))
    if failure=='guard':
        with (hooks/'sitecustomize.py').open('a') as stream:
            stream.write('from jscc import sharing_startup\n'
                         'def failed_guard(*args,**kwargs):\n'
                         '    raise RuntimeError("injected guard failure")\n'
                         'sharing_startup.run_diagnostic=failed_guard\n')
    preflight=run_entry(path,env,'--preflight')
    assert preflight.returncode==0,preflight.stderr
    result=run_entry(path,env,'--controller')
    log=(path.parent/'worker.log').read_text()
    receipt=json.loads((path.parent/'startup/startup-receipt.json').read_text())
    if failure:
        assert result.returncode!=0,log
        assert receipt['status']=='startup_failed_no_science'
        assert not (path.parent/'outputs').exists()
        assert ('guard failure' if failure=='guard' else 'does not fit') in receipt['error']
    else:
        assert result.returncode==0,result.stderr+log
        assert receipt['status']=='startup_guard_passed'
        assert receipt['production_qualified'] is False
        assert receipt['fresh_initialization_restored'] and receipt['rng_restored']
        campaign=json.loads((path.parent/'outputs/campaign.json').read_text())
        assert (campaign['completed_updates'],campaign['completed_checkpoints'])==(24,17)
        assert campaign['status']=='complete'
        observations=json.loads((path.parent/'outputs/observations.json').read_text())
        assert sum(len(r['conditions']) for r in observations if r['request']['purpose']=='task')==108
        assert sum(len(r['conditions']) for r in observations if r['request']['purpose']=='objective')==192
        for learner in ('specialist_enc_l9','specialist_enc_l19','specialist_enc_fn','shared'):
            events=[json.loads(line) for line in (path.parent/'outputs'/learner/'metrics.jsonl').read_text().splitlines()]
            assert next(e for e in events if e['event_type']=='update')['payload']['completed_step']==1
    allocation=json.loads((path.parent/'allocation.json').read_text())
    assert allocation['cleanup_verified'] and allocation['charged_device_seconds']==0


def test_startup_request_cannot_mix_qualification_or_unknown_route(startup_contract):
    path,contract,_,_,_=startup_contract
    contract['qualification']={'path':'old','sha256':'old'}
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError,match='cannot supply'):
        controller.validate_contract(path)
    del contract['qualification']
    contract['entry_route']='unknown'
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError,match='unknown production entry'):
        controller.validate_contract(path)


def test_fit_counts_every_production_obligation():
    record={'sites':{str(i):{'stress_timings':[{'seconds':2}], 'task':[{'seconds':3}],
                           'objective':{'seconds':6}} for i in range(3)}}
    policy={'safety_multiplier':1.5,'unmeasured_seconds':dict.fromkeys(startup.UNMEASURED,10),
            'cleanup_reserve_seconds':20}
    fit=startup.estimate_remaining(record,policy)
    assert fit['estimated_remaining_seconds']==(1200*2+108*3+192)*1.5+40
    assert fit['coverage']['states']==17
    assert not fit['coverage']['heldout_measured_before_freeze']
    depth=startup.estimate_remaining(record,policy,q=400)
    assert depth['estimated_remaining_seconds']==(2400*2+108*3+192)*1.5+40
    assert depth['coverage']['scientific_updates']==2400


def test_q400_actual_device_name_gate(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import torch
    expected = 'GPU-00000000-0000-0000-0000-000000000000'
    path = tmp_path/'device-contract.json'
    path.write_text(json.dumps({'entry_route':startup.Q400_ROUTE, 'device_uuid':expected}))
    monkeypatch.setenv('SHARING_CONTROLLER_CONTRACT',str(path))
    monkeypatch.setenv('SHARING_CONTROLLER_CONTRACT_SHA256',ref(path)['sha256'])
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda:True)
    properties=SimpleNamespace(uuid=expected,name='NVIDIA H200')
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda index:properties)
    manifest={'synthetic_cpu':False,'resolved_config':{'model':{'device':'cuda','dtype':'bfloat16'}}}
    with pytest.raises(ValueError,match='actual RTX 5090'):
        controller.check_target_device(manifest)
    properties.name='NVIDIA GeForce RTX 5090'
    controller.check_target_device(manifest)
