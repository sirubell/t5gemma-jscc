"""Checksum-bound historical artifact interpretation, never live admission."""
import json
import hashlib
from pathlib import Path
import pytest
from jscc import sharing_admission as admission


@pytest.fixture
def saved_device(tmp_path):
    uuid = 'GPU-12345678-1234-1234-1234-123456789abc'
    lease_root = str(tmp_path/'leases')
    contract = dict(run_id='saved-run', device_uuid=uuid,
        device_lease=str(Path(lease_root)/f'device-{uuid}.lock'),
        payload_receipt={'sha256': 'payload-digest'}, admission={'sha256':'admission-digest'})
    payload = dict(component='saved-h200', sole_execution_owner='saved-owner',
        device_binding=dict(policy='slurm-one-h200-runtime-uuid-v1', gpu_class='H200',
            device_count=1, lease_root=lease_root, lease_filename_rule='device-{GPU_UUID}.lock',
            runtime_receipt_required=True))
    runtime = dict(schema='slurm-h200-device-binding-v1', status='BOUND_FROM_VERIFIED_SLURM_ALLOCATION',
        run_id=contract['run_id'], campaign_id=admission.CAMPAIGN, component=payload['component'],
        execution_owner=payload['sole_execution_owner'],
        payload_receipt={'path':'saved-payload','sha256':'payload-digest'},
        budget_admission={'path':'saved-admission','sha256':'admission-digest'},
        device_uuid=uuid,device_lease=contract['device_lease'],finish_before_epoch=admission.DEADLINE,
        device_name='NVIDIA H200',allocation_evidence={'saved':True},visibility_evidence={'saved':True},
        slurm_job_id='123',node='saved-node')
    path = tmp_path/'runtime.json'
    def write():
        path.write_text(json.dumps(runtime))
        contract['device_binding_receipt']={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    write()
    return contract,payload,runtime,write,path


def test_saved_runtime_ignores_current_scheduler_environment(saved_device,monkeypatch):
    contract,payload,_,_,_=saved_device
    monkeypatch.setenv('SLURM_JOB_ID','different-live-job')
    monkeypatch.setenv('SLURMD_NODENAME','different-live-node')
    admission._runtime_device(contract,payload,'saved-admission','saved-payload')


@pytest.mark.parametrize('field,value',[('slurm_job_id',''),('node',''),('device_uuid','GPU-short'),
    ('device_lease','wrong'),('device_name','NVIDIA A100'),('run_id','wrong'),
    ('allocation_evidence',{}),('visibility_evidence',{})])
def test_saved_runtime_required_bindings_fail_closed(saved_device,field,value):
    contract,payload,runtime,write,_=saved_device
    runtime[field]=value
    write()
    with pytest.raises(ValueError):
        admission._runtime_device(contract,payload,'saved-admission','saved-payload')


def test_saved_runtime_checksum_rejects_changed_bytes(saved_device):
    contract,payload,_,_,path=saved_device
    path.write_text('{}')
    with pytest.raises(ValueError):
        admission._runtime_device(contract,payload,'saved-admission','saved-payload')


def test_live_admission_rejects_before_any_artifact_read(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('artifact was read before policy rejection')
    monkeypatch.setattr(Path,'read_text',forbidden)
    with pytest.raises(ValueError,match='live historical admission'):
        admission.validate_overnight_admission({},True,query_live=True)
