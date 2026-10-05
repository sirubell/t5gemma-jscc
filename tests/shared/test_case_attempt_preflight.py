"""Pure selected-set attempt eligibility before any case execution."""
import json
from types import SimpleNamespace
import pytest
from jscc import shared_runs, experiment


@pytest.fixture
def cases(tmp_path, monkeypatch):
    root = tmp_path/'root'
    names = ['width-8-ratio-50', 'width-8-ratio-75']
    for name in names:
        (root/'cases'/name).mkdir(parents=True)
    monkeypatch.setattr(shared_runs, 'recipe_run_record', lambda _: {'cases': [{'case_id': name} for name in names]})
    return root, names


def attempt(root, name, status, identifier='0001'):
    case = root/'cases'/name
    target = case/'attempts'/f'training-{identifier}'
    target.mkdir(parents=True)
    (target/'attempt.json').write_text(json.dumps({'case': str(case), 'status': status, 'stage': 'training'}))
    return target


def tree(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('defect,message', [
    ('completed','already completed'), ('existing','explicit checkpoint continuation'),
    ('running','stopped-worker reconciliation'), ('locked','active training lock'),
    ('dangling_lock','active training lock'),
    ('occupied','already occupied'), ('orphan','required record'),
    ('status','ineligible for continuation')])
def test_whole_default_set_rejects_before_first_case_or_attempt(cases, monkeypatch, defect, message):
    root, names = cases
    second = root/'cases'/names[1]
    ex = {'root': str(root), 'attempt': '0001'}
    if defect in ('completed','existing','running','status'):
        attempt(root, names[1], {'existing':'failed','status':'unknown'}.get(defect,defect))
    elif defect == 'locked':
        (second/'.training-active').mkdir()
    elif defect == 'dangling_lock':
        (second/'.training-active').symlink_to(second/'absent')
    else:
        target = second/'attempts'/'training-0001'
        target.mkdir(parents=True)
        if defect == 'occupied':
            (target/'attempt.json').write_text(json.dumps({'status':'failed'}))
            # Existing records require explicit continuation before collision check.
            ex['attempt'] = '0002'
            (second/'attempts'/'training-0002').mkdir()
            (second/'attempts'/'training-0002'/'attempt.json').write_text(json.dumps({'status':'failed'}))
    before = tree(root)
    seen = []
    monkeypatch.setattr(experiment,'read_request',lambda _: (root/'request.yaml', {'workflow':'shared','operation':'train','execution':ex}))
    monkeypatch.setattr(experiment,'_bound',lambda *args: {})
    monkeypatch.setattr(shared_runs,'train_recipe_case',lambda *args,**kwargs: seen.append(args))
    expected = 'explicit checkpoint continuation' if defect == 'occupied' else message
    with pytest.raises((ValueError,FileExistsError),match=expected):
        experiment.execute(root/'request.yaml')
    assert not seen
    assert tree(root) == before
    assert not (root/'cases'/names[0]/'attempts').exists()
    assert not (root/'cases'/names[0]/'training.json').exists()


def test_fresh_default_set_keeps_declared_order_and_explicit_subset(cases, monkeypatch):
    root,names = cases
    assert shared_runs.preflight_recipe_cases(root) == names
    assert shared_runs.preflight_recipe_cases(root,[names[1]]) == [names[1]]
    seen=[]
    monkeypatch.setattr(experiment,'read_request',lambda _: (root/'request.yaml',{'workflow':'shared','operation':'train','execution':{'root':str(root),'attempt':'new'}}))
    monkeypatch.setattr(experiment,'_bound',lambda *args: {})
    monkeypatch.setattr(shared_runs,'train_recipe_case',lambda root,case,**kwargs:seen.append(case))
    experiment.execute(root/'request.yaml')
    assert seen == names
    assert not list(root.glob('cases/*/attempts'))


def test_continuation_checks_running_status_before_checkpoint_lookup(cases,monkeypatch):
    root,names=cases
    attempt(root,names[1],'running')
    def forbidden(*args,**kwargs):
        raise AssertionError('continuation examined before running status')
    monkeypatch.setattr(shared_runs,'_resume_reference',forbidden)
    with pytest.raises(ValueError,match='stopped-worker reconciliation'):
        shared_runs.preflight_recipe_cases(root,[names[1]],attempt_id='0002',resume='checkpoint')


def test_explicit_stopped_worker_reconciliation_allows_continuation(cases,monkeypatch):
    root,names=cases
    previous=attempt(root,names[1],'running')
    (root/'cases'/names[1]/'.training-active').mkdir()
    shared_runs.reconcile_recipe_attempt(root,names[1],'0001')
    assert json.loads((previous/'attempt.json').read_text())['status']=='interrupted'
    assert not (root/'cases'/names[1]/'.training-active').exists()
    reference=SimpleNamespace(path='recorded-checkpoint')
    monkeypatch.setattr(shared_runs,'_resume_reference',lambda *args,**kwargs:reference)
    assert shared_runs.preflight_recipe_cases(root,[names[1]],attempt_id='0002',resume='checkpoint')==[names[1]]


def test_existing_continuation_identifier_is_rejected(cases,monkeypatch):
    root,names=cases
    attempt(root,names[0],'interrupted','0002')
    monkeypatch.setattr(shared_runs,'_resume_reference',lambda *args,**kwargs:SimpleNamespace(path='checkpoint'))
    with pytest.raises(FileExistsError,match='already occupied'):
        shared_runs.preflight_recipe_cases(root,[names[0]],attempt_id='0002',resume='checkpoint')


def test_per_case_lock_recheck_rejects_state_changed_after_preflight(cases,monkeypatch):
    root,names=cases
    assert shared_runs.preflight_recipe_cases(root,[names[0]])==[names[0]]
    attempt(root,names[0],'completed')
    monkeypatch.setattr(shared_runs,'_manifest',lambda _: {})
    monkeypatch.setattr(shared_runs,'_case_descriptions',lambda *args:{names[0]:{}})
    def forbidden(*args,**kwargs):
        raise AssertionError('attempt reserved after eligibility changed')
    monkeypatch.setattr(shared_runs,'reserve_attempt',forbidden)
    with pytest.raises(ValueError,match='already completed'):
        shared_runs.train_recipe_case(root,names[0],attempt_id='0002')
    assert not (root/'cases'/names[0]/'.training-active').exists()
    assert not (root/'cases'/names[0]/'attempts/training-0002').exists()


@pytest.mark.parametrize('selected', [[],['unknown'],['width-8-ratio-50']*2])
def test_invalid_selection_has_no_attempt_effects(cases,selected):
    root,_=cases
    with pytest.raises(ValueError,match='distinct declared cases'):
        shared_runs.preflight_recipe_cases(root,selected)
    assert not list(root.glob('cases/*/attempts'))


def test_finalization_without_terminal_continuation_has_no_effects(cases):
    root,names=cases
    with pytest.raises(ValueError,match='exact terminal checkpoint'):
        shared_runs.preflight_recipe_cases(root,names,finalize_terminal=True)
    assert not list(root.glob('cases/*/attempts'))
