"""Explicit offline reconciliation of durable scheduler intents.

The caller supplies recorded authoritative evidence. No scheduler is queried or
invoked here. Snapshots are immutable; SUBMISSION.json is never rewritten.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import re

from .activation_replay import canonical_digest, file_digest

TERMINAL = {'completed', 'failed', 'cancelled', 'rejected', 'preflight-failed'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


@contextmanager
def _locked(directory):
    import fcntl
    with (directory/'.reconciliation.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def _read(bundle, directory):
    from .submission import validate_bundle
    validate_bundle(bundle)
    _require(directory.resolve() == Path(bundle['output']).resolve(), 'receipt/output differs')
    receipt = json.loads((directory/'SUBMISSION.json').read_text())
    _require(receipt['bundle_identity'] == bundle['identity'], 'receipt identity differs')
    saved = json.loads((directory/'bundle.json').read_text())
    _require(saved == bundle, 'saved bundle differs')
    _require(receipt.get('bundle_sha256') == file_digest(directory/'bundle.json'), 'bundle bytes differ')
    _require(receipt.get('payload_sha256') == file_digest(directory/'job.sbatch'), 'payload bytes differ')
    return receipt


def _latest(directory):
    files = sorted((directory/'reconciliation').glob('[0-9]*.json'))
    return json.loads(files[-1].read_text()) if files else None


def _save(directory, record):
    from .submission import _sync_directory, write_json
    folder = directory/'reconciliation'
    folder.mkdir(exist_ok=True)
    _sync_directory(directory)
    prior = _latest(directory)
    record['sequence'] = 1 if prior is None else prior['sequence']+1
    record['previous_sha256'] = None if prior is None else canonical_digest(prior)
    record['recorded_at'] = datetime.now(timezone.utc).isoformat()
    path = folder/f'{record["sequence"]:08d}.json'
    # Under the lock only a fully synced atomic file becomes visible. A crash
    # leaves an ignored temporary file, never an empty history entry.
    _require(not path.exists(), 'immutable reconciliation entry exists')
    write_json(path, record)
    return record


def _intent(bundle, receipt, job):
    intent = {key: job.get(key) for key in ('attempt_identity', 'attempted_at', 'arguments')}
    intent.update(bundle_identity=bundle['identity'], group=job['group'],
                  payload_sha256=receipt['payload_sha256'])
    # Legacy receipts have no generated identity/time. Exact original receipt
    # bytes form the immutable intent anchor; absence never permits inference
    # from a nearby job name or time alone.
    intent['submission_sha256'] = receipt['_sha256']
    return intent


def _resolve(bundle, receipt, job, lookup):
    retained = job.get('job_id')
    def unknown(basis):
        if retained and job['status'] in ('accepted', 'submitted'):
            return {'status': 'accepted', 'job_id': retained, 'basis': 'original-receipt'}
        return {'status': 'unknown', 'basis': basis}
    intent = _intent(bundle, receipt, job)
    if not isinstance(lookup, dict) or lookup.get('intent') != intent:
        return unknown('missing-exact-intent')
    if lookup.get('authoritative') is not True or lookup.get('query_status') != 'complete':
        return unknown('query-incomplete-or-failed')
    matches = lookup.get('matches')
    if not isinstance(matches, list):
        return unknown('missing-matches')
    if len(matches) == 1:
        match = matches[0]
        if (isinstance(match, dict) and match.get('intent') == intent
                and re.fullmatch(r'[1-9][0-9]*', str(match.get('job_id', '')))):
            return {'status': 'accepted', 'job_id': str(match['job_id']), 'basis': 'unique-authoritative-match'}
    if not matches and lookup.get('nonacceptance') == {
            'exhaustive': True, 'final': True, 'acceptance_window_closed': True}:
        return {'status': 'not-accepted', 'basis': 'authoritative-final-nonacceptance'}
    return unknown('ambiguous-or-insufficient')


def _retry_metadata(previous):
    return {key: previous[key] for key in ('retry_authorization', 'new_bundle_identity', 'recovery_claims', 'recovery_approvals')
            if previous and key in previous}


def _approve_recovery(record, authorization, new_identity, groups):
    from copy import deepcopy
    _require(not record.get('conflicts'), 'authoritative evidence conflict blocks recovery')
    approvals = dict(record.get('recovery_approvals', {}))
    _require(approvals.get(new_identity, authorization) == authorization, 'recorded retry authority is immutable')
    approvals[new_identity] = authorization
    claims = dict(record.get('recovery_claims', {}))
    for group in groups:
        _require(claims.get(str(group), new_identity) == new_identity, 'group already claimed by another recovery')
        claims[str(group)] = new_identity
    approved = deepcopy(record)
    approved.setdefault('retry_authorization', authorization)
    approved.setdefault('new_bundle_identity', new_identity)
    approved.update(recovery_claims=claims, recovery_approvals=approvals)
    return approved


def reconciliation_intents(bundle, receipt_dir):
    """Read exact anchors to include in recorded evidence, including legacy."""
    directory = Path(receipt_dir)
    receipt = _read(bundle, directory)
    receipt['_sha256'] = file_digest(directory/'SUBMISSION.json')
    return [_intent(bundle, receipt, job) for job in receipt['jobs']]


def reconcile_job(bundle, receipt_dir, *, evidence):
    directory = Path(receipt_dir)
    with _locked(directory):
        receipt = _read(bundle, directory)
        receipt['_sha256'] = file_digest(directory/'SUBMISSION.json')
        _require(isinstance(evidence, dict) and evidence.get('schema') == 'scheduler-reconciliation-evidence-v1'
                 and evidence.get('bundle_identity') == bundle['identity']
                 and isinstance(evidence.get('reference'), str) and evidence['reference'].strip(),
                 'exact recorded scheduler evidence required')
        digest = canonical_digest(evidence)
        previous = _latest(directory)
        if previous and previous.get('evidence_sha256') == digest:
            _require(not previous.get('conflicts'), 'authoritative task evidence conflict remains unresolved')
            return previous
        conflict_resolution = evidence.get('conflict_resolution')
        resolving = bool(previous and previous.get('conflicts'))
        if previous and previous.get('conflicts'):
            _require(not any(c.get('kind') == 'group' for c in previous['conflicts']),
                     'group identity conflict requires separate exact lifecycle investigation')
            _require(isinstance(conflict_resolution, dict)
                     and conflict_resolution.get('authoritative') is True
                     and conflict_resolution.get('reference')
                     and conflict_resolution.get('conflict_sha256') == canonical_digest(previous),
                     'explicit authoritative conflict resolution required')
        lookups = evidence.get('groups', [])
        _require(isinstance(lookups, list), 'group evidence must be a list')
        indices = [row.get('group') for row in lookups if isinstance(row, dict)]
        _require(len(indices) == len(lookups) and len(indices) == len(set(indices))
                 and all(type(i) is int and 0 <= i < len(bundle['groups']) for i in indices),
                 'duplicate or invalid group evidence')
        jobs = {job['group']: job for job in receipt['jobs']}
        groups, tasks, conflicts = [], [], []
        for i, group in enumerate(bundle['groups']):
            job = jobs.get(i)
            if job is None:
                resolution = ({'status': 'not-attempted', 'basis': 'durable-stopped-intent'}
                              if receipt['status'] == 'stopped-no-retry' else
                              {'status': 'unknown', 'basis': 'submitter-not-durably-stopped'})
            else:
                lookup = next((row for row in lookups if row['group'] == i), None)
                resolution = _resolve(bundle, receipt, job, lookup)
            # Acceptance is monotonic across snapshots even if later lookup
            # data is delayed or fails. A conflicting ID is explicit failure.
            prior = previous['groups'][i] if previous else None
            if prior is None and job and job.get('job_id') and job['status'] in ('accepted', 'submitted'):
                prior = {'status': 'accepted', 'job_id': job['job_id'], 'basis': 'original-receipt', 'group': i}
            if prior:
                if prior['status'] == 'not-accepted':
                    if resolution['status'] == 'accepted':
                        conflicts.append({'kind': 'group', 'group': i, 'index': None, 'prior': prior, 'new': resolution})
                    resolution = {key: prior[key] for key in ('status', 'basis')}
                if prior['status'] == 'accepted':
                    if (resolution['status'] == 'not-accepted'
                            or resolution.get('job_id', prior['job_id']) != prior['job_id']):
                        conflicts.append({'kind': 'group', 'group': i, 'index': None, 'prior': prior, 'new': resolution})
                    resolution = {key: prior[key] for key in ('status', 'job_id', 'basis')}
            groups.append(dict(resolution, group=i))
            for index in group['indices']:
                row = {'group': i, 'index': index, 'status': 'unknown'}
                if resolution['status'] in ('not-accepted', 'not-attempted'):
                    row.update(status='rejected', basis=resolution['basis'])
                authoritative_task = None
                if resolution['status'] == 'accepted':
                    candidates = [t for t in evidence.get('tasks', []) if
                                  t.get('group') == i and t.get('index') == index]
                    if len(candidates) == 1:
                        t = candidates[0]
                        if (t.get('job_id') == resolution['job_id'] and t.get('authoritative') is True
                                and t.get('status') in TERMINAL | {'pending', 'queued', 'running'} and t.get('reference')
                                and t.get('query_status') == 'complete'
                                and t.get('scheduler_task_id') == resolution['job_id']+'_'+str(index)):
                            authoritative_task = t
                            if t['status'] in TERMINAL:
                                row.update(status=t['status'], basis='authoritative-task', evidence=t)
                if previous:
                    prior_task = next(t for t in previous['tasks'] if t['group'] == i and t['index'] == index)
                    affected = resolving and any(c['group'] == i and c['index'] == index for c in previous['conflicts'])
                    if affected:
                        _require(authoritative_task is not None and authoritative_task['status'] in TERMINAL,
                                 'conflict resolution needs exact authoritative terminal task evidence')
                    elif prior_task['status'] in TERMINAL:
                        if authoritative_task and authoritative_task['status'] != prior_task['status']:
                            conflicts.append({'group': i, 'index': index, 'prior': prior_task, 'new': authoritative_task})
                            row = dict(row, status='unknown', basis='authoritative-conflict')
                        else:
                            row = prior_task
                tasks.append(row)
        if previous and previous.get('retry_authorization'):
            for old, new in zip(previous['groups'], groups, strict=True):
                _require(old['status'] == new['status'] and old.get('job_id') == new.get('job_id'),
                         'reconciliation contradicts authorized retry; reservations remain retained')
        record = {'schema': 'submission-reconciliation-v1', 'bundle_identity': bundle['identity'],
                  'submission_sha256': receipt['_sha256'], 'evidence_sha256': digest,
                  'evidence': evidence, 'groups': groups, 'tasks': tasks,
                  'all_tasks_terminal': not conflicts and all(t['status'] in TERMINAL for t in tasks),
                  'retry_eligible_groups': [g['group'] for g in groups if g['status'] in ('not-accepted', 'not-attempted')],
                  'retry_requires_new_exact_authorization': True}
        record.update(_retry_metadata(previous))
        if conflicts:
            record['conflicts'] = conflicts
            from .submission import source_consumers
            registry = source_consumers(bundle)
            registry_path = registry.directory/(canonical_digest(str(Path(bundle['source']['path']).resolve()))+'.json')
            consumers = json.loads(registry_path.read_text())['consumers'] if registry_path.exists() else {}
            record['source_already_released'] = consumers.get(bundle['identity'], {}).get('released', False)
        saved = _save(directory, record)
        if conflicts:
            limitation = ('; historical source release is immutable and cannot be undone'
                          if saved.get('source_already_released') else '; source hold requires explicit resolution')
            raise ValueError('authoritative evidence conflict/contradiction recorded'+limitation)
        return saved


def finalize_source_consumer(bundle, receipt_dir, *, release=False):
    """Explicitly terminalize all-task evidence; release is a separate choice."""
    from .submission import source_consumers
    directory = Path(receipt_dir)
    with _locked(directory):
        _read(bundle, directory)
        record = _latest(directory)
        if record is None:
            raise ValueError('durable reconciliation record required')
        _require(record and record['submission_sha256'] == file_digest(directory/'SUBMISSION.json')
                 and not record.get('conflicts') and record['all_tasks_terminal'], 'every bound task needs durable terminal evidence')
        registry = source_consumers(bundle)
        # Registry API is intentionally reused unchanged. Inspect persisted
        # state to make recovery after a crash between terminal/release idempotent.
        registry_path = registry.directory/(canonical_digest(str(Path(bundle['source']['path']).resolve()))+'.json')
        consumers = json.loads(registry_path.read_text())['consumers'] if registry_path.exists() else {}
        if bundle['identity'] not in consumers:
            return {'state': 'unreserved', 'released': False}
        row = consumers[bundle['identity']]
        _require(row['owner'] == bundle['output'], 'source consumer owner differs')
        if row['state'] not in TERMINAL:
            registry.record_state(bundle['source']['path'], bundle['identity'],
                                  'completed' if all(t['status'] == 'completed' for t in record['tasks']) else 'failed',
                                  owner=bundle['output'], evidence=str(directory/'reconciliation'/f'{record["sequence"]:08d}.json'))
        if release and not row['released']:
            registry.release(bundle['source']['path'], bundle['identity'], owner=bundle['output'])
        return {'state': 'terminal', 'released': release or row['released']}


def authorize_retry_job(bundle, receipt_dir, *, new_bundle, authorization):
    """Explicit exact authorization retires only proven unaccepted case fences.

    No jobs are submitted; original task reservations and archived case fences
    remain. Retry submission still uses the existing exact submission approval.
    Changed science/source/resources requires its separate revision procedure.
    """
    from .submission import CASE_SCHEMA, _sync_directory, validate_bundle
    _require(bundle.get('schema') == CASE_SCHEMA and new_bundle.get('schema') == CASE_SCHEMA,
             'old submission execution retired')
    directory = Path(receipt_dir)
    validate_bundle(new_bundle)
    with _locked(directory):
        receipt = _read(bundle, directory)
        record = _latest(directory)
        if record is None:
            raise ValueError('durable reconciliation record required')
        _require(record and record['submission_sha256'] == file_digest(directory/'SUBMISSION.json')
                 and not record.get('conflicts') and all(g['status'] != 'unknown' for g in record['groups']), 'fully reconciled attempt required')
        _require(isinstance(authorization, dict) and authorization.get('approved') is True
                 and authorization.get('operation') == 'authorize-recipe-retry'
                 and authorization.get('bundle_identity') == bundle['identity']
                 and authorization.get('new_bundle_identity') == new_bundle['identity']
                 and authorization.get('source_commit') == bundle['source']['commit']
                 and authorization.get('output') == new_bundle['output']
                 and isinstance(authorization.get('user_authorization_reference'), str)
                 and authorization['user_authorization_reference'].strip(), 'new explicit exact retry authorization required')
        _require(bundle['schema'] == new_bundle['schema'], 'retry schema differs')
        _require(new_bundle['identity'] != bundle['identity'] and new_bundle['output'] != bundle['output']
                 and new_bundle['attempt'] != bundle['attempt'], 'fresh output and attempt required')
        keys = ('source', 'source_binding', 'python', 'cache', 'root', 'resources')
        keys += ('run_output', 'run_identity', 'input_binding', 'combined')
        for key in keys:
            _require(bundle[key] == new_bundle[key], 'retry changes source/science/budget: '+key)
        old_indices = []
        dependency_key = 'aftercorr'
        for group in new_bundle['groups']:
            matches = [i for i, old in enumerate(bundle['groups'])
                       if old['indices'] == group['indices']
                       and old.get('stage') == group.get('stage')
                       and old['concurrency'] == group['concurrency']]
            _require(len(matches) == 1 and matches[0] not in old_indices,
                     'retry changes group task/resource budget or duplicates group')
            old_indices.append(matches[0])
        _require(old_indices, 'retry selects no groups')
        for position, old_index in enumerate(old_indices):
            group = new_bundle['groups'][position]
            predecessor = bundle['groups'][old_index][dependency_key]
            if predecessor is None:
                inherited = bundle['groups'][old_index].get('accepted_dependency_job_id')
                _require(group[dependency_key] is None and group.get('accepted_dependency_job_id') == inherited,
                         'retry dependency differs')
            elif predecessor in old_indices:
                mapped = old_indices.index(predecessor)
                _require(mapped < position and group[dependency_key] == mapped
                         and not group.get('accepted_dependency_job_id'), 'retry dependency differs')
            else:
                predecessor_record = record['groups'][predecessor]
                _require(predecessor_record['status'] == 'accepted' and group[dependency_key] is None
                         and group.get('accepted_dependency_job_id') == predecessor_record['job_id'],
                         'retry must retain accepted scheduler dependency')
        old_tasks = {}
        for group in record['groups']:
            shape = bundle['groups'][group['group']]
            stages = ['training', 'evaluation'] if shape['stage'] == 'combined' else [shape['stage']]
            for index in shape['indices']:
                case = bundle['cases'][index]
                for stage in stages:
                    if stage in case['commands']:
                        old_tasks[case['case_id'], stage] = (case, group)
        fences = []
        selected = 0
        selected_commands = []
        for position, group in enumerate(new_bundle['groups']):
            stages = ['training', 'evaluation'] if group['stage'] == 'combined' else [group['stage']]
            for index in group['indices']:
                case = new_bundle['cases'][index]
                for stage in stages:
                    if stage in case['commands']:
                        attempt = f"{new_bundle['attempt']}-g{position}-i{index}"
                        selected_commands.append((case, stage, case['commands'][stage], attempt))
        scheduled_keys = {(case['case_id'], stage) for case, stage, _, _ in selected_commands}
        _require(len(new_bundle['cases']) == len(bundle['cases']), 'retry changes case set')
        for old_case, case in zip(bundle['cases'], new_bundle['cases'], strict=True):
            _require({k: v for k, v in old_case.items() if k not in ('commands', 'requests')}
                     == {k: v for k, v in case.items() if k not in ('commands', 'requests')},
                     'retry changes retained effective science')
            _require(old_case['commands'].keys() == case['commands'].keys()
                     == old_case['requests'].keys() == case['requests'].keys(), 'retry changes retained stages')
            for stage, command in case['commands'].items():
                if (case['case_id'], stage) not in scheduled_keys:
                    _require(command == old_case['commands'][stage]
                             and case['requests'][stage] == old_case['requests'][stage],
                             'retry changes retained training/checkpoint request or command')
        for case, stage, command, expected_attempt in selected_commands:
            selected += 1
            _require((case['case_id'], stage) in old_tasks, 'retry adds science')
            old_case, group = old_tasks[case['case_id'], stage]
            _require(group['status'] in ('not-accepted', 'not-attempted'), 'accepted groups cannot be retried')
            for key in ('path', 'effective_config_sha256'):
                _require(case[key] == old_case[key], 'retry changes effective science')
            def without_attempt(request):
                from copy import deepcopy
                result = deepcopy(request)
                result['execution']['attempt'] = '<fresh-attempt>'
                return result
            request = case['requests'][stage]
            old_request = old_case['requests'][stage]
            _require(without_attempt(request) == without_attempt(old_request),
                     'retry changes request/science/budget')
            _require(request['execution']['attempt'] == expected_attempt
                     and request['execution']['attempt'] != old_request['execution']['attempt'],
                     'retry requires deterministic fresh stage attempt')
            for candidate, arguments in ((bundle, old_case['commands'][stage]), (new_bundle, command)):
                request_path = str(Path(candidate['output'])/'requests'/f"{stage}-{case['index']}.yaml")
                _require(arguments == [candidate['python'], candidate['source']['path']+'/experiment.py',
                                       '--config', request_path], 'retry changes command/request ownership')
            path = Path(old_case['path'])/'scheduler-reservations'/(stage+'.json')
            archive = path.parent/(stage+'.retired-'+bundle['identity']+'.json')
            if path.exists():
                try:
                    owner = json.loads(path.read_text()).get('identity')
                except (ValueError, AttributeError):
                    metadata = path.stat()
                    proven = any(row.get('path') == str(path) and row.get('identity') == bundle['identity']
                                 and row.get('device') == metadata.st_dev and row.get('inode') == metadata.st_ino
                                 for row in receipt.get('created_stage_fences', []))
                    _require(proven and group['status'] == 'not-attempted'
                             and receipt['status'] == 'stopped-no-retry', 'incomplete stage fence ownership unproven')
                    owner = bundle['identity']
                _require(owner == bundle['identity'], 'stage fence belongs to another attempt')
            else:
                _require(archive.exists() or group['status'] == 'not-attempted', 'missing retained stage fence')
            fences.append((path, archive))
        _require(selected > 0, 'retry selects no tasks')
        # Record approval before any retirement. A crash may leave some fences
        # retired; the same action resumes, while all uncertain fences stay held.
        approved = _approve_recovery(record, authorization, new_bundle['identity'], old_indices)
        if record.get('recovery_approvals', {}).get(new_bundle['identity']) != authorization:
            record = _save(directory, approved)
        for path, archive in fences:
            if path.exists():
                _require(not archive.exists(), 'retired fence already exists')
                path.rename(archive)
                _sync_directory(path.parent)
        return {'new_bundle_identity': new_bundle['identity'], 'submission_performed': False,
                'authorization_record': record['sequence'], 'retired_fences': [str(a) for _, a in fences]}


def prepare_retry_bundle(bundle, receipt_dir, *, output, attempt, groups=None) -> dict[str, Any]:
    """Review-only subset of reconciled unaccepted groups, unchanged science."""
    from copy import deepcopy
    from .submission import CASE_SCHEMA, normalize_path
    _require(bundle.get('schema') == CASE_SCHEMA, 'old submission execution retired')
    directory = Path(receipt_dir)
    with _locked(directory):
        _read(bundle, directory)
        record = _latest(directory)
        if record is None:
            raise ValueError('durable reconciliation record required')
        _require(record and not record.get('conflicts') and all(g['status'] != 'unknown' for g in record['groups']), 'fully reconciled attempt required')
        _require(not record.get('conflicts'), 'authoritative evidence conflict blocks recovery')
        eligible = [i for i in record['retry_eligible_groups'] if str(i) not in record.get('recovery_claims', {})]
        selected = eligible if groups is None else groups
        _require(isinstance(selected, list) and selected and len(set(selected)) == len(selected)
                 and all(type(i) is int and i in eligible for i in selected),
                 'only eligible unaccepted groups may be selected')
        _require(isinstance(attempt, str) and re.fullmatch(r'[A-Za-z0-9_-]+', attempt), 'safe fresh attempt required')
        new = deepcopy(bundle)
        new['output'] = normalize_path(str(output), bundle['root'])
        _require(new['output'] != bundle['output'], 'fresh output required')
        new['retry_of'] = {'bundle_identity': bundle['identity'], 'submission': str(directory),
                           'reconciliation_sha256': canonical_digest(record), 'groups': selected}
        dependency_key = 'aftercorr'
        remap = {old: index for index, old in enumerate(selected)}
        new['groups'] = []
        for old in selected:
            group = deepcopy(bundle['groups'][old])
            dep = group[dependency_key]
            if dep is not None:
                if dep in remap:
                    _require(remap[dep] < remap[old], 'retry dependencies must precede consumers')
                    group[dependency_key] = remap[dep]
                else:
                    _require(record['groups'][dep]['status'] == 'accepted', 'missing accepted dependency')
                    group[dependency_key] = None
                    group['accepted_dependency_job_id'] = record['groups'][dep]['job_id']
            new['groups'].append(group)
            stages = ['training', 'evaluation'] if group['stage'] == 'combined' else [group['stage']]
            for index in group['indices']:
                for stage in stages:
                    command = new['cases'][index]['commands'].get(stage)
                    if command:
                        case = new['cases'][index]
                        case['requests'][stage]['execution']['attempt'] = f'{attempt}-g{remap[old]}-i{index}'
                        case['commands'][stage] = [new['python'], new['source']['path']+'/experiment.py',
                            '--config', str(Path(new['output'])/'requests'/f'{stage}-{index}.yaml')]
        _require(attempt != bundle['attempt'], 'fresh attempt required')
        new['attempt'] = attempt
        new.pop('identity')
        new['identity'] = canonical_digest(new)
        return new


def require_retry_authorization(bundle):
    """An explicitly prepared recovery cannot bypass its exact approval gate."""
    if 'retry_of' not in bundle:
        return
    directory = Path(bundle['retry_of']['submission'])
    with _locked(directory):
        record = _latest(directory)
        if record is None:
            raise ValueError('durable reconciliation record required')
        approval = record.get('recovery_approvals', {}).get(bundle['identity'], {})
        _require(record and not record.get('conflicts')
                 and all(record.get('recovery_claims', {}).get(str(i)) == bundle['identity'] for i in bundle['retry_of']['groups'])
                 and approval.get('approved') is True
                 and approval.get('operation') == 'authorize-recipe-retry',
                 'new explicit exact retry authorization required before submission')
