# Fixed source preview and consumer lifetime

Current `recipe-case-submission-v1` submissions bind a clean ordinary checkout/worktree at a full
Git SHA, the tracked `uv.lock` digest, and the selected Python interpreter and
installed distribution versions. Preparation, submission and launch check that
binding before data/model execution. These checks read small metadata; they do
not download assets, load weights or hash immutable tensors. Dirty tracked or
untracked nonignored source is rejected. Ignored caches, run outputs and local
research notes do not make the checkout dirty.

Create a clean worktree from the reviewed commit, install its locked environment
before binding it, and author the existing job YAML's `source.commit`,
`source.path` and `python` for that checkout. Same-SHA consumers may share a
checkout and its unchanged runtime. Different concurrent revisions require
another pinned checkout/worktree. A worktree is sufficient; no campaign/release
hierarchy is required. Routine scientific changes remain in the existing YAML.

```bash
uv run --locked python scripts/submit_recipe.py prepare \
  --job job.yaml --limits approved-limits.yaml --output /fresh/bundle.json
uv run --locked python scripts/submit_recipe.py preview --bundle /fresh/bundle.json
uv run --locked python scripts/submit_recipe.py check --bundle /fresh/bundle.json
```

Use concrete absolute paths under the approved host root. Preview prints the
recorded finite cases, exact public train/evaluate commands, resources, selected
scientific work, task counts and own-task dependencies. It invokes no scheduler.
The preview payload equals the saved submission payload; dependency ID `1` is a
labeled placeholder. Ordinary-task submission arrays are not provided here.

New preparation, action, preview, render, scheduler generation, submission,
launch and retry accept only `recipe-case-submission-v1`. The retired
`shared-codec-submission-v1` route rejects before output reservation, acquired
input loading, registry/history/claim/fence changes, scheduler calls or execution.
Saved legacy bundles remain available to `check`, `intents`, `reconcile`,
`terminal` and `release`. `check` preserves prepared/model/config/source/runtime
integrity checks, rather than merely validating the bundle identity. Changed
source bytes may make an old prepared binding incompatible; use its retained
exact source where needed. Saved bundle, payload, receipt and source hashes are
never rewritten or converted. Retiring current execution does not stop jobs or
release held pinned execution sources.

Bound effective settings come from `bundle.json`, so later authored YAML edits
cannot change a queued case. Editing the bundle or prepared/config bytes fails
identity checks. Changing science/source/budget requires a new reviewed binding
and an explicitly linked case/revision; a rejected check is not an exact retry.
Saved bundles lacking `source_binding` cannot pass this preflight; retain their
original source-specific readers where needed. New work requires fresh current
case preparation.

## Callable interfaces

`jscc.source_lifetime.capture_source_binding(source, commit, python)` returns a
JSON-serializable `fixed-source-binding-v1` with full SHA, canonical checkout path,
lock digest, interpreter path/resolved binary metadata, Python/platform/prefix and
installed distribution versions. `check_source_binding(binding)` rechecks exact
identity. It works independently of task science: COCO/HellaSwag consumers can
call it without changing their configs, entry points, channels or learner.
Interpreter symlinks are supported. Package bytes are assumed immutable under
this lifetime contract; version metadata cannot detect arbitrary same-version
illicit mutation. An explicit audit is needed for that scenario. Startup checks
alone cannot prevent a developer from changing a checkout after validation.

`jscc.recipe_submission.submission_preflight(bundle)` checks the bundle's
source/runtime binding and existing effective/prepared config contract.
`preview_job(bundle, bundle_path)` returns the concrete preview and payload.
`source_consumers(bundle)` selects the single host registry at
`<job.root>/.source-consumers`, outside the checkout and scientific outputs.
All submitters and maintenance owners on that host must use that same registry.

`SourceConsumers(directory)` also supports non-recipe consumers:

```python
binding = capture_source_binding(checkout, approved_full_sha, python_path)
registry = SourceConsumers(host_root / '.source-consumers')
registry.reserve(binding, consumer_id, owner=owner_id, reference=attempt_receipt)
registry.record_state(checkout, consumer_id, 'queued', owner=owner_id, evidence=receipt)
# Before actual data/model execution:
check_source_binding(binding)
registry.record_state(checkout, consumer_id, 'running', owner=owner_id, evidence=launch_receipt)
# Only after ALL tasks bound to this consumer are durably terminal:
registry.record_state(checkout, consumer_id, 'completed', owner=owner_id, evidence=terminal_receipt)
registry.release(checkout, consumer_id, owner=owner_id)
# Atomically exclude new reservations while maintaining checkout/runtime:
with registry.maintenance(checkout, python=python_path):
    perform_authorized_checkout_or_runtime_maintenance()
```

Consumer IDs are unique retained attempts, not reusable case names. Ownership and
all state events remain after release. POSIX advisory locks serialize updates
across independent processes and release on exit. Do not remove the registry or
its lock file to bypass a hold. Each unreleased consumer holds the whole checkout
and runtime, even when another same-binding consumer finishes. Reserve rejects
a changed binding on a held checkout. Maintenance must hold `maintenance(checkout)` across checkout mutation. If
replacing an environment or interpreter, pass `python=python_path` too: it
checks consumers on every checkout sharing that environment/interpreter. New
reservations remain blocked until the context exits. `require_released` is a
read-only diagnostic, not an atomic mutation gate. Git itself is not intercepted.

## Submission and recovery obligations

Existing `submit_job` reserves fresh outputs, writes the bundle/payload and
submission receipt, then performs preflight and reserves the source consumer
before the first scheduler call. Its consumer ID is the bundle identity; owner
is the exact output path; reference is `SUBMISSION.json`. Successful mock/live
submissions record `queued`. Exceptions after reservation conservatively record
`unresolved`; accepted IDs and original intent remain in the submission receipt.
A source or prepared/config preflight failure records type/message/stage with zero scheduler calls
and retains the output reservation. No automatic retry or release occurs.

`run_job` writes an exclusive per-group/case preflight marker. Wrong/dirty source,
changed runtime/config, or loaded helper checkout mismatch records a failed
marker before GPU/model execution. Duplicate invocation cannot overwrite it. The CLI also retains JSON/identity
validation failures beside the trusted bundle path without using a mutated
bundle's output path.
A failed task never releases a shared submission: other rows may still be
queued/running. The owning submission session must reconcile every task before
recording terminal state and release. Neither a CLI error nor one task's exit
proves that the whole consumer is terminal.

Supported held states are `reserved`, `queued`, `running`, `unresolved`.
Terminal states are `completed`, `failed`, `cancelled`, `rejected`,
`preflight-failed`; each requires a durable evidence reference and the same owner.
For a grouped consumer, a terminal state asserts **all** its tasks are terminal.
Active state updates cannot regress running/uncertain state to queued/reserved.
A lookup failure, missing response or ambiguous scheduler match stays unresolved.
The [released scheduler reconciliation](#explicit-scheduler-reconciliation-and-recovery) below consumes retained authoritative evidence and aggregates terminal task receipts. The owning operator must explicitly record terminal
state and release; this module never infers completion from age, source checks,
output existence or one accepted/rejected job. Crashes between receipt/registry
writes require inspecting both durable records; default to retaining the hold.

Offline Git fixtures and mock scheduler results establish these software
contracts only. Real scheduler acceptance, target runtime/GPU fit, acquisition,
full-weight performance and scientific outcomes need their own authorized checks.

## Reviewed public recipe case arrays

Use `recipe-case-submission-v1` for the released public recipe run commands. One
`action` prepares source-bound small records from acquired registrations, freezes
selected cases and renders their exact commands/resources before optional mock
submission. With no scheduler option it previews only. A later live submission
requires recorded approval for the exact bundle identity, source and output;
preparation or a mock result supplies no live authority.

```yaml
schema: recipe-case-submission-v1
recipe: /host/source/configs/recipes/changed.yaml
registrations: /host/assets/registrations
run_output: /host/runs/new-comparison
root: /host
source:
  path: /host/source
  commit: FULL_REVIEWED_40_CHARACTER_SHA
python: /host/source/.venv/bin/python
cache: /host/hf-cache
output: /host/submissions/new-comparison
resources:
  partition: h200q
  account: research
  qos: h200q_1g
  node: hgx-cn01
  gpus: 1
  cpus: 16
  memory_gb: 170
  walltime_minutes: 120
cases: [width-32-ratio-50]
evaluation_cases: [width-32-ratio-50]
concurrency: 1
combined: false
attempt: reviewed-01
```

These paths and resource names are examples; use the existing approved limits
YAML for the applicable host/account. Limits retain `root`, `resources`,
`max_concurrency` and `max_array_tasks`. Finite per-task limits and per-array
concurrency remain distinct from the saved historical H200 four-allocated-GPU
account policy and scientific budgets. This implementation queries no live policy
and adds no aggregate controller. `cases: null` explicitly requests every declared
case. A subset or singleton retains its original recipe indices; it never expands
into all cases. `evaluation_cases: []` adds no evaluation. Every evaluation case
must be a selected training case.

```bash
uv run --locked python scripts/submit_recipe.py action \
  --job reviewed-cases.yaml --limits approved-limits.yaml \
  --output /host/reviews/new-comparison.json
# The same action can use fake scheduler responses, after scope is reviewed:
uv run --locked python scripts/submit_recipe.py action \
  --job reviewed-cases.yaml --limits approved-limits.yaml \
  --output /host/reviews/new-comparison-mock.json \
  --mock-results fake-responses.json
```

Preparation binds one fresh logical recipe root; later preview/check/render and
submission consume the saved bundle. Public training reads bound effective
settings through `--output`, with explicit `--case` and retained attempt identity.
Public evaluation uses `--run`, its own case and the exact recorded frozen terminal
policy. The command runner checks checkpoint role/step/hash/size and own successful
training before evaluator/model construction. Initial terminal evaluation differs
from explicit development or frozen-YAML supplements; this routine graph adds
only authorized initial terminal requests.

Separate train/evaluation arrays use matching stable indices and Slurm
`aftercorr:<training-array-id>` so each evaluation waits for its own successful
training task. With `combined: true`, compatible approved stages share one
allocation and run as separate commands with separate outcomes. Group/task counts,
selected declared optimizer work and presentations are displayed independently.
The preview payload is byte-identical to the saved submission payload. Scheduler
IDs appear only after an actual adapter response; template IDs are placeholders.

Fresh submission/output and case reservations persist exact bundle/payload hashes
and command intent before the adapter is called. `SUBMISSION.json` retains each
group's arguments, stdout/stderr/exit, classification and actual accepted job ID.
Rejected, ambiguous, interrupted and partially accepted attempts retain their
reservations. Unknown acceptance blocks retry. Per-task durable command/launch
receipts retain public-command outcomes; none automatically releases the shared
source consumer. Reconciliation must establish durable terminal evidence for
**every** task before explicit registry terminal state/release. Queue age, an
output directory or one failed case does not establish this predicate.

## Explicit scheduler reconciliation and recovery

A stopped or lost submission response retains `SUBMISSION.json`, the exact
`bundle.json` and `job.sbatch` bytes, case stage fences and task reservations.
New intents include a unique `attempt_identity`, UTC `attempted_at`, exact
arguments and payload digest, persisted before the scheduler adapter runs.
Accepted IDs remain accepted. The reconciler performs no scheduler query or
submission; an operator supplies recorded scheduler evidence for the exact
intent. A similarly named job or a nearby timestamp alone is insufficient.

```bash
python scripts/submit_recipe.py intents --bundle submission/bundle.json
python scripts/submit_recipe.py reconcile --bundle submission/bundle.json --evidence scheduler-evidence.json
```

`reconcile_job(bundle, receipt_dir, *, evidence)` supports both saved bundle
schemas. `reconciliation_intents` returns the exact anchors to copy into the
evidence. Legacy intents with missing identity/time retain those missing values
and include the original submission SHA256; they require an exact authoritative
record and never invent a timestamp. Evidence has this shape (IDs here are fake):

```json
{
  "schema": "scheduler-reconciliation-evidence-v1",
  "bundle_identity": "exact bundle identity",
  "reference": "retained scheduler query/accounting receipt",
  "groups": [{
    "group": 0,
    "intent": {"attempt_identity": "exact retained value", "attempted_at": "exact retained value", "arguments": ["exact", "arguments"], "bundle_identity": "exact bundle identity", "group": 0, "payload_sha256": "exact payload digest", "submission_sha256": "exact original receipt digest"},
    "authoritative": true,
    "query_status": "complete",
    "matches": [{"job_id": "123", "intent": {"copy": "the complete exact intent above"}}]
  }],
  "tasks": [{"group": 0, "index": 0, "job_id": "123", "scheduler_task_id": "123_0", "authoritative": true, "query_status": "complete", "status": "completed", "reference": "retained task accounting evidence"}]
}
```

One authoritative exact match establishes acceptance. Multiple matches, failed
queries, delayed accounting and incomplete coverage remain `unknown`. Empty
matches establish `not-accepted` only with
`"nonacceptance":{"exhaustive":true,"final":true,"acceptance_window_closed":true}`:
the recorded evidence must prove the entire acceptance window closed, rather
than merely show the job absent from the current queue. An old rejection status
alone does not override this requirement. Groups with no persisted invocation
intent are `not-attempted` only when the submission is durably stopped. Every
accepted array index needs its own exact authoritative terminal accounting
record; a completed array parent or one task cannot establish completion of all
bound tasks. Scheduler terminal state establishes lifetime safety, not model or
scientific success.

Snapshots under `reconciliation/00000001.json`, etc., retain supplied evidence,
original receipt digest, all group/task dispositions and a digest link to the
previous snapshot. A file lock serializes actions and atomic synced writes keep
incomplete snapshots out of history. Repeating the latest exact evidence returns
the same snapshot. Acceptance and supported task terminal evidence persist across
later delayed queries. Original receipts and reservations are not overwritten.

Retry eligibility and operational authority are distinct. An unknown group blocks
all fresh current-case retry preparation. Accepted groups are never selected again. Prepare a
reviewable subset of only eligible groups, retaining accepted predecessor IDs:

```bash
python scripts/submit_recipe.py prepare-retry --bundle submission/bundle.json --output retry-bundle.json --retry-output /absolute/approved/root/retry-submission --attempt 0002
python scripts/submit_recipe.py authorize-retry --bundle submission/bundle.json --new-bundle retry-bundle.json --authorization retry-approval.json
```

`prepare_retry_bundle` changes only the operational output, attempt and selected
unaccepted groups. Evaluation-only retries retain the original training attempt
and recorded checkpoint policy. `authorize_retry_job` requires an approval with
`approved:true`, `operation:"authorize-recipe-retry"`, exact old `bundle_identity`,
`new_bundle_identity`, `source_commit`, new `output` and a nonempty
`user_authorization_reference`. It checks unchanged source/runtime, effective
science, resources, concurrency and dependencies. The action records approval
before retiring only eligible case stage fences into retained
`<stage>.retired-<old identity>.json` archives; task reservations remain intact.
Saved legacy retry metadata remains readable; new legacy retry preparation and
authorization reject before mutation. A new current-case submission still uses
the existing explicit exact `submit-recipe` approval. None of these recovery
commands submits automatically. Changed science/source/budgets use the existing
revision/case rules. Completed or accepted training is never automatically rerun.

When every bound task has durable terminal evidence, explicitly record the
consumer terminal state and optionally release it:

```bash
python scripts/submit_recipe.py terminal --bundle submission/bundle.json
python scripts/submit_recipe.py release --bundle submission/bundle.json
```

`finalize_source_consumer(..., release=False)` reuses `SourceConsumers` and retains
its registry history. `release=True` is the explicit release action; repeating it
is safe after interruption. Accepted tasks keep the source held until each task
has terminal accounting evidence. Definitively unaccepted or stopped unattempted
groups need no fabricated launch result, including groups after partial acceptance
and a reservation write failure before scheduler invocation. Reservations and
scientific output stay retained; no automatic cancellation or release is added.

Recovery approval also records immutable `recovery_claims` per original group and
`recovery_approvals` per child bundle. Once claimed, a group cannot fund a second
recovery from that original receipt, even under a new approval. Reconcile the
child's own stopped attempt to prepare its next recovery. Disjoint unclaimed
groups may receive separate exact approvals; later evidence retains all claims
and approvals. Inherited accepted dependencies remain fixed across successive
current-case recovery generations. Saved legacy claims, approvals and inherited
dependencies remain lifecycle evidence.

An exact authoritative task result contradicting retained terminal evidence
writes a durable conflict snapshot before reporting failure. Missing or failed
queries still retain supported prior evidence. Conflict snapshots block terminal,
release, retry preparation/approval and recovery submission. To resolve a task
conflict, supply complete corrected authoritative terminal evidence for each
conflicted task and `conflict_resolution` with `authoritative:true`, a retained
`reference` and the exact canonical `conflict_sha256` of the blocking snapshot.
Group acceptance/nonacceptance identity contradictions require a separate exact
lifecycle investigation and stay blocked. If the source was already released,
the conflict explicitly reports that historical release is immutable; it cannot
undo the provider's recorded release or promise retrospective protection.

The producer records exclusive stage-fence creation path, bundle/group/task
identity and inode/device before writing the stage JSON. A durably stopped,
never-invoked group may archive an incomplete fence only when this creation
proof matches its current file. Malformed bytes alone and another attempt's
empty fence never permit retirement. Approval/group claims precede individual
retirements, so interruption resumes only the same exact child recovery and
keeps archived bytes unchanged.
