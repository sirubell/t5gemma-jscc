# Fixed source preview and consumer lifetime

Existing recipe submissions bind a clean ordinary checkout/worktree at a full
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

Use concrete absolute paths under the existing approved host root. Preview prints
the full source/runtime binding, effective configs and their digests, selected
paths, per-task GPU/CPU/RAM/walltime, stable trajectory indices, per-array caps,
dependencies, task counts, and separate per-location/case/comparison update and
presentation budgets. `selected_rows_request` is a selection request, not an
observed eligible-row count. No scheduler is invoked. The payload equals the one
used by `submit`; dependency ID `1` in command templates is a labeled placeholder
for an earlier group's actual receipt. This preview covers existing recipe job
inputs; general task/case arrays and evaluation pairing are downstream.

Bound effective settings come from `bundle.json`, so later authored YAML edits
cannot change a queued case. Editing the bundle or prepared/config bytes fails
identity checks. Changing science/source/budget requires a new reviewed binding
and an explicitly linked case/revision; a rejected check is not an exact retry.
Old bundles without `source_binding` need fresh preparation, not silent migration.

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

`run_job` writes an exclusive per-trajectory preflight marker. Wrong/dirty source,
changed runtime/config, or loaded helper checkout mismatch records a failed
marker before GPU/model execution. Duplicate invocation cannot overwrite it. The CLI also retains JSON/identity
validation failures beside the trusted bundle path without using a mutated
bundle's output path.
A failed trajectory never releases a shared submission: other rows may still be
queued/running. The owning submission session must reconcile every task before
recording terminal state and release. Neither a CLI error nor one task's exit
proves that the whole consumer is terminal.

Supported held states are `reserved`, `queued`, `running`, `unresolved`.
Terminal states are `completed`, `failed`, `cancelled`, `rejected`,
`preflight-failed`; each requires a durable evidence reference and the same owner.
For a grouped consumer, a terminal state asserts **all** its tasks are terminal.
Active state updates cannot regress running/uncertain state to queued/reserved.
A lookup failure, missing response or ambiguous scheduler match stays unresolved.
Scheduler reconciliation, terminal receipt aggregation and general arrays belong
to the downstream submission owner. That owner must explicitly record terminal
state and release; this module never infers completion from age, source checks,
output existence or one accepted/rejected job. Crashes between receipt/registry
writes require inspecting both durable records; default to retaining the hold.

Offline Git fixtures and mock scheduler results establish these software
contracts only. Real scheduler acceptance, target runtime/GPU fit, acquisition,
full-weight performance and scientific outcomes need their own authorized checks.
