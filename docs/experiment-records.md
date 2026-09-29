# Durable experiment records

`jscc.experiment_records` validates and stores local evidence using only Python's
standard library. It never imports a model, tensor loader, GPU runtime or online
tracker. This is an explicit new contract, `experiment-records-v1`; historical
`metrics.jsonl` files are not silently interpreted or rewritten as this schema.
The helper's CPU tests establish evidence mechanics. The versioned baseline runner integrates actual learner/evaluator production,
artifact reconciliation and completion. Full-weight telemetry, GPU timing accuracy
and scientific quality still require their own acceptance receipts.

## Events and measurements

Every event has exactly `schema`, `event_type`, `run_id`, `phase_id`, `task`,
`timestamp`, `units`, `identity`, and `payload`. `task` is `coco` or `hellaswag`.
`identity` has exact source, resolved configuration, data and parent references:
`{source, config, data, parent}`. Parent may be null for an initial phase. These
references must resolve to the retained source/configuration/data/parent evidence
in the experiment's inventory; a descriptive recipe name alone does not preserve
source bytes. `units` declares units explicitly and `timestamp` records acquisition
time. Unknown fields and unsupported schema versions are rejected.

A scalar measurement is `{ "value": 0, "reason": null }` when observed, or
`{ "value": null, "reason": "not measured on CPU" }` when unavailable. Zero is
an actual observation, not a placeholder. Nonfinite numbers are prohibited in
JSON; a failed/nonfinite attempt records unavailable measurements with reasons
and the explicit failure flag. A genuinely zero-LR optimizer call is not skipped.

An `update` payload requires:

| Group | Required values |
| --- | --- |
| Axes | `site_id`, `attempted_step`, `completed_step`, `final_step`, `phase_start_step`, `site_step`, `sweep` |
| `exposure` | `source_tokens`, `target_tokens`, `sequences`, `padded_tokens`, `valid_tokens`, `cumulative_valid_tokens`, `latent_width`, `hidden_coordinates`, `hidden_coordinates_allocated`, `memory_coordinates`, `memory_coordinates_allocated` |
| `objective` | `kind` (`local` or `combined`), measured `total`, `components` keyed by `K` and `R` |
| Optimizer | `lr_used`, `lr_next` arrays by parameter group; `gradient` with measured `pre_clip`, `post_clip`, `clip_threshold`, boolean `clipped`; booleans `nonfinite`, `skipped`; measured `update_l2` |
| `snr` | `condition`, measured `min`, `mean`, `max`, exact `draw_ref` |
| `timing` | measured `step_seconds`, `elapsed_seconds`, `data_cache_seconds`, `throughput`; `throughput_denominator`, `scope` |
| `memory` | measured `allocated_bytes`, `reserved_bytes`; `reset_scope` |

Exposure counters are nonnegative integers except the two memory-coordinate fields, which
are measurements so an absent receiver-memory stream remains unavailable with
`not_applicable`. Hidden real-coordinate counts are not bits or Eb/N0. Global
update axes do not reset at a staged phase boundary; `phase_start_step` records
the offset. `site_step` and `sweep` allow a shared learner's global update to be
compared at matched site exposure with a specialist. Cumulative token exposure
is global and includes attempted work. Successful updates advance completed
step; failed/skipped attempts do not. The caller must stop the affected arm on
these events; the helper marks their completion incomplete.

Each objective component contains measured `raw`, `weighted`, `numerator` and
an integer `denominator`. Measured raw values must equal numerator/denominator.
Successful combined objective is `K + 0.1 R`; successful local objective is
`0.1 R`, with all K measurements unavailable and denominator zero. Objective
curves must retain their kind rather than comparing local and combined totals.
Update-L2 is required on the first update of a phase, every global tenth update,
and its final update; intermediate values are null with a sampling reason.
Timing scopes must describe asynchronous boundaries, cache/data work and the
throughput denominator. Memory scopes describe the high-water reset boundary;
unavailable GPU memory is not zero occupancy.

A `cost` payload contains `category`, measured `seconds`, `attribution`
(`first_use` or `reuse`), `site_id`, `scope`, `concurrency` and `device_count`.
Separate setup, acquisition, training, validation, task scoring, I/O, transfer,
failed attempts and allocation scopes rather than silently aggregating them.
A `footprint` payload records parameter/storage counts separately from runtime:
`scope` (`per_site`, `bank_total` or `shared`), `site_id`, measured
`parameter_count`, measured `checkpoint_bytes`, and `checkpoint_refs` (artifact
paths). Available counts are nonnegative integers; unavailable counts retain
reasons. Measured checkpoint bytes require unique nonempty references and must
reconcile with the referenced tensor sizes in the retained or omitted inventory. This is unique checkpoint-content footprint: references aliasing identical SHA-256/byte-count content make completion incomplete rather than double-counting aliases.
Omitted tensor sizes remain metadata assertions, not verified local bytes. A
parameter count is a producer assertion bound to the event identity; it does not
prove device memory usage. Bank totals require an explicit producer scope.
Missing/non-tensor references or byte-count disagreement make completion
incomplete. Retained and omitted artifact paths must be globally unique.

A `failure` payload contains a nonempty `reason` and retained evidence `reference`.

## Observation identity and reuse

`observation_identity(request)` hashes canonical JSON with the distinct
`codec-observation-v1` namespace. It does not change the existing evaluation
comparability identity. Its exact request fields are:

```text
source config data parent checkpoint panel site role task protocol noise scorer
layout precision backend conditions expected_items learner_kind step site_step
purpose objective_kind comparison details
```

`purpose` is `task` or `objective`. `objective_kind` is null for task observations,
and `local` or `combined` for objective observations. Complete objective panels
require measured R with weight 0.1 and, for combined panels, measured K with
weight 1. Local K stays unavailable with denominator zero. An all-unavailable
panel cannot satisfy completion.

`comparison` requires immutable control references for `architecture`,
`initialization`, `training_data`, `objective`, `exposure` and `schedule`. The
full observation digest includes these controls. Shared/specialist comparisons
match these controls even when full run configurations differ. Vanilla may use
explicit `not_applicable` references and remains a distinct named contrast.
Integration derives the references from accepted controls; this helper validates
the required mapping and preserves it, without certifying scientific matching. `role` is `trained`, `heldout` or `diagnostic`.
`learner_kind` is `shared`, `specialist`, `initialization` or `vanilla`. Held-out
specialists, including l14, are rejected. `details` requires `prompt`,
`native_policy`, `draws` and `outputs` references; retain exact manifests/hashes
for input construction, prefix/generation behavior, the frozen RNG/draw request and the predeclared output/reference policy. The `outputs` request field identifies that output contract, not hashes of future generated result bytes. Actual output hashes and realized draw receipts belong in the retained artifact inventory. This keeps expected observation identities computable before acquisition; never rewrite a scheduled identity after seeing results.
Requests bind both global `step` and matched-exposure `site_step`.

New codec observations have the exact ordered condition grid
`["no_noise", -6, 0, 6, 12, 18]`. Vanilla uses only `["vanilla"]` and the matching
learner kind. Vanilla bypass and no-noise codec execution have separate identities.

An observation payload contains `request`, its computed `identity`, `status`
(`complete`, `incomplete` or `failed`), `conditions`, and `reuse` (null or a receipt).
Each condition row contains `condition`, `requested`, `completed`, `failed`,
`failure_reason`, `denominator`, `metrics`, `items`, and `objectives`. The aggregate denominator is
exactly completed items; objective components retain their separate token/vector
numerators and denominators. Task rows require metric measurements. Objective
rows require K/R components. Items retain `item_id`, `source_id` and task-specific
scores, predictions, tokens or captions; source groups support paired analysis. HellaSwag task items require raw/normalized correctness, prediction, score and tokens; their aggregate raw/normalized accuracies must reconcile. COCO requires raw/clean captions, CIDEr, EOS/cap flags and tokens; aggregate CIDEr must reconcile with per-item scores. Empty conditions retain unavailable aggregates.
Unknown failed counts use null and a required nonempty `failure_reason`; known counts may retain an explicit reason. When failed counts are known, pending counts are represented by requested minus completed minus failed; no
pending or failed condition can be complete. Missing or duplicate conditions,
items, wrong denominators and changed identities fail validation.

A reuse receipt contains `identity`, `status: "complete"`, `reference`, and
measured `acquisition_seconds`. Reuse requires the full identical observation
identity and a complete current receipt. Comparison compatibility does not grant
result reuse. The prior acquisition remains attributable even when a later
consumer performs only cache reads.

## Atomic files, inventories and derived completion

Public APIs are `measure`, `validate_event`, `validate_observation`,
`observation_identity`, `canonical_bytes`, `append_event`, `read_events`,
`write_json_atomic`, `artifact_ref`, `verify_inventory`, and `completion_status`.
Validation raises `RecordError`; I/O failures propagate to stop the caller.

`append_event(path, event)` validates the existing log, writes the complete new
JSONL file to a temporary sibling, checks the write count, flushes/fsyncs, replaces
the destination, then fsyncs its directory. This favors clear atomic single-writer
semantics over unbounded-log throughput. Use one writer per run; concurrent
writers are unsupported. Partial files are retained on failure, no automatic
retry or repair occurs, and the old file remains unchanged before replacement.
A directory-fsync failure can occur after replacement: it is still a failed
operation whose crash durability is not certified. Do not resume that arm merely
because the destination is readable. A truncated existing log is rejected.

`artifact_ref(path, root, kind)` binds a relative path, `kind` (`metadata` or
`tensor`), exact `sha256` and `bytes`. It hashes bytes without loading tensors.
An inventory is:

```json
{
  "schema": "experiment-records-v1",
  "mode": "metadata-only",
  "artifacts": [],
  "omitted_tensors": []
}
```

Both lists contain artifact references. Explicitly inventory omitted tensors in
a metadata mirror; their bytes are unavailable there. Tensor-complete inventories
cannot omit tensors. Verification rejects changed bytes/size, missing files,
duplicate paths and paths escaping the inventory root. Hashes identify bytes;
they do not reconstruct missing checkpoint tensors. No copy or backup is made.

A per-run completion manifest has `schema`, `run_id`, `phases`,
`expected_observations` (full observation digests) and `expected_artifacts`
(relative paths). Each phase is `{phase_id, updates, start_step,
start_valid_tokens}`. For a second 200-update phase starting after 200 updates,
use `start_step: 200`; its final step is 400. Manifest phases are ordered: each later start step must
equal the preceding planned end, and its starting valid-token count must equal
the preceding starting count plus actual attempted token exposure. Overlaps,
resets, gaps and reversal are rejected. The first phase may explicitly start at
nonzero offsets for a partial baseline record; that does not certify omitted
prior phases. The schedule producer must expand
accepted quarter checkpoints/objective panels and midpoint/terminal task panels
into the expected identities and artifacts. The helper does not infer a scientific
protocol from a run name or invent panel sizes.

`completion_status(manifest, events, inventory, root)` verifies artifact bytes,
requires every declared completed update in order, rejects duplicate/sparse
cadence, checks cumulative token exposure, and compares every expected observation
and artifact against actual evidence. It returns complete/incomplete with missing
lists and reasons; it never trusts a caller's success flag. Use a separate manifest
for each run/arm even when phase names repeat. Incomplete independent arms do not
become a complete cohort. Metadata-only completion can describe the mirrored
records, but `tensor_bytes_verified` remains false. The tensor byte flag verifies retained bytes only, not a tensor loader, state compatibility or actual recovery. Restore validation requires separate execution evidence.

## Verification and integration boundary

Run focused checks from the repository root with the project's pinned environment:

```bash
uv run --offline --no-sync python -m pytest -q tests/test_experiment_records.py
uv run --offline --no-sync ruff check jscc/experiment_records.py tests/test_experiment_records.py
uv run --offline --no-sync pyright jscc/experiment_records.py tests/test_experiment_records.py
```

Tests inject short writes, file-fsync and replacement failures; reject truncated
logs, incompatible reuse, missing panels, unsupported fields and corrupt inventory
bytes; distinguish null, actual zero and failed updates; cover COCO and HellaSwag;
and confirm importing the helper loads no torch/Transformers modules. Actual
emission and complete accepted cadence must be confirmed by CPU integration.
No tests here establish full-weight telemetry or accelerator fit. Records remain
local, with no automatic upload, transfer, retention expiry or pruning.

## Render saved local evidence

The report command reads either one run directory or a campaign directory whose
immediate children are run directories. Each run supplies `metrics.jsonl`,
`inventory.json`, and `manifest.json` as specified above. An optional retained
`completion.json` must equal the recomputed completion. The renderer verifies
input artifacts before creating its output directory and refuses an existing
nonempty output directory.

```bash
uv run --offline --no-sync python scripts/render_experiment_report.py \
  --records runs/observed/example-campaign --output runs/observed/example-report
```

Open `index.html` in the output directory. PNG plots have companion CSV tables;
`analysis-settings.json` binds input hashes and bootstrap settings, and
`image-inventory.json` lists image hashes. Plot families cover objective components
against completed updates and source-valid tokens, LR used/next, gradients,
clipping, sampled update-L2, timing, memory, task/condition/site quality, paired
shared-versus-specialist differences, held-out initialization/vanilla comparisons,
worst observed trained-site results, and cost categories. Local and combined
objectives and different tasks remain separate. Missing values break lines;
partial condition results remain visible in tables but are not plotted as a
complete quality estimate. Phase transitions and initialization points are marked.

Paired HellaSwag comparisons match site-local steps and input/scoring identities,
not shared versus specialist global step numbers. Descriptive intervals use
10,000 paired source-group resamples, seed 0, row-weighted estimates, and the
2.5th/97.5th percentiles. The full item/source mapping must match. These intervals
condition on saved weights, development items and fixed draws; they do not measure
training-seed variability or correct multiplicity. COCO caption/scorer evidence
is retained and CIDEr curves are rendered; the renderer does not recompute CIDEr
or automatically select cross-arm scientific contrasts. No architecture is
promoted from these plots.

Costs remain separated by declared category and first-use/reuse attribution.
Overlapping command-wall and subphase categories are not added into a physical
total. `bank_total` must be an explicit audited producer/resource-ledger scope;
the renderer does not infer bank membership from filenames. Repeated cost rows
with the same scope are unavailable as a total until disjointness is established.
Only declared inventory completeness is checked here: integration owns expansion
of the accepted protocol into every expected save and measurement.

The committed synthetic fixture builder exercises both tasks, six conditions,
quarter objective observations, initialization/midpoint/terminal task observations,
shared global/site-local axes, two staged phases, omitted CPU memory measurements,
and a deliberately failed COCO observation:

```bash
uv run --offline --no-sync python tests/fixtures/experiment_records/build_fixture.py /tmp/records-fixture-new
uv run --offline --no-sync python scripts/render_experiment_report.py \
  --records /tmp/records-fixture-new --output /tmp/records-report-new
uv run --offline --no-sync python -m pytest -q tests/test_experiment_records.py tests/test_experiment_report.py
```

Use fresh paths so prior failures and receipts remain available. The fixture's
small counts are synthetic mechanics tests, not replacements for accepted
scientific budgets or evidence that actual learner runs emit this schema.

For baseline architecture/LN comparisons, provide an explicit `comparisons.json`
in the report input root. Each contrast binds the exact left/right full
observation digests, preventing filename-based arm selection:

```json
{
  "schema": "experiment-comparisons-v1",
  "contrasts": [{
    "id": "direct-minus-residual-terminal",
    "left_observation": "<full observation digest>",
    "right_observation": "<full observation digest>",
    "metrics": ["normalized_correct", "raw_correct"],
    "allowed_control_differences": ["architecture", "initialization"]
  }]
}
```

This is an analysis specification, not authorization to execute or promote an
arm. Only architecture, initialization and objective are supported explicit
control differences. Training data, exposure, schedule and common evaluation
input/scoring/noise controls remain matched. Requested missing, incomplete,
ambiguous or incompatible observations produce blocked-analysis evidence.
Contrasts use each matched condition separately; no condition average selects a
checkpoint. Retain this input's hash and analysis settings with the report.

Parameter counts and checkpoint bytes use the `footprint` event described above,
with explicit per-site/shared/bank scope. The report displays these separately
from timing costs, preserves unavailable reasons and labels whether referenced
checkpoint bytes are retained or only declared in omitted-tensor metadata. It
does not infer bank membership, sum overlapping inventory aliases, or treat a
smaller parameter bank as measured speedup. The fixture's eight-byte binary
artifacts are synthetic byte-verification inputs, not serialized model tensors.

## Connected baseline evidence

See [encoder-final baseline](encfn-baseline.md) for strict producer integration.
`tests/test_baseline_pipeline.py` exercises all four topologies through actual
CPU updates, save/reload, six-condition model forwards, records and rendering.
`tests/test_baseline_cadence.py` executes 1,000 tiny-model updates across complete
400/200+200 trajectories. `tests/test_baseline_comparisons.py` verifies four actual
short arms and 144 explicit condition/metric contrast rows. These are synthetic
mechanics tests, with no pretrained download or real-data scientific claim.
