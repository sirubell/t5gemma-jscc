# Versioned encoder-final baseline capability

This is an opt-in software capability. It does not change the adopted residual
enc_l9 baseline, authorize a scientific run, or establish numerical acceptance
on full-weight BF16 hardware. COCO and HellaSwag retain their ordinary
`train.py`/`evaluate.py` workflows, including decoder receiver-memory behavior.

The new learner uses online frozen-teacher/student forwards for full-vocabulary
K and complete padded sequence replay for local R. Its fixed objectives are
K + 0.1R and 0.1R. Local K remains unavailable, with a null value and reason.
Accumulation uses target-token K and valid-sequence R denominators over the
whole effective batch. Neither the teacher nor the backbone acquires gradients;
the student suffix stays in the autograd graph.

## Preview and prepared execution

```bash
uv run --offline --no-sync python scripts/encfn_baseline.py
uv run --offline --no-sync python scripts/encfn_baseline.py --output /tmp/baseline-plan.json
```

The initial plan contains four cells, 1,600 updates and 12 task assessments.
`--selected-cell D-none --selected-cell R-none --owner-approval DECISION_ID`
compiles the conditional 2,800-update/22-assessment plan. It does not execute it
or constitute owner approval. The YAML `configs/studies/encfn_baseline_v2.yaml`
is a protocol description, not an input to the legacy study expander.

A later accepted package can execute exactly one prepared segment:

```bash
uv run --offline --no-sync python scripts/encfn_baseline.py \
  --execute /prepared/arm.json --output /new/run-directory
```

Preparation must supply schema `prepared-encfn-baseline-v2`, a hash-bound resolved
`config` file, `cell`, `strategy`, `run_id`, `pairing_id`, and all 400 ordered
`updates`. Each update is a list of immutable batch references with `path`,
`sha256`, and `view_sha256` from `baseline_protocol.batch_identity`. The
`objective_validation` list binds exactly 128 sequences; `task_request` binds
the exact 256-item development panel and six conditions. Input files retain
native targets, source masks, padded layout and task/view/image metadata.
`data_ids` retains original membership. No iterator cycling or ID substitution
is performed. Scientific membership, exclusion and hardware acceptance are
separate preparation gates.

`source_inventory` covers every `jscc/*.py` recursively plus this runner,
`uv.lock` and `pyproject.toml`. `source_archive` is a retained ZIP reference
with path/hash; the runner compares every inventory byte with the archive and
executed source. `state_metadata` binds source/config/data/initialization/stream/
protocol/parent identities and explicit lineage. The initialization reference
is the immutable actual codec state dictionary, not a seed-only assertion.
Preparation must copy the same core tensor snapshot between each family's norm
variants. The runner checks initialization bytes and exact topology on load.
The comparison mapping explicitly binds architecture, initialization, training data,
objective, exposure and schedule control references; it never infers equivalence
from shape or cell names.

Conditional local segments additionally require role-keyed `replays` with v2
consumer requirements and an applicable parity receipt. `parent_checkpoint`
and `parent_expected` identify the exact reconstruction-prefix state at 200;
continuation carries optimizer, scheduler, scaler, RNG and ordered stream
state. The checkpoint also binds pairing ID, draw schema, PyTorch runtime, device
and dtype; changing the continuation noise policy is rejected before output. `reused_assessments` binds any physically deduplicated task observation
to its original validated checkpoint and receipt, with identical codec tensors
and source/config/protocol/initialization/noise provenance. Reuse additionally
requires exact full request identity, panel/scorer/noise/layout settings, all six
conditions and actual complete item/source-family counts. There is no automatic
conditional launch or automatic failed-job retry.

## APIs and artifacts

- `models.split_model.resolve_encoder_site` resolves post-final-norm `enc_fn`
  separately from raw last-block output. `SplitModel.at_site` moves one insertion
  with exception-safe restoration and keeps the codec object unchanged.
- `activation_replay.producer_spec_v2`, `capture_sequences`, and `open_replay`
  preserve complete padded sequences and verify retained source, shard bytes,
  roles, native prefixes, views, precision and applicable compatibility evidence.
  Functional cached suffix replay is rejected. Historical v1 equality is intact.
- `experiment_schedule.compile_baseline_plan` compiles finite physical segments.
  `NoiseKey` binds pairing purpose/site/local batch/view/layout/stream/condition;
  runtime AWGN receipts preserve realized draw digests and shapes. The CPU
  `paired_noise` helper is a separate explicit CPU draw schema, not a promise
  that another device/dtype has identical random values.
- `experiment_state.save_state`, `open_state`, `restore_state` implement immutable
  hash-bound `experiment-state-v2` / `single_site_v2` states. `branch_metadata`
  permits only the declared 200/400 reconstruction-prefix continuations.
  Legacy weight transfer and iterator-restart resume remain unchanged.
- `baseline_protocol.BaselineLearner.update/validate` perform actual tensor work;
  `run_baseline` executes a finite segment with initialization, quarter checkpoints,
  objective observations and mandatory task assessments. `evaluate_checkpoint`
  uses existing COCO/HellaSwag adapters and gates site/identity before weight load.

Runs retain `run.json`, `metrics.jsonl`, immutable `step_*.pt`,
`checkpoints.json`, per-condition objective JSON, draw receipts, task evaluation
artifacts and `completion.json`. Any nonfinite, skipped optimizer call, logging
failure, invalid save or missing mandatory assessment stops that learner.
Attempted exposure remains charged; no extra replacement update is scheduled.
The zero-LR first AdamW call still advances moments and completed updates.

The learner emits plain `experiment-records-v1` update, failure, objective, cost
and per-site footprint mappings through
an optional callback. `evaluation.observation_event_payload` projects executable
requests into the agreed `codec-observation-v1` identity and plain observation
mapping, preserving the full original request. Final durable records/plots and
completion-inventory integration belongs to the CPU integration stage. CPU
receipts and a normal exit cannot substitute for that gate.

## Verification boundaries

Focused tests exercise tiny real Transformers routes, disk replay parity,
online gradients and unequal microbatch reductions, finite schedules, corrupt
and partial saves, strict old source guards, six-condition evidence and full
continuous-versus-restored optimizer trajectories. The runner is also exercised
with a synthetic CPU model and mandatory assessment callback. Those tests do
not download data, verify real source-family exclusion, certify BF16 behavior,
measure GPU fit or establish downstream scientific quality.

The prepared local consumer additionally binds its parity receipt to the current
executed learner source, the intended optimization versus objective-validation
capability, the exact resolved backbone/site/precision and each prepared view
digest. Evaluation reopens immutable checkpoint bytes and rejects a mutated
in-memory payload before it loads codec weights.
