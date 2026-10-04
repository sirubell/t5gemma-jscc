# Ordinary tasks at explicit experiment roots

Both supported tasks accept an exact fresh output root:

```bash
uv run --locked python train.py --config configs/tasks/coco.yaml --output runs/coco-case
uv run --locked python evaluate.py --run runs/coco-case
uv run --locked python train.py --config configs/tasks/hellaswag.yaml --output runs/hellaswag-case
uv run --locked python evaluate.py --run runs/hellaswag-case
```

These commands execute training/evaluation and require the applicable experiment authorization. `--check` still prints the composed task/model YAML without creating outputs or loading a model. The explicit output replaces `run.output_dir` in the bound effective configuration. Task/model-design references remain relative to the declaring YAML. Task objectives, channel support, CUDA/Colab device behavior and scoring adapters retain their existing policies.

A root is reserved atomically before model construction. An existing path fails, even after an interrupted attempt. There is no automatic retry, overwrite or checkpoint copy. Training saves its authoritative artifacts in `attempts/training-0001/`; the root retains their links. The singleton layout is:

```text
runs/coco-case/
  experiment.json              # source, runtime, lock, logical root and context
  case.json                    # effective identity, authored texts/references, budget/input declarations
  config.yaml                  # complete effective task/model settings
  training.json                # exact training-attempt link, created before execution
  checkpoint.json              # eligible default checkpoint path/role/step/hash/size
  attempts/training-0001/
    attempt.json               # independent outcome and retained failure details
    config.yaml, data_ids.json, run.json, completion.json
    best.pt, last.pt, step_*.pt, metrics.jsonl, ...
  evaluations/evaluation-0001/
    attempt.json, config.yaml, results.json, ...
```

All small JSON writes use the existing atomic durable writer. A crash can leave a reservation with a `running` outcome; this is unresolved evidence and never permits automatic retry. Case binding can leave an unfinished reservation if preparation fails. Retain it and diagnose the failure. Later edits to authored task/model YAML have no effect: evaluation reads the hashed complete effective configuration and the checkpoint's saved config, requiring them to match. Changing the bound configuration itself fails.

Default evaluation uses the recorded `best.pt` required by ordinary selection, with its exact saved step and digest. It never searches for newest or falls back to `last.pt`. A training time limit or an incomplete fixed-presentation budget has a partial outcome and no default target. Missing, changed or mismatched checkpoints fail before model construction. Fixed-presentation ordinary training retains its full-budget gate. Existing fixed-step callers remain supported through explicit checkpoint selection:

```bash
uv run --locked python evaluate.py --run runs/coco-case \
  --checkpoint step_000500.pt --expected-step 500 --attempt step500
```

Explicit checkpoint selections and evaluation overrides are supplemental requests. A supported partial attempt can be evaluated by an explicit checkpoint filename; it cannot become the default result. Overrides stay restricted to the existing evaluation settings and are frozen within that evaluation attempt. Each evaluation has its own outcome; completed condition results survive a later scoring failure. A second evaluation needs an explicit fresh `--attempt` identifier, while the default `0001` reservation prevents an accidental repeated invocation. Training/evaluation/collection records are separate, so failed collection cannot alter successful training or evaluation.

The new ordinary root route rejects `--resume` until exact data-position restoration is released by its continuation owner. Calls without `--output` retain the legacy timestamped run path and historical resume behavior; those iterator restarts are not exact continuation. Recipe dispatch, automatic assets, scientific multi-case execution, scheduler/source preflight and collection commands belong to their respective downstream integrations.

## Callable record/API release

`jscc.ordinary_runs` exposes these interfaces. Paths returned by reservation/binding are absolute. Identifiers are plain names without separators. The JSON schema marker is `experiment-run-v1`.

| Callable | Contract |
|---|---|
| `reserve_root(output, *, kind="ordinary-task", context=None)` | Exclusive fresh logical root; saves source/runtime/lock identity. No reused root or inferred retry. |
| `bind_case(root, config, *, authored_config=None, case_id=None)` | Exclusive singleton binding at root, or fresh `cases/<case_id>` child. Saves complete config, authored/reference text and identity, declared model/data/initializer settings and update/batch/accumulation budget. Does not execute a case. |
| `bound_config(case)` | Reads complete saved config after checking its bound hash/schema. Does not read authored YAML. |
| `reserve_attempt(case, stage, *, attempt_id="0001", request=None)` | Exclusive named stage attempt; `training`, `evaluation` or `collection`; persists request/source/runtime before engine invocation. Returns `evaluations/evaluation-<id>` or `attempts/<stage>-<id>`. |
| `validate_reserved_attempt(output, stage)` | Requires running matching-stage reservation and no engine config output yet; atomically claims `.engine-started` and returns the bound config. Engines use this before setup to reject repeated/concurrent execution. |
| `finish_attempt(output, status, *, details=None)` | Atomic final outcome: `completed`, `failed`, `interrupted`, `partial`. An already finalized attempt cannot be rewritten. |
| `train_task(config, output, *, authored_config=None, resume=None)` | Fresh ordinary singleton; returns the logical root. Saves training link/outcome and publishes an eligible exact default target. Rejects unreleased exact resume. |
| `evaluate_run(root, checkpoint_name=None, overrides_path=None, expected_step=None, *, attempt_id="0001")` | Consumes an ordinary bound case; returns exact evaluation attempt. Default target verified through one open checkpoint descriptor; explicit checkpoint/override retained as supplemental. |
| `is_recorded_root(path)` | Detects root/case markers for public CLI dispatch. |

The existing engines add keyword-only `run_directory` (and evaluation `expected_config`/`expected_checkpoint`) for a caller-owned reserved attempt; they validate the reservation/config before setup. Ordinary data membership/prompt evidence stays authoritative in `data_ids.json`/`prompt_evidence.json`. `run.json` adds setup identity with actual model/processor metadata, selected row count when available and communication-state tensor shape/dtype/digests (including scalar state); non-tensor custom-module extra state is explicitly marked unavailable in the small identity summary and remains authoritative in the checkpoint; it does not hash immutable full backbone weights. Completion records retain actual optimizer updates/presentations, separately from declared budgets. Unavailable metadata is `null`, not guessed.

Downstream owners consume these commits before editing shared CLI/engine files. Recipe consumers may reuse root/case/attempt reservations, but own their scientific input/count/checkpoint contracts and execution policy. Continuation must replace the fresh-only gate only after releasing durable cursor/state compatibility and tests. Fixed-source launch checks and lifetime coordination remain with their owner; the recorded source/runtime alone is provenance, not a launch authorization or source-lifetime guarantee.
