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

The recorded ordinary route supports explicit exact continuation under the policy below. Calls without `--output` retain the legacy timestamped run path and historical iterator restart behavior. Recipe dispatch, automatic assets, scientific multi-case execution, scheduler/source preflight and collection commands belong to their respective owners.

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
| `train_task(config, output, *, authored_config=None, resume=None, attempt_id=None)` | Fresh ordinary singleton, or explicit continuation into a new retained attempt at the same bound case. Returns the logical root. Fresh defaults to `0001`; continuation defaults to `resume-<parent-id>-<saved-step>`, with exclusive reservation. An explicit callable `attempt_id` selects another fresh name. |
| `evaluate_run(root, checkpoint_name=None, overrides_path=None, expected_step=None, *, attempt_id="0001")` | Consumes an ordinary bound case; returns exact evaluation attempt. Default target verified through one open checkpoint descriptor; explicit checkpoint/override retained as supplemental. |
| `is_recorded_root(path)` | Detects root/case markers for public CLI dispatch. |

The existing engines add keyword-only `run_directory` (and evaluation `expected_config`/`expected_checkpoint`) for a caller-owned reserved attempt; they validate the reservation/config before setup. Ordinary data membership/prompt evidence stays authoritative in `data_ids.json`/`prompt_evidence.json`. `run.json` adds setup identity with actual model/processor metadata, selected row count when available and communication-state tensor shape/dtype/digests (including scalar state); non-tensor custom-module extra state is explicitly marked unavailable in the small identity summary and remains authoritative in the checkpoint; it does not hash immutable full backbone weights. Completion records retain actual optimizer updates/presentations, separately from declared budgets. Unavailable metadata is `null`, not guessed.

Downstream owners consume these commits before editing shared CLI/engine files. Recipe consumers may reuse root/case/attempt reservations, but own their scientific input/count/checkpoint contracts and execution policy. Continuation consumers must retain the durable cursor/state compatibility checks and focused tests below. Fixed-source launch checks and lifetime coordination remain with their owner; the recorded source/runtime alone is provenance, not a launch authorization or source-lifetime guarantee.


## Exact ordinary continuation

Set `data.num_workers: 0` in the task YAML **before starting** a resumable ordinary run. HellaSwag's supplied ordinary task already uses zero workers; the supplied COCO task uses four workers and needs this YAML change for exact continuation. Normal fresh training remains supported with multiple workers, fixed-presentation studies and other custom channels, but their metadata states the concrete unsupported-continuation reason. Their checkpoints do not qualify for this exact route. A previously created unsupported run cannot acquire a cursor by changing its YAML later.

```bash
# COCO or HellaSwag, using an unchanged complete effective task YAML:
uv run --locked python train.py --config my-coco.yaml --output runs/coco-case
uv run --locked python train.py --config my-coco.yaml --output runs/coco-case \
  --resume runs/coco-case/attempts/training-0001/last.pt

uv run --locked python train.py --config configs/tasks/hellaswag.yaml --output runs/hellaswag-case
uv run --locked python train.py --config configs/tasks/hellaswag.yaml --output runs/hellaswag-case \
  --resume runs/hellaswag-case/attempts/training-0001/step_001000.pt
```

These are separate explicitly requested operations, not experiment launch receipts. Continuation requires an existing bound case and a finalized parent training attempt (`interrupted`, `failed`, `partial` or `completed`), with an explicitly selected checkpoint from that case. A crashed attempt still marked `running` needs its outcome reconciled before continuation. There is no latest-file search, automatic retry, overwrite or budget extension. Changed scientific settings, source or runtime require a separate case/revision. `train_task(..., resume=path, attempt_id="0002")` provides an explicit fresh attempt name for callers; the public existing `--resume` interface uses the deterministic default name and rejects its reuse.

`ordinary-continuation-v1` checkpoints restore communication weights including memory codec/channel extra state, AdamW, scheduler, scaler, Python/NumPy/Torch/CUDA RNG, best-selection/early-stop state, cumulative presentation/token/channel counters and `ordinary-epoch-batches-v1` data cursor together. The cursor saves one realized epoch's ordered batch indices and the next consumed batch; no tokenized batches or per-update tensor packages are needed. Fresh shuffle draws and short final batches retain the prior PyTorch policy. A restored iterator consumes no extra training RNG draw; after its saved epoch finishes, the next epoch draws its ordinary fresh permutation. COCO's sampled target and HellaSwag's dynamic padding remain unchanged. Gradient accumulation restarts at the next optimizer update, with gradients cleared before processing its microbatches.

If a valid durable checkpoint is after update1000 and interruption occurs after unsaved1500, the new attempt restores1000 weights, optimizer, RNG, counters and data cursor and next performs1001. Later metrics and later `best.pt` state in the parent attempt cannot override the explicitly selected checkpoint's training state. Best-selection references preserve the best that existed at that checkpoint, even if a later unsaved prefix published a different best.

Each existing scheduled save publishes one immutable `checkpoint_<step>.pt` with a durable `.pt.json` digest/size/step/role receipt. `last.pt`, `best.pt` and existing `step_<step>.pt` names are hardlink aliases to that save; this retains checkpoint-time best identity without duplicating tensor bytes. Checkpoint writes use a temporary file, flush/fsync and atomic replacement. The receipt is published before aliases; a failed/partial write or unpublished file cannot qualify for continuation. Earlier attempts/checkpoints/results retain their bytes and path identities. No emergency save or new checkpoint cadence is introduced. Local filesystems must support hardlinks within the attempt directory.

Continuation checks full effective config, source revision/code digest, runtime/packages/lock/numerical policy, device/CUDA topology, ordered data membership/size/cache fingerprint and prompt evidence, model/tokenizer metadata, initial communication identity, custom channel source and optimizer parameter mapping. Pinned immutable large model/data bytes remain under the existing acquisition/receipt contract; continuation does not add a full-backbone or full-dataset hash pass. Missing required continuation identities and unsupported loaders fail rather than claim exactness. The supported loader is the ordinary map-style zero-worker, non-dropping RandomSampler/SequentialSampler route with default global RNG. Multi-worker/prefetch/persistent-worker state, custom samplers/generators and fixed-presentation studies are unsupported. Legacy checkpoints without a valid cursor/receipt are retained but cannot be upgraded implicitly.

Built-in AWGN/identity and the supplied deterministic attenuation example support exact continuation. Another custom channel may opt in with class attribute `supports_exact_continuation = True` only when all mutable runtime state is in `state_dict` (including JSON-compatible `get_extra_state`/`set_extra_state`) and randomness comes from the captured global generators. Private generators, external simulator state and mutable hidden attributes require their own complete supported state representation; absent the explicit declaration, continuation is rejected. Validation and feature observations restore communication/channel state, RNG and model mode even on failure; they do not advance the training simulator. Weight-only initialization/phase transfer remains a separate API and cannot be combined with exact continuation.

CPU tiny interrupted/uninterrupted tests cover both tasks, data cycles/short tails/accumulation, isolated validation/features, custom channel extra state and NumPy randomness, the actual1000/1500 scenario, checkpoint-time best ancestry and failed saves/mismatch rejection. These establish the supported local fixture contract. They do not establish GPU performance, cross-environment bitwise equality, full-weight scientific quality or historical checkpoint compatibility.
