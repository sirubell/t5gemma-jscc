# Original13 training integration contracts

This extends the retained six-encoder A/E/R implementation with a separate,
versioned original13 plan. It does not change legacy `EncoderStagesPlan`.
The M1 interfaces include deliberately failing consumer stubs; this revision
cannot train or score an original13 model until those consumers are integrated.

## Plan and input binding

`Original13Plan` in `jscc/encoder_stages_plan.py` binds ordered complete native16
views, input and full development-panel identities, width and reconstruction
percentage. Production uses 10000 views, 13 ordered sites and 130000 updates.
Ratios are 50, 75 and 90; widths are 2304, 1152 and 512. `resolved()` is the
versioned plan schema; `from_resolved()` rejects changed or unsupported contracts.
`update_at(index)` supplies the actual site/view/noise key and phase-local LR.
`objective_at(index)` supplies the resolved KL/hidden/memory coefficients.
The routing consumer must use those coefficients in forward, backward and records.

Each phase uses AdamW peak LR 2e-4, weight decay 0.002500001398541448 and cosine
with zero minimum and `max(1, phase_updates // 20)` warmup. The first update in
each phase consumes zero LR; reaching the phase horizon sets the next LR to zero.
There is one weights-retaining Adam/LR reset. The runner owns its construction.

Checkpoint states include initialization, four quarter saves in each phase,
separate pre/post-reset files, and functional10%: 11 files per uninterrupted
trajectory. Only pre-reset and functional10% request observations. There is no
extra terminal copy or automatic pruning. Saves occur at complete site cycles.

`bind_original13_inputs` verifies retained tensor-file/view hashes and native
row order one batch at a time, fresh initializer hashes and the inherited recipe.
It binds the exact existing256-question object, including source families, with
canonical identity `d83b0856429b508910fab8a42453490b9d4bce3c52e78e17291b64046116980f`.
The object's historical panel digest is retained as a separate field.
Preparation produces nine resolved plans, width configs, observation/terminal
requests and source/input identities. It loads no backbone and launches nothing.

```bash
uv run --locked python scripts/original13_training.py bind \
  --inputs /absolute/path/to/inputs/prepared.json \
  --inputs-sha256 INPUT_FILE_SHA256 --prepared /absolute/new-prepared.json
uv run --locked python scripts/original13_training.py check \
  --prepared /absolute/new-prepared.json --prepared-sha256 PREPARED_FILE_SHA256
```

The training CLI owns `bind` and `check` at M1; model construction and the
train/resume command surface remain pending integration. The existing
`scripts/encoder_stages.py` retains the legacy A/E/R CLI.

## Routing consumer API

`jscc/original13_routing.py` is routing-owned. Its stubs fail explicitly.

- `create_original13_learner(model, *, plan, metadata, event_sink)` returns a
  learner with the documented counters, exposure ledger, canonical parameters,
  optimizer/scheduler/scaler and `update([batch], batch_identities=[binding])`.
  Update consumes the scheduled LR, increments scheduler/progress/exposure once,
  persists an actual update event and stops on nonfinite/skipped work. Reuse the
  existing numerical-update and bound-batch mechanics.
- `canonical_codec(model)` returns the only communication parameter owner.
- `export_codec(model)` and `load_codec(model, weights)` export/load its flat
  strict body, verifying shared aliases; no frozen backbone or duplicate roles.
- `observation_state(model)` exports detached serializable idle route/noise,
  mode, mask and channel-counter state. `restore_runtime_state(model, state)`
  restores it during continuation and rejects unsupported active states.

The parameter names in the export must match the canonical module's names.
The envelope checks this at save/load; the routing consumer owns model alias
and role-stream correctness. The learner factory receives the initial model;
`Original13TrainingState` installs the authoritative optimizer and scheduler.

Metadata supplied by preparation/model loading must include source, config,
data, initialization, stream, protocol, prepared and model-revision identities.
`stream_identity` equals the input manifest hash. Source identity hashes the
source inventory; the release receipt separately binds the full Git commit.

## Envelope, continuation and observations

`Original13TrainingState` in `jscc/encoder_stages.py` owns save/restore/reset.
`experiment_state.py` retains schema `experiment-state-v2`, adding kind
`original13_v1`; legacy kinds are unchanged. The body uses routing export/load
callbacks. The envelope saves full optimizer/scheduler/scaler, Python/NumPy/
Torch/CUDA RNG, exact next site/view, phase/reset parent, valid exposure receipts,
route/noise state and prior segment ancestry. Wrong source, plan, input,
partial state, nonfinite weights/moments and incompatible counters are rejected.
Supported continuation is within the same backend; cross-backend bitwise
identity is not promised.

`run_original13_training(model, processor, *, plan, batches, metadata,
observation_request, output, guard, routing=None, observe=None, resume=None)`
runs into a fresh output directory. `resume` is an explicit `CheckpointRef`,
not a best/latest search. A segment logs only its newly completed updates and
retains its parent checkpoint and ancestry; global exposure is restored once.
Pre-reset continuation resets before the next update. An observation at or
before the resumed step stays pending for explicit prior-segment reconciliation;
it is not silently retried. Resume of an already terminal state is rejected.

The evaluation consumer owns `jscc/encoder_stages_evaluation.py` and
`scripts/original13_evaluation.py`. Legacy terminal open/evaluate functions were
moved to that module and re-exported from the training module.

`observe_original13(model, processor, *, state, request, output, guard)` receives
the exact saved state/request and returns a receipt with `status='complete'`
and `request_identity=canonical_digest(request)`. It isolates/restores routes,
private noise, mode and counters. It may raise `ObservationScoringError` only
for isolated scoring/output failure after restoration. Other exceptions are
fatal. The runner restores global RNG and checks the checkpoint plus canonical
weights, optimizer, scheduler/scaler, gradients, counters, exposure and route
state before continuing a scoring failure. No automatic retry is permitted.
Failed required observations retain pending completion even when training freezes.

The preparation request binds enc_l9/enc_fn/dec_l0/dec_l24, no_noise/-6 and256
questions. Initial terminal requests bind13 sites × two conditions and10042
questions, five-shot/candidate16/FP32 scoring. Optional0/+6/+18 support remains
separate. Real scorer/loader acceptance belongs to the consumer and final join.

CPU tests with a mechanical routing adapter prove plan/state/runner behavior,
including actual optimizer LR consumption and continuation. They do not prove
real encoder/decoder routes, gradients, aliasing, scorer behavior or full-weight
hardware fit. Final integration requires those real consumer implementations.
