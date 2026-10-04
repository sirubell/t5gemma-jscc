# Shared codec recipe contracts

The YAML resolver prepares complete execution descriptions. It does not launch
training. The retained original13 training/evaluation commands remain callable;
`original13_training.py --recipe path.yaml` additionally verifies that YAML
matches the retained prepared execution before training/resume. Changed semantics
fail with a clear integration requirement;
general recipe execution is connected by the downstream training, data, state,
observation and submission deliveries before final integration.

```bash
uv run --locked python scripts/experiment_recipe.py \
  --config configs/recipes/original13.yaml \
  --model-config /existing/pinned/backbone/config.json \
  --output /fresh/output/resolved.json
```

Use the `config.json` from the selected locally cached backbone. Resolution reads
its encoder/decoder layer counts and hidden dimensions without downloading or
loading weights. A revision recorded in metadata must match the task model pin.
The runtime loader must validate locations against the loaded backbone again;
a JSON file alone does not establish loaded runtime identity. Output is exclusive
creation: choose a fresh file for each resolution. References in recipe YAML are
relative to that YAML; references in its task YAML retain the existing resolver's
semantics. The resolved output contains the composed model, task, data, optimizer,
channel, schedule, save states, scoring requests and all derived counts.

`updates_per_location` is authoritative. For N ordered locations and Q updates
per location, native16/accumulation1 produces N×Q optimizer updates, Q×16 reused
base-stream presentations and N×Q×16 site presentations. Dataset selection counts
are independent. Every complete view traverses locations in order before the
stream advances. Repeated data content is allowed. Unsupported batch semantics,
independent declarations of derived budgets, invalid/duplicate boundaries, empty
selections and fractional phase/save quotas fail explicitly.

The retained example resolves nine independent trajectories with 13 locations,
Q=10,000, widths 2304/1152/512 and reconstruction percentages 50/75/90. Each has
130,000 updates, 160,000 base-stream presentations and 2,080,000 site presentations.
Quarter-phase states plus initialization, post-reset and the functional 10%
observation give 11 checkpoint states. Pre/post reset remain distinct at one
update. Two 256-question observations preserve the chosen four sites and NN/−6
conditions. Encoder hidden nMSE is .1; decoder hidden/memory are .05 each;
functional training adds KL1.0. These values are example settings, not validator
literals. Generalized consumers must honor the resolved coefficients.

## Callable boundaries

- `jscc.experiment_recipe.resolve_recipe(path, model_config_path=...)` returns
  a versioned mapping with `trajectories`, `model_topology`, `matrix_counts`,
  `identity` and `launch_authorized: false`. Each trajectory binds a complete
  `config`, ordered `locations`, `updates_per_location`, width/ratio, phases,
  objectives, counts, checkpoint states, observation and evaluation requests.
  Its identity covers those settings; routine YAML edits need no schema bump.
- `jscc.recipe_plan.RecipePlan(description, views, study_pairing_id)` verifies
  the resolved trajectory identity at setup, copies settings and binds exactly Q
  immutable `BatchView` records supplied by the input owner. `update_at(index)`
  returns `ScheduledUpdate` with site, indexed view, phase-local LR and retained
  `original13-role-awgn-v1` noise key. `objective_at(index)` resolves role weights;
  `progress(completed, phase_start)` returns next location/view and exposure
  counters, rejecting progress across an unrecorded reset. This scheduling API
  does not construct a model, optimizer or checkpoint.
- Data owner: `jscc.original13_inputs.load_original13_prepared(path, checksum=...,
  width=..., ratio=..., load_batches=True, allow_pending_assets=False)` retains
  the existing `(manifest, config, plan, batches, initial, metadata, data_ids)`
  return. `initialize_original13(config, initial, evaluation=False, synthetic=False)`
  returns `(processor, model)` using local assets. General data delivery must be
  indexed/seekable: view index is `completed_updates // N`; consuming an
  observation must not advance training. `BatchView` carries view/mask/layout
  identities and sequence/token counts; real batch tensors must match it.
- Provenance owner: `jscc.original13_provenance` owns `stages_source_inventory`,
  `model_asset_inventory`, `bind_model_assets`, `verify_model_assets`,
  `original13_execution_guard` and `verify_original13_execution_source`.
  Integrity/pinning policies can change here without modifying tensor generation
  or checkpoint serialization. The current inventory covers every `jscc` module.
- State owner: `jscc.original13_state.Original13TrainingState(learner, plan=...,
  metadata=..., routing=...)` supplies `save(path) -> ValidatedState`,
  `restore(CheckpointRef) -> ValidatedState`, and `reset(pre_reset_state)`.
  `validate_original13_state(payload)` remains the checkpoint trust-boundary
  validator. `jscc.experiment_state` owns byte references, serialization, full
  optimizer/scheduler/scaler/RNG restore. `jscc.original13_training.run_original13_training`
  retains explicit `routing`, `observe`, `resume`, `config`, `data_ids`, `guard`
  hooks and campaign/freeze receipts. Observations run only at recorded states;
  scoring failures continue only after verified restoration. Generalized state
  validation must be released by this owner before consumers adopt new plans.
- Observation owner: `jscc.recipe_observation_policy.checkpoint_policy` resolves
  integral per-location save fractions into physical update/state pairs.
  `jscc.original13_scoring` owns runtime isolation, scoring policy and panel
  scoring; `jscc.encoder_stages_evaluation` owns requests, terminal selection,
  orchestration, receipts and reporting. `_score_original13_panel` is retained as
  a legacy orchestration injection point; a future API change needs handover.
- Submission owner: `scripts/experiment_recipe.py.main(argv)` is the preparation
  entry point; `scripts/original13_training.py.main(argv)` retains explicit
  bind/check/train/resume operation and exact source/input/checkpoint arguments.
  Submission must consume the resolved JSON and bind the exact accepted runtime
  source, input/model assets, approved resource limits and fresh receipt location.
  This foundation adds no scheduler call, deployment or resubmission behavior.

## Downstream file ownership

This manifest allocates whole files so dependency-ready workers can proceed in
isolated checkouts from the reviewed foundation commit. Shared-file changes
require explicit release. Legacy facade files may only be changed by integration
unless listed below. Tests follow the same owner as the behavior they check.

| Delivery | Owned runtime files | Owned tests |
|---|---|---|
| YAML training | `jscc/original13_routing.py`, `jscc/recipe_plan.py`, `jscc/models/split_model.py`, `jscc/encoder_online_reconstruction.py` | `test_original13_routes`, `test_original13_objectives`, new `test_recipe_training` |
| Cached data | `jscc/original13_inputs.py`, `jscc/encoder_stages_preparation.py`, `jscc/data/hellaswag.py`, `jscc/data/hellaswag_prompts.py`, `jscc/presentation.py` | `test_encoder_stages_preparation`, data/presentation tests, new `test_recipe_data` |
| Hash costs | `jscc/original13_provenance.py`, `jscc/baseline_protocol.py`, `jscc/sharing_protocol.py` | `test_baseline_protocol`, `test_sharing_protocol`, new `test_recipe_provenance` |
| Optimizer/resume | `jscc/original13_state.py`, `jscc/original13_training.py`, `jscc/experiment_state.py` | `test_original13_plan_state`, `test_experiment_state`, new `test_recipe_resume` |
| Checkpoints/observations | `jscc/recipe_observation_policy.py`, `jscc/original13_scoring.py`, `jscc/encoder_stages_evaluation.py` | `test_original13_observations`, `test_original13_evaluation`, new `test_recipe_observations` |
| Submission | `scripts/experiment_recipe.py`, `scripts/original13_training.py`, `scripts/original13_evaluation.py`, new `jscc/recipe_submission.py`, `scripts/submit_recipe.py` | new `test_recipe_submission` |
| Final integration | `jscc/experiment_recipe.py`, `jscc/encoder_stages.py`, `jscc/encoder_stages_plan.py`, `jscc/config.py`, `configs/recipes/`, this document | `test_experiment_recipe`, `test_original13_joined` |

The state owner releases training runner observation hooks to the observation
owner only after its own accepted delivery; whole-file ownership otherwise stays
with state until integration. The recipe resolver/config/legacy plan stay with
integration; downstream proposals for their changes are recorded in delivery
receipts, not concurrent edits. Navigation files belong to the navigation ticket.
No owner changes the frozen original13 experiment branch, packages or campaign.
Source commit and local evidence locations are recorded in the canonical recipe
contract ticket; ignored local receipts do not travel in Git.
