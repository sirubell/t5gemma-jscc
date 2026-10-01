# Finite shared-codec pilot

The explicit `D-N` candidate uses the retained `direct_affine` implementation:
one biased Linear from D1152 to B and one biased Linear from B to D1152,
with identity outer boundaries, no residual blocks, no FiLM and no dropout.
The named model/task configurations prepare B512, B1152 and B2304. Preparing
a configuration does not establish hardware acceptance or authorize a run.

This path retains effective batch64 through physical batch16 with four
accumulated microbatches. It uses the same ordered 200 complete parent views
at encoder layers9,19 and final-normalized output. Each specialist makes
200 updates; the shared learner makes600 round-robin updates. Its learning-rate
horizon is600 and each specialist's is200. The finite SharingPlan owns these
horizons and quarter checkpoints, independently of ordinary training fields.
Native batch16 with800 updates per site is a different recipe and is rejected.

Within each width, all four learners start from the same fresh seed0 CPU codec
initialization with fresh optimizers. The specialists also supply the selected
architecture/site baseline comparison; they are not duplicated by another
baseline campaign. Existing short screens and unrelated trained checkpoints
do not substitute for these matched specialists. Cross-run reuse requires exact
recipe, initialization, source, ordered data/noise, exposure, optimizer/LR and
evaluation identities.

The lifecycle retains17 states,108 task condition panels on256 development
documents,192 objective condition panels on128 items, vanilla, freeze and the
post-freeze geometry observations. Layer14 remains excluded until trained-state
freeze. A full10,042-document evaluation is separate work, not the development
panel or evidence of convergence.

`jscc.sharing_binding.bind_production` creates a fresh immutable package from
the reviewed `sharing-production-inputs-v1` input manifest. The chosen model
snapshot path, width, recipe/pairing/decision identities and cap must be explicit.
`scripts/sharing_production.py --preview CONTRACT` validates and prints its
external GNU timeout command; `--controller` additionally requires exact-source
qualification/measurement, an existing task-7 reservation, live exclusive lease
and idle admission. Production contracts and owner proposals must bind the
same `finish_before_epoch`; a late allocation is rejected before its claim.

`scripts/sharing_qualification.py` supplies the separate bounded D-N diagnostic.
Its `--preflight` mode is CPU-only and its `--preview` mode launches nothing.
The controller/worker require actual target preflight, immutable three-width
campaign admission, task-7 holds and external process-group watchdog. Each width
executes24 physical optimizer calls at the three trained sites, including the
exact reviewed same-shape guard and stress shapes, three checkpoint perturb/
reload checks, and full development/objective panels. A diagnostic receipt
never declares the full production lifecycle qualified.

H200 and5090 can use the same16x4 scientific route and frozen source. Target
model paths, runtime/backend, source/config/input bindings and hardware
acceptance are verified on each execution host. One device's fit or runtime
measurement does not accept another device. The production lifecycle currently
executes the four learners sequentially in one width job; independent queued
stages would require new checkpoint/obligation handoff integration. Three width
jobs are supported without splitting that lifecycle or duplicating training.

Ordinary COCO and HellaSwag workflows remain documented in the main README.
