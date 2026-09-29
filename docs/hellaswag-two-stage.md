# HellaSwag enc_l9 local-reconstruction pilot

This software prepares and runs a four-arm, single-site pilot under the pinned
five-shot corrected-baseline-v2 recipe. The historical 10,000-update result is
context, not the A control. A new plan requires explicit positive U, L and F
with U = L + F. No command in this workflow schedules a job.

## Prepare without loading a model

```bash
uv run --locked python scripts/hellaswag_two_stage.py prepare \
  --base configs/tasks/hellaswag.yaml --output-dir /path/to/fresh-plan \
  --u U --l L --f F
uv run --locked python scripts/hellaswag_two_stage.py plan \
  --a /path/to/fresh-plan/a.yaml --b /path/to/fresh-plan/b.yaml \
  --c /path/to/fresh-plan/c.yaml --d /path/to/fresh-plan/d.yaml
```

Replace U/L/F with an approved numeric budget; there is no built-in 2,000 or
80/20 choice. `--functional-lr` may explicitly set a shared fresh LR for C/D;
otherwise both use the base recipe's LR. Export copies the model/data/prompt
and evaluation recipe, sets each arm's schedule to its own update horizon,
saves only the terminal step, and disables early stopping. C/D use the same
phase-two batches and AWGN seed, beginning at B's presentation count in the
same permutation stream. The plan check compares the complete C/D functional
training and evaluation settings, A/B/C/D model/data design, and U=L+F.

The prepared files are complete flat resolved YAML recipes. Keep the
`plan.json`, source revision and exact generated files with an experiment
request; do not edit one arm in isolation after checking the plan. Preparation
does not verify target hardware, available model/data bytes, checkpoint weights
or a GPU-hour budget.

## Execute only under a separately approved experiment budget

Each command below is independent. `--max-seconds` is mandatory and applies to
setup plus its operation. It is a soft wall deadline checked between bounded
operations; a long model load, kernel or serialization call cannot be
interrupted by Python. An external scheduler wall limit remains the hard cap.
An over-cap functional phase preserves a PARTIAL completion receipt and raises
a timeout, so its CLI exits nonzero rather than reporting a successful arm.
The cache directory must be fresh and empty. Capture additionally requires a
retained serialized-shard byte cap; manifest and transient serialization bytes
are outside this cap, so use an external filesystem quota for a hard disk
limit. Before loading data it prints an activation-only
upper bound from declared presentations, source token cap, input width and
backbone dtype; that bound excludes tokens/masks and serialization overhead.
The manifest records actual shard bytes. A full padded cache can be very large
and its first-use capture/storage/I/O cost may outweigh local replay savings.

```bash
uv run --locked python scripts/hellaswag_two_stage.py capture \
  --config /path/to/fresh-plan/b.yaml --cache-dir /path/to/cache \
  --max-seconds SECONDS --max-cache-bytes BYTES
uv run --locked python scripts/hellaswag_two_stage.py local \
  --config /path/to/fresh-plan/b.yaml --cache-dir /path/to/cache \
  --max-seconds SECONDS
uv run --locked python scripts/hellaswag_two_stage.py functional \
  --arm A --config /path/to/fresh-plan/a.yaml --max-seconds SECONDS
uv run --locked python scripts/hellaswag_two_stage.py functional \
  --arm C --config /path/to/fresh-plan/c.yaml \
  --parent /path/to/B-run/last.pt --max-seconds SECONDS
uv run --locked python scripts/hellaswag_two_stage.py functional \
  --arm D --config /path/to/fresh-plan/d.yaml --max-seconds SECONDS
```

Capture executes the frozen encoder only through the after-layer-9 boundary.
It stops before the codec hook and saves the full padded activation, binary
mask, input tokens, target tokens and source row IDs for every presentation
microbatch. The cache preserves batch shape and view order; it does not shuffle
individual tokens or deduplicate repeated rows. The manifest binds the source
digest, model/tokenizer revision, data/prompt policy, split, precision, channel,
presentation stream, token/view hashes and shard checksums. Loading checks
identity and training/selection disjointness; each shard is checked before
use. A failed capture leaves `partial.json` and no complete manifest.

B reads the complete cache and executes `SplitModel._roundtrip`, which shares
the online codec, per-sample masked power normalization, AWGN and precision
boundary. Its objective is the existing per-sample masked nMSE reduction. It
does not execute the teacher, suffix or decoder. It records a terminal
`last.pt` in the standard checkpoint format, so the common evaluator can read
it. A partial B run writes `FAILED_OR_PARTIAL` and has no terminal B
checkpoint. B records its initial communication weights and hash. A and D
record the same initialization evidence through the ordinary fixed-stream
training path; compare these receipts before interpreting outcomes.

C loads only B's codec/channel weights. Its generated recipe declares the
planned B scientific-recipe digest and terminal step; the loader requires the
actual B checkpoint and complete receipt to match this declaration. It also
rejects a wrong source digest, data IDs, optional explicit cache declaration
or phase-two stream offset. C gets
fresh AdamW, scheduler, scaler, best score and update counter. `--resume` is
not a phase-transfer interface. D starts from the same seed as A/B and uses
C's exact phase-two data/noise stream and fresh LR horizon. Both functional
phases use the existing full teacher/student training loop.

Evaluate each terminal checkpoint with the same evaluator and fixed panel:

```bash
uv run --locked python evaluate.py --run /path/to/arm-run --checkpoint last.pt
```

Use the train-derived selection panel for pilot choices and retain official
validation for the declared assessment. Report terminal checkpoint step,
presentations, valid vectors/tokens, SNR exposure, first-use setup/capture,
cache storage/I/O, B replay, C/D functional and evaluation cost. C inherits B's
capture and replay cost once; separately report physical campaign cost.
Numerical equivalence or quality tolerance requires a declared criterion and
real evaluation, beyond the offline CPU gates.
