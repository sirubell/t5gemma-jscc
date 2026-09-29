# COCO visited-trajectory diagnostic

This no-training follow-up replays the 128 matched-image trajectories from the
completed fixed32 query/endpoint screen. It does not rerun donor controls,
reference-caption variants, or a quality benchmark. All four modes and all 32
images are retained. The prior artifact hashes and the new source/runtime hashes
are bound in a newly prepared manifest; historical receipts remain unchanged.

## Selected states and comparisons

The decoder sequence includes native BOS. A state is the exact prefix before a
recorded next-token decision. Select before the first complete generated newline
sequence, after that sequence if a further decision was visited, and immediately
before EOS or the final allowed token under the 64-token cap. Deduplicate identical
prefixes while retaining all event names. A newline emitted as token64 has no
visited after-newline state. Missing newline and unvisited after-final-token
states are explicit; no continuation after EOS is scored.

For each trajectory, replay native cached generation once per communication role,
forcing the saved next tokens only after capturing raw model logits and the
native greedy choices. The forcing also permits a bypass teacher to follow the
student's exact prefixes even if the teacher would have stopped earlier. Those
are explicitly conditional teacher probes, not a claim that the teacher visited
the student's states autonomously. Student greedy replay disagreement is retained
as a diagnostic outcome rather than silently replacing the frozen trajectory.

At each distinct selected prefix, run the existing full-forward endpoint helper
for the same student and bypass teacher. Vanilla reuses its own teacher tensors.
Report raw-logit cache/full deltas, top1 agreement, full-vocabulary directional
KL, EOS probability/rank/margin, top5 and exact top1 tie counts for each path.
Teacher-to-student KL uses identical prefixes and float32 logits, temperature 1.
No numerical acceptance threshold is invented for full-weight BF16 results.

Native generation supplies BOS, cache positions and attention preparation. The
wrapper applies the production attention context; the endpoint helper explicitly
enters it on direct backbone calls. No-noise transmission preserves the normal
codec and normalization rules. Decoder cached replay checks exactly one receiver
memory transmission; independent full forwards require receiver memory through
the existing helper. Caches never cross inputs or teacher/student roles.

## Preparation and execution

Local evidence-only preparation needs no model weights or dataset load:

```bash
uv run --locked python scripts/coco_trajectory_diagnostic.py --prepare \
  --evidence runs/observed/20260929-coco-job-18387 \
  --manifest /path/to/new-trajectory-manifest.json
uv run --locked python scripts/coco_trajectory_diagnostic.py --check-only \
  --evidence runs/observed/20260929-coco-job-18387 \
  --manifest /path/to/new-trajectory-manifest.json
```

The manifest derives exact counts from the prior generation tokens. It also binds
all three evidence files, current Python source, the entry script, project lockfile
and installed runtime packages. `--check-only` validates this local preparation;
it does **not** attest to remote checkpoint/model availability. Target deployment
must prepare a target-runtime manifest after source is frozen. Changed source,
runtime or evidence requires a fresh manifest, never editing historical receipts.

Explicit run mode additionally requires `--selection`, `--checkpoint-index`,
`--endpoint-manifest`, `--output`, and `--max-seconds` (plus optional
`--reserve-seconds`, default 60). The endpoint manifest is freshly prepared with the
existing endpoint CLI using real target checkpoint/config/ID/archive bindings.
Its verification checks checkpoint bytes, native-v2 recipe and frozen selection.
The trajectory runner also checks checkpoint receipt equality to the prior screen,
tokenizer EOS/newline IDs, native BOS and exact processed input tensor hashes.
The script only runs locally; it never submits jobs or downloads assets. Offline
Hugging Face flags are set before model/data loading.

## Budgets and durable evidence

The current fixed evidence requires 128 trajectories, 191 distinct selected
states, 565 logical requests, 565 encoder calls and 5,218 decoder calls. This
includes 224 cached replays (128 students plus 96 codec bypass teachers), and 341
full-prefix requests. There are 89 missing-newline trajectories. A selected state
is not equivalent to one decoder call: cached replay traverses every recorded
decision, including the terminating EOS/final64th-token decision. The internal run ceiling is 1,200 seconds; callers must also bound
pre-run verification and process lifetime externally.

Every request first fsyncs a partial attempt. A successful request writes and
fsyncs its raw `.pt` output, then records the artifact SHA-256 and completion.
A failure records its error and leaves existing receipts/artifacts intact.
Trajectory-start records bind input hashes, prior request ID, checkpoint and
prefix selection; trajectory summaries retain raw captions and repetition flags.
Actual encoder/decoder hooks count attempted calls, including failures. The
completion status requires exact expected request and call counts.

A POSIX alarm covers run setup, model/data loading, inference and artifact writes;
new operations stop at the reserved flush interval. Initial manifest/checkpoint
verification happens before this run timer, so a scheduler/process timeout must
bound the complete command too. The request and call ceilings are calculated
from immutable trajectories. Raw artifacts have a hard 2 GiB retained-byte cap;
JSON receipts are additional small metadata. No retry, adaptive smaller batch,
or resumed scientific replacement is automatic. A failed run remains failed.

Focused CPU tests use tiny randomly initialized T5Gemma2 models and establish
state boundaries, native cache/full routing, receiver-memory behavior, finite
checks and durable failure receipts. They do not establish BF16 correctness,
caption quality, target runtime or GPU memory fit.

Target preparation uses the existing path-bindings JSON without manually writing
identity hashes. The trajectory CLI invokes the existing endpoint manifest
builder and verifier, then binds the resulting manifest SHA-256:

```bash
uv run --locked python scripts/coco_trajectory_diagnostic.py --prepare \
  --evidence /path/to/copied-job-18387-evidence \
  --selection /path/to/frozen-selection.json \
  --checkpoint-index /path/to/frozen-checkpoint-index.json \
  --bindings /path/to/existing-bindings.json \
  --endpoint-manifest /path/to/new-endpoint-manifest.json \
  --manifest /path/to/new-trajectory-manifest.json
```

For `--check-only` and `--run`, supply the same selection, checkpoint index and
endpoint manifest, omitting `--bindings`. Both manifests must match the deployed
source and target packages. Evidence-only local preparation deliberately cannot
stand in for this target-verified manifest.
