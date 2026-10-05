# COCO saved trajectories and comparison helpers

[`jscc/coco_trajectory.py`](../jscc/coco_trajectory.py) retains an offline
saved-evidence reader, state selection, workload accounting and reusable raw
cache/full comparison helpers. The fixed32/step500 trajectory preparation and
execution commands, candidate-manifest builder and executor are retired from
current source. Historical evidence, command receipts and held source packages
keep their original identities; no replacement executor or current-source
historical reproduction is supplied.

## Read existing evidence

`read_evidence(directory)` reads `identity.json`, `status.json` and `items.jsonl`
without loading model weights or a dataset. It requires completion status
`complete` with exactly 1,024 completed requests and verifies the identity file's
SHA-256 against `status.json`'s `manifest_sha256`. It requires exactly 768 complete
endpoint rows with one consistent EOS/newline token contract, and exactly 128
complete matched-image generation trajectories covering 32 image IDs by all four
modes (`vanilla`, `enc_l9`, `enc_fn`, `dec_l8`). Each generation checkpoint receipt
must equal its mode's saved identity. It derives visited states from the recorded
tokens and returns fresh SHA-256 values for all three evidence files. These
checks preserve the prior reader's exact contract; they do not verify checkpoint
weight availability or individual raw artifact bytes.

For local receipt inspection, the retained API is:

```python
from pathlib import Path
from jscc.coco_trajectory import read_evidence, workload

evidence = read_evidence(Path("/absolute/path/to/saved-screen"))
counts = workload(evidence)
```

`workload` derives trajectory, distinct selected-state, logical-request, decoder
and encoder call counts from those tokens and modes, plus missing-newline counts
and zero training updates. Vanilla needs one comparison role; codec modes need
student and bypass-teacher roles. Each cached replay traverses every recorded
next-token decision; full-prefix calls add one call per selected state. These are
expected comparison counts, not proof that a new execution completed.

## Selected states and raw comparisons

`selected_states` retains native BOS and selects exact prefixes before the first
complete generated newline sequence, after it only if another decision was
visited, and before EOS or the final token under the default 64-token cap. It
rejects post-EOS continuation and prematurely stopped trajectories. Identical
prefixes are deduplicated while their event names remain. Missing newline and an
unvisited after-final-token state are explicit; multi-token newline is treated as
one complete sequence.

`ForcedTrajectory` checks the native prefix, records the unforced greedy choice,
and forces only the known next token after finite raw logits are available.
`cached_replay` captures selected raw logits during native cached generation,
requires exact saved-token and selected-state coverage, and checks exactly one
receiver-memory transmission per coded decoder trajectory. Its forward hook is
removed on both success and failure. Inputs and caches remain separate for every
input and student/teacher role. A bypass teacher following student prefixes is a
conditional probe; it need not have visited those states autonomously.

The retained [`forward_endpoint`](coco-query-endpoint-diagnostic.md) helper
provides a full forward on the same prefix with production attention and
receiver-memory rules. `finite`, `logit_metrics` and `compare` reject nonfinite
vectors or mismatched vocabularies, and report raw-logit deltas, top1 agreement,
directional full-vocabulary KL, EOS probability/rank/margin, top5 and exact top1
tie counts. Float32 metric computation and temperature 1 remain unchanged. No
full-weight BF16 tolerance or quality claim follows from these helpers.

## Historical receipt interpretation and validation

The saved fixed screen previously yielded 128 trajectories, 191 distinct states,
565 logical requests and encoder calls, 5,218 decoder calls, and 89 missing-newline
trajectories. Those historical counts describe that evidence, not a default for
new inputs. Its retired executor used a 1,200-second run ceiling, a 2 GiB raw
artifact cap, reserve/alarm checks, fsynced partial attempts and per-artifact
hash/completion receipts. Failed and stopped attempts remain historical outcomes;
those producer mechanisms are not retained as a current launch path.

Source/runtime/checkpoint/tokenizer/native-prefix/input bindings in saved records
must remain intact. A changed source requires fresh future bindings rather than
editing an old scientific hash or receipt. Reader acceptance does not attest to
remote assets, scheduler/process limits, target runtime, GPU fit or scientific
quality.

CPU tests retain visited-state boundaries, raw logits before forcing, real tiny
native cache/full comparisons and receiver-memory behavior. Small local receipt
fixtures exercise complete matrix, identity hash, tokenizer and checkpoint gates.
The private historical matrix test remains explicitly skipped when its saved
evidence is absent from the portable checkout.
