# COCO query and endpoint helpers

The fixed32/step500 COCO preparation and execution family has been retired from
current source. Its two dedicated diagnostic commands, identity-manifest builder,
checkpoint/data loaders, executor, journal and POSIX alarm are no longer provided.
The reusable scientific helpers remain in [`jscc/coco_diagnostic.py`](../jscc/coco_diagnostic.py).
Saved scientific artifacts, historical command receipts and held source packages
retain their original identities. Interpret earlier execution against its recorded
source; current source does not reproduce the frozen executor.

## Retained query and prefix contracts

`prepare_pair` builds matched and donor query inputs with the same four demo
images, captions and order. Only the fifth image changes. It requires equal text
IDs and masks, equal processed demo crops, different query crops, and consistent
per-image and full processor crop counts. RGB and tensor hashes describe the
actual inputs. `diagnostic_inputs` moves every source tensor to the backbone
device and casts only image pixels to its dtype, preserving integer IDs and masks.

`prefix_variants` retains the complete caption, removes a terminal period (and
marks a no-op when absent), and appends newline. `prefix_inputs` excludes EOS from
the lexical prefix and invokes the backbone's native shift with a synthetic EOS
label. The resulting decoder input is native BOS followed by every lexical prefix
token; its last logit predicts the token **after** that complete prefix. A prefix
exceeding the default 65-position limit is returned as unsupported, never truncated.

`forward_endpoint` captures raw full-vocabulary logits in a no-gradient full
forward with the wrapper's attention and transmission contexts. Decoder routes
require receiver-memory transmission. `score_logits` rejects nonfinite logits,
and reports top1 plus EOS and newline-first-token probability/rank. Multi-token
newline has no single-token sequence probability: that field stays null.
`teacher_student_kl` uses finite float32 logits and directional full-vocabulary KL
at temperature 1; comparisons must use identical inputs and decoder prefixes.

`generate_caption` retains native greedy, cached, 64-new-token generation and
receiver-memory checks. It reports tokens, raw/clean captions, observed decoder
calls, EOS/truncation and repetition flags. `reference_alignment` is descriptive
lexical F1; it is not CIDEr or a report-set quality estimate.
`feature_signatures` and `tensor_sha256` hash tensor shape, dtype and contiguous
native storage bytes, including BF16 bits. Existing float32 hashes keep their
original format. File `sha256`, `bytes_sha256` and object-only `read_json` remain
available for saved-receipt readers.

## Saved screen receipts and budget semantics

The historical screen used 32 archived selection IDs, four modes (`vanilla`,
`enc_l9`, `enc_fn`, `dec_l8`), two query controls and one generation plus three
endpoint probes per pair: **1,024 logical requests**. The first four IDs formed
an included 128-request timing stage. A generation can invoke the decoder many
times; requests and decoder calls are distinct counts. Vanilla used the archived
`enc_fn` identity with communication bypass; codec modes used no-noise transmission.

Historical selection/checkpoint records and independent training-source archive,
config, run, data-ID, runtime and current-at-execution source hashes retain their
original meaning. They are not rewritten to bind changed source. Earlier
`items.jsonl` records contain partial attempts and terminal complete, unsupported
or failed rows; `status.json` records observed counts and the stop reason.
Raw endpoint `.pt` artifacts have their own hash receipts. Unsupported long
prefixes and stopped or failed runs remain explicit outcomes. The saved trajectory
reader accepts only the exact completed prior screen described in the
[trajectory guide](coco-trajectory-diagnostic.md).

`RequestBudget` remains a helper with the fixed 1,024-attempt ceiling. It requires
finite positive time and a smaller nonnegative reserve, refuses new work at the
reserve boundary, counts attempts before computation, and separately tracks
completed requests and reported decoder/encoder calls. A failed attempt still
consumes its slot. It does not implement the retired process alarm or durable
executor, and no new budget policy or execution command is introduced here.

Retained CPU tests cover query swaps, native prefixes, finite/scoring semantics,
input placement, receiver memory, reserve/attempt accounting and native tensor
hashes. Tiny randomly initialized models establish software behavior; they do
not establish full-weight BF16 agreement, caption quality or GPU fit.
