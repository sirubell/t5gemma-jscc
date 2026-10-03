# Native16 long encoder-sharing validation

This ticket 21 study validates three encoder positions before the separately
planned all-split expansion. It changes neither the shared default model nor
COCO, decoder or memory behavior. Use `scripts/sharing_long_study.py`, with an
exact prepared package and a pinned source checkout. Exporting inputs or a
package never starts a GPU job.

Each of the three predeclared widths (512, 1152, 2304) uses one biased Linear
per codec half, no external normalization, FiLM off, native BF16 forward,
FP32 full-vocabulary KL and masked per-sequence reconstruction, and simultaneous
`K + 0.1 R`. Training uses normalized masked AWGN with independent per-sequence
uniform SNR draws from −6 to +18 dB. Native physical batch 16, accumulation one,
AdamW LR 0.0002 and weight decay 0.002500001398541448 remain explicit.
The named task files reference the existing named model designs.

Per width, three fresh seed-zero specialists train 6,400 updates / 102,400
presentations each. **One** shared codec trains 19,200 updates / 307,200 total
presentations. Shared sites repeat the fixed order `enc_l9, enc_l19, enc_fn` per
optimizer update. Every site consumes exactly the same ordered complete batches
and site-local noise draws as its specialist; there is no reuse of trained
baseline weights. Source-valid, target-valid and padded tokens are counted
separately from sequence presentations.

| Shared step | Total presentations | Per-site updates (layer 9, layer 19, final) |
|---|---:|---|
| 6,400 | 102,400 | 2,134 / 2,133 / 2,133 |
| 12,800 | 204,800 | 4,267 / 4,267 / 4,266 |
| 19,200 | 307,200 | 6,400 / 6,400 / 6,400 |

Specialists save at 3,200 and 6,400; the shared trajectory saves at the three
steps above. Including common initialization gives ten states per width.
Each learner's LR schedule is fixed before update zero: 5% linear warmup and
cosine decay across its complete 6,400 or 19,200 horizon. The initial zero-LR
update still counts and updates AdamW moments. No update is appended after the
terminal zero LR; no heuristic early stopping or checkpoint selection occurs.

The full approved optimization pool uses deterministic seeded epoch
permutations (`20260920`), crossing epoch tails to retain complete 16-row
batches. Preparation rebuilds prompts and native prefixes from pinned raw data;
it cannot cycle the old q200 subset. The ordered 102,400-presentation stream is
shared across widths and learners. Selection comes from the fixed 128 objective
rows in the separate training-split selection pool. Development consists of the
previously inspected 256 official-validation rows. Full 10,042-row official
validation has also been inspected already and is never described as a new
untouched test set. Exact raw-data, family and demo audits accompany a package.

Sparse selection observations compute only no-noise K/R: three common-initial
sites, each specialist at its two saves, and all three trained shared sites at
its three saves (18 panels). There are no in-training development task panels.
Training ends with `training_frozen`; full task evaluation is still pending.
The existing inventory-backed freeze verifies all four complete learners,
ordered exposure and exact checkpoint bytes before layer 14 can be assessed.
Layer 14 never participates in training or selection.

Final evaluation uses the three terminal specialist checkpoints at their own
sites and the **same terminal shared checkpoint** at layers 9, 19, final and
held-out layer 14. The seven complete codec no-noise panels and one matched full vanilla bypass
reference run first, followed by
seven panels each at −6, 0, 6, 12 and 18 dB. Each panel covers all 10,042 rows with
five-shot lm-eval, FP32 scoring, compact per-item evidence, native BF16/SDPA and
batch 16. A failed panel stops later work and preserves partial evidence. There
is no best-checkpoint fallback or automatic retry. This makes 42 codec panels plus one full vanilla panel per width. The vanilla
reference uses the same source, scorer and runtime; a 256-row baseline or a
reference from a different runtime cannot substitute.

Prepare real batches with the existing CPU-only script, supplying the audited
full-validation JSON with `--final-task FINAL_TASK.json --final-task-sha256 SHA256`.
The script validates and retains that exact request as `final_task_request`;
missing final requests cannot bind a long package. Then bind each named
long task through `jscc.sharing_binding.bind_production`, using the explicit
`sharing-dn-native16-long-v1` protocol and
`K+0.1R-native-effective16-fresh-v1` recipe. The raw input manifest retains its
preparation provenance; the bound learner protocol declares the changed cadence.
The binding must include the immutable `final_task_request` reference.

```bash
uv run --locked --no-sync python scripts/sharing_long_study.py preview --prepared PREPARED/prepared.json
uv run --locked --no-sync python scripts/sharing_long_study.py train --prepared PREPARED/prepared.json --output FRESH_TRAIN_RUN --approved-source EXACT_COMMIT --timeout-seconds TRAIN_TIMEOUT
uv run --locked --no-sync python scripts/sharing_long_study.py evaluate --prepared PREPARED/prepared.json --training-run EXACT_TRAIN_RUN --output FRESH_FINAL_RUN --approved-source EXACT_COMMIT --timeout-seconds EVAL_TIMEOUT
```

GPU entry commands require a clean tracked checkout at the approved commit and a
Slurm allocation. Use the scheduler time limit plus an external `timeout` around
the command; timeout values are per job. The entry uses the recorded training
path rather than searching for a recent checkpoint. The simplest execution is
three width training jobs and three matching final evaluation job dependent on its own completed/frozen training job. The widths
and configurations are fixed before launch; held-out results never retune
other widths. A cross-width training barrier is unnecessary. Splitting individual learners would require collecting
four jobs into the same verified freeze; that extra scheduling is deferred.
No H200 smoke job, cumulative ledger or automatic publication is introduced.
The inherited event writer rewrites each learner log per update; its long-horizon
CPU/I/O cost is unmeasured and must be included in provisional timing.
Actual submission waits for approval of the exact source, commands and finish
window. CPU lifecycle tests establish software behavior, not native16 H200
throughput, full-weight width-wide memory fit or scientific performance.
