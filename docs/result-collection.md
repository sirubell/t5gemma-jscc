# Compact result collection

Use one model-free local action for an ordinary COCO/HellaSwag experiment root
or a public recipe root (including singleton, multiple cases and retained
interrupted attempts):

```bash
uv run --locked python scripts/collect_results.py --run runs/coco-case \
  --snapshot collection-snapshots/coco-01 --output collected/coco-01
uv run --locked python scripts/collect_results.py --run runs/hellaswag-case \
  --snapshot collection-snapshots/hellaswag-01 --output collected/hellaswag-01
uv run --locked python scripts/collect_results.py --run runs/comparison \
  --snapshot collection-snapshots/comparison-01 --output collected/comparison-01
```

Snapshot and destination must be outside the scientific source root and separate
from each other. A snapshot is exclusive and binds the selected bytes. Repeat
the exact command to retry that snapshot. Use new snapshot/output names to collect
new results or newly grown scalar streams. A separate authorized mount/staging
operation may expose remote bytes as `--run`; this action itself makes no network
connection. `--source-uri ssh://host/absolute/run` records their exact retrieval
root without connecting to it.

Default output contains `exports/results.jsonl` (actual final, supplemental and
intermediate panel/condition/case rows), `exports/scalars.jsonl` (every recorded
scalar/event row in source order within each metrics file), `exports/status.json`,
`exports/errors.json`, and `exports/inputs.json`. JSON Lines retain nested
objective measures, units, LR, exposure, site/condition, actual denominators,
source/checkpoint/request identity and per-attempt ancestry. `sequence` and
`raw_metrics` identify the exact original row/file/hash/size. No missing rows,
smoothing, downsampling or inferred metrics are added. Non-update events retain
compact recorded content, including failure events. Bulk per-item and sparse
full-batch audit fields stay in the referenced raw file.

Small frozen configs and existing source/runtime/input/checkpoint identity,
status, timing, attempt, campaign, durable checkpoint receipts, reconciliation
and result/request records are copied under
`files/<source-relative-path>`. Input ID lists stay at source with exact hash,
size and retrieval reference in `inputs.json`. Recorded tensor hashes are
metadata and do not mean copied tensor verification or backup. Ordinary current
training, earlier eligible default checkpoint and explicit supplemental ancestry
remain independent records. Recipe engine records live below `training-N/engine`.
Pending observations, including `pending_prior_segment`, remain pending. Recipe
`gate.final_eligible` and root result status refer only to the selected initial
requests; collection does not establish campaign quality or training/evaluation
success.

Full raw metrics, per-item/prediction/prompt/input exports, debug logs, checkpoints,
models, data and caches stay at source. `manifest.json.remote` preserves their
exact retrieval references and observed byte sizes, with recorded hashes when
available, explicitly marked as unverified metadata. There is no byte ceiling,
bulk default, deletion or expiry policy. Select additional files explicitly:

```bash
uv run --locked python scripts/collect_results.py --run runs/comparison \
  --snapshot collection-snapshots/paired-01 --output collected/paired-01 \
  --select cases/width-32-ratio-50/evaluations/evaluation-0001/panels/enc_l2_no_noise/items.jsonl
```

`--select` takes source-relative files, not directory patterns. If a selected
file is absent when binding, its unknown bytes/hash remain missing; acquire it
and use a new snapshot. If a known selected file disappears later, retry may
restore/copy the original exact bytes. A source with changed bytes fails the
snapshot check. Source files are read stably (device/inode/size/mtime and content
digest); growing reads and unterminated JSON Lines reject instead of dropping a
row. The snapshot covers stable per-file bytes, not one atomic campaign-wide
instant. Concurrent scientific record publication can require a fresh snapshot.
Final frozen raw metrics must match their recorded digest.

`collection.json` and immutable `receipts/<pass-id>.json` record selected,
copied, verified, reused, missing and failed counts/bytes, unknown byte counts,
snapshot/destination identity, exact error paths and elapsed seconds. Known
source byte totals survive failures. Copied byte counts include bytes staged even
when snapshot verification fails; verified/reused byte counts are actual readback
sizes. A recorded hash with unknown size can recover, while its selected byte
count remains explicitly unknown. Unchanged verified destination copies are
reused only after hashing them again; damaged copies are recopied. Publication
uses temporary files, expected source-byte checks and destination readback
verification. Partial passes retain all earlier receipts and successful copies.
CLI exits 1 for an incomplete collection and 0 for complete selected copies.
Do not treat this exit code as scientific acceptance.

A `.collection-lock` excludes concurrent writers. A hard process death may leave
that lock: confirm the old worker stopped before explicit lock reconciliation;
no automatic liveness inference or stale-lock deletion occurs. Preparation
failures retain an adjacent `.collection-snapshot-*/preparation-failure.json`.
These local fixture/mount mechanics do not establish live transfer performance.

Python consumers can call `prepare_snapshot(root, snapshot, selected=(),
source_uri=None)`, `collect_snapshot(snapshot, destination)`, or the combined
`collect_results(root, destination, snapshot=..., selected=(), source_uri=None)`
from `jscc.result_collection`. None loads models or changes scientific records.
