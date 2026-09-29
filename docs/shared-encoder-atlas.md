# Shared encoder S0 atlas and replay acceptance

`scripts/shared_encoder_atlas.py` is a bounded diagnostic for the adopted residual H1152/B512 encoder codec. It answers whether clean encoder representations at zero-based layers4,9,19 can be captured and passed through the same codec implementation with exact online/local replay parity. It does not train shared weights, run task evaluation, establish a quality advantage or establish full functional-gradient parity. The production enc_l9-only replay/cache guards remain unchanged.

## Exact scope

- 128 distinct query rows:64 optimization rows, then64 disjoint train-derived selection rows. Atlas capture uses8 batches of16,3 sites,384 site-presentation records.
- Clean capture removes the installed communication hook temporarily, installs observation hooks at4/9/19, and stops immediately after19. There is no codec/channel transmission or decoder/suffix execution. Temporary hooks are removed and the original hook restored on success and exceptions.
- Six batch64 parity pairs: each of3 sites under no noise and fixed replay-audited AWGN at+6dB. The first64 train views are freshly padded together, rather than repacked from the batch16 cache. Each online forward has exactly one active communication insertion and stops at that boundary. Replay uses the same activation shape/mask and actual AWGN draw hash; comparisons cover activation, reconstruction, nMSE, every codec gradient and valid/allocated coordinates.
- Exact equality is the acceptance rule. A mismatch records its maximum absolute error and tensor hashes and fails; tolerances are never widened automatically.
- Zero optimizer updates; no l14 activation inspection. Both native source truncation and target tokenization occur only during frozen-view preparation, using the existing five-shot builder. Capture never truncates a supplied view.

The CLI supervises a child process and kills it at the declared wall ceiling, at most600seconds including child startup and model loading. The runner additionally checks deadlines at stage boundaries and hooks. The default serialized-output ceiling is1GiB; a smaller ceiling may be specified down to1KiB. No smaller batch, shorter input, changed codec or retry is substituted after failure. A production-sized batch64 parity check may not fit a given host; that is a failed fit observation.

## Prepare frozen views

Use a **resolved/frozen H1-derived configuration** and the pinned run's `data_ids.json`, not a newly sampled split. The script requires the adopted residual codec, HellaSwag enc_l9, normalized AWGN and explicit native five-shot prompt policy. It validates the resolved configuration hash across preparation/capture.

```bash
uv run --locked --no-sync python -m scripts.shared_encoder_atlas \
  --config /absolute/path/frozen-s0.yaml \
  --prepare-views \
  --data-ids /absolute/path/pinned-run/data_ids.json \
  --output /absolute/path/new-s0-views \
  --max-seconds 600 \
  --max-cache-bytes 1073741824
```

Preparation is a separate bounded CPU/data operation and does not load backbone weights. It fixes the **first64 entries in the supplied train-ID ordering and first64 in the supplied selection-ID ordering**, writing `selected_rows.json` before tokenizer/data work. It uses the existing `PromptBuilder` over the pinned dataset revision and full supplied train demonstration pool. Only128 query views are tokenized. Query rows and queries sharing a source with a candidate demonstration are excluded by the native builder; selection rows are never demonstrations. The existing policy preserves left-truncated2048-token sources and512-token targets.

`views.json` records original data IDs, selected row order/group, five demo IDs and sources, source/document/prompt hashes, token arrays and hashes, source lengths, native preprocessing digest, dataset fingerprint, tokenizer name/revision, pad ID, resolved-config hash, script/source digest and one hash per view. This file is required explicitly for capture. Inspect its receipts and freeze it with the release before the GPU invocation. Do not hand-edit token hashes to make a changed view pass.

## Capture and parity

```bash
uv run --locked --no-sync python -m scripts.shared_encoder_atlas \
  --config /absolute/path/frozen-s0.yaml \
  --views /absolute/path/new-s0-views/views.json \
  --output /absolute/path/new-s0-result \
  --max-seconds 600 \
  --max-cache-bytes 1073741824
```

The output directory must not already exist. Model loading uses the pinned configuration/tokenizer revision; ordinary project model loading may access its configured caches/network. Deployment should supply accepted caches and offline environment policy when required. The script does not transfer files, submit jobs or change runtime dependencies.

The artifact is **`shared-encoder-s0-atlas-v1`**, not the production `enc-boundary-microbatches-v1` schema. It contains:

| File | Meaning |
|---|---|
| `status.json` | Running stage or terminal COMPLETE/FAILED_OR_PARTIAL, elapsed time, completed counts and zero updates |
| `identity.json`, `config.json`, `views.json` | Resolved input/source/runtime intent and frozen view bindings |
| `atlas-00.pt` through `atlas-07.pt` | Full padded input/mask/label/ID tensors, view IDs and native-dtype clean activations for exactly three sites |
| `parity.json` | Completed comparisons, exact error/hash results, actual AWGN hashes and payload counts |
| `manifest.json` | Published only after all eight captures and six parity pairs pass, with hashes/counts/timing/memory and per-shard statistics |

Statistics include per-sequence RMS, near-zero-energy counts and channelwise mean/variance separately for every microbatch/site; selection and optimization groups remain identifiable. These are descriptive summaries, not site-specific normalization calibration. This smallest S0 does not compute covariance/rank sketches. No sample norm or clean hidden tensor is passed to the receiver codec as a scale side channel. Raw clean targets enter only reconstruction loss and diagnostics.

`model_load_seconds`, `capture_seconds`, `parity_seconds` and terminal elapsed seconds have separate meanings. CUDA peak allocated/reserved memory is measured from before model load and includes the loaded frozen backbone. The local replay path avoids a backbone forward but retains the backbone in memory; do not describe this as a backbone-free learner.

Call `read_complete(path)` from the script to verify a completed artifact's schema, count/site contract and file hashes. A missing completion manifest, a FAILED_OR_PARTIAL receipt, any parity mismatch, nonfinite activation/loss/gradient, byte-cap failure or killed worker is a failed/incomplete acceptance attempt. Partial files are retained as evidence and are not reusable production caches. A CLI hard timeout removes any completion manifest, keeps the most recent progress fields, and records `HardWallTimeout`.

Success only permits reviewing a separately scoped shared-versus-specialist training proposal. It does not launch one.

## Local verification

```bash
uv run --locked --no-sync --extra dev ruff check \
  scripts/shared_encoder_atlas.py tests/test_shared_encoder_atlas.py
uv run --locked --no-sync --extra dev pyright \
  scripts/shared_encoder_atlas.py tests/test_shared_encoder_atlas.py
uv run --locked --no-sync --extra dev python -m pytest -q \
  tests/test_shared_encoder_atlas.py
```

Tests use a real tiny20-layer Transformers backbone, not fabricated hook outputs. They cover all three clean sites, all six parity conditions, masks/traffic, actual replay noise, local gradients, unchanged weights, exception/deadline cleanup, finite checks, frozen native view preparation, incomplete/corrupt artifacts and unchanged production enc_l9 rejection. CPU acceptance is not a full-weight CUDA result.
