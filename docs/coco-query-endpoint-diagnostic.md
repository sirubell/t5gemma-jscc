# COCO query and endpoint diagnostic

This local diagnostic examines whether three completed step-500 COCO codecs preserve query-image dependence and the decoder distribution after a fixed reference caption. It performs no training or remote work. The frozen screen has 32 archived selection IDs, four modes (`vanilla`, `enc_l9`, `enc_fn`, `dec_l8`), matched and cyclic-permuted query images, one greedy generation, and three full-forward endpoint probes per pair. That is **1,024 logical requests**. The first four IDs constitute the included **128-request timing stage**. A request is not a decoder call: each generation can invoke the decoder repeatedly.

The selection and checkpoint index are retained in `archive/2026-09-24/research/coco-diagnostic-offline-20260922/{FIXED-SELECTION,CHECKPOINTS}.json`. The `vanilla` identity uses the archived `enc_fn` checkpoint/config receipt, while `SplitModel.transmission(bypass=True)` bypasses its communication path. The three codec modes use no-noise transmission. The diagnostic never substitutes an alternate checkpoint, shrinks the matrix, retries a failed request, or changes precision/backend when a request fails.

## Prepare and check identities without loading a model

First obtain local copies of the actual four checkpoint paths. The archived index contains remote path labels and SHA-256 receipts, but no checkpoint weight bytes. Prepare an explicit `bindings.json` with paths to each route's saved `config.yaml`, `run.json`, and `data_ids.json`, plus the checkpoint bytes. The vanilla and `enc_fn` entries may use the same files. Example structure:

```json
{
  "historical_source": {
    "archive": "/absolute/path/to/h200-workers4/execution.tar.gz",
    "manifest": "/absolute/path/to/h200-workers4/source-manifest.json"
  },
  "modes": {
    "vanilla": {"checkpoint": "/absolute/path/to/enc_fn/step_000500.pt", "config": "/absolute/path/to/enc_fn/config.yaml", "run": "/absolute/path/to/enc_fn/run.json", "data_ids": "/absolute/path/to/enc_fn/data_ids.json"},
    "enc_l9": {"checkpoint": "/absolute/path/to/enc_l9/step_000500.pt", "config": "/absolute/path/to/enc_l9/config.yaml", "run": "/absolute/path/to/enc_l9/run.json", "data_ids": "/absolute/path/to/enc_l9/data_ids.json"},
    "enc_fn": {"checkpoint": "/absolute/path/to/enc_fn/step_000500.pt", "config": "/absolute/path/to/enc_fn/config.yaml", "run": "/absolute/path/to/enc_fn/run.json", "data_ids": "/absolute/path/to/enc_fn/data_ids.json"},
    "dec_l8": {"checkpoint": "/absolute/path/to/dec_l8/step_000500.pt", "config": "/absolute/path/to/dec_l8/config.yaml", "run": "/absolute/path/to/dec_l8/run.json", "data_ids": "/absolute/path/to/dec_l8/data_ids.json"}
  }
}
```

The matching historical archive is the saved `coco-native-v2-20260921/download/h200-workers4/remote/execution.tar.gz`, paired with that directory's `source-manifest.json`. The preparation command hashes every archive member against the manifest and recomputes the historical `training_source_digest` from archived `jscc/*.py`, `train.py`, `uv.lock`, and `pyproject.toml`. That digest must equal each route's saved `run.json`; this does not equate historical training bytes with the current diagnostic source. It also verifies checkpoint hashes against `CHECKPOINTS.json`, config hashes against `run.json`, saved ID equality/disjointness, recipe routes, and model/data revisions. The new manifest binds all current `jscc/*.py`, the diagnostic CLI, `train.py`, `uv.lock`, `pyproject.toml`, and installed package versions. A later edit invalidates it.

From the project root, with the pinned environment already present:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 UV_CACHE_DIR=/private/tmp/uv-coco-diagnostic \
  uv run --locked --no-sync python scripts/coco_query_endpoint_diagnostic.py \
  --prepare --selection /absolute/path/to/FIXED-SELECTION.json \
  --checkpoint-index /absolute/path/to/CHECKPOINTS.json \
  --bindings /absolute/path/to/bindings.json \
  --manifest /absolute/path/to/new-identity-manifest.json

HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 UV_CACHE_DIR=/private/tmp/uv-coco-diagnostic \
  uv run --locked --no-sync python scripts/coco_query_endpoint_diagnostic.py \
  --check-only --selection /absolute/path/to/FIXED-SELECTION.json \
  --checkpoint-index /absolute/path/to/CHECKPOINTS.json \
  --manifest /absolute/path/to/new-identity-manifest.json
```

Both commands read local files only; they do not instantiate a model, decode an image, or load a dataset. Their `identities_verified_not_executed` result means file and recipe identities passed. It does not establish that weights fit the intended GPU or that image/model caches are ready. `--prepare` requires a new output manifest path and will not overwrite one.

## Explicit run

Only after the selected machine's weights, model/data caches, source, time cap, and execution authorization have been confirmed, use a **new** output directory and an explicit hard budget:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 UV_CACHE_DIR=/private/tmp/uv-coco-diagnostic \
  uv run --locked --no-sync python scripts/coco_query_endpoint_diagnostic.py \
  --run --selection /absolute/path/to/FIXED-SELECTION.json \
  --checkpoint-index /absolute/path/to/CHECKPOINTS.json \
  --manifest /absolute/path/to/new-identity-manifest.json \
  --output /absolute/path/to/new-diagnostic-output \
  --max-seconds 2400 --reserve-seconds 300
```

The example budget is a command illustration, **not** an approved allocation or measured runtime. The run forces Hugging Face offline mode, starts its wall clock before loading the cached dataset/model, stops starting requests at the reserve boundary, and uses a POSIX alarm at the hard deadline. A host scheduler/allocation limit is still needed: Python signals cannot guarantee immediate interruption of an in-flight CUDA kernel. The first-four-ID stage measures all four modes and exits with partial evidence if a conservative projection does not fit the remaining budget. No automatic second job or extension is attempted.

Each attempt is written as a `partial` JSONL item before computation, then appended again with a terminal `complete`, `unsupported`, or `failed` status. Request attempts consume the 1,024 ceiling even when unsuccessful. `status.json` is replaced after each item with observed request/decoder/encoder counts and stop reason. The output holds raw endpoint logits in individual `.pt` files with SHA-256 receipts. `complete_with_unsupported` means one or more captions exceeded the 65-position native shifted prefix limit; those prefixes are preserved and never truncated. A 128-request timing stop or deadline stop has a separate status and retains all partial evidence.

For each ID, the source prompt retains the same four demo images, captions, and order. Only the fifth image changes to the next frozen query ID. The runner verifies equal text IDs/masks, equal processed demo crops, different query crops, image hashes, per-image crop counts, and full input tensor hashes. It rechecks Karpathy validation/test membership and saved train/demo/report disjointness from cached data before model loading. The endpoint reference is exactly `answer[0]` from the recorded selection row; an empty first entry is an error. The three variants are the complete text, text without a terminal period (with a recorded no-op if absent), and text plus newline. Tokenizer EOS is excluded from the lexical prefix. A synthetic EOS is used only to invoke the backbone's native shift so the probed decoder input is BOS followed by every lexical prefix token. The final logit thus predicts the token **after** the complete prefix.

Endpoint results use raw full-forward logits under the wrapper's attention context and record exact prefix IDs, decoder inputs/mask, EOS and newline-first-token probability/rank, top-1, full logits, and same-input teacher/student KL. Multi-token newline has no single-token sequence probability; its first-token probability is recorded and the sequence field is null. There is no cached endpoint replay. Generation uses native greedy 64-new-token settings and records token IDs, raw/clean captions, EOS/truncation/repetition flags, original and donor references, descriptive lexical F1 against each reference set, and matched/permuted output-change summaries. Boundary and receiver-memory feature hashes distinguish changed activations from changed captions. Lexical F1 is not CIDEr; this selection screen is not a report-set quality estimate or evidence that an EOS training change would be sufficient.
