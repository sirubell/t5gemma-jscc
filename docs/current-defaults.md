# Current research defaults

On September 29, 2026 the owner adopted the latest completed September comparison recipes as the current defaults. `configs/tasks/coco.yaml` and `configs/tasks/hellaswag.yaml` now resolve the shared `configs/model.yaml` to encoder layer 9 (zero-based), H1152/B512 residual codec, external LayerNorm none and FiLM off. Receiver-memory LayerNorm remains both for explicitly selected decoder variants. Smoke recipes share this model design with their own tiny budgets and data policies.

| Setting | COCO | HellaSwag |
|---|---|---|
| Historical recipe | Native-v2 full13 enc_l9, final step 500 | Five-shot64 enc_l9, final step 10,000 |
| Training prompt | Four fixed image/caption demos plus query | `prompt-alignment-v1`, five_shot, seed 20260920, source cap 2048 |
| Decoder input | Native model preparation | Native model preparation, protocol `corrected-baseline-v2-native-decoder-inputs` |
| Microbatch × accumulation | 16 × 4 | 64 × 1 |
| Updates / presentations | 500 / 32,000 | 10,000 / 640,000 |
| LR / schedule / warmup | 0.0002 / 500 / 25 | 0.0002 / 10,000 / 500 |
| Minimum LR ratio | 0.1 | 0 |
| KL / nMSE weights | 1.0 / 0.1 | 1.0 / 0.1 |
| Selection | 512 held-out images; final step; 0/6/12 dB; monitor KL | 512 training-split holdout rows; every 2,000 updates; no-noise; monitor loss |
| Final evaluation | 2,000 report images, 64 new tokens, batch 16 | 10,042 official validation examples, five-shot, fp32-v1 scoring, batch 64 |
| Codec conditions | no-noise, -6, +6, +18 dB | no-noise, -6, +6, +18 dB |

Both saved recipes use codec-only evaluation (`vanilla: false`, `mode: codec_only`); their historical vanilla controls were separate. Request a vanilla-only evaluation explicitly with `mode: vanilla_only` in an evaluation override when needed. Do not infer the existence of a matching vanilla result from codec scores.

The prompt-builder version `prompt-alignment-v1` does **not** mean the old PAD-prefix decoder protocol. The adopted HellaSwag run used native decoder preparation. The historical plain-context task and encoder-final-norm / external-LayerNorm-both design are no longer current defaults. Named studies may intentionally override the model design, and saved checkpoints retain their own complete configurations.

## Source identities and portability

The source recipes were saved by these completed runs:

- HellaSwag: `20260921-032958-five-shot64-enc_l9-cf5d23`; resolved config SHA-256 `07814af9b6f74c93b03657d205e429810984235731becd09b989233253ac976b`; training source SHA-256 `def4a5ab05a453d7b47c10a9542454f1c30e42044502413c88026048cd4b47af`.
- COCO: `20260922-015144-coco-native-ref6-32k-enc_l9-s0-3a7275`; resolved config SHA-256 `65bdce03db43891cbd4ba06ba086db3559c7c358dcfa91de67d9e23fa7a3bfe1`; training source SHA-256 `537701245e2e481b4610d75a9d0f2b62b0a77f362805b749c5afa896d429fffe`.

Portable files change the run name/output path, compose the shared model file, express COCO's `flash_math` policy as a runtime override, and omit the derived `codec.input_dim` populated by the model loader. The historical HellaSwag memory override's explicit `snr_film: false` is inherited from the shared disabled setting. Data, training, evaluation and resolved active model semantics are otherwise retained. These current files have different bytes and hashes from the original resolved configurations. Archive receipts bind the originals; new runs must record their own source/configuration identities.

Fresh data selection under the pinned revisions/seeds reproduces the configured selection procedure; claiming exact historical presentations still requires checking saved IDs and their digests. Default adoption does not establish that historical checkpoint bytes or dataset caches are currently available. It does not establish GPU fit, convergence or scientific equivalence between changed implementations.

## New work and execution boundary

The owner has authorized local implementation of the COCO query/endpoint diagnostic and the single-site HellaSwag two-stage pilot. This does not authorize actual GPU inference, feature capture, timing trials, training or scheduler submissions. Agree on verified artifacts, target hardware, explicit budgets and stop criteria before those actions. Local check/preparation commands must not load pretrained models.

Both ordinary task checks remain available without loading models or data:

```bash
uv run --locked --no-sync python train.py --config configs/tasks/coco.yaml --check
uv run --locked --no-sync python train.py --config configs/tasks/hellaswag.yaml --check
```

For two-stage comparisons, create a separately named plan with explicit U/L/F budgets. Do not silently substitute the 10,000-update historical score for a new shorter A arm, and do not treat the tentative 2,000-update or 80/20 proposal as execution approval.

The [COCO diagnostic guide](coco-query-endpoint-diagnostic.md) describes local identity preparation and the fixed 32-ID screen. The [HellaSwag two-stage guide](hellaswag-two-stage.md) describes four-arm preparation, sequence capture, local replay and explicit phase transfer. They keep preparation separate from execution; no tool submits scheduler jobs.
