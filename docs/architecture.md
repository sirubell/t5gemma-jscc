# Architecture and experiment semantics

This describes the current implementation. Both task entry points and common configuration fields are in the [README](../README.md).

## Split model

The T5Gemma-2 backbone is frozen and kept in eval mode. SplitModel inserts a hook at a configured encoder/decoder location, sends the representation through the codec/channel, and returns the reconstructed representation to the model.

The implementation uses standard PyTorch/Transformers operations and the configured device/dtype. It contains no H200-only kernels or scheduler requirement. Hardware compatibility and batch-memory fit are execution concerns described in the [running guide](running.md).

```text
hidden [B,T,D] → codec.encode → z [B,T,K]
              → power normalization → channel(z, snr_db)
              → codec.decode → reconstructed [B,T,D]
```

D comes from the backbone configuration; K is codec.bottleneck_dim. Both codec halves consist of Linear layers and residual blocks. External LayerNorm is configurable; internal block LayerNorm remains enabled. Optional receiver FiLM conditions the first decoder hidden representation on SNR.

## Explicit codec architecture experiments

`codec.architecture` defaults to `residual_mlp` when omitted, preserving saved historical recipes and residual checkpoint tensor keys. The shared baseline uses this architecture. Named designs live in `configs/models/`:

- `enc_l9_residual.yaml`: current two-block residual reference.
- `enc_l9_zero_block.yaml`: `residual_mlp` with zero residual blocks; each half still contains two Linear layers, with no activation between them.
- `enc_l9_direct_affine.yaml`: `direct_affine`, one `Linear(D, B)` encoder and one `Linear(B, D)` decoder, including biases.

The direct design requires zero residual blocks, external LayerNorm `none`, FiLM disabled and zero dropout. Its retained `hidden_dim` and activation configuration do not add hidden layers. These restrictions make the minimal architecture experiment explicit; they are not a change to the shared default. Each half is affine, while the complete transmission path still includes nonlinear power normalization and the configured channel.

At D1152/B512, main-codec parameter counts including biases are 14,473,088 for the two-block residual design, 3,837,824 for zero-block factorization and 1,181,312 for direct affine. These are model parameters, not communication bits or coordinates. B=D removes dimensional compression but still applies normalization/channel effects; B>D expands the transmitted representation. Compare task quality alongside valid channel coordinates and measured cost.

## Shared baseline

COCO and HellaSwag reference `configs/model.yaml` for the same backbone revision, encoder-layer-9 split, codec architecture and channel defaults. This also applies to paired CPU and H200 smoke recipes; device/dtype vary by execution environment, not by task. Task YAMLs own data, training and evaluation. The loader composes them once and saves the complete result in each run/checkpoint.

| Component | Baseline |
|---|---|
| Split | Encoder after layer 9 (zero-based) |
| Codec hidden width | 1152 |
| Bottleneck width | 512 |
| Residual blocks | Two in each codec half; each block computes x + F(x) |
| Activation / dropout | GELU / 0 |
| External LayerNorm | None at the hidden-stream boundary; internal block norms remain enabled |
| Channel | AWGN with per-sample power normalization |
| SNR-FiLM | Disabled |

The owner adopted these completed September recipe settings on September 29; see [current defaults](current-defaults.md) for the exact lineage. This baseline makes the task designs consistent; it is not an experimentally established optimum. Task-specific preprocessing, targets, training budgets and metrics still differ. A study may explicitly override the split or codec settings.

FiLM remains implemented as an opt-in experiment via `codec.snr_film: true` in a separately named model design file. Point the relevant experiment's task YAML at that file, keeping the shared default off for normal development and collaboration. When disabled, no FiLM module or parameters are created and codec decoding does not depend on SNR. The channel still uses SNR to generate noise. `film_hidden` and `clean_film_snr` have no numerical effect while FiLM is off.

With FiLM enabled, an SNR-conditioned MLP applies `(1 + gamma) * hidden + beta` after the first codec decoder Linear. Its final layer starts at zero, making modulation initially identity. This is receiver-side conditioning, not a change to the physical channel interface.

For multimodal encoder splits, after_embed and before_first_layer are pre-hooks on the first text layer. They run after vision features replace image placeholder tokens. A hook on the token embedding itself would be too early: later image scatter could overwrite the reconstructed image positions. The regression tests explicitly distinguish these placements.

For a decoder split after layer k, transmitter layers 0..k use original encoder memory. Receiver layers k+1..end use one shared reconstructed memory tensor from a second, independently trained codec. By default the memory codec inherits the main `codec` settings; an explicit `codec.memory` mapping may override fields such as `layernorm` while inheriting the remaining shared design. The memory is transmitted once per uncached decoder forward; cached generation builds receiver cross-attention K/V from that reconstruction and does not retransmit memory on later tokens. A split after the last decoder layer or final norm needs no memory stream because the receiver has no cross-attention layers. Decoder hidden states still pass through their own codec at the selected boundary.

This corrects the former clean-memory bypass under the [accepted system definition](adr/0001-transmission-boundary.md). Historical global-memory coding also perturbed transmitter decoder layers, so it represents a different protocol. The additional codec increases parameter count and communication use; equal bottleneck width does not imply equal total cost across encoder and decoder splits.

Greedy generation (num_beams=1) is supported. Beam-specific physical noise sharing is not defined, so the wrapper rejects beam search when receiver memory is transmitted. Reusing populated K/V caches across inputs or channel conditions is unsupported. Autoregressive token feedback is assumed available to the transmitter; feedback traffic/latency is not modeled.

Evaluation records channel_uses_real for hidden and memory streams: real latent coordinates actually sent, including padding, prompts and repeated evaluation calls. These are evaluator execution counts, not bits or a deduplicated per-context rate (HellaSwag scores multiple candidates). Per-sample power normalization applies independently to each stream. Teacher-forced full sequences and autoregressive one-token hidden transmissions have different normalization domains; quantify this limitation before interpreting robustness curves.

The shared default now uses an encoder split, so its encoder output memory passes through the codec/channel before the decoder consumes it. Decoder splits remain supported for explicit experiments.

## Training objective

Each microbatch runs a no-grad teacher with codec/channel bypassed, then a student through the codec/channel. Gradients can pass through frozen downstream layers to the codec. Custom channel parameters with requires_grad=True also enter the optimizer.

Loss is kl_weight × KL(teacher || student) + mse_weight × nMSE:

An optional `training.loss_weights: {kl: 1.0, hidden: 0.05, memory: 0.05}` explicitly weights the three components. When present, this mapping takes precedence over legacy `kl_weight` and `mse_weight`; the reconstruction terms are not averaged again. It requires all three finite nonnegative coefficients. For decoder runs with both streams, `{kl: 1.0, hidden: 0.05, memory: 0.05}` matches the legacy `kl_weight: 1.0, mse_weight: 0.1` objective. The same weights apply to training, selection and component logging. Compare unweighted components and task scores across weight variants, rather than ranking different weighted totals. Defaults are unchanged.


- KL uses positions whose labels are not −100, computes probabilities in float32 and applies the configured temperature.
- In the legacy helper call without a mask, nMSE divides global hidden-state reconstruction MSE by the original representation's global mean squared value. The corrected baseline passes a stream-specific valid-position mask and computes a per-sample nMSE, excluding padding before averaging samples.
- With receiver memory coding, corrected-baseline nMSE is the equal mean of the hidden-stream and memory-stream per-sample masked nMSE means, keeping the configured reconstruction weight unchanged. KL gradients train both codecs.
- The corrected baseline aggregates KL over all valid target tokens in the effective batch, uses the same aggregation in validation, and samples one SNR per training sample. Its allocated latent-coordinate count and valid payload count are recorded separately.
- Teacher forcing uses the backbone's native `prepare_decoder_input_ids_from_labels` method. For pinned T5Gemma 2 this prepends decoder BOS2 and maps ignored labels to PAD0. Custom backbones without this API retain an explicit-start/PAD fallback. Earlier corrected-v1 training used the top-level fallback PAD0 while native HellaSwag evaluation used BOS2; preserve those runs as a separate protocol. Native preparation aligns the start-token contract, not the task-specific training versus five-shot evaluation prompts.

A step is an optimizer update. Microbatches accumulate gradients before clipping, AdamW and cosine scheduling. `training.max_steps` is the stop budget; optional `training.schedule_steps` keeps the learning-rate horizon independent for short diagnostics and must be at least `max_steps`. Validation evaluates configured SNR conditions, then restores training mode.

## Task data

| Task | Training inputs/targets | Task evaluation |
|---|---|---|
| COCO | Images and demonstration captions as prompt; caption target | Autoregressive captions, Java PTB tokenizer, CIDEr |
| HellaSwag | Five-shot prompt; correct ending target | lm-eval likelihood of candidate endings, acc/acc_norm |

COCO selects disjoint train/demo/validation/report IDs using Karpathy splits and saves them in each run. Evaluation reuses the checkpoint's report IDs. These IDs are not claimed to match historical reports.

New HellaSwag runs reserve num_validation rows from the official training split for checkpoint selection, using data.selection_seed (default 0) independently of the training seed. The remaining rows supply optimization examples; num_train limits those remaining rows only. All split variants share the same selection IDs. Official validation is reserved for final evaluation. Saved legacy IDs without validation_split retain the old official-validation selection protocol when resuming; they are not silently migrated.

## Checkpoints and randomness

best.pt is replaced only when the monitored metric improves by the configured relative min_delta. It need not be the checkpoint with the smallest floating-point loss across every validation. last.pt is updated after validation, with optional extra save_steps.

A checkpoint includes hidden codec, optional receiver-memory codec, channel, optimizer, scheduler, scaler, configuration, data IDs and selected RNG state. It does not include the frozen backbone. Resume reconstructs that backbone and restores the saved recipe; the data iterator restarts, so exact uninterrupted batch order is not guaranteed.

Changing current YAML defaults does not alter historical checkpoints. Encoder checkpoints retain their saved design. Old decoder checkpoints requiring receiver memory but lacking a memory codec cannot load into the corrected boundary; evaluate them with archived historical source for diagnostics, or train a new corrected checkpoint. Do not initialize a random memory codec and report it as the old model.

Validation and evaluation conditions isolate/reset Python, NumPy and PyTorch RNG. A custom simulator's private RNG or cross-call state must provide its own seed/reset behavior.

See [validation coverage](validation.md), [migration decisions](migration.md), and the [channel integration guide](channel-integration.md) before extending these paths.

### HellaSwag decoder payload accounting

The decoder HFLM adapter records actual unpadded continuation token sequences
before harness batching. Each forward resolves its padded inputs back to a unique
actual continuation length and supplies a separate 2D payload-validity mask.
Ambiguous length matches fail instead of inferring length from PAD token values.
The temporary mask is restored even on exceptions and does not reset condition
counters. HFLM sorting, native input preparation, attention, scoring and full-shaped
AWGN draws remain unchanged. Decoder token-wise power does not use this mask to
couple positions; this correction affects valid-coordinate accounting.

Older direct-HFLM decoder counts may include padding and retain their historical
meaning. New HellaSwag result metadata identifies `actual-harness-continuation-lengths-v2`.
Allocated coordinates, valid coordinates and deduplicated physical traffic are
separate quantities. CPU tiny-model tests cover real harness mixed-length batches;
full-weight target-hardware acceptance remains separately recorded.

### Scoped SDPA backend policy

`model.sdpa_backend_policy` defaults to `auto`, preserving existing recipes.
The opt-in `flash_math` policy allows PyTorch Flash Attention and math SDPA,
excluding efficient and cuDNN SDPA within `SplitModel.forward()` and
`SplitModel.generate()` (including generation's encoder pass). PyTorch selects
an eligible backend; this setting does not guarantee Flash Attention is used.
The context restores the caller's backend flags on exit, including exceptions.
Direct calls to `model.base` do not inherit it: diagnostics or adapters using
those paths must explicitly enter `with model.attention_context():`.

This policy was introduced after a long-prefix COCO diagnostic on WS RTX 5090
with PyTorch 2.10 CUDA 12.8 isolated an efficient-SDPA discrepancy. It is not a
claim that H200 or other framework versions exhibit the same issue. New target
hardware still requires correctness and throughput preflight. This policy does
not change model precision, masks, power normalization or codec routing.

Current COCO task and smoke recipes opt in via `runtime.sdpa_backend_policy`,
leaving the shared model recipe and historical HellaSwag settings unchanged.
Configuration validation currently permits `flash_math` only for COCO because
HellaSwag harness adapters call the backbone directly and do not yet scope this
policy. An `auto` wrapper preserves any explicitly supplied outer SDPA context,
which lets diagnostics force math-only execution without changing dispatch globally.

### Explicit functional phase reset

The HellaSwag two-stage planner can opt into `--include-reset-control`, adding fresh functional E1 and weight-only child E to A/B/C/D. E1 uses the local-stage recipe with functional execution; E uses C's phase-two recipe. The E command requires its exact terminal parent checkpoint and `--parent-sha256`. Functional-parent transfer validates complete native functional receipts, current source, saved recipe, data IDs, exact presentation boundary and checkpoint bytes. It creates fresh optimizer/scheduler/scaler state rather than resuming training. Historical B-local transfer stays separate and unchanged. CPU lifecycle tests establish these semantics; full-weight GPU acceptance is recorded separately.

## Code responsibilities

`losses.py` contains numerical KL/reconstruction reductions. `training_objectives.py` owns the shared batch-statistics type, explicit/legacy objective weighting, and effective-batch aggregation used by functional and local reconstruction training. `training.py` owns teacher/student forwards, validation, optimizer-loop orchestration and checkpoint lifecycle. `local_reconstruction.py` owns cached reconstruction and phase-parent validation. This separation does not change objective formulas, stream masks, reduction order or schedules.

The existing public objective helpers remain re-exported from `training.py` for historical scripts and top-level instrumentation; new shared-objective code should import from `training_objectives.py`. Functional training still invokes its imported aggregate/scaled-loss names so existing instrumentation can wrap those entry points. Private globals inside moved helpers belong to the objective module.

COCO endpoint and trajectory diagnostics share input placement through `coco_diagnostic.diagnostic_inputs`: all source tensors move to the backbone device, and only image pixels adopt the backbone dtype. Diagnostic budgets, output schemas and distinct evidence-hash formats retain their existing definitions.
