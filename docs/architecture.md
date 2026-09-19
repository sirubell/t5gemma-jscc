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

## Shared baseline

COCO and HellaSwag reference `configs/model.yaml` for the same backbone revision, encoder-final-norm split, codec architecture and channel defaults. This also applies to paired CPU and H200 smoke recipes; device/dtype vary by execution environment, not by task. Task YAMLs own data, training and evaluation. The loader composes them once and saves the complete result in each run/checkpoint.

| Component | Baseline |
|---|---|
| Split | Encoder after final norm; no layer index is required |
| Codec hidden width | 1152 |
| Bottleneck width | 512 |
| Residual blocks | Two in each codec half; each block computes x + F(x) |
| Activation / dropout | GELU / 0 |
| External LayerNorm | Both input and output; internal block norms remain enabled |
| Channel | AWGN with per-sample power normalization |
| SNR-FiLM | Disabled |

This baseline makes the task designs consistent; it is not an experimentally established optimum. Task-specific preprocessing, targets, training budgets and metrics still differ. A study may explicitly override the split or codec settings.

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
| HellaSwag | Context prompt; correct ending target | lm-eval likelihood of candidate endings, acc/acc_norm |

COCO selects disjoint train/demo/validation/report IDs using Karpathy splits and saves them in each run. Evaluation reuses the checkpoint's report IDs. These IDs are not claimed to match historical reports.

New HellaSwag runs reserve num_validation rows from the official training split for checkpoint selection, using data.selection_seed (default 0) independently of the training seed. The remaining rows supply optimization examples; num_train limits those remaining rows only. All split variants share the same selection IDs. Official validation is reserved for final evaluation. Saved legacy IDs without validation_split retain the old official-validation selection protocol when resuming; they are not silently migrated.

## Checkpoints and randomness

best.pt is replaced only when the monitored metric improves by the configured relative min_delta. It need not be the checkpoint with the smallest floating-point loss across every validation. last.pt is updated after validation, with optional extra save_steps.

A checkpoint includes hidden codec, optional receiver-memory codec, channel, optimizer, scheduler, scaler, configuration, data IDs and selected RNG state. It does not include the frozen backbone. Resume reconstructs that backbone and restores the saved recipe; the data iterator restarts, so exact uninterrupted batch order is not guaranteed.

Changing current YAML defaults does not alter historical checkpoints. Encoder checkpoints retain their saved design. Old decoder checkpoints requiring receiver memory but lacking a memory codec cannot load into the corrected boundary; evaluate them with archived historical source for diagnostics, or train a new corrected checkpoint. Do not initialize a random memory codec and report it as the old model.

Validation and evaluation conditions isolate/reset Python, NumPy and PyTorch RNG. A custom simulator's private RNG or cross-call state must provide its own seed/reset behavior.

See [validation coverage](validation.md), [migration decisions](migration.md), and the [channel integration guide](channel-integration.md) before extending these paths.
