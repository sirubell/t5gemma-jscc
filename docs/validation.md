# Validation coverage

Earlier real-environment checks established execution for the recipes recorded below. The later receiver-only decoder-memory correction and HellaSwag training-holdout selection are covered by CPU regression tests; full-weight checks of these changes are still required before a new research sweep. Historical execution does not establish current model quality or convergence.

## Automated tests

```bash
uv sync --locked --extra dev
uv run --locked --extra dev pyright
uv run --locked --extra dev ruff check .
uv run --locked --extra dev python -m pytest -q
uv run --locked --extra dev python -m pytest -q tests/test_coco.py
uv run --locked --extra dev python -m pytest -q tests/test_hellaswag.py
uv run --locked --extra dev python -m pytest -q tests/test_sweep_routes.py
```

The suite uses CPU tensors and tiny real T5Gemma-2 modules without pretrained downloads. Coverage includes shared-design composition, runtime overrides, saved flat-config independence, task alignment with FiLM off, study expansion/export, train/eval pairing, and the absence of FiLM parameters when disabled. New tests verify sender-clean/receiver-transmitted memory, shared memory realizations, cached generation without repeated memory transmission, gradients to both codecs, checkpoint/resume, and disjoint HellaSwag training/selection rows with legacy-ID compatibility.

Pyright and Ruff are pinned project dev dependencies in pyproject.toml and uv.lock. Use the project uv commands above rather than a machine's Mason/global executables. Third-party dynamic registry/dataset boundaries are explicitly typed; missing-import diagnostics are not globally disabled.

| Area | Coverage |
|---|---|
| COCO | Data separation, multimodal forward/backward/generation, image channel routing after vision scatter, receiver-only memory transmission |
| HellaSwag | Padding/labels and actual HFLM seq2seq likelihood through the codec |
| Shared core | LayerNorm choices, power/SNR, channel replacement, gradients and frozen backbone, RNG isolation, scheduler, checkpoint and resume |
| Configs/studies | All supplied plan counts, task filters, seeds, isolated overrides, relative paths, portable resolved exports, preview without execution, failed-train and same-index evaluation routing |
| Sweep routes | Both tasks at all 13 split locations and all three bottleneck widths; small CPU forward/backward/generation, finite losses, nonzero codec gradients and frozen backbone |

The image-route fixture assigns a nonzero image projector: Transformers initializes an untrained projector to zero, which otherwise erases all pixel differences. This fixture change does not modify pretrained model loading.

The COCO regression checks that image pixels affect transmitted representations, AWGN changes image positions, and the first text layer receives the reconstructed image positions. A control probe moving the hook back to token embeddings must fail because later image scatter overwrites those positions. At decoder splits, transmitter layers retain original encoder memory, while receiver layers use separately transmitted memory; decoder-memory tests cover that boundary and cached generation.

## Real-weight execution

Historical observations below retain the recipes used at execution time. The later study-array smoke validates the current shared encoder-final-norm / LayerNorm-both / FiLM-off baseline on full weights; it does not relabel historical results as belonging to that baseline.

| Environment/task | Observed result |
|---|---|
| CPU workstation, both tasks | Small real-weight training, checkpoint reload and task evaluation passed |
| RTX 5090 CUDA/BF16 | Both tasks passed current-baseline full-weight smoke: four updates, batch 1, accumulation two, validation, checkpoint reload and all three small evaluation conditions; training peak allocated memory was 4.80/4.34 GiB for COCO/HellaSwag |
| H200 BF16, both tasks | Short real-weight train/validation/evaluation passed |
| H200 HellaSwag research recipe | Early stopping at step 16000; best checkpoint at 13500 evaluated on all 10042 examples, five-shot, 11 conditions |
| H200 COCO research recipe | Training log reached step 5293/6000 before scheduler timeout; last/best saved at 5000; full task evaluation did not start |
| H200 short check after type cleanup | Both tasks completed four updates, accumulation two, validation, checkpoint reload and all three small evaluation conditions in one 1 min 37 sec job |
| H200 current-baseline study arrays | Separate train/evaluate jobs for both tasks all completed with exit 0; each trained four updates and evaluated no-noise / 0 dB / vanilla on two samples, using the matching step-4 checkpoint; overall execution interval was 1 min 30 sec |

The H200 runs used PyTorch 2.10.0+cu128 and reported BF16 support with compute capability (9, 0). COCO generation/CIDEr passed small-scale checks, but the final long-run COCO codec was not evaluated. The owner elected to end the executability check rather than continue it.

The RTX 5090 check used driver 570.211.01, PyTorch 2.10.0+cu128, capability (12, 0), and BF16 support. It ran sequentially without Slurm and exited successfully. Project-managed Pyright and Ruff also passed on that Linux workstation after moving legacy macOS metadata sidecars out of the source tree and excluding non-source backups. Full-batch and all-variant GPU memory limits remain unmeasured.

## Practical limits

- Forced scheduler timeout does not save the latest in-memory update. The interrupted COCO run lost the updates after its last saved checkpoint.
- The main research YAMLs do not set max_minutes; no graceful time-budget stop was configured in that interrupted run.
- Tiny-model tests establish routes and gradients, not image-caption quality.
- FP8, compile, shared encoder computation and batch/worker alternatives have not been benchmarked for this project.
- A collaborator's physical channel has not yet been supplied or validated.
- Beam search with receiver-memory coding is not supported. Tests use greedy generation; feedback communication and per-context deduplication of evaluation channel-use counts remain outside the simulator.

## What a sweep test does and does not prove

1. Config/plan tests verify expansion, names, paths and train/eval pairing without GPU work.
2. Sweep-route tests run all 26 split entries and six bottleneck entries on small 26-layer Transformers models. They preserve selected indices and bottleneck widths, but reduce hidden size, vocabulary and image resolution. They test computation paths, not GPU memory or statistical performance.
3. A full-weight smoke on the target GPU checks kernels, data/processor integration, memory at the selected batch, checkpoint reload and actual task metrics. Shared-baseline smoke has passed on H200 and RTX 5090, but does not cover every variant or larger-batch memory fit. Include memory-heavy configurations before a large sweep.
4. Full-duration multi-seed sweeps collect research evidence. They are not required just to check the software and should only run for an explicit research question.

Private job IDs, machine/account details, full metrics and artifact paths remain in ignored docs/local/experiments.md and docs/local/ws-validation.md on the owner's checkout.

## Study timing and HellaSwag evidence

Training writes local `metrics.jsonl` regardless of W&B configuration: loss, KL, nMSE, learning rate, sampled update time and GPU allocated-memory peak. Interval wall throughput includes intervening validation/checkpoint overhead; elapsed loop time excludes model/data setup. Validation rows include their own duration. W&B remains optional and disabled in the default task recipe.

Evaluation records per-condition wall time. HellaSwag writes harness metadata and per-example samples beside aggregate results; its dataset name/revision follows the saved data configuration. These records support paired analysis and execution provenance. Noise is still sampled per forward call/candidate batch, not a shared physical realization for all endings of a question.

## 2026-09-19 closeout scope

The speed implementation has full-weight RTX5090 paired acceptance on encoder layer9 and decoder layer8 with receiver memory: BF16 frozen backbone, FP32 trainable codecs/Adam,16×2, three real variable batches and three nonzero controlled updates, no-noise and explicit fixed AWGN. The original nondeterministic reference failed its own gradient/update replay tolerance. The accepted policy enables `training.deterministic_algorithms: true`, CUBLAS workspace `:4096:8`, and controlled branch RNG reset. This changes execution reproducibility, not the model objective; it is not a guarantee across devices or software versions.

Streamed backward and valid-only full-vocabulary KL were equivalent under that policy. Actual AWGN production calls confirmed both switches and the validation path. The other eleven split positions passed smaller batch2 train/reload/payload smoke; these are not full16×2 numerical or quality tests. COCO, H200, full-length corrected training and broad seed robustness were not tested in this closeout.

HellaSwag fixed batch8 replay was exact on the512 development panel, but crossbatch BF16 forward sensitivity remains: a large candidate-score anomaly arose before scoring and was greatly reduced by a small FP32-forward probe. Keep evaluator precision, batch composition/panel and backend explicit. A fixed-batch result must not be described as batch-invariant or a full-precision reference. Owner-specific raw evidence and final gates are in ignored local research records.


The final closeout also found a native input-preparation mismatch: the old
training helper used PAD0 when the top-level start-token field was absent,
whereas T5Gemma2's native method and the HellaSwag adapter used decoder BOS2.
`model_inputs` now delegates to the native model API; custom backbones without
that API retain the prior fallback. This is explicitly versioned as
`corrected-baseline-v2-native-decoder-inputs` in the prospective study recipe.
The failing regression was reproduced before the fix. Native-prefix paired
CUDA checks, all13 route checks and production timing were rerun within the
bounded closeout budget. Tensor differences were zero; scalar losses agreed
within the preregistered tolerance. Old PAD-prefix results remain historical,
and no native-v2 task-quality improvement has been measured. Existing training
labels/context and five-shot evaluator prompts are not claimed to be identical.
Use recorded source revisions when reproducing older recipe/protocol labels.
