# Research scope and open questions

The September report and project cleanup are complete. This document records research questions and possible comparisons, not an approved experiment plan or report deadline. The owner has selected local implementation of the COCO query/endpoint diagnostic and HellaSwag enc_l9 two-stage pilot; actual experiment jobs still await confirmation. The main research question remains how split location changes task quality and communication cost across COCO and HellaSwag. New model runs, including timing preflights, require an agreed question, protocol and budget.

## Current working baseline

The [shared baseline](architecture.md#shared-baseline) lives in configs/model.yaml, referenced by both task YAMLs. It aligns COCO and HellaSwag model design and disables SNR-FiLM, providing a consistent starting point for development and physical-channel integration. The September 29 default adoption follows the completed enc_l9 recipes; [current defaults](current-defaults.md) records their training and evaluation lineage. Historical results must retain the exact architecture and protocol that produced them.

An initial architecture can be a research prototype without being an optimized design. Rigor comes from a clear system model, justified comparisons, controlled protocols and conclusions bounded by the evidence. Exhaustively testing every hyperparameter is not required.

## Interpret existing evidence

For each useful figure or experiment, record:

| Field | Question |
|---|---|
| Research question | What comparison or claim does this result address? |
| Provenance | Which source version, configuration and checkpoint produced it? |
| Protocol | Which data split, demos, seeds, power normalization and evaluation settings were used? |
| Signal path | Which representation was transmitted, and was any encoder memory kept clean? |
| Outcome | Which task metrics, loss curves and resource measurements are available? |
| Comparability | Which other runs used the same relevant conditions? |
| Evidence gap | Can the claim be supported now, is reevaluation enough, or is new training necessary? |

Keep personal artifact paths and the report-to-result map in ignored `docs/local/research/`; the completed historical evidence and source snapshots are in the MTK archive outside this Git project. Classify results as usable comparisons, diagnostic observations, or unresolved evidence. Historical defects qualify conclusions; they do not justify silently reinterpreting old scores. Missing artifacts may warrant retraining when a comparison matters and the owner agrees to its scope. Early Llama work is a separate stage, not a direct T5Gemma baseline.

## Possible paper outline

1. Problem and motivation: task performance under constrained communication resources.
2. System model: model split, transmitted symbols, channel assumptions and available SNR information.
3. Method: frozen backbone, codec, objectives and training procedure.
4. Experimental protocol: datasets, baselines, data separation, budgets, seeds and metrics.
5. Results: comparisons supported by consistent protocols, including negative or inconclusive findings.
6. Limitations and open questions: missing controls, modeling assumptions and untested settings.

This outline is a possible way to present supported findings; it does not set a writing or experiment deadline.

## Questions for later experiments

### Bottleneck and model capacity

Separate hidden width (codec capacity/compute) from bottleneck width (transmitted representation size) and block count (depth). A compact study could compare narrower/current/wider bottlenecks at selected splits, holding other choices fixed. Report task quality against communication use and compute, rather than searching for one best score.

Bottleneck dimensions are not bits. Define real/complex packing, token count and channel uses per sample before making bitrate or bandwidth claims. Decoder generation can have different transmission lengths from teacher-forced training.

### Residual connections

Each block currently computes x + F(x). Setting n_res_blocks to zero changes depth, parameter count and nonlinearity as well as removing residual blocks. To isolate the value of the skip connection, compare matched blocks with and without the addition while holding the rest of the architecture fixed. A matched non-residual variant is a future experiment, not an existing config option.

### SNR-FiLM

FiLM is available but disabled by default. An earlier lack of visible benefit does not establish that training was too short. First inspect optimization curves, parameter updates and whether modulation varies with SNR; then compare separately trained on/off models with matched task, split, bottleneck, data and training budget. Removing FiLM from an already-trained model is not equivalent to training a no-FiLM baseline.

If useful, a constant-condition FiLM control can help distinguish extra capacity from useful SNR information. Receiver SNR mismatch is another candidate: compare true channel SNR against the SNR reported to FiLM. The current wrapper couples them, so such a study would require an explicit interface change.

FiLM and SNR-adaptive JSCC are established ideas; adding conditioning alone does not establish novelty. Potential contributions must be compared against relevant work, especially for intermediate language-model representations and task-level communication tradeoffs:

- [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871)
- [SNR-adaptive deep joint source-channel coding for wireless image transmission](https://arxiv.org/abs/2102.00202)

## Future design discussion

The shared codec and FiLM-off baseline provide a reference for comparisons. Physical-channel tensor, power, SNR and state conventions, FiLM, and other architectural variations remain candidate topics. Select any new experiment only after comparing the relevant historical source, configuration, data, prompt, transmission and evaluation semantics with the owner; no candidate here is approved for execution.
