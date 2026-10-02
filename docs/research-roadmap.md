# Research scope and open questions

## Research workflow (2026-10-02)

Use a lightweight master's research workflow. The goal for **Tuesday, October 6, 2026** is to advance a shared codec across all original split points; the three encoder positions provide initial validation, not a permanent scope restriction. This is a research goal, not a completion guarantee or additional GPU authorization. Consult the local [HANDOFF](local/research/HANDOFF.md) and [research tickets](local/research/wayfinder/codec-research/map.md) for current decisions, execution status and results. The earlier context below does not mean that experiments have yet to start.

1. Start by identifying the task, current owner, source commit, existing results and next step. Pin the experiment branch and full Git SHA; develop in a separate, short-named worktree such as `development`. Preserve running source pins; the existing [local layout](local/research/analysis/git-deployment-layout-audit-20261001/WORKFLOW.md) retains historical paths.
2. On servers, use a pinned checkout, necessary configuration/input/checkpoint/model assets and the existing direct command. Set up with `uv sync --locked`, then use that environment's fixed `.venv/bin/python`; an already validated host runtime can be reused. Avoid rebuilding deployment infrastructure or repeating full audits for each configuration.
3. Run focused tests for changed behavior; broaden coverage for shared-core changes or actual failures. Retain essential GPU-assignment, input/checkpoint identity and readability, nonfinite-stop and save-success checks, plus per-job termination limits and actual elapsed time. Do not add cumulative GPU-remaining reports; preserve historical accounting.
4. Parallel work may cover disjoint scopes, with one integration/deployment owner. Make one explicit handoff rather than repeatedly prompting idle workers. Update tickets, evidence/result paths, scientific conclusions and completion times promptly. Track training, evaluation and collection/report/upload separately: collection or upload failures do not erase verified scientific success, while missing required evaluation still prevents declaring the whole experiment complete.
5. Explain results to the user in Traditional Chinese and write worker prompts in English. Use full architecture names, such as “one Linear layer per transmitter/receiver, without external LayerNorm,” rather than only D-LN. Local sessions prefer GPT-6.1 Sol, High, Standard, without Fast; this preference does not itself change session settings.

New canonical sessions reach this section through AGENTS.md. Check whether a new worktree includes this document version. **Older frozen checkouts do not automatically inherit new documentation**; read the canonical guide without changing the experiment SHA. For external server/Colab sessions, the owner supplies this guide, the [Colab runbook](running.md#colab-runbook-2026-10-02) and the necessary task summary. Ignored `docs/local/` files do not accompany a clone. An external clone cannot see documentation that has not been committed and published.

Suggested short worker prompt:

> Read the current research workflow and assigned task handoff. Keep the experiment source pinned. Use the existing command and validated Python runtime, run focused checks for changed behavior, and report scientific status, elapsed time, artifact paths, and the next concrete blocker. Coordinate disjoint ownership with the integration owner; do not redesign the workflow or launch extra jobs.

These workflow preferences are not blanket authorization. Follow the task's established scope; identify any genuinely missing action/target without copying custom permission rules into documentation.

## Earlier research context

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
