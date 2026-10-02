# Research scope and open questions

## 研究工作流程（2026-10-02）

採用輕量的碩士研究流程。目標是在 **2026年10月6日（星期二）** 推進「所有原始 split points 共用 codec」研究；三個 encoder 位置是初步驗證，不是永久範圍。這是研究目標，不是完成保證或額外 GPU 授權。最新決策、執行狀態與結果以本機 [HANDOFF](local/research/HANDOFF.md) 和 [研究 tickets](local/research/wayfinder/codec-research/map.md) 為準；下方舊研究背景不代表目前尚未開始實驗。

1. 開始時只確認任務、目前 owner、source commit、既有結果及下一步。固定實驗 branch 與完整 Git SHA，開發使用獨立、短名稱 worktree（例如 `development`）。不要修改正在執行的 pinned source；既有 [本機配置](local/research/analysis/git-deployment-layout-audit-20261001/WORKFLOW.md) 保留歷史路徑。
2. Server 使用固定 checkout、必要 config／input／checkpoint／model assets，以及已存在的直接執行指令。先 `uv sync --locked` 建立環境，再使用該環境固定的 `.venv/bin/python`；已有驗證過的 host runtime 可沿用。不要為每個 config 重建部署框架或重做全套稽核。
3. 改動行為才跑相應的 focused tests；共用核心變動或實際失敗需要時再擴大。每個 job 保留必要的 GPU assignment、input/checkpoint 身分與可讀性、nonfinite 停止、儲存成功檢查，以及每 job 終止上限與真實耗時。不要新增累積 GPU 剩餘額度報表；歷史 accounting 原樣保留。
4. 可並行處理不重疊的工作，由一位 owner 負責整合與 deployment。明確交接一次；不要反覆詢問閒置 worker。完成後立即更新 ticket、evidence/result 路徑、科學結論與完成時間。Training、evaluation、collection/report/upload 分開記錄狀態；收集或上傳失敗不能抹去已驗證成功的科學工作，缺少必要 evaluation 時也不能宣稱整體完成。
5. 對使用者用繁體中文，worker prompts 用英文；架構寫全名，例如「transmitter/receiver 各一層 Linear、無外部 LayerNorm」，不只寫 D-LN。Local sessions 偏好 GPT-6.1 Sol、High、Standard、非 Fast；這是使用者設定偏好，文件不會替 session 切換模型。

新 canonical session 從 AGENTS.md 進入本節。新 worktree 應確認是否含此文件版本；**舊 frozen checkout 不會自動繼承新文件**，可直接讀 canonical 文件而不修改實驗 SHA。外部 server／Colab session 需由 owner 提供本指南、[Colab runbook](running.md#colab-runbook2026-10-02) 與必要任務摘要；ignored `docs/local/` 不會隨 clone 出現。不要假設文件尚未 commit／publish 時外部 clone 能讀到。

可交給 worker 的簡短英文 prompt：

> Read the current research workflow and assigned task handoff. Keep the experiment source pinned. Use the existing command and validated Python runtime, run focused checks for changed behavior, and report scientific status, elapsed time, artifact paths, and the next concrete blocker. Coordinate disjoint ownership with the integration owner; do not redesign the workflow or launch extra jobs.

上述是流程偏好，不是 blanket authorization；沿用本次任務已給的 scope，缺少具體 action/target 時只指出真正缺口，不複製自訂 permission 規則。

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
