# Study preview, export and optional transport

A study describes independent ordinary task configurations. Preview uses the single entry with `operation: preview` and `execution.plan: {kind: study, path: PLAN}`. Optional plan.task filters COCO or HellaSwag. Optional plan.output exclusively exports a fresh directory. Preview/export never trains or invokes a scheduler; study is a plan kind rather than another operation. See [requests](operations-workflow.md).

Plans own name, declaring-file-relative task_configs, unique nonnegative seeds and named experiments with model_overrides. `expand_study` reuses ordinary config composition and split/design validation. Baseline and bottleneck are current generic plans. Historical encfn/outer-LN protocol schemas remain reader/protocol records and do not silently become generic studies.

Exports write complete one-operation train requests, an index manifest and explicit training links. All model/task references resolve before export, so the prepared directory can move between hosts. Complete effective scientific configuration and source attribution accompany the manifest. No implicit resource budget, latest-run search or scientific launch is introduced.

Optional `jscc.study_task` and `scripts/slurm_study.sh` transport one exact manifest index. Training generates an explicit root/attempt/success_link request; evaluation reads its recorded training link and generates a fresh linked request with recorded checkpoint/expected_step/overrides. Different manifests have distinct request filenames and roots. Existing output/link collisions, symlink escapes and duplicate training links reject. Evaluation-only manifests may share training links while retaining distinct evaluation attempts/outputs.

The optional `scripts/slurm_job.sh` executes supplied explicit requests sequentially. It does not infer a train/evaluate workflow from a task filename. Scheduler resource/dependency/array values remain external declared settings, and exported studies do not authorize a sweep.
