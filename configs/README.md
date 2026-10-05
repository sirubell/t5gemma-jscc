# Authored configuration

Public requests use `experiment.py --config FILE` and `jscc-experiment-v1`. Supplied ordinary/science examples select check; a shared check needs a local pinned backbone config.json. Use [operations](../docs/operations-workflow.md) to author explicit prepare/train/resume/evaluate/collect/preview/submit requests.

| Directory | Purpose |
|---|---|
| models | Reusable named model/split/codec/channel designs, including default and shared direct-affine |
| experiments/ordinary | COCO/HellaSwag ordinary science with data, clock, selection and scoring |
| experiments/shared | Dedicated effective shared base and ordered-site/quota/objective/panel science |
| requests | Recorded evaluation/collection requests; task-specific evaluation options remain separate files |
| studies | Nonexecuting study/protocol plans; historical protocol schemas are not generic studies |
| smoke | Explicit bounded check examples and their preserved CPU/H200 runtime settings |

Paths are declaring-file-relative. A complete effective science configuration is recorded at binding. `execution.root` owns the operation output; inherited `run.output_dir` is not an implicit root. Width-only families share `shared_direct_affine.yaml` and explicit `experiment.codec.bottleneck_dim`; q200/effective64, q400/local-files-only and native16-long retain their distinct protocol values. Historical saved config and evidence bytes are immutable.

Shared native source 2048/target 512, batch16/accumulation1, direct-affine/native arithmetic, two phases and one optimizer/LR reset remain supported. Unsupported or inactive shared overrides reject; ordinary selection/stopping keys are not shared controls. Shared COCO uses `coco_base.yaml` and its check example, with acquired image, caption, demo, reference, processor and scorer identities. Checks do not load assets or launch training. Shared COCO and HellaSwag use whole-panel single-case parallel evaluation through `evaluation.parallel`; exact controls and retry/failure semantics are in the operations guide.
