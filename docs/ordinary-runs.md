# Ordinary roots and exact continuation

Ordinary COCO and HellaSwag use `experiment.py --config REQUEST` with `workflow: ordinary`. They retain their own data/model/loss/scoring implementations; they do not inherit shared sites, native16 quotas or frozen-terminal semantics. See [operation fields](operations-workflow.md) for complete requests.

Authored science composes a declaring-file-relative named `model_config`, task/data/training/evaluation settings and optional supported runtime overrides. Routine width choices may specify only `codec.bottleneck_dim` alongside the named design. Other architectural changes select a named model design. Supported custom channels retain their `module:Class` and kwargs consumers.

`operation: check` composes without model/data loading or output. `prepare` uses an explicit fresh root and empty preparation mapping to bind science without training. `train` uses explicit root/attempt; a previously prepared root must match exactly and have no previous training. No occupied failure is implicitly retried.

The singleton case lives at its logical root. Science is fully resolved in config.yaml and hashed by case.json. Attempts own runtime/input records, metrics and checkpoints. A training request may publish a fresh success_link only after returning; its destination must be disjoint from the logical root. Link publication failure preserves the completed scientific attempt.

COCO preserves image/data/demo partition guards, image-aware teacher targets, caption/CIDEr scoring, receiver-only reconstructed memory and valid stream-mask/gradient denominators. HellaSwag preserves disjoint optimization/selection membership and candidate scoring. Both preserve ordinary max/schedule/save/selection/stopping controls and the recorded eligible best checkpoint.

`resume` requires the bound root, explicit new attempt and a continuation mapping containing exact checkpoint path, owning parent_attempt, role, step, size_bytes and sha256. Ordinary role is `ordinary-post-update`. The underlying loader also restores deterministic data position, optimizer/LR/scaler/RNG, counters and selection state. Repeat work since the last valid save; do not combine an older state with later counters. Source/runtime/config/channel and durable byte guards remain unchanged. Legacy weight-only files are not exact continuation states.

`evaluate` requires a recorded root and new attempt. Its empty evaluation mapping selects recorded eligible best; explicit checkpoint filename and expected_step select an exact supplemental target. Evaluation-only overrides select existing task-correct settings. Distinct codec/vanilla requests retain separate attempt/output identities. Mutable authored training YAML is never consulted to find a newer checkpoint.

Training, evaluation and collection each preserve their own completion/failure records. Missing scoring or copy evidence cannot erase completed training. A failed or interrupted attempt remains retained. Publication/reconciliation of an already completed continuation preserves its previous checkpoint/run ancestry without repeating optimizer work.
