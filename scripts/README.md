# Script catalog

Scripts provide concrete diagnostics, local input preparation, scheduler transport and saved-evidence tools. The public scientific request migration is owned by the single-entry workflow: `experiment.py --config REQUEST.yaml`, with one YAML-selected operation per invocation. The entry dispatches the current ordinary/shared COCO and HellaSwag consumers; retaining a diagnostic here does not authorize a scientific run.

| Scripts | Purpose |
|---|---|
| `benchmark_speed.py`, `benchmark_evaluation.py` | Bounded throughput or evaluation-capacity measurements for exact recorded inputs/checkpoints. Preserve effective configuration, optimizer restoration, payload and telemetry evidence. |
| `shared_encoder_atlas.py`, `hellaswag_replay_preflight.py` | Explicit zero-update activation/replay diagnostics with input identities, masks, paired noise and gradient parity. |
| `objective_optimizer_equivalence.py`, `split_wiring_smoke.py` | Controlled objective/optimizer equivalence and small split wiring diagnostics. Their restored hooks and raw parameter/payload evidence remain part of their contracts. |
| `ordering_score_diagnostic.py`, `backbone_anomaly_probe.py` | Historical ordering/companion scoring and anomaly localization. Original captures and source identities retain their saved meanings. |
| `coco_*diagnostic.py`, `coco_capacity_check.py`, `coco_native_preflight.py`, `coco_route_smoke.py` | Bounded image, route, cache, precision, memory and replay/scorer diagnostics. They do not establish full-weight shared COCO acceptance. |
| `five_shot_preflight.py`, `exact64_preflight.py`, `prompt_alignment_benchmark.py`, `payload_precision_probe.py` | Purpose-specific capacity, gradient, prompt and saved payload/precision checks. |
| `prepare_shared_inputs.py`, `prepare_native16_inputs.py` | Explicit local input preparation with exact source, cache, membership, initializer and panel bindings. No training or submission. |
| `evaluate_historical_weight_screen.py` | Historical saved-checkpoint evaluator restricted to dec_l20 step500/1000 and its frozen fixture, source and global request budget. It is not an ordinary/shared authoring frontend. |
| `render_shared_terminal_report.py`, `render_experiment_report.py` | Offline bound terminal/scalar/pairing reports and verified local PNG/CSV/HTML evidence. |
| `index_*.py`, `extract_compact_samples.py`, `verify_manifest.py`, `compare_evidence_json.py`, `recompute_*.py` | Offline indexing, containment/hash verification, comparison and recomputation of named saved artifact formats. |
| `submit_recipe.py`, `slurm_job.sh`, `slurm_study.sh`, `slurm_outer_ln.sh` | Optional CASE/study transport and reconciliation. All current scientific commands use experiment.py and explicit one-operation YAML requests; acquire/register/prepare/collect use the same entry. |
| Other `slurm_*.sh` | Optional transport around concrete diagnostics. The obsolete rejection-only COCO/exact64/five-shot group wrappers are retired; frozen bundles retain their exact original source wrappers. |

Scoring/intervention helpers now live in `jscc/diagnostic_scoring.py`; VJP, feature and pooled gradient helpers in `jscc/gradient_diagnostics.py`; codec family/role initialization in the existing `jscc/models/codec.py`. Indexed collation is in `jscc/shared_inputs.py`, trainable-gradient admission in `jscc/qualification.py`, and named saved prompt/COCO/reconstruction/baseline readers in `jscc/checkpoint_readers.py`. Historical finite-update configuration/order instrumentation is a test fixture provider in `tests/model_helpers.py`.

The campaign-specific original13, staged/pilot, sharing, native-policy and baseline execution fronts have retired. Their useful readers, helper bodies and scientific assertions transfer to the providers above; historical schemas, noise strings, parameter keys, checkpoint/data/evidence bytes and frozen source holds are preserved. A moved provider creates a new source identity and does not relabel an old result.

For offline COCO recomputation:

```bash
uv run --locked python scripts/recompute_coco.py path/to/captions_no_noise.jsonl
```

For a separately authorized speed benchmark:

```bash
uv run --locked python scripts/benchmark_speed.py --checkpoint path/to/checkpoint.pt --output path/to/benchmark --variant streamed --batch-size 16 --accumulation 2
```

The checkpoint supplies configuration and data IDs. Each repeat reports effective configuration and its hash separately from checkpoint identity; restored optimizer LR/weight-decay state retains its original meaning. A diagnostic result is not GPU fit, throughput, quality or scientific acceptance for another workflow.
