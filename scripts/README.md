# Script catalog

Run commands from the repository root with the locked uv environment. The primary experiment interface is `train.py --config <task.yaml> --output <fresh-root>` for one COCO or HellaSwag run and `evaluate.py --run runs/<run-directory>` for its checkpoint. `study.py` can preview or export a multi-run plan when needed; it does not run the plan. No study manifest is required for an individual run.

| Scripts | Purpose |
|---|---|
| [`coco_query_endpoint_diagnostic.py`](coco_query_endpoint_diagnostic.py) | Verify local COCO checkpoint/source identities, then explicitly execute a bounded query-image and endpoint diagnostic. Preparation/check modes do not load models. See the [diagnostic guide](../docs/coco-query-endpoint-diagnostic.md). |
| [`coco_trajectory_diagnostic.py`](coco_trajectory_diagnostic.py) | Replay saved COCO generation trajectories and compare cached/full-prefix states under a bounded diagnostic. Requires explicit source/checkpoint/evidence bindings; [guide](../docs/coco-trajectory-diagnostic.md). |
| [`shared_encoder_atlas.py`](shared_encoder_atlas.py) | Prepare fixed HellaSwag views, then explicitly capture encoder-site activations and verify local replay. Zero optimizer updates; this is not shared-codec training. See the [atlas guide](../docs/shared-encoder-atlas.md). |
| [`hellaswag_two_stage.py`](hellaswag_two_stage.py) | Prepare/check an enc_l9 A/B/C/D plan, optionally adding the functional reset control; separate explicit capture, local reconstruction and functional phase commands. See the [two-stage guide](../docs/hellaswag-two-stage.md). No command submits a job. |
| [`hellaswag_replay_preflight.py`](hellaswag_replay_preflight.py) | Check one unchanged 64-example enc_l9 batch against local replay, with no-noise/fixed AWGN, explicit parity gates and zero optimizer updates. Config-only mode loads no model/data; [guide](../docs/hellaswag-replay-preflight.md). GPU execution requires a separately bounded allocation. |
| `compare_evidence_json.py`, `recompute_coco.py` | Reusable offline evidence comparison and COCO CIDEr recomputation utilities. The latter consumes exported per-image JSONL evidence. |
| `index_*.py`, `extract_compact_samples.py`, `verify_manifest.py`, `recompute_*.py` | Evidence indexing, extraction, verification and recomputation for specific saved artifact formats. Check each script's inputs before reuse. |
| Other retained diagnostic, policy and Slurm scripts | Inspect or explicitly execute their documented bounded work. Their old source bindings and approvals remain historical evidence, not current execution authorization. |
| [`experiments/`](experiments/README.md) | The relocated evening and main-weight screen modules, with exact old-to-new module names. |

For a reusable offline recomputation, for example:

```bash
uv run --locked python scripts/recompute_coco.py path/to/captions_no_noise.jsonl
```

Keep historical manifests, source hashes and result paths with their archived evidence. A new execution needs a fresh recipe and source binding; moving a script does not revise an earlier result or authorize another run.

## Versioned baseline and local records

- `encfn_baseline.py` previews the finite four-cell plan, executes one explicitly prepared segment, or writes exact comparisons for a complete campaign. See [baseline guide](../docs/encfn-baseline.md).
- `render_experiment_report.py` verifies saved scalar/artifact inventories and creates local PNG/CSV/HTML reports. See [records guide](../docs/experiment-records.md). It does not load models or launch jobs.

## Retired training surfaces

The closeout production harness and prompt-alignment pair executor are retired from current source. Native preflight retains only imported indexing/collation and gradient-check helpers; prompt-policy retains strict `sha`/`validate_run` readers. Their module training executors are unsupported. The prompt-alignment shell retains its independent benchmark branch, and the prompt-policy shell retains exact64 preflight. Standalone `evaluate_prompt_alignment.py`, fixed prompt YAMLs and historical plans/results/source packages remain available. These validators retain source gates and may require the matching historical source environment. Future training uses explicit ordinary roots and exact recorded checkpoints; retirement does not rewrite scientific receipts or release held sources.
