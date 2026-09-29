# Script catalog

Run commands from the repository root with the locked uv environment. The primary experiment interface is `train.py` for one COCO or HellaSwag run and `evaluate.py --run runs/<run-directory>` for its checkpoint. `study.py` can preview or export a multi-run plan when needed; it does not run the plan. No study manifest is required for an individual run.

| Scripts | Purpose |
|---|---|
| `compare_evidence_json.py`, `recompute_coco.py` | Reusable offline evidence comparison and COCO CIDEr recomputation utilities. The latter consumes exported per-image JSONL evidence. |
| `index_*.py`, `extract_compact_samples.py`, `verify_manifest.py`, `recompute_*.py` | Evidence indexing, extraction, verification and recomputation for specific saved artifact formats. Check each script's inputs before reuse. |
| Other root-level diagnostic, preflight, policy, closeout and Slurm scripts | Reproduce or inspect bounded historical work. Some start training, evaluation or scheduler jobs; their old source bindings and approvals are historical evidence, not current execution authorization. |
| [`experiments/`](experiments/README.md) | The relocated evening and main-weight screen modules, with exact old-to-new module names. |

For a reusable offline recomputation, for example:

```bash
uv run --locked python scripts/recompute_coco.py path/to/captions_no_noise.jsonl
```

Keep historical manifests, source hashes and result paths with their archived evidence. A new execution needs a fresh recipe and source binding; moving a script does not revise an earlier result or authorize another run.
