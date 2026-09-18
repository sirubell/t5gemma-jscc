# Configuration layout

```text
configs/
  model.yaml                   # One shared backbone/split/codec/channel design
  tasks/
    coco.yaml                  # Data, training and evaluation defaults
    hellaswag.yaml
    hellaswag_diagnostic.yaml  # Five-setting pre-H200 HellaSwag diagnostic
    hellaswag_corrected.yaml   # Three-route corrected-protocol baseline
  smoke/
    coco_cpu.yaml              # Tiny budgets on CPU / float32
    hellaswag_cpu.yaml
    coco_h200.yaml             # Tiny budgets on CUDA / bfloat16
    hellaswag_h200.yaml
  evaluation/
    smoke.yaml                 # Small evaluation-only overrides, either task
    wireless.yaml              # Custom-channel evaluation example
  studies/
    baseline.yaml
    splits.yaml
    bottleneck.yaml
    hellaswag_diagnostic.yaml  # 5 settings, WS/5090 only before formal H200
    hellaswag_corrected.yaml   # 3 routes, prepared before H200 submission
    smoke_h200.yaml
```

## Which file should change?

- Change the shared architecture in model.yaml, or copy it to a named model design for an explicit variation.
- Change task data/budgets/metrics in tasks/coco.yaml or tasks/hellaswag.yaml. They both reference ../model.yaml.
- Use smoke/ for execution checks. CPU and H200 have different execution precision/budgets but share the model design. FiLM is off.
- Pass evaluation/ files to evaluate.py with --config. They override evaluation settings from a checkpoint, not the checkpoint's model architecture.
- Use studies/ to list a set of experiments without duplicating model/task configurations. Study model_overrides apply equally to both tasks.

`codec.memory` is an optional nested override for the receiver-memory codec. It inherits the main codec design and can pin a memory-specific boundary norm; a flat codec configuration keeps the historical shared setting.

The *_h200.yaml files are ordinary CUDA/bfloat16 smoke recipes, not H200-only model implementations. A compatible RTX 5090 can try the same small recipes directly with Python; see [standalone GPU execution](../docs/running.md#standalone-gpu-workstations-including-rtx-5090). Keep task-specific batch/resource adjustments in task recipes and record the actual hardware used.

## Supplied study plans

| Plan | Variations per task | Seeds | Training configurations |
|---|---:|---|---:|
| baseline | Shared model | 0 | 2 |
| splits | 13 encoder/decoder locations | 0 | 26 |
| bottleneck | 256, 512, 1024 | 0 | 6 |
| smoke_h200 | Shared model, H200 smoke budgets | 0 | 2 |
| hellaswag_diagnostic | Five HellaSwag split/norm settings | 0 | 5 |
| hellaswag_corrected | Corrected protocol: enc_fn, enc_l9, dec_l8 | 0 | 3 |

Each training configuration can have a separate evaluation job. Evaluation SNRs do not multiply the number of trained models. All supplied plans keep FiLM off.

The split plan lists the 13 historical locations under the current shared codec design with boundary-aware normalization: `post` at `enc_emb`, `both` at `enc_fn`, and `none` at raw encoder/decoder residual streams. Internal residual-block LayerNorm and FiLM-off remain shared. Decoder receiver layers use a second memory codec, so count both streams and their parameters. This is not a reproduction of historical clean-memory or globally memory-coded results. `hellaswag_diagnostic.yaml` is a five-setting, 4,000-update pre-H200 plan; it keeps a 20,000-update schedule horizon and fixes decoder memory normalization while varying the main decoder codec. Plans are editable definitions, not evidence that choices are optimal or requests to execute them. Before a fixed-budget research comparison, settle the training budget and selection protocol, including HellaSwag's current early-stop setting.

The `hellaswag_diagnostic` plan is the gate before a formal H200 study. It is HellaSwag-only, stops after 4,000 optimizer updates while retaining a 20,000-update schedule, uses effective batch 32 (`16 x 2`), validates the full fixed 512-row selection holdout, and evaluates a fixed 512-example panel at `no_noise`, `-6`, `18`, plus `vanilla`. Its five rows change only the split and main codec boundary norm; `codec.memory.layernorm: both` is held constant for decoder rows.

`hellaswag_diagnostic` is legacy diagnostic evidence. The corrected baseline is a
separate protocol and must be reported separately from both the historical
`hellaswag.yaml` runs and the five-row diagnostic. Use
`configs/tasks/hellaswag_corrected.yaml` with
`configs/studies/hellaswag_corrected.yaml` for the three planned routes:
`corrected_enc_fn` (`enc_fn` / main `both`), `corrected_enc_l9`
(`enc_l9` / main `none`), and `corrected_dec_l8`
(`dec_l8` / main `none`, memory `both`). All three use T5Gemma 2,
B512/H1152, FiLM off, seed 0, batch `16 x 2`, 4,000 optimizer updates,
a 20,000-step schedule horizon, a 512-row validation holdout, and
`no_noise`/`-6`/`18` plus vanilla evaluation. The plan has not been submitted
to H200. Its resolved task config carries the explicit
`protocol: corrected-baseline-v1` label for run manifests and downstream
result indexing.

## Examples

```bash
uv run --locked python train.py --config configs/tasks/coco.yaml --check
uv run --locked python train.py --config configs/tasks/hellaswag.yaml --check
uv run --locked python study.py --config configs/studies/splits.yaml
uv run --locked python study.py --config configs/studies/splits.yaml --task hellaswag
uv run --locked python study.py --config configs/studies/smoke_h200.yaml --output runs/studies/smoke-h200
uv run --locked python study.py --config configs/studies/hellaswag_diagnostic.yaml --task hellaswag
uv run --locked python study.py --config configs/studies/hellaswag_corrected.yaml --task hellaswag
uv run --locked python study.py --config configs/studies/hellaswag_corrected.yaml --task hellaswag --output runs/studies/hellaswag-corrected-h200
```

References inside a task/study YAML are relative to that file. Direct task runs write to the project runs/ directory; smoke runs retain their configured output directory. Exported studies instead collect artifacts beneath their prepared directory. See [study execution](../docs/studies.md) for the manifest and job workflow.

The corrected HellaSwag plan's export command only prepares three resolved
configs and a manifest. The expected future H200 entry point is the exported
manifest with matching train/evaluate arrays, for example
`--array=0-2%3` for one GPU per route and an `aftercorr` dependency for the
evaluation array. Site-specific partition/account/QOS options still belong at
submission time. Do not treat this documentation as evidence that an H200 job
has been sent.
