# Configuration layout

```text
configs/
  model.yaml                   # One shared backbone/split/codec/channel design
  tasks/
    coco.yaml                  # Data, training and evaluation defaults
    hellaswag.yaml
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
    smoke_h200.yaml
```

## Which file should change?

- Change the shared architecture in model.yaml, or copy it to a named model design for an explicit variation.
- Change task data/budgets/metrics in tasks/coco.yaml or tasks/hellaswag.yaml. They both reference ../model.yaml.
- Use smoke/ for execution checks. CPU and H200 have different execution precision/budgets but share the model design. FiLM is off.
- Pass evaluation/ files to evaluate.py with --config. They override evaluation settings from a checkpoint, not the checkpoint's model architecture.
- Use studies/ to list a set of experiments without duplicating model/task configurations. Study model_overrides apply equally to both tasks.

The *_h200.yaml files are ordinary CUDA/bfloat16 smoke recipes, not H200-only model implementations. A compatible RTX 5090 can try the same small recipes directly with Python; see [standalone GPU execution](../docs/running.md#standalone-gpu-workstations-including-rtx-5090). Keep task-specific batch/resource adjustments in task recipes and record the actual hardware used.

## Supplied study plans

| Plan | Variations per task | Seeds | Training configurations |
|---|---:|---|---:|
| baseline | Shared model | 0 | 2 |
| splits | 13 encoder/decoder locations | 0 | 26 |
| bottleneck | 256, 512, 1024 | 0 | 6 |
| smoke_h200 | Shared model, H200 smoke budgets | 0 | 2 |

Each training configuration can have a separate evaluation job. Evaluation SNRs do not multiply the number of trained models. All supplied plans keep FiLM off.

The split plan lists the 13 locations used by the historical launcher, but uses the NEW shared codec settings (including LayerNorm both and FiLM off). It is not a reproduction of historical results. Decoder splits retain clean encoder memory and have different communication semantics from encoder splits. These plans are editable study definitions, not evidence that the choices are optimal or requests to execute them.

## Examples

```bash
uv run --locked python train.py --config configs/tasks/coco.yaml --check
uv run --locked python train.py --config configs/tasks/hellaswag.yaml --check
uv run --locked python study.py --config configs/studies/splits.yaml
uv run --locked python study.py --config configs/studies/splits.yaml --task hellaswag
uv run --locked python study.py --config configs/studies/smoke_h200.yaml --output runs/studies/smoke-h200
```

References inside a task/study YAML are relative to that file. Direct task runs write to the project runs/ directory; smoke runs retain their configured output directory. Exported studies instead collect artifacts beneath their prepared directory. See [study execution](../docs/studies.md) for the manifest and job workflow.
