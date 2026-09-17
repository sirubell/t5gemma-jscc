# Migration decisions

The refactor established one research project with shared COCO/HellaSwag training and evaluation. It preserves the frozen T5Gemma-2, configurable split, residual MLP codec, power normalization, AWGN, optional receiver FiLM, KL+nMSE, AdamW and cosine scheduling.

## Code mapping

| Historical area | Current module |
|---|---|
| Separate task training scripts | jscc/training.py |
| Compressor implementations | jscc/models/codec.py |
| Backbone/split wrappers | jscc/models/split_model.py |
| Channel | jscc/models/channel.py |
| Task data | jscc/data/ |
| COCO and HellaSwag evaluation scripts | jscc/evaluation.py |
| Shell argument combinations | Shared model YAML + task YAML, resolved before train.py/evaluate.py |

## Intentional changes

- The current shared baseline uses encoder final norm, external LayerNorm both and FiLM off for both tasks. Earlier COCO examples used decoder layer 24, external LayerNorm none and FiLM on. Existing checkpoints/results retain that earlier design; they are not relabeled as evidence for the new baseline.
- Both tasks use the same configurable LayerNorm residual codec. Historical HellaSwag fixed mean/std calibration was not carried over, so this is not an equivalent rerun of that configuration.
- Steps and the learning-rate schedule count optimizer updates. Effective batch still depends on microbatch size and accumulation.
- Checkpoint selection uses the configured validation metric rather than separate selection jobs. Additional saved steps can be evaluated explicitly.
- COCO validation/report subsets are generated and saved in the new run; they are not guaranteed to equal historical report panels.
- HellaSwag uses the correct ending for distillation. New runs hold out selection rows from training, reserving official validation for final evaluation; legacy saved IDs retain the original overlapping protocol.
- LoRA, token mixing and rate gates were not ported. Receiver-only encoder-memory coding was added after the owner clarified the communication boundary; it differs from historical global-memory coding.
- Cluster configuration is outside model code. Runs require no Git-clean or hash approval gate.
- Source task YAMLs now reference one shared model_config file. Saved run/checkpoint configurations are still complete flat dictionaries; old flat configurations and checkpoint loading do not require conversion.
- The old flat config filenames moved into tasks/, smoke/ and evaluation/. Update command paths; no duplicate alias files are retained. Relative model references and output directories were adjusted so direct task/smoke runs keep their previous artifact locations. Studies add explicit model overrides and export complete configs without changing the shared model file.

## Artifact compatibility

New checkpoints contain codec/channel state, optimizer state, configuration and data IDs. Historical checkpoint formats and codec structures are not automatically imported.

Private source locations and historical artifact paths are recorded in docs/local/legacy.md when present. Keep an index of purpose, source/configuration, checkpoint, results and comparability before moving old data. Legacy results are not automatically comparable after changes to architecture or sample selection.

See [architecture](architecture.md) for current behavior and [validation](validation.md) for what has been exercised.
