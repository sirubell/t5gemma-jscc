# HellaSwag full-weight replay parity preflight

This probe checks the current HellaSwag `enc_l9` recipe on one complete
fixed-stream training microbatch. It keeps the configured 64×1 batch, pinned
model/data and BF16 CUDA precision. It makes **zero optimizer updates** and
does not write an activation cache or evaluate task accuracy. A passing probe
does not establish B→C transfer, memory fit for training, quality or speed.
The resolved recipe must match `configs/tasks/hellaswag.yaml` in task,
protocol, seed, model, split, codec, channel, data and evaluation settings.
Only training step/selection/logging caps and the run output may differ; the
batch policy, objective, learning rate, data/prompt policy and noise seeds
remain pinned. The receipt records both the baseline resolved-config digest
and the scientific-identity digest. At runtime, the loaded backbone and
derived codec input must each have width 1152. Revision strings and source
digests identify the intended bytes; they do not independently verify every
cached model or dataset file's content.

First inspect the recipe without loading model weights, data or CUDA:

```bash
uv run --locked python scripts/hellaswag_replay_preflight.py \
  --config configs/tasks/hellaswag.yaml \
  --output /path/to/fresh/check.json --max-seconds 60 --check-only
```

After a target-specific GPU budget and isolated runtime are approved, execute
with a new receipt path. Set `HF_HOME` to the verified pinned cache location.
The command forces Hugging Face offline mode; absent model or dataset bytes
fail rather than download. The example soft deadline is a **proposed ceiling**,
not a measured runtime or execution authorization:

```bash
uv run --locked python scripts/hellaswag_replay_preflight.py \
  --config configs/tasks/hellaswag.yaml \
  --output /path/to/fresh/parity.json --max-seconds 900
```

The caller must enforce a scheduler wall limit and filesystem quota. Python
checks its soft deadline between operations and cannot interrupt a stalled
load or GPU kernel. The output JSON is created before loading and records a
failure or the conditions completed so far. Do not reuse an output path.

The probe takes the first 64 row IDs from the declared permutation stream.
It captures the frozen encoder's full `[64,T,1152]` tensor immediately before
the codec, then runs the production encoder split hook through the codec and
stops before the suffix and decoder. Local replay uses the production
`local_batch_values` path with a CPU copy of that activation and its source
mask. It checks no noise and one deterministic AWGN stream, resetting that
stream for the online and replay paths. For each condition it requires exact
activation equality, equal valid-sample counts, finite losses and codec
gradients, an identical audited AWGN draw, at least one nonzero codec
gradient, and no backbone gradients.
BF16 predeclared limits are activation bitwise equality, nMSE absolute
`1e-5`/relative `1e-4`, and per-parameter gradient absolute `1e-5`/relative
`1e-3`. The receipt retains raw absolute errors and gradient reference,
observed and difference norms. These are engineering parity gates for this
same-host execution, not cross-host or task-quality tolerances.

Timing synchronizes CUDA around capture, online and replay work and records
peak allocated/reserved GPU memory. The first data load may tokenize the full
training pool even though the probe consumes one batch; its setup time is
inside `--max-seconds`. No checkpoint or completion claim from this probe
should be substituted for the separate bounded U/L/F phase-transfer smoke.
