# Custom wireless channel integration

This guide is for collaborators implementing a physical wireless simulator. The project's normal research workflow is in the [README](../README.md). Integration starts by replacing the channel between codec encoding and decoding.

Both tasks reference the same `configs/model.yaml`: an encoder-final-norm residual-codec baseline with SNR-FiLM disabled. All supplied training and smoke recipes reference it. Start channel integration with this baseline. The simulator still accepts SNR to model noise; it does not need to implement or coordinate FiLM. FiLM is reserved for an explicit later experiment.

An H200 account is not needed to develop the channel. The tensor checker and automated tests run on CPU; full model checks can run on a compatible standalone CUDA GPU such as RTX 5090. Follow the [workstation guide](running.md#standalone-gpu-workstations-including-rtx-5090), starting with a small batch. The static analysis tools are installed with `uv sync --locked --extra dev`.

## Contract

Implement torch.nn.Module.forward(z, snr_db). See the [attenuation example](../examples/wireless_channel.py) and [built-in AWGN](../jscc/models/channel.py).

| Item | Contract |
|---|---|
| Input z | Real tensor [batch, tokens, bottleneck] |
| Output | Received representation with the same shape, device and dtype |
| Power | With normalize_power enabled, the caller normalizes mean power over all token/bottleneck elements in each sample |
| SNR | One numeric value per batch; None is the clean path, returned unchanged by the example |
| Precision | Research recipes use CUDA/bfloat16; internal float32/complex computation is possible if the output restores the input dtype |
| Gradients | End-to-end training needs gradients back to z; evaluation may use a non-differentiable simulator |

AWGN adds real Gaussian noise with standard deviation 10 ** (−snr_db / 20), assuming approximately unit signal power. Define the relationship again when introducing complex I/Q, modulation or coding.

## Integration steps

1. Create an importable Python module and an nn.Module class with constructor options.
2. Copy [wireless.yaml](../configs/evaluation/wireless.yaml); set channel.type to module:ClassName and channel.kwargs to your options.
3. Check small tensors before loading a model:

   ```bash
   uv run --locked python -m examples.check_channel --config configs/evaluation/wireless.yaml
   ```

   Use your copied filename when applicable. This checker currently uses CPU/float32 and 0 dB. A message reporting no usable gradient means evaluation-only use until resolved, even if the process exits normally. Also test CUDA/BF16, clean operation, other SNRs and sequence lengths before training.

4. Evaluate a saved codec with the replacement channel:

   ```bash
   uv run --locked python evaluate.py --run runs/<coco-run> --config configs/evaluation/wireless.yaml
   uv run --locked python evaluate.py --run runs/<hellaswag-run> --config configs/evaluation/wireless.yaml
   ```

   A replacement channel is a new instance. Parameters from another channel checkpoint are not automatically imported.

   Evaluation restores the checkpoint's saved model settings. An older FiLM-enabled checkpoint stays FiLM-enabled even after editing a current YAML. Use a checkpoint saved with the intended shared baseline when comparing simulators; historical COCO decoder-split results are not equivalent to the new encoder-split baseline.

5. For training, copy `configs/model.yaml` to a named design file and change its channel.type and channel.kwargs. Preserve normalize_power, train_noise, train_snr_range and clean_film_snr; keep codec.snr_film false. Point each task YAML's model_config at the new design file. Both tasks then use the same simulator and codec. Trainable channel parameters are included with codec parameters in the optimizer.

## Agree on the physical interpretation

- Representation and channel uses: the current interface sends floating-point representations, not a defined bitstream. Specify real/complex packing, modulation and symbols per token.
- SNR: distinguish real-dimension power, complex-symbol power, Es/N0 and Eb/N0; identify where power is normalized.
- State: generation may call the channel repeatedly. Define when fading, packet state and randomness reset.
- Split: encoder splits modify encoder representations. Decoder-only splits keep encoder memory clean, so they describe a different physical system.
- Clean/vanilla: no-noise keeps the codec; vanilla bypasses codec/channel. clean_film_snr is inactive in the default FiLM-off baseline; it only conditions a clean path when FiLM is explicitly enabled.

Only AWGN and the simple attenuation example have been validated so far. An initial collaboration deliverable is one channel module plus its power, SNR and state definitions; full model experimentation can follow when the interface is confirmed.
