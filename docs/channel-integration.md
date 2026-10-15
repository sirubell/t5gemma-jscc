# Plugging in a channel

A channel replaces what happens between the codec's transmitter (encoder) and receiver (decoder). The same channel works for one-site and shared (13-site) runs, at encoder and decoder sites, in training and evaluation.

## Contract

```python
class MyChannel(torch.nn.Module):
    normalizes_power = False   # optional; True = you handle transmit power, the caller does not normalize
    snr_mode = "per_sample"    # optional; "per_batch" = you receive one float per call

    def forward(self, z, snr_db):
        # z: real tensor [batch, tokens, b]
        # snr_db: None (no noise), float, or tensor [batch, 1, 1] (per_sample mode)
        # return a tensor with the same shape and dtype as z
        # optionally set self.aux_loss = <scalar tensor> for an extra training loss
```

| Item | Rule |
|---|---|
| Power | By default the caller scales `z` to unit average power per sample (encoder streams: over valid tokens; decoder hidden stream: per token). Set `normalizes_power = True` if your mapping (e.g. a QAM constellation) fixes power itself. |
| SNR | AWGN uses noise std `10^(-SNR/20)` per real dimension at unit signal power. Define your own convention (Es/N0, per complex symbol) and keep it consistent. Training draws SNR uniformly from `channel.train_snr_range`. |
| Trainable parameters | Optimized with the codec, saved in checkpoints, put in `train()`/`eval()` mode with the codec (BatchNorm/Dropout behave correctly). |
| Auxiliary loss | `self.aux_loss` is added with weight `training.channel_aux_weight` (default 0). |
| Gradients | Training needs gradients to reach `z` (straight-through estimators are fine). Evaluation-only channels may be non-differentiable. |
| Decoder sites | Two streams pass through the channel: the decoder hidden state and the encoder memory used by receiver layers. Both use the same SNR. |

## Steps

1. Put your module in an importable package (for example `examples/` or your own package on `PYTHONPATH`).
2. Check it without a model: `python -m examples.check_channel your.module:YourChannel`.
3. Use it in a run by overriding the channel in an experiment YAML:

   ```yaml
   base: ../hellaswag.yaml
   run: {name: shared13-qam}
   channel:
     type: examples.qam_channel:LearnedQAMChannel
     kwargs: {tau: 1.0}
   training:
     channel_aux_weight: 0.01
   ```

4. Train: `python experiment.py train configs/examples/your-file.yaml` (or through `scripts/job.sh` on the cluster).

`examples/qam_channel.py` is a worked example: a trainable mapping of coordinate pairs to 16-QAM symbols (straight-through Gumbel-softmax), complex AWGN and a trainable demapper. It uses `normalizes_power`, `snr_mode = "per_batch"` and `aux_loss`. It keeps float32 symbol logits for every coordinate pair (16 per pair), so it needs several extra GB: on a shared 32 GB RTX 5090 it ran at batch 8 but ran out of memory at batch 16.

## Adapting a simulator written for one vector per sample

Simulators built for image classifiers often take `[batch, m]`. Here every token carries a `b`-dimensional vector, so either treat `[batch * tokens, b]` as the batch (reshape in and out inside `forward`), or pack coordinates into symbols as the QAM example does. Note the cost: a sequence has hundreds to thousands of tokens.

## Agree on the physical meaning

- What one transmitted unit is (real coordinate, complex symbol, bit) and how many channel uses a sample costs.
- Which SNR definition is used.
- When channel state (fading, randomness) resets during autoregressive generation, where the channel is called once per generated token at decoder sites.
