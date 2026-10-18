"""Per-site hidden-state statistics for ``codec.site_norm`` (forward passes of the frozen backbone only).

    python -m jscc.site_stats CONFIG N_BATCHES OUT.pt

Over training batches 0..N-1 and valid tokens (encoder: attention mask, which covers image tokens in COCO;
decoder: labels != -100), per-dimension mean and std at each of the config's sites and of the encoder
memory (``memory``, what decoder sites also send through the codec). Writes ``{"note", "stats": {stream:
{"mean", "std", "tokens"}}}``, the format ``codec.site_stats`` reads.
"""
import sys

import torch

from .model import decoder_stack, encoder_stack, parse_site


@torch.no_grad()
def collect(model, data, sites, n_batches):
    D = int(model.base.config.decoder.hidden_size)
    dev = next(model.base.parameters()).device
    sums = {s: [0, torch.zeros(D, dtype=torch.float64, device=dev), torch.zeros(D, dtype=torch.float64, device=dev)]
            for s in list(sites) + ["memory"]}
    masks = {}

    def add(stream, hidden):
        hidden = hidden[0] if isinstance(hidden, tuple) else hidden
        flat = hidden.double()[masks["dec" if stream.startswith("dec_") else "enc"]]
        a = sums[stream]
        a[0] += flat.shape[0]
        a[1] += flat.sum(0)
        a[2] += flat.pow(2).sum(0)

    enc, dec, handles = encoder_stack(model.base), decoder_stack(model.base), []
    for s in sites:
        stack_name, where, index = parse_site(s)
        if (stack_name, where) == ("enc", "emb"):
            handles.append(enc.layers[0].register_forward_pre_hook(lambda m, args, s=s: add(s, args[0])))
        elif (stack_name, where) == ("enc", "fn"):
            handles.append(enc.norm.register_forward_hook(lambda m, args, out, s=s: add(s, out)))
        else:
            stack = enc if stack_name == "enc" else dec
            handles.append(stack.layers[index].register_forward_hook(lambda m, args, out, s=s: add(s, out)))
    handles.append(model.base.get_encoder().register_forward_hook(
        lambda m, args, out: add("memory", out.last_hidden_state)))
    try:
        for k in range(n_batches):
            inputs, labels = data.model_inputs(data.train_batch(k), model)
            masks["enc"], masks["dec"] = inputs["attention_mask"].bool(), labels != -100
            with data.autocast(model):
                model(**inputs)
    finally:
        for h in handles:
            h.remove()
    out = {}
    for s, (n, s1, s2) in sums.items():
        mean = s1 / n
        out[s] = {"mean": mean.float().cpu(), "std": (s2 / n - mean ** 2).clamp_min(0).sqrt().float().cpu(), "tokens": n}
    return out


def main(argv):
    import os

    from .config import load_config
    from .model import build_model
    from .tasks import load_task_data
    config_path, n_batches, out = argv[0], int(argv[1]), argv[2]
    config = load_config(config_path)
    config["codec"]["site_norm"] = "none"  # the statistics are measured without a site norm
    processor, model = build_model(config, local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1")
    model.eval()
    stats = collect(model, load_task_data(config, processor), config["training"]["sites"], n_batches)
    note = (f"Per-site hidden-state mean/std (per dim, valid tokens) of the frozen backbone on {config['task']} "
            f"training batches 0-{n_batches - 1} (batch {config['training']['batch_size']}); memory = encoder output.")
    torch.save({"note": note, "stats": stats}, out)
    print(note, {s: v["tokens"] for s, v in stats.items()})


if __name__ == "__main__":
    main(sys.argv[1:])
