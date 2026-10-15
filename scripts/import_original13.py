"""Wrap an archived original13 checkpoint as a v2 run directory for score-parity evaluation.

    python scripts/import_original13.py OLD_CHECKPOINT.pt RUN_DIR [--sites enc_l9 dec_l12 dec_l24]
    python experiment.py evaluate RUN_DIR --checkpoint imported.pt

The archived file (``kind: original13_v1``) stores the shared direct-affine codec as a flat
state dict under ``model``; v2 reads ``communication.codec``. Only the codec weights are copied.
"""
import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from jscc.config import load_config, save_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("--sites", nargs="+")
    parser.add_argument("--conditions", nargs="+", default=["no_noise", "-6"])
    args = parser.parse_args()
    old = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if old.get("kind") != "original13_v1":
        raise SystemExit(f"expected an original13_v1 checkpoint, got kind={old.get('kind')!r}")
    codec = old["model"]
    if set(codec) != {"encoder.0.weight", "encoder.0.bias", "decoder.0.weight", "decoder.0.bias"}:
        raise SystemExit(f"unexpected codec keys: {sorted(codec)}")
    config = load_config(REPO / "configs/hellaswag.yaml")
    config["run"]["name"] = args.run.name
    config["codec"]["bottleneck_dim"] = int(codec["encoder.0.weight"].shape[0])
    config["evaluation"]["conditions"] = [c if c == "no_noise" else float(c) for c in args.conditions]
    if args.sites:
        config["evaluation"]["sites"] = args.sites
    args.run.mkdir(parents=True, exist_ok=False)
    config["run"]["root"] = str(args.run.parent.resolve())
    save_config(config, args.run / "config.yaml")
    torch.save({"communication": {"codec": codec, "channel": {}}, "imported_from": str(args.checkpoint.resolve())},
               args.run / "imported.pt")
    print(f"wrote {args.run}: b={config['codec']['bottleneck_dim']}, sites={args.sites or 'all 13'}")


if __name__ == "__main__":
    main()
