#!/usr/bin/env python3
"""Offline CIDEr recomputation from exported COCO per-image evidence (no GPU/Java).

Run: uv run python scripts/recompute_coco.py captions_no_noise.jsonl
The saved PTB tokens fix the tokenizer output; raw references/captions are also
retained in each row for independently re-running PTBTokenizer when Java exists.
"""
import argparse
import json
from pathlib import Path

from pycocoevalcap.cider.cider import Cider


def recompute(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    if not records or len({row["image_id"] for row in records}) != len(records):
        raise ValueError("Need nonempty, unique per-image evidence")
    references = {row["image_id"]: row["ptb_references"] for row in records}
    captions = {row["image_id"]: row["ptb_caption"] for row in records}
    score, per_image = Cider().compute_score(references, captions)
    differences = [abs(float(value) - row["cider"]) for row, value in zip(records, per_image)]
    return {"cider": float(score), "num_images": len(records),
            "maximum_saved_score_difference": max(differences),
            "per_image": [{"image_id": row["image_id"], "cider": float(value)}
                          for row, value in zip(records, per_image)]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captions")
    args = parser.parse_args()
    print(json.dumps(recompute(args.captions), indent=2))
