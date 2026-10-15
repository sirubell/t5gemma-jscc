"""Show recent results: the runs index and the status line of each recent run.

    python scripts/results.py [RUNS_ROOT] [-n 20]
"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("root", nargs="?", default="runs", type=Path)
parser.add_argument("-n", type=int, default=20)
args = parser.parse_args()
index = args.root / "INDEX.md"
if index.exists():
    lines = index.read_text().splitlines()
    print("\n".join(lines[:2] + lines[2:][-args.n:]))
runs = sorted((p.parent for p in args.root.glob("*/summary.json")), key=lambda p: p.stat().st_mtime)[-args.n:]
print()
for run in runs:
    s = json.loads((run / "summary.json").read_text())
    t = s.get("training", {})
    print(f"{run.name:45s} {s['status']:16s} updates={t.get('updates')} last_val={t.get('last_validation')}")
    print(f"    {run / 'SUMMARY.md'}")
