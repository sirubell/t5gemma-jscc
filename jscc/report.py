"""Human-readable run summaries and the shared results index.

Every run directory gets ``summary.json`` and ``SUMMARY.md``; every finished,
failed or interrupted stage appends one line to ``<runs root>/INDEX.md``.
"""
import json
import subprocess
import time
from pathlib import Path


def source_state(repo: Path) -> dict:
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def read_metrics(run: Path):
    path = run / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _fmt(value):
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def score_table(results: dict, task: str) -> str:
    metric = "cider" if task == "coco" else "acc_norm"
    conditions = sorted({k.split("|")[1] for k in results if "|" in k}, key=lambda c: (c != "no_noise", c))
    sites = list(dict.fromkeys(k.split("|")[0] for k in results if "|" in k))
    lines = [f"| site | {' | '.join(conditions)} |", "|---|" + "---|" * len(conditions)]
    for s in sites:
        cells = [results.get(f"{s}|{c}", {}).get(metric) for c in conditions]
        lines.append(f"| {s} | " + " | ".join("" if v is None else f"{100 * v:.2f}" if metric == "acc_norm"
                                               else f"{v:.4f}" for v in cells) + " |")
    if "vanilla" in results:
        v = results["vanilla"][metric]
        lines.append(f"\nVanilla (no split): {100 * v:.2f}" if metric == "acc_norm" else f"\nVanilla (no split): {v:.4f}")
    return "\n".join(lines)


def write_summary(run: Path, config: dict, status: str, **fields):
    run = Path(run)
    old = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else {}
    summary = {**old, **fields, "run": str(run), "task": config["task"], "status": status,
               "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    metrics = read_metrics(run)
    validations = [m for m in metrics if m["type"] == "validation"]
    phases = [m for m in metrics if m["type"] == "phase_end"]
    summary["training"] = {
        "updates": max((m["total"] for m in metrics if "total" in m), default=0),
        "phases": [{k: p[k] for k in ("phase", "reason", "update", "best")} for p in phases],
        "last_validation": validations[-1]["mean"] if validations else None,
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    lines = [f"# {run.name}", "", f"Status: **{status}**  ", f"Updated: {summary['updated']}  ",
             f"Task: {config['task']}  Sites: {', '.join(config['training']['sites'])}  ",
             f"Code: {summary.get('source', {}).get('commit', '?')[:10]}"
             f"{' (dirty)' if summary.get('source', {}).get('dirty') else ''}", ""]
    if summary.get("error"):
        lines += ["## Error", "", "```", summary["error"], "```", ""]
    t = summary["training"]
    lines += ["## Training", "", f"Updates: {t['updates']}", ""]
    for p in t["phases"]:
        lines.append(f"- {p['phase']}: stopped by {p['reason']} at update {p['update']}, best validation {_fmt(p['best'])}")
    if validations:
        lines += ["", "Validation (mean over sites and conditions):", ""]
        lines += [f"- {v['phase']} update {v['update']}: {_fmt(v['mean'])}{' *' if v['improved'] else ''}"
                  for v in validations[-10:]]
    for name, results in summary.get("evaluations", {}).items():
        lines += ["", f"## Evaluation: {name}", "", score_table(results, config["task"])]
    (run / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return summary


def append_index(run: Path, config: dict, stage: str, status: str, note: str = ""):
    index = Path(config["run"]["root"]) / "INDEX.md"
    if not index.exists():
        index.write_text("| time | run | stage | status | note |\n|---|---|---|---|---|\n")
    with index.open("a") as f:
        f.write(f"| {time.strftime('%Y-%m-%d %H:%M')} | {Path(run).name} | {stage} | {status} | {note} |\n")
