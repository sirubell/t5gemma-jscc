"""Execute exactly one prepared study entry in a train or evaluation job."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def _contained(directory, value):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("study paths must be nonempty relative filenames")
    path = (directory / value).resolve()
    if directory not in path.parents:
        raise ValueError("study path escapes manifest directory")
    return directory / value


def _manifest_paths(manifest_path, entries, phase):
    directory = manifest_path.parent
    links, outputs = [], []
    for position, row in enumerate(entries):
        if "index" in row and (type(row["index"]) is not int or row["index"] != position):
            raise ValueError("study entry index must equal its positional index")
        if "config" in row:
            _contained(directory, row["config"])
        link = _contained(directory, row["run_link"])
        if phase == "train" and link.resolve() in [item.resolve() for item in links]:
            raise ValueError("study training links must be distinct")
        links.append(link)
        outputs.append(directory / "runs" / manifest_path.name / f"{position:04d}")
        if row.get("evaluation_link"):
            _contained(directory, row["evaluation_link"])
    for link in links:
        resolved_link = link.resolve()
        for output in outputs:
            resolved_output = output.resolve()
            if (resolved_link == resolved_output or resolved_output in resolved_link.parents
                    or resolved_link in resolved_output.parents):
                raise ValueError("study links and training roots must be disjoint")
    return links, outputs


def run_task(manifest_path, index, phase, checkpoint=None, expected_step=None, evaluation_config=None, *, attempt=None):
    if phase not in ("train", "evaluate"):
        raise ValueError("phase must be train or evaluate")
    if phase == "train" and checkpoint is not None:
        raise ValueError("checkpoint is only supported for evaluate")
    if attempt is not None and phase != "evaluate":
        raise ValueError("attempt is only supported for evaluate")
    if evaluation_config and phase != "evaluate":
        raise ValueError("evaluation config is only supported for evaluate")
    if expected_step is not None:
        if phase != "evaluate":
            raise ValueError("expected step is only supported for evaluate")
        if not checkpoint:
            raise ValueError("expected step requires an explicit checkpoint")
        if type(expected_step) is not int or expected_step < 1:
            raise ValueError("expected step must be a positive integer")
    checkpoint = checkpoint or "best.pt"
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest["version"] != 1:
        raise ValueError("Unsupported study manifest version")
    if index < 0 or index >= len(manifest["runs"]):
        raise ValueError("Study index is out of range")
    links, outputs = _manifest_paths(manifest_path, manifest["runs"], phase)
    entry = manifest["runs"][index]
    directory = manifest_path.parent
    link = links[index]
    root = Path(__file__).resolve().parents[1]
    import yaml
    from .experiment import SCHEMA
    if phase == 'train':
        if link.exists() or link.is_symlink() or outputs[index].exists() or outputs[index].is_symlink():
            raise FileExistsError('This study entry already has a completed training run; export a new plan')
        authored = yaml.safe_load((directory/entry['config']).read_text())
        if authored.get('schema') == SCHEMA:
            from .config import load_config
            science = load_config(directory/entry['config'])
        else:
            science = authored
        request = {'schema': SCHEMA, 'workflow': 'ordinary', 'operation': 'train', 'experiment': science,
                   'execution': {'root': str(outputs[index]), 'attempt': '0001', 'success_link': str(link)}}
    else:
        run = link.read_text().strip()
        if not run:
            raise ValueError('Training run link is empty')
        evaluation = {'checkpoint': checkpoint}
        if expected_step is not None:
            evaluation['expected_step'] = expected_step
        if evaluation_config:
            evaluation['overrides'] = str(Path(evaluation_config).resolve())
        request = {'schema': SCHEMA, 'workflow': 'ordinary', 'operation': 'evaluate',
                   'execution': {'root': run, 'attempt': attempt or '0001', 'evaluation': evaluation}}
        if entry.get('evaluation_link'):
            result_link = _contained(directory, entry['evaluation_link'])
            if result_link.exists() or result_link.is_symlink():
                raise FileExistsError('This study entry already has a completed evaluation')
            request['execution']['success_link'] = str(result_link)
    requests = directory/'requests'
    requests.mkdir(exist_ok=True)
    filename = requests/f'{manifest_path.name}-{index:04d}-{phase}-{attempt or "0001"}.yaml'
    with filename.open('x') as handle:
        yaml.safe_dump(request, handle, sort_keys=False)
    command = [sys.executable, str(root/'experiment.py'), '--config', str(filename)]
    print(f"{phase}: index={index} task={entry['task']} experiment={entry['experiment']} seed={entry['seed']}", flush=True)
    subprocess.run(command, cwd=root, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "evaluate"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--checkpoint", help="checkpoint filename (default: best.pt)")
    parser.add_argument("--expected-step", type=int, help="require this exact saved optimizer step")
    parser.add_argument("--attempt", help="explicit evaluation attempt name (default: 0001)")
    parser.add_argument("--evaluation-config", help="evaluation-only overrides, e.g. a shared vanilla panel")
    args = parser.parse_args()
    if args.attempt is not None:
        run_task(args.manifest, args.index, args.phase, args.checkpoint, args.expected_step,
                 args.evaluation_config, attempt=args.attempt)
    elif args.evaluation_config:
        run_task(args.manifest, args.index, args.phase, args.checkpoint, args.expected_step, args.evaluation_config)
    else:
        run_task(args.manifest, args.index, args.phase, args.checkpoint, args.expected_step)


if __name__ == "__main__":
    main()
