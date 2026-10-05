"""Fake caller execution checks fresh naming, replay, and retired boundaries."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def job_fixture(tmp_path, job_id="fixture-job"):
    bindir = tmp_path / ".local" / "bin"
    bindir.mkdir(parents=True)
    log = tmp_path / "commands.jsonl"
    uv = bindir / "uv"
    uv.write_text(f'''#!{sys.executable}
import json, sys
from pathlib import Path
args = sys.argv[1:]
with Path({str(log)!r}).open("a") as stream:
    stream.write(json.dumps(args)+"\\n")
if "train.py" in args:
    root = Path(args[args.index("--output")+1])
    assert not root.exists(), "wrapper precreated final root"
    root.mkdir(parents=True, exist_ok=False)
    link = Path(args[args.index("--run-path-file")+1])
    with link.open("x") as stream:
        stream.write(str(root.resolve())+"\\n")
''')
    uv.chmod(0o755)
    for name in ("hostname", "nvidia-smi", "java"):
        tool = bindir / name
        tool.write_text("#!/bin/bash\nexit 0\n")
        tool.chmod(0o755)
    return dict(os.environ, HOME=str(tmp_path), SLURM_SUBMIT_DIR=str(tmp_path), SLURM_JOB_ID=job_id), log


def invoke_job(environment, configs):
    return subprocess.run(["bash", str(ROOT / "scripts" / "slurm_job.sh"), *configs],
                          env=environment, capture_output=True, text=True)


def test_job_argument_ordinals_preserve_duplicate_names_and_identical_arguments(tmp_path):
    environment, log = job_fixture(tmp_path)
    configs = ["directory a/task.yaml", "directory b/task.yaml", "directory a/task.yaml"]
    result = invoke_job(environment, configs)
    assert result.returncode == 0, result.stderr
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    trains = [row for row in commands if "train.py" in row]
    evaluations = [row for row in commands if "evaluate.py" in row]
    assert [row[row.index("--config")+1] for row in trains] == configs
    assert [row[row.index("--output")+1] for row in trains] == [
        f"runs/jobs/fixture-job/training-{index:04d}" for index in range(1, 4)]
    for index, (training, evaluation) in enumerate(zip(trains, evaluations), 1):
        link = tmp_path / f"runs/jobs/fixture-job/links/{index:04d}.run.txt"
        assert training[training.index("--run-path-file")+1] == str(link.relative_to(tmp_path))
        assert evaluation[evaluation.index("--run")+1] == link.read_text().strip()
    before = log.read_bytes()
    result = invoke_job(environment, configs)
    assert result.returncode != 0
    assert log.read_bytes() == before


@pytest.mark.parametrize("used", ["last_root", "dangling_root", "last_link", "dangling_link", "alias_links"])
def test_job_preflights_all_targets_before_any_execution(tmp_path, used):
    environment, log = job_fixture(tmp_path)
    job = tmp_path / "runs/jobs/fixture-job"
    job.mkdir(parents=True)
    root, link = job / "training-0002", job / "links/0002.run.txt"
    if used == "alias_links":
        (job / "links").symlink_to(job / "training-0001", target_is_directory=True)
    elif "root" in used:
        root.symlink_to(tmp_path / "missing") if used.startswith("dangling") else root.mkdir()
    else:
        link.parent.mkdir()
        link.symlink_to(tmp_path / "missing") if used.startswith("dangling") else link.write_text("used")
    result = invoke_job(environment, ["same.yaml", "same.yaml"])
    assert result.returncode != 0
    assert not log.exists()
    assert not (job / "training-0001").exists()


@pytest.mark.parametrize("job_id", ["../escape", "", "has space", "/absolute"])
def test_job_rejects_invalid_identity_before_infrastructure(tmp_path, job_id):
    environment, log = job_fixture(tmp_path, job_id)
    result = invoke_job(environment, ["task.yaml"])
    assert result.returncode != 0
    assert not log.exists() and not (tmp_path / "runs").exists()


@pytest.mark.parametrize("script,phase", [("slurm_prompt_alignment.sh", "pair"),
                                         ("slurm_prompt_policy64.sh", "train"),
                                         ("slurm_prompt_policy64.sh", "evaluate")])
def test_retired_shell_phases_reject_before_environment_or_execution(tmp_path, script, phase):
    environment, log = job_fixture(tmp_path)
    environment.pop("SLURM_SUBMIT_DIR", None)
    result = subprocess.run(["bash", str(ROOT / "scripts" / script), "unused-release", "unused-root", phase, "unused-item"], env=environment,
                            capture_output=True, text=True)
    assert result.returncode == 2 and "retired" in result.stderr.lower()
    assert not log.exists()


def test_retired_modules_and_retained_helpers():
    assert not (ROOT / "scripts/closeout_production.py").exists()
    assert not (ROOT / "scripts/prompt_alignment_pair.py").exists()
    from scripts import native_preflight_train, prompt_policy_job
    assert not hasattr(native_preflight_train, "main") and not hasattr(prompt_policy_job, "main")
    assert all(hasattr(native_preflight_train, name) for name in
               ("IndexedRows", "IndexedCollator", "checked_gradient_names"))
    assert all(hasattr(prompt_policy_job, name) for name in ("sha", "validate_run"))


@pytest.mark.parametrize("phase", ["capacity", "unknown"])
def test_outer_ln_rejects_unsupported_phase_before_setup(tmp_path, phase):
    environment, log = job_fixture(tmp_path)
    gpu = tmp_path / ".local/bin/nvidia-smi"
    gpu.write_text(f"#!{sys.executable}\nfrom pathlib import Path\nPath({str(log)!r}).touch()\n")
    output = tmp_path / "output"
    # Missing release and runtime identity must not obscure phase rejection.
    environment.pop("SLURM_JOB_ID", None)
    result = subprocess.run(["/bin/bash", str(ROOT / "scripts/slurm_outer_ln.sh"),
                             str(tmp_path / "missing-release"), "unused-study", phase,
                             str(output)], env=environment, capture_output=True, text=True)
    assert result.returncode == 2
    assert ("retired" if phase == "capacity" else "Unknown phase") in result.stderr
    assert not output.exists() and not log.exists()


@pytest.mark.parametrize("phase", ["scoring", "train", "evaluate", "vanilla"])
def test_outer_ln_supported_dispatch_retains_recorded_targets(tmp_path, phase):
    shell_version = subprocess.check_output(["/bin/bash", "-c", "echo ${BASH_VERSINFO[0]}"], text=True)
    if phase == "scoring" and int(shell_version) < 4:
        pytest.skip("Retained scoring uses empty arrays under nounset, unsupported by Bash 3.2")
    environment, log = job_fixture(tmp_path)
    bindir = tmp_path / ".local/bin"
    # Only argv is captured: no study, scoring engine or GPU process executes.
    for name in ("bash", "uv", "nvidia-smi"):
        tool = bindir / name
        tool.write_text(f'''#!{sys.executable}
import json, sys
from pathlib import Path
with Path({str(log)!r}).open("a") as stream:
    stream.write(json.dumps([Path(sys.argv[0]).name, *sys.argv[1:]])+"\\n")
''')
        tool.chmod(0o755)
    release = tmp_path / "release"
    inputs = release / "outer-input"
    inputs.mkdir(parents=True)
    runs = {"enc_l9": "/recorded/encoder-attempt", "dec_l8": "/recorded/decoder-attempt"}
    (inputs / "scoring-runs.json").write_text(json.dumps(runs))
    prepared, output = tmp_path / "prepared", tmp_path / "output"
    result = subprocess.run(["/bin/bash", str(ROOT / "scripts/slurm_outer_ln.sh"),
                             str(release), str(prepared), phase, str(output)],
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    dispatch = [row for row in commands if row[0] != "nvidia-smi"]
    manifest = str(prepared / "study-manifest.proposed.json")
    fixed = ["--fixed-step", "--checkpoint", "step_005000.pt", "--expected-step", "5000"]
    if phase == "scoring":
        expected = []
        for route in ("enc_l9", "dec_l8"):
            row = ["uv", "run", "--locked", "--no-sync", "python", "-m",
                   "scripts.outer_ln_scoring_check", "--run", runs[route], "--checkpoint",
                   "step_000080.pt", "--samples", str(inputs / "scoring-documents.jsonl"),
                   "--output", str(output / f"scoring-{route}")]
            if route == "enc_l9":
                row += ["--tensor-fixtures", str(inputs / "anomaly-fixtures.json")]
            expected.append(row)
    elif phase == "train":
        expected = [["bash", "scripts/slurm_study.sh", "train", manifest]]
    elif phase == "evaluate":
        expected = [["bash", "scripts/slurm_study.sh", "evaluate", manifest, *fixed,
                     "--attempt", "codec"]]
    else:
        expected = [["bash", "scripts/slurm_study.sh", "evaluate",
                     str(prepared / "vanilla-manifest.proposed.json"), *fixed,
                     "--attempt", "vanilla", "--evaluation-config", str(prepared / "vanilla.yaml")],
                    ["uv", "run", "--locked", "--no-sync", "python", "-m", "jscc.outer_ln_gate",
                     "--manifest", manifest, "--output", str(output / "first-wave-gate.json")]]
    assert dispatch == expected


def test_native_preflight_execution_rejects_without_model_imports():
    # Importing the retained digest stays usable; execution fails before setup.
    from scripts.native_preflight_eval import digest
    assert digest({"b": 2, "a": 1}) == digest({"a": 1, "b": 2})
    code = """import runpy, sys
class RejectRuntime:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'jscc', 'lm_eval'}:
            raise AssertionError('runtime import before rejection: ' + fullname)
sys.meta_path.insert(0, RejectRuntime())
runpy.run_module('scripts.native_preflight_eval', run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode != 0 and "execution is retired" in result.stderr
    assert "runtime import before rejection" not in result.stderr


@pytest.mark.parametrize("task", ["coco", "hellaswag"])
def test_study_public_recorded_chain_with_distinct_evaluations(tmp_path, monkeypatch, task):
    """Actual record/reservation APIs; explicitly fake training and scoring engines."""
    import torch
    import yaml
    import train
    import evaluate
    from jscc.config import load_config
    from jscc.ordinary_runs import validate_reserved_attempt
    from jscc.study_task import run_task
    from jscc.outer_ln_gate import resolve_training_run
    from scripts.coco_native_study import verify_training_completion

    config = load_config(ROOT / "configs/tasks" / f"{task}.yaml")
    config["training"].update(max_steps=2, save_steps=[2])
    authored = tmp_path / "task.yaml"
    authored.write_text(yaml.safe_dump(config))
    manifest = tmp_path / "manifest.json"
    entry = {"index": 0, "task": task, "experiment": "software-fixture", "seed": 0,
             "config": authored.name, "run_link": "links/training.txt", "evaluation_link": "links/codec.txt"}
    manifest.write_text(json.dumps({"version": 1, "runs": [entry]}))
    checkpoint = "step_000002.pt"
    trained, scored = [], []

    def fake_training(effective, *, run_directory):
        run = Path(run_directory)
        assert validate_reserved_attempt(run, "training") == effective
        (run / "config.yaml").write_text(yaml.safe_dump(effective))
        for name in ("best.pt", checkpoint):
            torch.save({"step": 2, "config": effective, "codec": {"fixture": torch.ones(1)}}, run / name)
        (run / "completion.json").write_text(json.dumps({"status": "FULL_BUDGET_COMPLETED", "reason": "max_steps", "step": 2,
            "optimizer_updates": 2, "presentations": 4}))
        trained.append(run)
        return run

    def fake_evaluation(training, selected_checkpoint, overrides, *, run_directory, **kwargs):
        output = Path(run_directory)
        assert validate_reserved_attempt(output, "evaluation") == kwargs["expected_config"]
        assert Path(training) == trained[0] and selected_checkpoint == checkpoint
        assert kwargs["expected_step"] == 2
        (output / "results.json").write_text(json.dumps({"status": "fixture", "optimizer_step": 2}))
        scored.append(output)
        return output

    original_subprocess = subprocess.run
    def invoke(command, **kwargs):
        if len(command) < 2 or Path(command[1]).name not in ("train.py", "evaluate.py"):
            return original_subprocess(command, **kwargs)
        assert kwargs["check"] is True
        if Path(command[1]).name == "train.py":
            train.main(command[2:])
        else:
            evaluate.main(command[2:])

    monkeypatch.setattr("jscc.training.train", fake_training)
    monkeypatch.setattr("jscc.evaluation.evaluate", fake_evaluation)
    monkeypatch.setattr(subprocess, "run", invoke)
    run_task(manifest, 0, "train")
    root = Path((tmp_path / entry["run_link"]).read_text().strip())
    assert root == tmp_path / "runs/manifest.json/0000"
    assert resolve_training_run(root, expected_config=authored)[:2] == (root, trained[0])
    if task == "coco":
        evidence = verify_training_completion(tmp_path, {"runs": [entry]}, 0,
            {"updates": 2, "presentations_per_run": 4, "fixed_checkpoint": checkpoint})
        assert evidence["run"] == str(trained[0]) and evidence["logical_root"] == str(root)
    for attempt in ("codec", "vanilla"):
        entry["evaluation_link"] = f"links/{attempt}.txt"
        manifest.write_text(json.dumps({"version": 1, "runs": [entry]}))
        run_task(manifest, 0, "evaluate", checkpoint, 2, attempt=attempt)
    assert scored == [root / f"evaluations/evaluation-{attempt}" for attempt in ("codec", "vanilla")]
    for output in scored:
        record = json.loads((output / "attempt.json").read_text())
        assert record["status"] == "completed" and record["case"] == str(root)
    with pytest.raises(FileExistsError):
        run_task(manifest, 0, "evaluate", checkpoint, 2, attempt="vanilla")
    assert len(trained) == 1 and len(scored) == 2
