"""Fixed-step evaluation rejects ambiguous checkpoints before model construction."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from jscc.study_task import run_task


ROOT = Path(__file__).resolve().parents[1]


def test_study_task_propagates_checkpoint_and_step(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"version": 1, "runs": [{
        "task": "hellaswag", "experiment": "fixed", "seed": 0, "run_link": "run.txt",
    }]}))
    (tmp_path / "run.txt").write_text(str(tmp_path / "actual-run"))
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: calls.append(command))
    run_task(manifest, 0, "evaluate", checkpoint="step_004000.pt", expected_step=4000)
    assert calls[0][-4:] == ["--checkpoint", "step_004000.pt", "--expected-step", "4000"]


@pytest.mark.parametrize("options", [
    ["--fixed-step"], ["--fixed-step", "--checkpoint", "step_004000.pt"],
    ["--fixed-step", "--expected-step", "4000"], ["--expected-step", "4000"],
])
def test_launcher_rejects_incomplete_fixed_step(tmp_path, options):
    result, argv = invoke_launcher(tmp_path, "evaluate", options)
    assert result.returncode != 0
    assert argv is None


def invoke_launcher(tmp_path, phase, options):
    # An isolated HOME prevents the launcher finding the owner's real uv.
    bindir = tmp_path / ".local/bin"
    bindir.mkdir(parents=True)
    recorded = tmp_path / "argv.json"
    fake = bindir / "uv"
    fake.write_text(f"#!{sys.executable}\nimport json, sys\nfrom pathlib import Path\n"
                    f"Path({str(recorded)!r}).write_text(json.dumps(sys.argv[1:]))\n")
    fake.chmod(0o755)
    environment = dict(os.environ, HOME=str(tmp_path), SLURM_SUBMIT_DIR=str(ROOT), SLURM_ARRAY_TASK_ID="2")
    result = subprocess.run(["bash", str(ROOT / "scripts/slurm_study.sh"), phase,
                             "prepared path/manifest.json", *options], env=environment,
                            capture_output=True, text=True)
    return result, json.loads(recorded.read_text()) if recorded.exists() else None


def test_launcher_fixed_step_argv(tmp_path):
    result, argv = invoke_launcher(tmp_path, "evaluate", [
        "--fixed-step", "--checkpoint", "step_004000.pt", "--expected-step", "4000"])
    assert result.returncode == 0, result.stderr
    assert argv == ["run", "--locked", "--no-sync", "python", "-m", "jscc.study_task", "evaluate",
                    "--manifest", "prepared path/manifest.json", "--index", "2",
                    "--checkpoint", "step_004000.pt", "--expected-step", "4000"]


@pytest.mark.parametrize("phase", ["train", "evaluate"])
def test_launcher_legacy_arguments(tmp_path, phase):
    result, argv = invoke_launcher(tmp_path, phase, [])
    assert result.returncode == 0, result.stderr
    assert argv is not None
    assert argv[-6:] == ["jscc.study_task", phase, "--manifest", "prepared path/manifest.json", "--index", "2"]


@pytest.mark.parametrize("state", [{}, {"step": 3999}, {"step": "4000"}, {"step": 4000.0}, {"step": True}])
def test_invalid_checkpoint_step_fails_before_model(tmp_path, monkeypatch, state):
    import jscc.evaluation as evaluation
    torch.save(state, tmp_path / "step_004000.pt")
    monkeypatch.setattr(evaluation, "build_model", lambda *args: pytest.fail("must reject before model construction"))
    with pytest.raises(ValueError, match="step"):
        evaluation.evaluate(tmp_path, "step_004000.pt", expected_step=4000)


def test_missing_checkpoint_has_no_fallback(tmp_path, monkeypatch):
    import jscc.evaluation as evaluation
    torch.save({"step": 4000}, tmp_path / "best.pt")
    monkeypatch.setattr(evaluation, "build_model", lambda *args: pytest.fail("must reject before model construction"))
    with pytest.raises(FileNotFoundError):
        evaluation.evaluate(tmp_path, "step_004000.pt", expected_step=4000)


def test_tiny_checkpoint_with_exact_internal_step(tmp_path):
    from jscc.checkpoint_policy import validate_checkpoint_step
    checkpoint = tmp_path / "step_004000.pt"
    torch.save({"step": 4000, "codec": {"weight": torch.ones(1)}}, checkpoint)
    validate_checkpoint_step(torch.load(checkpoint, weights_only=True), 4000)


def test_evaluate_cli_propagates_expected_step(monkeypatch):
    import evaluate
    import jscc.evaluation as evaluation
    calls = []
    monkeypatch.setattr(evaluation, "evaluate", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--run", "run", "--checkpoint", "step_004000.pt",
                                    "--expected-step", "4000"])
    evaluate.main()
    assert calls == [(("run", "step_004000.pt", None), {"expected_step": 4000})]


@pytest.mark.parametrize("phase,checkpoint,step", [
    ("evaluate", None, 4000), ("evaluate", "step_004000.pt", 0),
    ("evaluate", "step_004000.pt", -1), ("train", "step_004000.pt", 4000),
])
def test_study_runner_rejects_invalid_step_options(tmp_path, phase, checkpoint, step):
    with pytest.raises(ValueError, match="step"):
        run_task(tmp_path / "missing-manifest.json", 0, phase, checkpoint, step)


def test_study_cli_propagates_expected_step(monkeypatch):
    import jscc.study_task as study_task
    calls = []
    monkeypatch.setattr(study_task, "run_task", lambda *args: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["study_task", "evaluate", "--manifest", "manifest.json", "--index", "2",
                                    "--checkpoint", "step_004000.pt", "--expected-step", "4000"])
    study_task.main()
    assert calls == [("manifest.json", 2, "evaluate", "step_004000.pt", 4000)]


@pytest.mark.parametrize("phase,options", [
    ("train", ["--fixed-step", "--checkpoint", "step_004000.pt", "--expected-step", "4000"]),
    ("evaluate", ["--checkpoint", "step_004000.pt", "--expected-step", "0"]),
    ("evaluate", ["--checkpoint", "step_004000.pt", "--expected-step", "wrong"]),
    ("evaluate", ["--checkpoint"]),
    ("evaluate", ["--unknown"]),
])
def test_launcher_rejects_invalid_options(tmp_path, phase, options):
    result, argv = invoke_launcher(tmp_path, phase, options)
    assert result.returncode != 0
    assert argv is None


def test_direct_api_requires_explicit_checkpoint_even_if_best_has_expected_step(tmp_path, monkeypatch):
    import jscc.evaluation as evaluation
    torch.save({'step': 20000}, tmp_path / 'best.pt')
    monkeypatch.setattr(evaluation.torch, 'load', lambda *a, **kw: pytest.fail('must reject before load'))
    with pytest.raises(ValueError, match='explicit checkpoint'):
        evaluation.evaluate(tmp_path, expected_step=20000)


def test_direct_cli_requires_explicit_checkpoint(monkeypatch):
    import evaluate
    monkeypatch.setattr(sys, 'argv', ['evaluate.py', '--run', 'run', '--expected-step', '20000'])
    with pytest.raises(SystemExit) as exc:
        evaluate.main()
    assert exc.value.code == 2
