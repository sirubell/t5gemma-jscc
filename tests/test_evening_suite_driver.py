import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts.experiments import evening_suite_driver as runner


def fixture(tmp_path, monkeypatch, remaining=20000):
    (tmp_path / "source").mkdir()
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "AUTHORIZATION.json").write_text(
        json.dumps(
            {
                "execution_authorized": True,
                "plan_id": "EVENING_SMALL_SUITE_V1",
                "gpu_work_stop_at_unix": 100 + remaining,
                "max_gpu_minutes": 240,
            }
        )
    )
    for name in ["execution-source", "input-manifest"]:
        (tmp_path / f"source/{name}.json").write_text("{}")
    weight = tmp_path / "fake.pt"
    weight.write_bytes(b"unit-test-no-model")
    rows = [
        {
            "route": route,
            "step": step,
            "run": str(tmp_path),
            "path": str(weight),
            "sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
        }
        for route, steps in [
            ("enc_l9", [1000, 5000, 10000]),
            ("dec_l0", [10000]),
            ("dec_l20", [1000, 5000, 10000]),
        ]
        for step in steps
    ]
    (tmp_path / "checkpoints/ws-receipts.json").write_text(json.dumps(rows))
    monkeypatch.setattr(runner.sys, "argv", ["driver", "--root", str(tmp_path)])
    monkeypatch.setattr(runner.time, "time", lambda: 100)
    return weight


def test_bounded_schedule_failure_is_not_retried(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw["timeout"]))
        return SimpleNamespace(returncode=1 if len(calls) == 2 else 0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    runner.main()
    assert len(calls) == 10 and sum(x[1] for x in calls) == 6000
    assert sum("--vjp-cap" in c for c, _ in calls) == 3
    assert all("evening_pilot" not in " ".join(c) for c, _ in calls)
    result = json.loads((tmp_path / "driver-completion.json").read_text())
    assert (
        result["outcomes"][1]["status"] == "FAILED" and result["optimizer_updates"] == 0
    )
    with pytest.raises(FileExistsError):
        runner.main()
    assert len(calls) == 10


def test_deadline_and_wrong_checkpoint_prevent_dispatch(tmp_path, monkeypatch):
    weight = fixture(tmp_path, monkeypatch, remaining=20)
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *a, **k: pytest.fail("must not dispatch")
    )
    runner.main()
    assert all(
        x["status"] == "NOT_RUN"
        for x in json.loads((tmp_path / "driver-completion.json").read_text())[
            "outcomes"
        ]
    )
    weight.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="Checkpoint bytes"):
        runner.main()
