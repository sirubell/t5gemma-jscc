"""Collection boundary fixtures mirror released ordinary/recipe record layouts.

No engine is run: experiment-run-v1 cases/attempts, recipe-run-v1 inventories,
ordinary results.json conditions and recipe panel receipts supply recorded facts.
Metrics include ordinary flat rows and recipe-training-event-v1 update envelopes.
"""
import hashlib
import json

import pytest

from jscc.result_collection import collect_results, collect_snapshot, prepare_snapshot


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(row) for row in path.read_text().splitlines()]


def singleton(root, task="hellaswag"):
    write(root / "experiment.json", {"schema": "experiment-run-v1", "kind": "ordinary-task", "root": str(root)})
    write(root / "case.json", {"schema": "experiment-run-v1", "case_id": None, "inputs": {"data": {"task": task}}})
    (root / "config.yaml").write_text(f"task: {task}\n")
    record = read(root / "case.json")
    record["config_sha256"] = hashlib.sha256((root / "config.yaml").read_bytes()).hexdigest()
    write(root / "case.json", record)
    attempt = root / "attempts/training-0001"
    write(attempt / "attempt.json", {"schema": "experiment-run-v1", "stage": "training", "attempt_id": "0001", "status": "completed", "case": str(root)})
    return attempt


@pytest.mark.parametrize("task", ["coco", "hellaswag"])
def test_ordinary_all_scalar_rows_and_companions_without_bulk(tmp_path, task):
    root = tmp_path / "run"
    attempt = singleton(root, task)
    rows = [{"step": i, "loss": 1 / i, "phase": "functional", "lr": .001, "train_tokens": 7 * i} for i in range(1, 4)]
    metrics = attempt / "metrics.jsonl"
    metrics.write_text("".join(json.dumps(row) + "\n" for row in rows))
    checkpoint = attempt / "best.pt"
    checkpoint.write_bytes(b"fixture-tensor-bytes")
    write(root / "checkpoint.json", {"path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "size": checkpoint.stat().st_size, "step": 3})
    child = root / "attempts/training-0002"
    write(child / "attempt.json", {"stage": "training", "attempt_id": "0002", "status": "interrupted", "request": {"resume": {"path": str(checkpoint), "sha256": read(root / "checkpoint.json")["sha256"]}}})
    write(child / "publication.json", {"schema": "ordinary-continuation-publication-v1", "status": "pending", "previous_training": {"path": str(attempt)}, "previous_checkpoint": read(root / "checkpoint.json"), "training": {"path": str(child)}, "checkpoint": None})
    evaluation = root / "evaluations/evaluation-0001"
    write(evaluation / "attempt.json", {"stage": "evaluation", "status": "failed", "attempt_id": "0001", "details": {"error_type": "RuntimeError", "message": "later scoring failed"}})
    table = {"task": task, "optimizer_step": 3, "conditions": [{"condition": "no_noise", "fixture_score": .7, "num_samples": 2}]}
    write(evaluation / "results.json", table)
    write(attempt / "data_ids.json", {"rows": [1, 2, 3]})
    (evaluation / "items.jsonl").write_text('{"item":1}\n')
    snapshot, destination = tmp_path / "snapshot", tmp_path / "collected"
    manifest = prepare_snapshot(root, snapshot, source_uri="ssh://fixture/run")
    assert manifest["schema"] == "result-collection-snapshot-v1"
    exports = lines(snapshot / "exports/scalars.jsonl")
    assert [row["event"] for row in exports] == rows
    assert all(row["raw_metrics"]["sha256"] == hashlib.sha256(metrics.read_bytes()).hexdigest() for row in exports)
    assert all(row["attempt_record"]["status"] == "completed" for row in exports)
    assert str(.7) in (snapshot / "exports/results.jsonl").read_text()
    assert "later scoring failed" in (snapshot / "exports/errors.json").read_text()
    paths = {row["path"] for row in manifest["artifacts"]}
    assert "files/config.yaml" in paths and "files/checkpoint.json" in paths
    assert not any(path.endswith(("metrics.jsonl", "best.pt", "items.jsonl", "data_ids.json")) for path in paths)
    remote = {row["path"]: row for row in manifest["remote"]}
    assert "attempts/training-0001/best.pt" in remote
    assert all(row["retrieval_ref"].startswith("ssh://fixture/run") for row in remote.values())
    receipt = collect_snapshot(snapshot, destination)
    assert receipt["status"] == "complete"
    assert checkpoint.read_bytes() == b"fixture-tensor-bytes"
    assert metrics.read_text() == "".join(json.dumps(row) + "\n" for row in rows)
    assert (destination / "files/config.yaml").read_bytes() == (root / "config.yaml").read_bytes()


def test_snapshot_retry_verifies_reuses_and_preserves_failed_pass(tmp_path):
    root = tmp_path / "run"
    attempt = singleton(root)
    checkpoint = attempt / "best.pt"
    checkpoint.write_bytes(b"expected")
    snapshot, destination = tmp_path / "snapshot", tmp_path / "destination"
    prepare_snapshot(root, snapshot, selected=["attempts/training-0001/best.pt"])
    checkpoint.unlink()
    missing = collect_snapshot(snapshot, destination)
    assert missing["status"] != "complete"
    assert missing["counts"]["missing"]["count"] == 1
    assert missing["counts"]["missing"]["bytes"] == len(b"expected")
    prior_receipts = {path: path.read_bytes() for path in destination.rglob("*.json") if path.name != "collection.json"}
    checkpoint.write_bytes(b"expected")
    recovered = collect_results(root, destination, snapshot=snapshot, selected=["attempts/training-0001/best.pt"])
    assert recovered["status"] == "complete"
    assert recovered["counts"]["reused"]["count"] > 0
    assert recovered["counts"]["copied"]["count"] == 1
    for path, content in prior_receipts.items():
        assert path.read_bytes() == content
    copied = destination / "files/attempts/training-0001/best.pt"
    copied.write_bytes(b"tampered")
    repaired = collect_snapshot(snapshot, destination)
    assert repaired["status"] == "complete"
    assert copied.read_bytes() == b"expected"
    assert repaired["counts"]["copied"]["count"] == 1
    checkpoint.write_bytes(b"different")
    copied.unlink()
    failed = collect_snapshot(snapshot, destination)
    assert failed["counts"]["failed"]["count"] == 1 and failed["status"] != "complete"
    assert checkpoint.read_bytes() == b"different"


def test_absent_selected_source_is_not_silently_removed_or_rebound(tmp_path):
    root = tmp_path / "run"
    singleton(root)
    snapshot, destination = tmp_path / "snapshot", tmp_path / "destination"
    manifest = prepare_snapshot(root, snapshot, selected=["missing.pt"])
    artifact = next(row for row in manifest["artifacts"] if row["path"] == "files/missing.pt")
    assert artifact["bytes"] is None and artifact["sha256"] is None
    receipt = collect_snapshot(snapshot, destination)
    assert receipt["counts"]["missing"]["count"] == 1
    (root / "missing.pt").write_bytes(b"new-identity")
    retry = collect_snapshot(snapshot, destination)
    assert retry["status"] != "complete"
    with pytest.raises(FileExistsError):
        prepare_snapshot(root, snapshot)


def test_recipe_attempt_ancestry_nested_scalars_and_partial_cases(tmp_path):
    root = tmp_path / "recipe"
    write(root / "experiment.json", {"schema": "experiment-run-v1", "kind": "shared-codec-recipe"})
    write(root / "recipe-run.json", {"schema": "recipe-run-v1", "cases": [{"case_id": "a"}, {"case_id": "b"}]})
    expected = []
    for case_id, state in [("a", "completed"), ("b", "interrupted")]:
        case = root / "cases" / case_id
        write(case / "case.json", {"schema": "experiment-run-v1", "case_id": case_id})
        (case / "config.yaml").write_text("task: hellaswag\n")
        record = read(case / "case.json")
        record["config_sha256"] = hashlib.sha256((case / "config.yaml").read_bytes()).hexdigest()
        write(case / "case.json", record)
        for number, status in [(1, "interrupted"), (2, state)]:
            attempt = case / f"attempts/training-{number:04d}"
            record = {"stage": "training", "attempt_id": f"{number:04d}", "status": status, "request": {"resume": {"path": "prior.pt", "sha256": "a" * 64}} if number == 2 else {}}
            write(attempt / "attempt.json", record)
            event = {"schema": "recipe-training-event-v1", "event_type": "update", "phase_id": "functional", "task": "hellaswag", "payload": {"completed_step": number, "site_id": "enc_l2", "lr_used": [.001], "lr_next": [.0005], "exposure": {"presentations": 2}, "objective": {"components": {"R": {"numerator": 4, "denominator": 2}}}}}
            engine = attempt / "engine"
            engine.mkdir()
            write(attempt / "training.json", {"engine_path": str(engine)})
            write(engine / "campaign.json", {"schema": "recipe-training-v1", "status": status, "segment_parent": record["request"].get("resume"), "observations": [{"status": "pending_prior_segment", "step": 1}]})
            (engine / "metrics.jsonl").write_text(json.dumps(event) + "\n")
            expected.append(event)
            if number == 2:
                observation = engine / "observation_000002"
                write(observation / "result.json", {"status": "partial", "request": {"scope": "development", "checkpoint": {"path": "step_000002_functional.pt", "step": 2, "role": "functional_observation"}, "scoring_policy": "fp32-v1"}, "panels": [{"site": "enc_l2", "condition": "no_noise", "status": "complete", "requested": 2, "completed_items": 2, "metrics": {"acc": .5}}]})
        write(case / "evaluations/evaluation-0001/attempt.json", {"stage": "evaluation", "status": "partial", "attempt_id": "0001"})
        write(case / "evaluations/evaluation-0001/panels/result.json", {"status": "partial", "panels": [{"site": "enc_l2", "condition": "no_noise", "status": "complete", "completed_items": 2, "metrics": {"acc": .5}}, {"site": "dec_l3", "condition": -6, "status": "failed", "completed_items": 0, "error": "fixture failure"}]})
    write(root / "evaluations/initial/results.json", {"schema": "recipe-evaluation-v1", "scope": "initial", "status": "incomplete", "cases": [{"case_id": "a", "status": "completed", "panels": {"requested": 2, "completed": 1, "failed": 1}}, {"case_id": "b", "status": "missing", "panels": {"requested": 2, "completed": 0, "pending": 2}}]})
    write(root / "evaluations/initial/gate.json", {"final_eligible": False, "status": "incomplete"})
    snapshot = tmp_path / "snapshot"
    prepare_snapshot(root, snapshot)
    scalars = lines(snapshot / "exports/scalars.jsonl")
    assert [row["event"] for row in scalars] == expected
    assert {row["case_id"] for row in scalars} == {"a", "b"}
    resumed = [row for row in scalars if row["attempt_record"]["attempt_id"] == "0002"]
    assert all(row["attempt_record"]["request"]["resume"]["sha256"] == "a" * 64 for row in resumed)
    tables = (snapshot / "exports/results.jsonl").read_text()
    assert "completed_items" in tables and "fixture failure" in tables
    table_rows = lines(snapshot / "exports/results.jsonl")
    aggregate = [row["row"] for row in table_rows if row["table"] == "cases"]
    assert aggregate == [{"case_id": "a", "status": "completed", "panels": {"requested": 2, "completed": 1, "failed": 1}}, {"case_id": "b", "status": "missing", "panels": {"requested": 2, "completed": 0, "pending": 2}}]
    assert all(row["event"]["payload"]["objective"]["components"]["R"]["denominator"] == 2 for row in scalars)
    statuses = (snapshot / "exports/status.json").read_text()
    assert all(value in statuses for value in ("completed", "interrupted", "partial", "pending_prior_segment"))
    observation_rows = [row for row in table_rows if row.get("scope") == "development"]
    assert len(observation_rows) == 2
    assert all(row["policy"]["checkpoint"]["role"] == "functional_observation" for row in observation_rows)


def test_cli_selected_companion_retry_accounting_and_lock(tmp_path, capsys):
    from scripts.collect_results import main

    root = tmp_path / "run"
    singleton(root)
    snapshot, destination = tmp_path / "snapshot", tmp_path / "destination"
    args = ["--run", str(root), "--snapshot", str(snapshot), "--output", str(destination),
            "--select", "config.yaml"]
    first = main(args)
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["status"] == "complete"
    counts = first["counts"]
    assert counts["selected"] == counts["verified"] == counts["copied"]
    assert counts["selected"]["bytes"] == sum(row["bytes"] for row in first["artifacts"])
    assert counts["missing"]["count"] == counts["failed"]["count"] == 0
    retry = main(args)
    capsys.readouterr()
    assert retry["counts"]["reused"] == counts["selected"]
    assert retry["counts"]["copied"]["count"] == 0
    assert first["receipt"] != retry["receipt"]
    before = (destination / "collection.json").read_bytes()
    (destination / ".collection-lock").mkdir()
    with pytest.raises(FileExistsError):
        collect_snapshot(snapshot, destination)
    assert (destination / "collection.json").read_bytes() == before
    (destination / ".collection-lock").rmdir()


def test_cli_missing_selection_returns_failure_with_durable_receipt(tmp_path, capsys):
    from scripts.collect_results import main

    root = tmp_path / "run"
    singleton(root)
    destination = tmp_path / "destination"
    with pytest.raises(SystemExit) as error:
        main(["--run", str(root), "--snapshot", str(tmp_path / "snapshot"),
              "--output", str(destination), "--select", "unavailable.pt"])
    assert error.value.code == 1
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["counts"]["missing"]["unknown_bytes"] == 1
    assert read(destination / "collection.json")["status"] == "incomplete"


@pytest.mark.parametrize("selection", ["../escape", "/absolute.pt"])
def test_selection_rejects_escape(tmp_path, selection):
    root = tmp_path / "run"
    singleton(root)
    with pytest.raises(ValueError):
        prepare_snapshot(root, tmp_path / "snapshot", selected=[selection])


def test_unterminated_scalar_snapshot_rejects_record_without_inventing_row(tmp_path):
    root = tmp_path / "run"
    attempt = singleton(root)
    (attempt / "metrics.jsonl").write_text('{"loss":0.2}')
    with pytest.raises(ValueError):
        prepare_snapshot(root, tmp_path / "snapshot")
    assert not (tmp_path / "snapshot").exists()
    assert (attempt / "metrics.jsonl").read_text() == '{"loss":0.2}'


def test_destination_and_source_containment_protect_scientific_bytes(tmp_path):
    root = tmp_path / "run"
    singleton(root)
    snapshot = tmp_path / "snapshot"
    prepare_snapshot(root, snapshot)
    with pytest.raises(ValueError):
        collect_snapshot(snapshot, root / "collected")
    external = tmp_path / "external"
    external.write_bytes(b"outside")
    (root / "escape.pt").symlink_to(external)
    with pytest.raises(ValueError):
        prepare_snapshot(root, tmp_path / "unsafe", selected=["escape.pt"])
    assert external.read_bytes() == b"outside"


def test_mutated_snapshot_export_cannot_be_verified(tmp_path):
    root = tmp_path / "run"
    singleton(root)
    snapshot, destination = tmp_path / "snapshot", tmp_path / "destination"
    prepare_snapshot(root, snapshot)
    (snapshot / "exports/status.json").write_text('{"invented":"success"}')
    result = collect_snapshot(snapshot, destination)
    assert result["status"] == "incomplete"
    assert result["counts"]["failed"]["count"] == 1
    assert not (destination / "exports/status.json").exists()
