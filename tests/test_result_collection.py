"""Collection boundary fixtures mirror released ordinary/recipe record layouts.

No engine is run: experiment-run-v1 cases/attempts, recipe-run-v1 inventories,
ordinary results.json conditions and recipe panel receipts supply recorded facts.
Metrics include ordinary flat rows and recipe-training-event-v1 update envelopes.
"""
import hashlib
import json
from pathlib import Path

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


def test_default_companion_retains_original_source_through_snapshot_retries(tmp_path):
    """G01: exports survive independently; companion repair needs original bytes."""
    root = tmp_path / "run"
    singleton(root)
    companion = root / "config.yaml"
    original = companion.read_bytes()
    snapshot, destination = tmp_path / "snapshot", tmp_path / "destination"
    manifest = prepare_snapshot(root, snapshot)
    entry = next(row for row in manifest["artifacts"] if row["path"] == "files/config.yaml")
    assert entry["kind"] == "companion"
    assert entry["source"] == str(companion)
    assert entry["sha256"] == hashlib.sha256(original).hexdigest()
    assert entry["bytes"] == len(original)
    assert all(Path(row["source"]).is_relative_to(root)
               for row in manifest["artifacts"] if row["kind"] == "companion")
    exports = [row for row in manifest["artifacts"] if row["kind"] == "export"]
    assert exports
    assert all(Path(row["source"]) == snapshot / row["path"] for row in exports)
    assert not (snapshot / "files/config.yaml").exists()
    snapshot_bytes = {path: path.read_bytes() for path in snapshot.rglob("*") if path.is_file()}

    companion.unlink()
    missing = collect_snapshot(snapshot, destination)
    assert missing["status"] == "incomplete"
    assert missing["counts"]["missing"] == {"count": 1, "bytes": len(original), "unknown_bytes": 0}
    assert all((destination / row["path"]).read_bytes() == Path(row["source"]).read_bytes()
               for row in exports)
    assert not (destination / entry["path"]).exists()
    prior_receipts = {Path(missing["receipt"]): Path(missing["receipt"]).read_bytes()}

    companion.write_bytes(original)
    recovered = collect_snapshot(snapshot, destination)
    assert recovered["status"] == "complete"
    assert recovered["counts"]["copied"]["count"] == 1
    copied = destination / entry["path"]
    assert copied.read_bytes() == companion.read_bytes() == original
    prior_receipts[Path(recovered["receipt"])] = Path(recovered["receipt"]).read_bytes()

    companion.unlink()
    reused = collect_snapshot(snapshot, destination)
    assert reused["status"] == "complete"
    assert reused["counts"]["reused"] == reused["counts"]["selected"]
    assert reused["counts"]["copied"]["count"] == 0
    assert not companion.exists()
    prior_receipts[Path(reused["receipt"])] = Path(reused["receipt"]).read_bytes()

    copied.write_bytes(b"damaged destination")
    missing_repair = collect_snapshot(snapshot, destination)
    assert missing_repair["status"] == "incomplete"
    assert missing_repair["counts"]["missing"]["count"] == 1
    assert copied.read_bytes() == b"damaged destination"
    prior_receipts[Path(missing_repair["receipt"])] = Path(missing_repair["receipt"]).read_bytes()

    companion.write_bytes(b"changed original")
    failed_repair = collect_snapshot(snapshot, destination)
    assert failed_repair["status"] == "incomplete"
    assert failed_repair["counts"]["failed"]["count"] == 1
    assert copied.read_bytes() == b"damaged destination"
    assert companion.read_bytes() == b"changed original"
    prior_receipts[Path(failed_repair["receipt"])] = Path(failed_repair["receipt"]).read_bytes()

    companion.write_bytes(original)
    repaired = collect_snapshot(snapshot, destination)
    assert repaired["status"] == "complete"
    assert repaired["counts"]["copied"]["count"] == 1
    assert copied.read_bytes() == companion.read_bytes() == original
    assert all(path.read_bytes() == content for path, content in prior_receipts.items())
    assert all(path.read_bytes() == content for path, content in snapshot_bytes.items())


@pytest.mark.parametrize("select_receipt", [False, True])
def test_panel_receipt_reference_retention_versus_explicit_byte_selection(tmp_path, select_receipt):
    """G02: selecting a receipt verifies its bytes, not its hashed siblings."""
    root = tmp_path / "run"
    singleton(root)
    evaluation = root / "evaluations/evaluation-0001/panels"
    panel_dir = evaluation / "enc_l2_no_noise"
    request = {"schema": "recipe-terminal-request-v1", "scope": "initial", "expected_items": 2,
               "sites": ["enc_l2"], "conditions": ["no_noise"]}
    parent_request = write(evaluation / "request.json", request)
    panel_request = write(panel_dir / "request.json", {
        **request, "schema": "recipe-panel-request-v1", "target_site": "enc_l2", "conditions": ["no_noise"],
    })
    items = panel_dir / "items.jsonl"
    items.write_text('{"item": 1}\n{"item": 2}\n')
    output_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in (panel_request, items)}
    panel = {"site": "enc_l2", "condition": "no_noise", "status": "complete",
             "requested_items": 2, "completed_items": 2, "denominator": 2,
             "metrics": {"acc": .5}, "output_hashes": output_hashes}
    receipt_path = write(panel_dir / "receipt.json", panel)
    receipt_bytes = receipt_path.read_bytes()
    receipt_reference = {"path": str(receipt_path), "sha256": hashlib.sha256(receipt_bytes).hexdigest()}
    parent_result = write(evaluation / "result.json", {
        "schema": "recipe-evaluation-receipt-v1", "status": "complete", "request": request,
        "panels": [{**panel, "receipt": receipt_reference}],
        "requested_panels": 1, "completed_panels": 1, "failed_panels": 0, "pending_panels": 0,
    })
    relative_receipt = receipt_path.relative_to(root).as_posix()
    snapshot, destination = tmp_path / "snapshot", tmp_path / "destination"
    manifest = prepare_snapshot(root, snapshot, selected=[relative_receipt] if select_receipt else [])
    artifacts = {row["path"]: row for row in manifest["artifacts"]}
    for path in (parent_request, parent_result, panel_request):
        assert artifacts["files/" + path.relative_to(root).as_posix()]["kind"] == "companion"
    if select_receipt:
        selected = artifacts["files/" + relative_receipt]
        assert selected["kind"] == "explicit"
        assert selected["source"] == str(receipt_path)
        assert selected["sha256"] == receipt_reference["sha256"]
        assert selected["bytes"] == len(receipt_bytes)
    else:
        assert "files/" + relative_receipt not in artifacts
        reference = next(row for row in manifest["remote"] if row["path"] == relative_receipt)
        assert reference["recorded_sha256"] == receipt_reference["sha256"]
        assert reference["retrieval_ref"] == receipt_path.as_uri()
        assert reference["verification"] == "metadata only; bytes not copied or verified"

    # A mismatched hashed sibling does not change collection's selected-byte success.
    items.write_text('{"changed": true}\n')
    assert hashlib.sha256(items.read_bytes()).hexdigest() != output_hashes[items.name]
    source_bytes = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    collected = collect_snapshot(snapshot, destination)
    assert collected["status"] == "complete"
    for path in (parent_request, parent_result, panel_request):
        assert (destination / "files" / path.relative_to(root)).read_bytes() == path.read_bytes()
    saved_panel = read(destination / "files" / parent_result.relative_to(root))["panels"][0]
    assert saved_panel["receipt"] == receipt_reference
    assert saved_panel["output_hashes"] == output_hashes
    assert lines(destination / "exports/results.jsonl")[0]["row"] == saved_panel
    copied_receipt = destination / "files" / relative_receipt
    if select_receipt:
        assert copied_receipt.read_bytes() == receipt_bytes
        assert read(copied_receipt)["output_hashes"] == output_hashes
        copied_row = next(row for row in collected["artifacts"] if row["path"] == "files/" + relative_receipt)
        assert copied_row["status"] == "verified" and copied_row["copied"]
    else:
        assert not copied_receipt.exists()
    assert not (destination / "files" / items.relative_to(root)).exists()
    assert all(path.read_bytes() == content for path, content in source_bytes.items())


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


def test_receipt_directory_symlink_rejects_before_copy_or_publication(tmp_path):
    root = tmp_path / "run"
    singleton(root)
    snapshot, destination, external = tmp_path / "snapshot", tmp_path / "destination", tmp_path / "external"
    prepare_snapshot(root, snapshot)
    destination.mkdir()
    external.mkdir()
    (destination / "receipts").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError):
        collect_snapshot(snapshot, destination)
    assert list(external.iterdir()) == []
    assert not (destination / "files").exists()
    assert not (destination / "collection.json").exists()


def test_missing_referenced_companion_unknown_size_restores_exact_hash(tmp_path):
    root = tmp_path / "run"
    singleton(root)
    expected = b'{"frozen":"identity"}'
    freeze = root / "freeze.json"
    write(root / "training.json", {"freeze": {"path": str(freeze), "sha256": hashlib.sha256(expected).hexdigest()}})
    snapshot, destination = tmp_path / "snapshot", tmp_path / "destination"
    manifest = prepare_snapshot(root, snapshot)
    entry = next(row for row in manifest["artifacts"] if row["path"] == "files/freeze.json")
    assert entry["bytes"] is None and entry["sha256"] == hashlib.sha256(expected).hexdigest()
    missing = collect_snapshot(snapshot, destination)
    assert missing["counts"]["missing"]["unknown_bytes"] == 1
    freeze.write_bytes(expected)
    retry = collect_snapshot(snapshot, destination)
    assert retry["status"] == "complete"
    assert retry["counts"]["selected"]["unknown_bytes"] == 1
    assert retry["counts"]["copied"]["bytes"] == len(expected)
    assert (destination / "files/freeze.json").read_bytes() == expected


def test_checkpoint_and_reconciliation_metadata_are_default_companions(tmp_path):
    root = tmp_path / "run"
    attempt = singleton(root)
    names = ("checkpoint_000002.pt.json", "last.pt.json", "best.pt.json", "reconciliation.json")
    for name in names:
        write(attempt / name, {"status": "partial", "identity": name})
    manifest = prepare_snapshot(root, tmp_path / "snapshot")
    paths = {row["path"] for row in manifest["artifacts"]}
    assert all("files/attempts/training-0001/" + name in paths for name in names)


def test_failed_copy_counts_actual_staged_bytes_and_expected_failure_bytes(tmp_path):
    root = tmp_path / "run"
    attempt = singleton(root)
    source = attempt / "best.pt"
    source.write_bytes(b"original")
    snapshot, destination = tmp_path / "snapshot", tmp_path / "destination"
    prepare_snapshot(root, snapshot, selected=["attempts/training-0001/best.pt"])
    changed = b"changed-longer"
    source.write_bytes(changed)
    receipt = collect_snapshot(snapshot, destination)
    assert receipt["status"] == "incomplete"
    assert receipt["counts"]["failed"]["bytes"] == len(b"original")
    assert receipt["counts"]["copied"]["bytes"] >= len(changed)
    failed = next(row for row in receipt["artifacts"] if row["status"] == "failed")
    assert failed["transferred_bytes"] == len(changed)
    assert not (destination / "files/attempts/training-0001/best.pt").exists()


def test_manifest_mutation_rejects_before_destination_creation(tmp_path):
    root = tmp_path / "run"
    singleton(root)
    snapshot, destination = tmp_path / "snapshot", tmp_path / "destination"
    prepare_snapshot(root, snapshot)
    path = snapshot / "manifest.json"
    manifest = read(path)
    manifest["artifacts"][0]["sha256"] = "0" * 64
    write(path, manifest)
    with pytest.raises(ValueError):
        collect_snapshot(snapshot, destination)
    assert not destination.exists()


def test_source_growth_during_copy_rejects_atomic_publication(tmp_path, monkeypatch):
    """Inject growth at the streaming boundary; no thread timing assumptions."""
    from pathlib import Path

    root = tmp_path / "run"
    attempt = singleton(root)
    source = attempt / "best.pt"
    source.write_bytes(b"initial")
    snapshot, destination = tmp_path / "snapshot", tmp_path / "destination"
    prepare_snapshot(root, snapshot, selected=["attempts/training-0001/best.pt"])
    original = Path.open

    def grow(path, mode="r", *args, **kwargs):
        opened = original(path, mode, *args, **kwargs)
        if path == source and mode == "rb":
            with original(source, "ab") as writer:
                writer.write(b"growth")
        return opened

    monkeypatch.setattr(Path, "open", grow)
    receipt = collect_snapshot(snapshot, destination)
    assert receipt["status"] == "incomplete"
    assert receipt["counts"]["failed"]["count"] == 1
    assert not (destination / "files/attempts/training-0001/best.pt").exists()
    monkeypatch.setattr(Path, "open", original)
    assert source.read_bytes() == b"initialgrowth"
