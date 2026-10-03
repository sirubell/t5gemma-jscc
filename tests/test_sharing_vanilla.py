"""Actual lifecycle/evaluator/harness/writer/reader; only dataset is in memory."""

import json

import pytest
import torch
from datasets import Dataset, DatasetDict

from jscc import evaluation
from jscc.sharing_run import run_sharing
from jscc.sharing_state import verify_study_freeze
from sharing_fixtures import sharing_batches, sharing_model, sharing_plan
from test_harness_payload import fixture_model
from test_shared_lifecycle import geometry_batches, task_template


@pytest.fixture
def real_evaluator_inputs(monkeypatch):
    torch.set_num_threads(1)

    def rows(count):
        return [
            {
                "ctx_a": "x",
                "ctx_b": "x",
                "activity_label": "x",
                "endings": ["x", "x x", "x x x", "x x x x"],
                "label": str(i % 4),
                "source_id": f"dev{i}",
                "ind": i,
            }
            for i in range(count)
        ]

    raw = DatasetDict(
        {"train": Dataset.from_list(rows(8)), "validation": Dataset.from_list(rows(2))}
    )
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: raw)
    # Reuse the real external harness task index across109 panel invocations.
    from lm_eval.tasks import TaskManager

    manager = TaskManager()
    monkeypatch.setattr("lm_eval.tasks.TaskManager", lambda: manager)
    tokenizer, _ = fixture_model()
    model = sharing_model("D-N")
    template = task_template()
    template.update(
        input_ids=[0, 1],
        source_family_ids=["dev0", "dev1"],
        backend=model.base.config.decoder._attn_implementation,
    )
    template["settings"].update(
        batch_size=8,
        num_fewshot=0,
        num_samples=2,
        scoring_policy="fp32-v1",
        evidence_mode="compact-v1",
        noise_seed=0,
    )
    template["data_settings"] = {"name": "fixture", "revision": "fixed"}
    return model, tokenizer, template


def test_actual_lifecycle_compact_vanilla(tmp_path, real_evaluator_inputs):
    model, tokenizer, template = real_evaluator_inputs
    root = tmp_path / "run"
    result = run_sharing(
        model,
        tokenizer,
        plan=sharing_plan(),
        update_batches=sharing_batches(),
        validation_batches=sharing_batches()[:1],
        metadata={
            "source_identity": "source",
            "config_identity": "config",
            "data_identity": "data",
            "model_revision": "tiny-cpu",
            "initialization_identity": "fresh-tiny",
            "stream_identity": "same-stream",
            "protocol_identity": "tiny-sharing",
            "recipe_identity": "K+.1R",
            "architecture_decision": "D-N",
        },
        task_template=template,
        output=root,
        geometry_batches=geometry_batches(),
        resource_guard=lambda: None,
    )
    assert result["status"] == "complete"
    assert (result["completed_updates"], result["completed_checkpoints"]) == (24, 17)
    assert len(list(root.rglob("*.pt"))) == 17
    assert verify_study_freeze(result["freeze_reference"])["synthetic"]
    observations = json.loads((root / "observations.json").read_text())
    task = [r for r in observations if r["request"]["purpose"] == "task"]
    assert sum(len(r["conditions"]) for r in task) == 108
    assert (
        sum(
            len(r["conditions"])
            for r in observations
            if r["request"]["purpose"] == "objective"
        )
        == 192
    )
    ordinary = [
        c for r in task for c in r["conditions"] if c["condition"] == "no_noise"
    ]
    assert len(ordinary) == 18
    ordinary_files = list(root.rglob("compact_no_noise.jsonl"))
    assert len(ordinary_files) == 18
    for artifact in ordinary_files:
        ordinary_rows = [json.loads(line) for line in artifact.read_text().splitlines()]
        assert all(p["hidden"] > 0 for row in ordinary_rows for p in row["valid_payload_coordinates"])
    report = json.loads((root / "sharing-comparison.json").read_text())
    assert report["vanilla_bypass"]["available"] is True
    rows, ids, families = evaluation._observation_items(
        root / "vanilla", "hellaswag", "vanilla"
    )
    assert ids == [0, 1] and families == ["dev0", "dev1"]
    assert all(
        p == {"hidden": 0, "memory": 0}
        for row in rows
        for p in row["valid_payload_coordinates"]
    )
    assert not (root / "vanilla/compact_no_noise.jsonl").exists()
    assert (
        json.loads((root / "vanilla/receipt.json").read_text())["status"] == "complete"
    )
    assert not torch.cuda.is_initialized()


def test_real_evaluator_rejects_bypass_mislabeled_no_noise(
    tmp_path, real_evaluator_inputs
):
    model, tokenizer, template = real_evaluator_inputs
    model.eval()
    with torch.no_grad(), model.transmission(bypass=True):
        with pytest.raises(ValueError, match="Per-item valid payload disagrees"):
            evaluation.evaluate_hellaswag(
                model,
                tokenizer,
                template["settings"],
                tmp_path,
                "no_noise",
                template["data_settings"],
            )
        assert model.valid_payload_counts == {"hidden": 0, "memory": 0}
