"""Current primitive assertions extracted without resurrecting legacy execution."""
import pytest
import torch
from datasets import Dataset, DatasetDict
from jscc import evaluation
from sharing_fixtures import sharing_model
from contracts.test_harness_payload import fixture_model
from shared.test_shared_lifecycle import task_template


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
