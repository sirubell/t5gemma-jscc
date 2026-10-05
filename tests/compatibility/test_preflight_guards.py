"""Preflight health must reject a partially disconnected trainable model."""
import pytest
import torch
from jscc.qualification import checked_gradient_names


def test_missing_gradient_cannot_pass_with_other_nonzero_gradients():
    model = torch.nn.Linear(2, 2)
    model.weight.grad = torch.ones_like(model.weight)
    with pytest.raises(RuntimeError, match='missing trainable gradient: bias'):
        checked_gradient_names(model)
    model.bias.grad = torch.ones_like(model.bias)
    assert checked_gradient_names(model) == ['weight', 'bias']
    model.bias.grad[0] = float('nan')
    with pytest.raises(FloatingPointError, match='nonfinite'):
        checked_gradient_names(model)


def test_indexed_helpers_preserve_native_policy_row_identity():
    from jscc.shared_inputs import IndexedCollator, IndexedRows

    dataset = [{"value": 4}, {"value": 9}]
    rows = IndexedRows(dataset, ["row-a", "row-b"])
    assert len(rows) == 2
    batch = IndexedCollator(lambda values: {"values": [v["value"] for v in values]})(
        [rows[1], rows[0]]
    )
    assert batch == {"values": [9, 4], "preflight_row_ids": ["row-b", "row-a"]}


def test_no_trainable_gradients_rejected():
    model = torch.nn.Linear(2, 2)
    model.requires_grad_(False)
    with pytest.raises(RuntimeError, match="no trainable gradients"):
        checked_gradient_names(model)


@pytest.fixture
def prompt_policy_run(tmp_path, monkeypatch):
    import json
    from jscc import presentation
    from jscc.checkpoint_readers import prompt_policy_sha as sha

    monkeypatch.setattr(presentation, "training_source_digest", lambda: "bound-source")
    (tmp_path / "config.yaml").write_text("fixed: config\n")
    checkpoint = tmp_path / "step_010000.pt"
    checkpoint.write_bytes(b"fixed checkpoint")
    documents = {
        "completion.json": {
            "status": "FULL_BUDGET_COMPLETED", "step": 10000,
            "optimizer_updates": 10000, "presentations": 640000,
            "presentation_budget_verified": True, "source_verified": True,
            "final_checkpoint": {"file": checkpoint.name, "sha256": sha(checkpoint)},
        },
        "run.json": {
            "resolved_config_sha256": sha(tmp_path / "config.yaml"),
            "training_source_sha256": "bound-source",
        },
        "launch-binding.json": {"prepared_config_sha256": "prepared-config"},
    }
    for name, contents in documents.items():
        (tmp_path / name).write_text(json.dumps(contents))
    return tmp_path, documents


def test_retained_prompt_policy_validator_accepts_exact_checkpoint(prompt_policy_run):
    from jscc.checkpoint_readers import validate_prompt_policy_run as validate_run

    run, _ = prompt_policy_run
    assert validate_run(run, 10000, "prepared-config") == run / "step_010000.pt"


@pytest.mark.parametrize("document,key,value,error", [
    ("completion.json", "status", "PARTIAL", "Incomplete fixed study budget"),
    ("completion.json", "step", 9999, "Incomplete fixed study budget"),
    ("completion.json", "optimizer_updates", 9999, "Incomplete fixed study budget"),
    ("completion.json", "presentations", 639999, "Incomplete fixed study budget"),
    ("completion.json", "presentation_budget_verified", False, "Incomplete fixed study budget"),
    ("completion.json", "source_verified", False, "Incomplete fixed study budget"),
    ("run.json", "training_source_sha256", "foreign-source", "Run config/source binding mismatch"),
    ("run.json", "resolved_config_sha256", "foreign-config", "Run config/source binding mismatch"),
    ("launch-binding.json", "prepared_config_sha256", "foreign-recipe", "Prepared config mismatch"),
    ("completion.json", "final_checkpoint", {"file": "best.pt", "sha256": "foreign"}, "Checkpoint binding mismatch"),
])
def test_retained_prompt_policy_validator_rejects_changed_binding(
    prompt_policy_run, document, key, value, error
):
    import json
    from jscc.checkpoint_readers import validate_prompt_policy_run as validate_run

    run, documents = prompt_policy_run
    documents[document][key] = value
    (run / document).write_text(json.dumps(documents[document]))
    with pytest.raises(ValueError, match=error):
        validate_run(run, 10000, "prepared-config")


def test_retained_prompt_policy_validator_rejects_changed_checkpoint(prompt_policy_run):
    from jscc.checkpoint_readers import validate_prompt_policy_run as validate_run

    run, _ = prompt_policy_run
    (run / "step_010000.pt").write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="Checkpoint binding mismatch"):
        validate_run(run, 10000, "prepared-config")
