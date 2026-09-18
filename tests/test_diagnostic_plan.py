"""The pre-H200 HellaSwag diagnostic is a fixed, reviewable five-run plan."""
from pathlib import Path

from jscc.config import load_config
from jscc.studies import expand_study


ROOT = Path(__file__).parents[1]
TASK = ROOT / "configs/tasks/hellaswag_diagnostic.yaml"
STUDY = ROOT / "configs/studies/hellaswag_diagnostic.yaml"


def test_diagnostic_task_keeps_a_short_stop_budget_and_long_schedule():
    config = load_config(TASK)
    training = config["training"]
    assert config["task"] == "hellaswag"
    assert training["max_steps"] == 4000
    assert training["schedule_steps"] == 20000
    assert (training["batch_size"], training["gradient_accumulation"]) == (16, 2)
    assert training["batch_size"] * training["gradient_accumulation"] == 32
    assert config["data"]["num_validation"] == 512
    assert training["validation_batches"] is None
    assert config["codec"]["snr_film"] is False
    assert config["codec"]["memory"] == {"layernorm": "both"}
    assert config["evaluation"] == {
        "snrs": ["no_noise", -6, 18],
        "batch_size": 8,
        "num_samples": 512,
        "num_fewshot": 5,
        "vanilla": True,
    }


def test_diagnostic_study_is_hellaswag_only_and_holds_memory_norm_fixed():
    plan = expand_study(STUDY)
    assert len(plan.runs) == 5
    assert {run.task for run in plan.runs} == {"hellaswag"}
    assert [run.experiment for run in plan.runs] == [
        "enc_fn", "enc_l9_old", "enc_l9_new", "dec_l8_old", "dec_l8_new"
    ]

    expected = {
        "enc_fn": ({"stack": "enc", "where": "after_final_norm"}, "both"),
        "enc_l9_old": ({"stack": "enc", "where": "after_layer", "index": 9}, "both"),
        "enc_l9_new": ({"stack": "enc", "where": "after_layer", "index": 9}, "none"),
        "dec_l8_old": ({"stack": "dec", "where": "after_layer", "index": 8}, "both"),
        "dec_l8_new": ({"stack": "dec", "where": "after_layer", "index": 8}, "none"),
    }
    for run in plan.runs:
        split, main_norm = expected[run.experiment]
        assert run.config["split"] == split
        assert run.config["codec"]["layernorm"] == main_norm
        assert run.config["codec"]["memory"] == {"layernorm": "both"}
        assert run.config["codec"]["snr_film"] is False
        assert run.config["training"]["max_steps"] == 4000
        assert run.config["training"]["schedule_steps"] == 20000
        assert run.config["evaluation"]["snrs"] == ["no_noise", -6, 18]
        assert run.config["evaluation"]["num_samples"] == 512
