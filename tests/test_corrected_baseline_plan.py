"""The corrected HellaSwag baseline stays separate from legacy evidence."""
from pathlib import Path

from jscc.config import load_config
from jscc.studies import expand_study


ROOT = Path(__file__).parents[1]
TASK = ROOT / "configs/tasks/hellaswag_corrected.yaml"
STUDY = ROOT / "configs/studies/hellaswag_corrected.yaml"


def test_corrected_task_has_the_fixed_shared_protocol():
    config = load_config(TASK)
    assert config["protocol"] == "corrected-baseline-v1"
    assert config["task"] == "hellaswag"
    assert config["seed"] == 0
    assert config["model"]["name"] == "google/t5gemma-2-1b-1b"
    assert config["codec"]["hidden_dim"] == 1152
    assert config["codec"]["bottleneck_dim"] == 512
    assert config["codec"]["snr_film"] is False
    training = config["training"]
    assert training["max_steps"] == 4000
    assert training["schedule_steps"] == 20000
    assert (training["batch_size"], training["gradient_accumulation"]) == (16, 2)
    assert config["data"]["num_validation"] == 512
    assert config["evaluation"] == {
        "snrs": ["no_noise", -6, 18],
        "batch_size": 8,
        "num_samples": 512,
        "num_fewshot": 5,
        "vanilla": True,
    }


def test_corrected_study_has_three_routes_and_explicit_memory_norm():
    plan = expand_study(STUDY)
    assert len(plan.runs) == 3
    assert {run.task for run in plan.runs} == {"hellaswag"}
    assert [run.experiment for run in plan.runs] == [
        "corrected_enc_fn", "corrected_enc_l9", "corrected_dec_l8"
    ]

    expected = {
        "corrected_enc_fn": ({"stack": "enc", "where": "after_final_norm"}, "both"),
        "corrected_enc_l9": ({"stack": "enc", "where": "after_layer", "index": 9}, "none"),
        "corrected_dec_l8": ({"stack": "dec", "where": "after_layer", "index": 8}, "none"),
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
