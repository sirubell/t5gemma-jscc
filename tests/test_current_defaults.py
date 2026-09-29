"""Prevent generic recipes from silently replacing the adopted completed protocols."""
from pathlib import Path

import pytest

from jscc.config import load_config, resolve_codec_configs
from jscc.evaluation_policy import evaluation_conditions


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("task", ["coco", "hellaswag"])
def test_current_tasks_share_the_completed_enc_l9_design(task):
    config = load_config(ROOT / f"configs/tasks/{task}.yaml")
    assert config["split"] == {"stack": "enc", "where": "after_layer", "index": 9}
    main, memory = resolve_codec_configs(config["codec"])
    assert (main["hidden_dim"], main["bottleneck_dim"], main["n_res_blocks"]) == (1152, 512, 2)
    assert main["layernorm"] == "none" and memory["layernorm"] == "both"
    assert main["snr_film"] is memory["snr_film"] is False
    assert config["model"]["revision"] == "dd0a2683227859151b1730ca3a63087df5b5f39b"
    assert config["training"]["batch_size"] * config["training"]["gradient_accumulation"] == 64
    assert config["training"]["streamed_backward"] and config["training"]["valid_only_kl"]
    assert evaluation_conditions(config["evaluation"]) == ["no_noise", -6, 6, 18]
    assert Path(config["run"]["output_dir"]) == ROOT / "runs"


def test_five_shot_default_is_training_policy_with_fixed_exposure_not_only_eval():
    config = load_config(ROOT / "configs/tasks/hellaswag.yaml")
    assert config["protocol"] == "corrected-baseline-v2-native-decoder-inputs"
    assert config["data"]["prompt_policy"] == {
        "version": "prompt-alignment-v1", "mode": "five_shot", "seed": 20260920,
        "source_max_length": 2048,
    }
    train = config["training"]
    assert train["max_steps"] == train["schedule_steps"] == 10000
    assert train["batch_size"] == 64 and train["gradient_accumulation"] == 1
    assert train["presentation_stream"]["total_presentations"] == 640000
    assert train["presentation_stream"]["policy"] == "epoch-permutations-v1"
    assert train["paired_randomness"]["policy"] == "step-microbatch-stream-v1"
    assert train["warmup_ratio"] * train["schedule_steps"] == 500
    assert config["evaluation"]["num_samples"] == 10042
    assert config["evaluation"]["scoring_policy"] == "fp32-v1"


def test_coco_default_retains_fixed_final_32k_protocol():
    config = load_config(ROOT / "configs/tasks/coco.yaml")
    train = config["training"]
    assert train["max_steps"] == train["schedule_steps"] == 500
    assert train["batch_size"] == 16 and train["gradient_accumulation"] == 4
    assert train["selection_steps"] == train["save_steps"] == [500]
    assert train["validation_snrs"] == [0, 6, 12]
    assert train["monitor"] == "kl" and train["min_lr_ratio"] == .1
    assert train["record_coco_presentations"] is True
    assert config["model"]["sdpa_backend_policy"] == "flash_math"
    assert (config["data"]["num_demos"], config["data"]["num_report"]) == (4, 2000)
    assert config["evaluation"]["max_new_tokens"] == 64
