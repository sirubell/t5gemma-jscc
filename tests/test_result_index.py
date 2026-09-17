from scripts.index_results import metric_rows


def test_unknown_sample_count_is_not_inferred_from_directory():
    rows = metric_rows({"split": "dec_l0", "snr": None,
                        "results": {"coco_CIDEr,none": 0.88}}, "june")
    assert rows[0]["sample_count"] is None
    assert rows[0]["value"] == 0.88
    assert "clean_memory_bypass_documented" in rows[0]["flags"]


def test_march_mixed_runs_remain_flagged():
    rows = metric_rows({"eval/no_noise_acc_norm": 0.56}, "march",
                       {"history_run": {"run_id": "a"}, "eval_run": {"run_id": "b"}})
    assert rows[0]["metric"] == "acc_norm"
    assert "mixed_run_provenance" in rows[0]["flags"]


def test_vanilla_is_not_labeled_codec_no_noise():
    row = metric_rows({"split": "vanilla", "snr": None,
                       "result": {"acc_norm,none": 0.64}}, "june")[0]
    assert row["condition"] == "vanilla"


def test_remote_memory_coding_and_selection_panel_are_distinct():
    from scripts.index_remote_results import observations
    row = observations({"split": "dec_l0", "codec_encoder_memory": True,
                        "acc_norm_codec": 0.6, "eval_samples": 10042})[0]
    assert row["sample_count"] == 10042
    assert "global_memory_coding_documented" in row["flags"]
    row = observations({"split": "dec_l20", "cider": 0.8, "snapshot": "step_004000",
                        "condition_id": "0_true", "provenance": {}})[0]
    assert "selection_panel_not_final_report" in row["flags"]
    assert row["sample_count"] is None
