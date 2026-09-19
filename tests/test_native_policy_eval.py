import json
import time

import pytest
import torch
import torch.nn.functional as F

from scripts.native_policy_eval import Evidence, paired_scores, stratified_panel


def test_panel_reproducible_source_and_length_coverage():
    rows = [{'doc_id': i, 'source': str(i % 2), 'input_length': i,
             'continuation_length': 512-i} for i in range(512)]
    first = stratified_panel(rows)
    assert first == stratified_panel(list(reversed(rows)))
    assert len(first) == len({r['doc_id'] for r in first}) == 192
    assert {r['stratum'][1] for r in first} == {0, 1, 2, 3}
    assert {r['source'] for r in first} == {'0', '1'}


def test_reservation_persisted_before_failure_and_cap(tmp_path):
    evidence = Evidence(tmp_path, time.monotonic()+30, cap=10)
    evidence.reserve(8, 'attempt')
    with pytest.raises(RuntimeError, match='cap'):
        evidence.reserve(3, 'overflow')
    assert evidence.reserved == 8
    assert json.loads((tmp_path/'reservations.jsonl').read_text())['requests'] == 8
    evidence.deadline = 0
    with pytest.raises(TimeoutError):
        evidence.reserve(1, 'late')
    assert evidence.reserved == 8


def test_fp32_scoring_upcasts_before_softmax_and_sum():
    logits = torch.tensor([[10.01, 11.12, 7.13], [9.22, 8.33, 10.42]], dtype=torch.bfloat16)
    targets = torch.tensor([1, 2])
    result = paired_scores(logits, targets)
    expected = F.log_softmax(logits.float(), -1)[torch.arange(2), targets]
    assert result['fp32_score'] == float(expected.sum())
    assert result['fp32_token_log_probs'] == expected.tolist()
    assert result['native_score'] == float(F.log_softmax(logits, -1)[torch.arange(2), targets].sum())
    assert result['logits_dtype'] == 'torch.bfloat16'


def test_panel_rejects_duplicate_or_insufficient_ids():
    with pytest.raises(ValueError):
        stratified_panel([])
    with pytest.raises(ValueError):
        stratified_panel([{'doc_id': 0}]*192)
