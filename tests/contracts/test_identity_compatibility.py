import copy

import pytest

from jscc.identity_compatibility import compatible_identity, identity_digest


def fixture_identity(address="1234"):
    return {"task_config": {"hellaswag": {
        "process_docs": "def process_docs(dataset): return dataset",
        "fewshot_config": {"process_docs": f"<function process_docs at 0x{address}>"}}},
        "prompts_digest": "prompts", "batch": 64, "source_sha256": {"eval": "source"}}


def key(value):
    return compatible_identity(value, identity_digest(value))


def test_process_address_only_is_compatible_without_mutating_evidence():
    a, b = fixture_identity(), fixture_identity("abcd")
    original = copy.deepcopy(a)
    assert identity_digest(a) != identity_digest(b)
    assert key(a) == key(b)
    assert a == original


@pytest.mark.parametrize("field,value", [("prompts_digest", "other"), ("batch", 32),
                                        ("source_sha256", {"eval": "changed"})])
def test_real_policy_or_evidence_changes_remain_different(field, value):
    a, b = fixture_identity(), fixture_identity()
    b[field] = value
    assert key(a) != key(b)


def test_task_source_and_function_name_are_not_erased():
    a, b = fixture_identity(), fixture_identity()
    b["task_config"]["hellaswag"]["process_docs"] += " # changed"
    assert key(a) != key(b)
    b = fixture_identity()
    b["task_config"]["hellaswag"]["fewshot_config"]["process_docs"] = "<function other at 0x1234>"
    assert key(a) != key(b)


def test_bad_hash_and_missing_source_rejected():
    a = fixture_identity()
    with pytest.raises(ValueError, match="hash mismatch"):
        compatible_identity(a, "invalid")
    del a["task_config"]["hellaswag"]["process_docs"]
    with pytest.raises(ValueError, match="requires recorded"):
        key(a)
