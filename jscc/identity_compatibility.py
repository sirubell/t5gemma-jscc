"""Compare archived identities without hashing a process-local callable address.

Raw identities and their hashes remain immutable. Only the known lm-eval
fewshot process_docs repr is canonicalized; task source and all evidence digests
remain in the compatibility key.
"""
import copy
import hashlib
import json
import re


def identity_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def compatible_identity(identity, expected_hash):
    if identity_digest(identity) != expected_hash:
        raise ValueError("Archived evaluation identity hash mismatch")
    result = copy.deepcopy(identity)
    task = result["task_config"]["hellaswag"]
    value = task["fewshot_config"]["process_docs"]
    if isinstance(value, str) and re.fullmatch(r"<function process_docs at 0x[0-9a-fA-F]+>", value):
        if not isinstance(task.get("process_docs"), str) or not task["process_docs"].startswith("def process_docs("):
            raise ValueError("Callable normalization requires recorded process_docs source")
        task["fewshot_config"]["process_docs"] = "<function process_docs>"
    return identity_digest(result)
