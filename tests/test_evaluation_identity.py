"""CPU-only identity serialization, including separate interpreter processes."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from tokenizers import AddedToken

from jscc.evaluation_policy import _identity_json_default, digest, evaluation_identity


def identity(metadata, **overrides):
    config = SimpleNamespace(to_dict=lambda: {"dtype": torch.bfloat16})
    model = SimpleNamespace(base=SimpleNamespace(
        parameters=lambda: iter([torch.zeros(1)]), config=config))
    tokenizer = SimpleNamespace(special_tokens_map={"eos_token": AddedToken("</s>")})
    settings = {"num_samples": 16, "num_fewshot": 5, "batch_size": 64, **overrides}
    adapter = SimpleNamespace(forward_logits_dtypes={"torch.bfloat16"}, max_length=32)
    return evaluation_identity(model, tokenizer, settings, {"revision": "fixed"}, metadata, adapter)


def test_actual_harness_nested_callable_across_processes():
    script = '''
import json
from lm_eval.tasks import TaskManager
from lm_eval.config.task import TaskConfig
from lm_eval.tasks._yaml_loader import load_yaml
from test_evaluation_identity import identity
from jscc.evaluation_policy import digest
recipe = load_yaml(TaskManager().task_index["hellaswag"].yaml_path, resolve_func=True)
recipe["fewshot_config"] = {"process_docs": recipe["process_docs"]}
metadata = {"configs": {"hellaswag": TaskConfig(**recipe).to_dict()}}
function = metadata["configs"]["hellaswag"]["fewshot_config"]["process_docs"]
assert callable(function)
fixed = identity(metadata)
print(json.dumps({"old": digest(json.loads(json.dumps(metadata, default=str))),
                  "fixed": fixed, "digest": digest(fixed)}))
'''
    env = {**{key: os.environ[key] for key in ("PATH", "HOME", "TMPDIR") if key in os.environ},
           "PYTHONPATH": os.pathsep.join([str(Path.cwd()), str(Path.cwd() / "tests")]),
           "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": ""}
    records = [json.loads(subprocess.check_output([sys.executable, "-c", script], env=env,
                                                text=True).splitlines()[-1]) for _ in range(3)]
    assert len({row["old"] for row in records}) > 1
    assert len({row["digest"] for row in records}) == 1
    fixed = records[0]["fixed"]
    assert fixed["version"] == "hellaswag-evaluation-identity-v2"
    function = fixed["task_config"]["hellaswag"]["fewshot_config"]["process_docs"]
    assert function["qualname"] == "process_docs"
    assert len(function["module_sha256"]) == 64
    assert fixed["frozen_model"]["config"]["dtype"] == "torch.bfloat16"
    assert fixed["tokenizer"]["special_tokens"]["eos_token"] == "</s>"


def test_same_named_function_helper_changes_identity(tmp_path):
    def load(directory, suffix):
        directory.mkdir()
        path = directory / "helper.py"
        path.write_text(f'def preprocess(text):\n    return text + {suffix!r}\n\ndef process_docs(rows):\n    return [preprocess(row) for row in rows]\n')
        spec = importlib.util.spec_from_file_location("helper", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.process_docs

    first = load(tmp_path / "a", "a")
    copied = load(tmp_path / "copy", "a")
    changed = load(tmp_path / "b", "b")
    def metadata(function):
        return {"configs": {"hellaswag": {"fewshot_config": {"process_docs": function}}}}
    baseline = identity(metadata(first))
    assert baseline == identity(metadata(copied))
    assert digest(baseline) != digest(identity(metadata(changed)))
    for overrides in ({"batch_size": 32}, {"num_fewshot": 0}, {"num_samples": 8},
                      {"scoring_policy": "fp32-v1"}):
        assert digest(baseline) != digest(identity(metadata(first), **overrides))


@pytest.mark.parametrize("value", [lambda value: value, len, object()])
def test_unsupported_values_fail_without_repr_fallback(value):
    with pytest.raises(TypeError, match="Evaluation identity|Unsupported evaluation identity"):
        json.dumps({"nested": [value]}, default=_identity_json_default)


def test_stateful_callable_rejected():
    def closure():
        return closure
    class Stateful:
        def __call__(self):
            return 1
    for value in (closure, Stateful(), Stateful().__call__):
        with pytest.raises(TypeError, match="stateless module-level"):
            _identity_json_default(value)


@pytest.mark.parametrize("source", [
    "def process_docs(rows=()):\n    return rows\n",
    "def process_docs(rows, *, suffix='x'):\n    return rows\n",
    "def process_docs(rows):\n    return rows\nprocess_docs.state = 1\n",
])
def test_module_function_with_semantic_state_rejected(tmp_path, source):
    path = tmp_path / "stateful.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("stateful", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(TypeError, match="stateless module-level"):
        _identity_json_default(module.process_docs)
