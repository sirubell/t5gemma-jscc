"""CPU checks for bounded oracle/noise interventions on real tiny Transformers."""
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from scripts.experiments.evening_eval import Intervention, compare, replay_epsilon, score
from test_harness_payload import fixture_model


def batch():
    return [{"sample_id": i // 4, "candidate": i % 4, "context": f"c{i}", "continuation": f"a{i}",
             "context_token_ids": [2, 4, 1] if i % 2 else [2, 1],
             "continuation_token_ids": [4, 1] if i % 2 else [4, 0, 1]}
            for i in range(64)]


class Budget:
    def __init__(self):
        self.used = 0

    def reserve(self, count, cell, phase):
        self.used += count


def adapter(model, tokenizer):
    from lm_eval.models.huggingface import HFLM
    from jscc.harness_payload import payload_harness_class
    return payload_harness_class(HFLM)(communication_model=model, pretrained=model.base,
                                      tokenizer=tokenizer, backend="seq2seq", batch_size=64, max_length=2048,
                                      softmax_dtype=torch.float32)


def test_oracle_streams_reproduce_production_vanilla_and_preserve_transmitter():
    torch.manual_seed(123)
    tokenizer, model = fixture_model()
    harness, budget = adapter(model, tokenizer), Budget()
    production, _, native = score(harness, model, batch(), budget, "production", "test")
    cc, cc_ev, _ = score(harness, model, batch(), budget, "CC", "test", {"hidden": "C", "memory": "C"})
    vanilla, _, _ = score(harness, model, batch(), budget, "vanilla", "test")
    oo, oo_ev, _ = score(harness, model, batch(), budget, "OO", "test", {"hidden": "O", "memory": "O"})
    co, co_ev, _ = score(harness, model, batch(), budget, "CO", "test", {"hidden": "C", "memory": "O"})
    assert compare(production, cc)["passed"]
    assert compare(vanilla, oo)["passed"]
    assert oo_ev["allocated"] == {"hidden": 0, "memory": 0}
    assert co_ev["allocated"]["hidden"] > 0 and co_ev["allocated"]["memory"] == 0
    assert cc_ev["valid"]["hidden"] == sum(len(r["continuation_token_ids"]) * 4 for r in batch())
    def h(ev):
        return next(e["input_sha256"] for e in ev["events"] if e["stream"] == "hidden")
    assert h(cc_ev) == h(co_ev) == h(oo_ev)
    assert all(e["codec_calls"] == e["channel_calls"] == 1 for e in cc_ev["events"])
    assert all(e["oracle_calls"] == 1 and e["codec_calls"] == 0 for e in oo_ev["events"])
    assert native["native_decoder"]["input_ids"][0][0] == 2
    assert budget.used == 320


def test_noise_valid_coordinates_invariant_to_padding_companions_and_stream_independent():
    keys = [[1, 0], [2, 0]]
    z = torch.zeros(2, 3, 4)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    a, proof = replay_epsilon(z, mask, keys, 20260921, "hidden")
    expanded = torch.zeros(1, 7, 4)
    b, again = replay_epsilon(expanded, torch.tensor([[1, 1, 0, 0, 0, 0, 0]], dtype=torch.bool),
                             keys[:1], 20260921, "hidden")
    c, _ = replay_epsilon(z, mask, keys, 20260921, "memory")
    assert torch.equal(a[0, :2], b[0, :2])
    assert proof[0]["valid_epsilon_sha256"] == again[0]["valid_epsilon_sha256"]
    assert not torch.equal(a, c)
    assert bool(torch.any(b[0, 2:] != 0))  # Padding still receives physical AWGN.


def test_selective_noise_pairing_and_restore_on_failure():
    tokenizer, model = fixture_model()
    harness, budget = adapter(model, tokenizer), Budget()
    proofs = {}
    for case, policies in (("AN", {"hidden": "A", "memory": "C"}),
                           ("NA", {"hidden": "C", "memory": "A"}),
                           ("AA", {"hidden": "A", "memory": "A"})):
        _, evidence, _ = score(harness, model, batch(), budget, case, "test", policies, 20260921)
        proofs[case] = {e["stream"]: e.get("noise") for e in evidence["events"]}
    assert proofs["AN"]["hidden"] == proofs["AA"]["hidden"]
    assert proofs["NA"]["memory"] == proofs["AA"]["memory"]
    assert proofs["AN"]["memory"] is None
    original = model._transmit
    with pytest.raises(RuntimeError):
        with Intervention(model, {"hidden": "C", "memory": "C"}, None, []).install():
            raise RuntimeError("fail")
    assert model._transmit == original


@pytest.mark.parametrize("module", ["evening_eval", "evening_gradients", "evening_pilot"])
@pytest.mark.parametrize("argv", [["--help"], ["--output", "unused-output", "--route", "dec_l20"]])
def test_retired_module_execution_rejects_before_imports_or_output(module, argv, tmp_path):
    # -S removes third-party packages: the retirement boundary must precede even
    # dependency imports, as well as model/data loading and output creation.
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "unused-output"
    argv = [str(output) if value == "unused-output" else value for value in argv]
    result = subprocess.run(
        [sys.executable, "-S", "-m", f"scripts.experiments.{module}", *argv],
        cwd=root, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1
    assert f"{module} executor is retired" in result.stderr
    assert "Traceback" not in result.stderr
    assert not result.stdout
    assert not output.exists()
