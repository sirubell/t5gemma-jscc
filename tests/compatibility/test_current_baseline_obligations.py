"""Scientific obligations transferred without restoring retired execution fronts."""

import jscc
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import runpy
import pytest
import torch
from torch.amp.grad_scaler import GradScaler
from typing import cast
from jscc import shared_inputs, shared_training, training, ordinary_runs
from jscc.shared_inputs import bind_batch_identity, batch_view
from jscc.shared_plan import RecipePlan
from jscc.shared_training import run_recipe_training
from jscc.activation_replay import canonical_digest
from shared.test_recipe_training import setup_recipe
from shared.test_recipe_resume import setup as setup_state
from ordinary.test_ordinary_continuation import (
    task_fixture,
    assert_state_equal,
    scientific_rows,
)
from contracts.test_numerical_policy import make_model







def test_native_audit_samples_exact_epsilon_and_hashes_bound_inputs_once(
    tmp_path, monkeypatch
):
    a = tmp_path / "a"
    a.mkdir()
    b = tmp_path / "b"
    b.mkdir()
    full, data, _ = setup_recipe(a, locations=["enc_fn"], q=8)
    sparse, other, _ = setup_recipe(b, locations=["enc_fn"], q=8)
    full.audit_policy = "full"
    original_digest = shared_inputs.tensor_digest
    hashes = []

    def count(value):
        hashes.append(value)
        return original_digest(value)

    monkeypatch.setattr(shared_inputs, "tensor_digest", count)
    binding = bind_batch_identity(data)
    second = bind_batch_identity(other)
    count_before = len(hashes)
    randn = torch.randn
    draws = []

    def capture(*args, **kwargs):
        value = randn(*args, **kwargs)
        draws.append(value.clone())
        return value

    monkeypatch.setattr(torch, "randn", capture)
    for step in range(2):
        start = len(draws)
        full.update([data], batch_identities=[binding])
        middle = len(draws)
        sparse.update([other], batch_identities=[second])
        assert middle > start and len(draws) - middle == middle - start
        for left, right in zip(draws[start:middle], draws[middle:]):
            assert torch.equal(left, right)
        assert len(hashes) == count_before

    def forbidden(_):
        pytest.fail("uncaptured epsilon must not be copied/hashed")

    monkeypatch.setattr("jscc.models.channel.tensor_digest", forbidden)
    sparse.update([other], batch_identities=[second])
    assert sparse.last_draws[0]["draws"] == []


def test_native_final_capture_and_binding_rejections_precede_attempt(tmp_path):
    obj, data, _ = setup_recipe(tmp_path, locations=["enc_fn"], q=8)
    binding = bind_batch_identity(data)
    with pytest.raises((TypeError, AttributeError)):
        obj.update([data], batch_identities=[binding.view_sha256])
    assert obj.attempted == 0
    with pytest.raises(ValueError):
        obj.update([data], batch_identities=[])
    assert obj.attempted == 0
    from jscc.shared_state import RecipeTrainingState
    from jscc.asset_identity import state_dict_identity
    from jscc.site_routing import export_codec

    metadata = {
        "source_identity": "tiny-source",
        "config_identity": canonical_digest(obj.schedule.description["config"]),
        "data_identity": canonical_digest({"rows": "tiny"}),
        "prepared_identity": "indexed-tiny",
        "observation_identity": canonical_digest(
            obj.schedule.description["observations"]
        ),
        "initialization_identity": state_dict_identity(export_codec(obj.model)),
        "model_revision": "tiny-original13",
    }
    obj.identity.update(
        config=metadata["config_identity"], data=metadata["data_identity"]
    )
    state = RecipeTrainingState(obj, plan=obj.schedule, metadata=metadata)
    for step in range(obj.final_step):
        if obj.completed == obj.schedule.stages[1].start:
            state.reset(state.save(tmp_path / "boundary.pt"))
        event = obj.update([data], batch_identities=[binding])
        assert event["payload"]["audit"]["noise_capture"] == (step in {0, 4, 7})
    assert obj.last_draws[0]["draws"]


def test_native_empty_mask_rejected_by_real_learner_before_exposure(tmp_path):
    obj, data, _ = setup_recipe(tmp_path, locations=["enc_fn"], q=8)
    data["attention_mask"][0] = 0
    obj.schedule = RecipePlan(
        obj.schedule.description, (batch_view(data),) * obj.schedule.q, "paired"
    )
    with pytest.raises(ValueError, match="empty real sequence"):
        obj.update([data])
    assert obj.attempted == obj.completed == obj.offset == 0
    assert obj.failed


@pytest.mark.parametrize(
    "fault",
    [
        "ambiguous_optimizer",
        "event_append",
        "guard",
        "deadline_after_save",
        "terminal_deadline",
        "terminal_save_deadline",
        "final_publication",
    ],
)
def test_current_runner_faults_never_publish_freeze(tmp_path, monkeypatch, fault):
    state, data, _ = setup_state(tmp_path / "setup")
    state.learner.model.memory_codec = None
    state.learner.model.universal_codec = False
    actual = []
    factory = shared_training.create_recipe_learner

    def create(*args, **kwargs):
        obj = factory(*args, **kwargs)
        actual.append(obj)
        if fault == "ambiguous_optimizer":

            def ambiguous(optimizer, *a, **kw):
                optimizer.step(*a, **kw)
                raise RuntimeError("ambiguous optimizer receipt")

            monkeypatch.setattr(obj.scaler, "step", ambiguous)
        return obj

    monkeypatch.setattr(shared_training, "create_recipe_learner", create)
    if fault == "event_append":
        real = Path.open

        class BrokenLog:
            def __init__(self, stream):
                self.stream = stream

            def write(self, _):
                raise OSError("event append failed")

            def close(self):
                self.stream.close()

        def opening(path, *a, **kw):
            stream = real(path, *a, **kw)
            return (
                BrokenLog(stream)
                if path.name == "metrics.jsonl" and a and a[0] == "xb"
                else stream
            )

        monkeypatch.setattr(Path, "open", opening)
    expired = [False]
    if fault == "terminal_save_deadline":
        from jscc.shared_state import RecipeTrainingState

        original_save = RecipeTrainingState.save

        def crossing_save(owner, path):
            saved = original_save(owner, path)
            if str(path).endswith("terminal.pt"):
                expired[0] = True
            return saved

        monkeypatch.setattr(RecipeTrainingState, "save", crossing_save)
    if fault == "final_publication":
        original_manifest = shared_training.write_manifest
        fired = [False]

        def broken_final(path, payload):
            if (
                Path(path).name == "campaign.json"
                and payload.get("status") == "training_frozen"
                and not fired[0]
            ):
                fired[0] = True
                raise OSError("final campaign publication failed")
            return original_manifest(path, payload)

        monkeypatch.setattr(shared_training, "write_manifest", broken_final)

    def guard():
        if expired[0] or (fault == "guard" and actual and actual[0].completed == 1):
            raise TimeoutError("allocation deadline")

    def saved(state, role):
        if fault == "deadline_after_save" or (
            fault == "terminal_deadline" and role == "terminal"
        ):
            expired[0] = True

    with pytest.raises((RuntimeError, OSError, TimeoutError)):
        run_recipe_training(
            state.learner.model,
            None,
            plan=state.plan,
            batches=[data] * state.plan.q,
            metadata=state.metadata,
            output=tmp_path / "run",
            guard=guard,
            synthetic=True,
            data_ids={"rows": "tiny"},
            on_checkpoint=saved,
            observe=lambda model, processor, **kw: {
                "status": "complete",
                "request_identity": canonical_digest(kw["request"]),
            },
        )
    result = json.loads((tmp_path / "run/campaign.json").read_text())
    assert result["status"] == "incomplete" and "error" in result
    if fault != "final_publication":
        assert not (tmp_path / "run/freeze.json").exists()
    else:
        assert (
            result["status"] == "incomplete"
        )  # Freeze bytes cannot erase failed publication.
    if fault in {"event_append", "ambiguous_optimizer"}:
        assert (
            actual[0].failed
            and actual[0].completed == (1 if fault == "event_append" else 0)
            and actual[0].attempted == 1
        )
        with pytest.raises(RuntimeError, match="terminal"):
            actual[0].update([data])
    if fault == "ambiguous_optimizer":
        assert all(moment["step"] == 1 for moment in actual[0].optimizer.state.values())


def test_ordinary_bf16_records_checkpoint_and_exact_continuation(tmp_path, monkeypatch):
    config, _, _ = task_fixture(
        tmp_path, monkeypatch, "hellaswag", max_steps=3, checkpoint=1, accumulation=1
    )
    config["model"].update(dtype="bfloat16", numerical_policy="codec_receiver_fp32")
    config["split"] = {"stack": "enc", "where": "after_final_norm"}
    config["channel"]["train_noise"] = False
    monkeypatch.setattr(training, "build_model", lambda cfg: (None, make_model()))
    base = ordinary_runs.train_task(deepcopy(config), tmp_path / "baseline")
    before = torch.load(base / "attempts/training-0001/last.pt", weights_only=True)
    original = training.append_metrics

    def interrupt(path, row):
        original(path, row)
        if row.get("phase") == "train" and row["step"] == 2:
            raise KeyboardInterrupt("unsaved work")

    monkeypatch.setattr(training, "append_metrics", interrupt)
    root = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        ordinary_runs.train_task(deepcopy(config), root)
    parent = root / "attempts/training-0001"
    checkpoint = parent / "step_000001.pt"
    immutable = checkpoint.read_bytes()
    monkeypatch.setattr(training, "append_metrics", original)
    ordinary_runs.train_task(
        deepcopy(config), root, resume=checkpoint, attempt_id="0002"
    )
    child = root / "attempts/training-0002"
    after = torch.load(child / "last.pt", weights_only=True)
    for key in ("codec", "optimizer", "scheduler", "scaler", "torch_rng", "step"):
        assert_state_equal(after[key], before[key])
    assert checkpoint.read_bytes() == immutable
    assert scientific_rows(child) == [
        r for r in scientific_rows(base / "attempts/training-0001") if r["step"] > 1
    ]
    rows = [json.loads(s) for s in (child / "metrics.jsonl").read_text().splitlines()]
    row = next(r for r in rows if r["phase"] == "train")
    assert row["loss"] == pytest.approx(
        row["kl"] + config["training"]["mse_weight"] * row["nmse"]
    )
    assert after["config"]["model"]["numerical_policy"] == "codec_receiver_fp32"
    assert after["optimizer"]["state"]


def test_bf16_saved_reader_rejects_request_and_checkpoint_policy(tmp_path):
    from jscc.evaluation import validate_evaluation_target
    from jscc.experiment_state import save_state, open_state, ValidatedState
    from jscc.models.split_model import resolve_encoder_site

    model = make_model()
    opt = torch.optim.AdamW(model.codec.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    scaler = GradScaler("cpu", enabled=False)
    cast(torch.Tensor, sum(p.square().sum() for p in model.codec.parameters())).backward()
    opt.step()
    scheduler.step()
    metadata = dict(
        source_identity="s",
        config_identity="c",
        data_identity="d",
        parent_identity=None,
        initialization_identity="i",
        stream_identity="stream",
        protocol_identity="reader-fixture",
        lineage=[],
        completed_updates=1,
        phase="both",
        snapshot_role="trained",
        numerical_policy="codec_receiver_fp32",
        model_state_contract="codec-only-stateless-channel-v1",
        comparison_controls={},
        site=asdict(resolve_encoder_site(model.base, model.split, "tiny")),
    )
    ref = save_state(
        tmp_path / "state.pt",
        model=model.codec,
        optimizer=opt,
        scheduler=scheduler,
        scaler=scaler,
        metadata=metadata,
        stream_state={"completed_updates": 1, "offset": 64},
    )
    state = open_state(ref, expected=metadata)
    request = {
        "checkpoint_sha256": ref.sha256,
        "learner_kind": state.payload["kind"],
        "data": "d",
        "comparison": {},
        "step": 1,
        "target_site": metadata["site"],
        "target_role": "trained",
        "settings": {"numerical_policy": "codec_receiver_fp32"},
        "precision": "bf16",
        "backend": model.base.config.decoder._attn_implementation,
        **{
            key: metadata[key + "_identity"]
            for key in ("config", "source", "protocol", "parent")
        },
    }
    validate_evaluation_target(state, request, model)
    with pytest.raises(ValueError, match="Observation numerical policy"):
        validate_evaluation_target(state, {**request, "settings": {}}, model)
    changed = deepcopy(state.payload)
    changed["metadata"].pop("numerical_policy")
    with pytest.raises(ValueError, match="Checkpoint numerical policy"):
        validate_evaluation_target(ValidatedState(changed, ref), request, model)


@pytest.mark.parametrize("fault", ["missing", "corrupted", "incomplete"])
def test_retained_paired_writer_rejects_saved_arm_damage(tmp_path, fault):
    from jscc.paired_report import write_baseline_comparisons
    from jscc.experiment_records import read_events, artifact_ref

    fixture = runpy.run_path(
        str(Path(jscc.__file__).resolve().parents[1] / "tests/fixtures/experiment_records/build_fixture.py")
    )
    source = fixture["build"](tmp_path / "source")
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    import shutil

    for cell in ("D-none", "D-LN", "R-none", "R-LN"):
        arm = campaign / cell
        shutil.copytree(source / "baseline-direct", arm)
        (arm / "run.json").write_text(
            json.dumps({"segment": {"strategy": "both", "cell": cell}})
        )
        inventory = json.loads((arm / "inventory.json").read_text())
        inventory["mode"] = "tensor-complete"
        (arm / "inventory.json").write_text(json.dumps(inventory))
    valid = write_baseline_comparisons(campaign, steps=(12,))
    assert len(valid["contrasts"]) == 4
    (campaign / "comparisons.json").unlink()
    arm = campaign / "D-LN"
    if fault == "missing":
        shutil.rmtree(arm)
    elif fault == "corrupted":
        inventory = json.loads((arm / "inventory.json").read_text())
        tensor = next(r for r in inventory["artifacts"] if r["kind"] == "tensor")
        (arm / tensor["path"]).write_bytes(b"corrupted")
    else:
        path = arm / "metrics.jsonl"
        events = read_events(path)
        events = [
            e
            for e in events
            if not (
                e["event_type"] == "observation"
                and e["payload"]["request"]["step"] == 12
            )
        ]
        path.write_text("".join(json.dumps(e) + "\n" for e in events))
        inventory = json.loads((arm / "inventory.json").read_text())
        for artifact in inventory["artifacts"]:
            if artifact["path"] == "metrics.jsonl":
                artifact.update(artifact_ref(path, arm, artifact["kind"]))
        (arm / "inventory.json").write_text(json.dumps(inventory))
    with pytest.raises((ValueError, FileNotFoundError)):
        write_baseline_comparisons(campaign, steps=(12,))
    assert not (campaign / "comparisons.json").exists()
