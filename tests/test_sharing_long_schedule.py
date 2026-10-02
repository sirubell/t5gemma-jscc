"""Formal native16 long schedule: finite stream, sparse selection, post-freeze final panels."""
from dataclasses import asdict, replace
import hashlib

import pytest

from jscc.models.split_model import ResolvedSite
from jscc.sharing_schedule import (
    BatchView, HELDOUT_SITE, NATIVE16_LONG_PROTOCOL, NATIVE16_PROTOCOL,
    NATIVE16_PROTOCOLS, SharingExposureLedger, SharingPlan, TRAINED_SITES,
    long_final_panels, native16_exposure,
)
from jscc.sharing_state import (
    build_sharing_state, create_study_freeze, validate_sharing_state, verify_study_freeze,
)


def plan(q=4, *, synthetic=True, protocol=NATIVE16_LONG_PROTOCOL):
    def digest(value):
        return hashlib.sha256(str(value).encode()).hexdigest()
    views = tuple(BatchView(digest(i), digest((i, "mask")), digest((i, "layout")),
                            16, 32 + i, 24 + i, 48 + i) for i in range(q))
    return SharingPlan(views, "long-paired", synthetic=synthetic, protocol_id=protocol)


def state(schedule, completed):
    return build_sharing_state(
        synthetic=schedule.plan.synthetic, learner=schedule.name,
        ordered_view_identities=[v.view_sha256 for v in schedule.plan.views],
        completed_updates=completed,
        source_valid_per_view=[v.source_tokens for v in schedule.plan.views],
        target_valid_per_view=[v.target_tokens for v in schedule.plan.views],
        padded_per_view=[v.padded_tokens for v in schedule.plan.views],
        batch_size=16, protocol_id=schedule.plan.protocol_id)


def test_formal_horizon_and_separate_legacy_identity():
    assert NATIVE16_PROTOCOLS == (NATIVE16_PROTOCOL, NATIVE16_LONG_PROTOCOL)
    assert native16_exposure(6400)["presentations_per_site"] == 102400
    formal = plan(6400, synthetic=False)
    assert formal.learner("shared").horizon == 19200
    assert formal.learner("shared").horizon * formal.batch_size == 307200
    with pytest.raises(ValueError, match="q6400"):
        plan(4, synthetic=False)
    with pytest.raises(ValueError, match="q6400"):
        build_sharing_state(synthetic=False, learner="shared",
            ordered_view_identities=[str(i) for i in range(4)], completed_updates=0,
            source_valid_per_view=[32] * 4, target_valid_per_view=[24] * 4,
            padded_per_view=[48] * 4, batch_size=16, protocol_id=NATIVE16_LONG_PROTOCOL)
    with pytest.raises(ValueError, match="complete16"):
        replace(formal, views=tuple(replace(v, sequences=2) for v in formal.views))
    with pytest.raises(ValueError, match="cycling"):
        replace(formal, views=(formal.views[0],) * 6400)
    assert plan(4, synthetic=False, protocol=NATIVE16_PROTOCOL).q == 4


def test_long_cadence_has_only_sparse_trained_objectives():
    p = plan(6400, synthetic=False)
    assert p.learner("specialist_enc_l9").checkpoint_steps == (3200, 6400)
    assert p.learner("shared").checkpoint_steps == (6400, 12800, 19200)
    assert len(p.saved_states) == len(set(p.saved_states)) == 10
    assert p.condition_panels == {"task": 0, "objective": 18}
    assert p.additional_vanilla_panels == 0
    assert all(o.purpose == "objective" and o.conditions == ("no_noise",)
               and o.site in TRAINED_SITES and o.stage == "pre_freeze" for o in p.observations)
    assert sum(o.learner == "common_initialization" for o in p.observations) == 3
    legacy = plan(4, protocol=NATIVE16_PROTOCOL)
    assert len(legacy.saved_states) == 17
    assert legacy.condition_panels == {"task": 108, "objective": 192}
    assert legacy.additional_vanilla_panels == 1


@pytest.mark.parametrize("completed,expected", [
    (6400, {"enc_l9": 2134, "enc_l19": 2133, "enc_fn": 2133}),
    (12800, {"enc_l9": 4267, "enc_l19": 4267, "enc_fn": 4266}),
    (19200, dict.fromkeys(TRAINED_SITES, 6400)),
])
def test_exact_formal_remainders_and_token_exposure(completed, expected):
    p = plan(6400, synthetic=False)
    shared = p.learner("shared")
    sharing, stream = state(shared, completed)
    assert sharing["completed_per_site"] == expected
    assert stream["offset"] == completed * 16
    assert stream["next_site"] == (TRAINED_SITES[completed % 3] if completed < 19200 else None)
    for site, count in expected.items():
        assert sharing["source_valid_per_site"][site] == sum(32 + i for i in range(count))
        assert sharing["target_valid_per_site"][site] == sum(24 + i for i in range(count))
        assert sharing["padded_per_site"][site] == sum(48 + i for i in range(count))
    assert sum(sharing["source_valid_per_site"].values()) != stream["offset"]


def test_fixed_cyclic_stream_matches_each_specialist_view_and_noise():
    p = plan()
    shared = p.learner("shared")
    ledger = SharingExposureLedger(shared)
    for index in range(shared.horizon):
        update = shared.update_at(index)
        assert update.site == TRAINED_SITES[index % 3]
        assert update.site_local_index == index // 3
        specialist = p.learner("specialist_" + update.site).update_at(index // 3)
        assert (update.view, update.noise_key) == (specialist.view, specialist.noise_key)
        ledger.record(update, receipt=f"shared-{index}")
    totals = ledger.finalize()
    for site in TRAINED_SITES:
        own = SharingExposureLedger(p.learner("specialist_" + site))
        for index in range(p.q):
            own.record(own.schedule.update_at(index), receipt=f"{site}-{index}")
        assert totals[site] == own.finalize()[site]
    assert shared.update_at(shared.horizon - 1).lr_factor > 0
    with pytest.raises(ValueError, match="horizon"):
        shared.update_at(shared.horizon)
    legacy = plan(protocol=NATIVE16_PROTOCOL).learner("shared")
    assert tuple(legacy.update_at(i).site for i in range(6)) == (
        "enc_l9", "enc_l19", "enc_fn", "enc_l19", "enc_fn", "enc_l9")


def test_final_panels_are_terminal_and_all_no_noise_first():
    panels = long_final_panels(6400)
    assert len(panels) == 42
    tags = [f"final:{o.learner}:{o.site}:{o.step}:{o.conditions[0]}" for o in panels]
    assert len(set(tags)) == 42
    assert [o.conditions[0] for o in panels] == [c for c in ("no_noise", -6, 0, 6, 12, 18) for _ in range(7)]
    for o in panels:
        assert o.purpose == "final" and o.stage == "post_freeze"
        assert o.step == (19200 if o.learner == "shared" else 6400)
        assert o.site in (*TRAINED_SITES, HELDOUT_SITE)
        if o.site == HELDOUT_SITE:
            assert o.learner == "shared"
    assert tags[0] == "final:specialist_enc_l9:enc_l9:6400:no_noise"
    assert tags[6] == "final:shared:enc_l14:19200:no_noise"


def policy():
    return {name: {"site": asdict(ResolvedSite(
        "after_final_norm" if name == "enc_fn" else "after_layer",
        None if name == "enc_fn" else int(name[5:]), "tiny-revision", 26, 8,
        "encoder.norm" if name == "enc_fn" else f"encoder.layers.{name[5:]}")),
        "role": "heldout_after_freeze" if name == HELDOUT_SITE else "trained"}
        for name in (*TRAINED_SITES, HELDOUT_SITE)}


def test_empty_geometry_is_only_allowed_for_long_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr("jscc.sharing_state._verified_runs", lambda *a, **k: ["a" * 64])
    obligations = {"task": ["task-final"], "objective": ["objective-final"], "geometry": []}
    reference = create_study_freeze(tmp_path / "freeze.json",
        protocol_identity=NATIVE16_LONG_PROTOCOL, recipe_identity="recipe",
        architecture_decision="decision", site_policy=policy(), trained_runs={},
        obligation_plan=obligations, synthetic=True)
    assert verify_study_freeze(reference)["obligation_plan"] == obligations
    with pytest.raises(ValueError, match="empty freeze obligation"):
        create_study_freeze(tmp_path / "legacy.json",
            protocol_identity=NATIVE16_PROTOCOL, recipe_identity="recipe",
            architecture_decision="decision", site_policy=policy(), trained_runs={},
            obligation_plan=obligations, synthetic=True)
    with pytest.raises(ValueError, match="empty freeze obligation"):
        create_study_freeze(tmp_path / "no-task.json",
            protocol_identity=NATIVE16_LONG_PROTOCOL, recipe_identity="recipe",
            architecture_decision="decision", site_policy=policy(), trained_runs={},
            obligation_plan={**obligations, "task": []}, synthetic=True)


@pytest.mark.parametrize("learner,completed", [("shared", 5), ("specialist_enc_l19", 4)])
def test_long_checkpoint_metadata_and_legacy_rotation_drift(learner, completed):
    sharing, stream = state(plan().learner(learner), completed)
    sites = policy() if learner == "shared" else {"enc_l19": policy()["enc_l19"]}
    metadata = {"completed_updates": completed, "phase": "both", "sharing": sharing,
                "evaluation_sites": sites, "protocol_identity": NATIVE16_LONG_PROTOCOL}
    if learner != "shared":
        metadata["site"] = sites["enc_l19"]["site"]
    payload = {"schema": "experiment-state-v2",
               "kind": "shared_encoder_v2" if learner == "shared" else "single_site_v2",
               "metadata": metadata, "stream": stream, "scheduler": {"last_epoch": completed}}
    validate_sharing_state(payload)
    if learner == "shared":
        assert sharing["completed_per_site"] == {"enc_l9": 2, "enc_l19": 2, "enc_fn": 1}
        sharing["completed_per_site"] = {"enc_l9": 1, "enc_l19": 2, "enc_fn": 2}
        with pytest.raises(ValueError, match="counters/order"):
            validate_sharing_state(payload)
