"""Strict sharing continuation metadata and inventory-backed held-out freeze.

These validators complement experiment_state's tensor/optimizer/RNG envelope;
sharing never changes the ordinary single-site state contract.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from .activation_replay import file_digest
from .sharing_schedule import DEPTH_PROTOCOL
from .experiment_records import canonical_bytes, completion_status, read_events, verify_inventory

SITES = ("enc_l9", "enc_l19", "enc_fn")
LEARNERS = ("shared", *(f"specialist_{site}" for site in SITES))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _count(value, label):
    _require(type(value) is int and value >= 0, f"invalid {label}")


def _updates(learner, q):
    if learner == "shared":
        return [SITES[(sweep + position) % 3] for sweep in range(q) for position in range(3)]
    _require(learner in LEARNERS, "unknown sharing learner")
    return [learner.removeprefix("specialist_")] * q


def build_sharing_state(*, synthetic, learner, ordered_view_identities,
                        completed_updates, source_valid_per_view,
                        target_valid_per_view, padded_per_view, batch_size, protocol_id=None):
    """Build deterministic stream counters from complete effective-batch counts.

The per-view vectors contain one aggregate token count per effective batch,
not individual examples. Their full ordered identities are frozen at step zero.
"""
    _require(type(synthetic) is bool, "synthetic must be explicit boolean")
    for value in (ordered_view_identities, source_valid_per_view, target_valid_per_view, padded_per_view):
        _require(isinstance(value, (list, tuple)), "sharing views/counts must be ordered sequences")
    identities = list(ordered_view_identities)
    q = len(identities) if synthetic else (400 if protocol_id == DEPTH_PROTOCOL else 200)
    if not synthetic and q == 400:
        _require(len(set(identities)) == 400, "q400 state requires400 distinct views")
    _require(q >= 4 and q % 4 == 0, "invalid sharing batch horizon")
    _require(type(batch_size) is int and batch_size in ((2, 64) if synthetic else (64,)), "sharing batch size mismatch")
    order = _updates(learner, q)
    _count(completed_updates, "completed updates")
    _require(completed_updates <= len(order), "sharing horizon exceeded")
    _require(len(identities) == q and all(isinstance(v, str) and v for v in identities)
             , "missing ordered views")
    vectors = {"source_valid_per_view": list(source_valid_per_view),
               "target_valid_per_view": list(target_valid_per_view),
               "padded_per_view": list(padded_per_view)}
    for name, values in vectors.items():
        _require(len(values) == q, f"wrong {name} length")
        for value in values:
            _count(value, name)
            _require(value > 0, f"empty {name}")
    _require(all(p >= s for p, s in zip(vectors["padded_per_view"], vectors["source_valid_per_view"])),
             "padded/source count mismatch")
    counts = {site: order[:completed_updates].count(site) for site in SITES}
    sharing = {"schema": "sharing-state-v1", "synthetic": synthetic, "q": q,
               "batch_size": batch_size, "learner": learner, "trained_sites": list(SITES),
               "horizon": len(order), "ordered_view_identities": identities,
               **vectors, "completed_per_site": counts,
               "source_valid_per_site": {s: sum(vectors["source_valid_per_view"][:n]) for s, n in counts.items()},
               "target_valid_per_site": {s: sum(vectors["target_valid_per_view"][:n]) for s, n in counts.items()},
               "padded_per_site": {s: sum(vectors["padded_per_view"][:n]) for s, n in counts.items()},
               "next_site": order[completed_updates] if completed_updates < len(order) else None}
    if not synthetic and q == 400:
        sharing["protocol_id"] = DEPTH_PROTOCOL
    stream = {"completed_updates": completed_updates, "offset": completed_updates * batch_size,
              "view_offsets": dict(counts), "next_site": sharing["next_site"]}
    return sharing, stream


def _site_policy(policy, *, learner="shared"):
    expected = {site: "trained" for site in SITES} if learner == "shared" else {
        learner.removeprefix("specialist_"): "trained"}
    if learner == "shared":
        expected["enc_l14"] = "heldout_after_freeze"
    _require(isinstance(policy, dict) and set(policy) == set(expected), "evaluation sites differ from sharing policy")
    topologies = set()
    for name, role in expected.items():
        declaration = policy[name]
        _require(isinstance(declaration, dict) and set(declaration) == {"site", "role"}
                 and declaration["role"] == role, "invalid sharing site role")
        site = declaration["site"]
        _require(isinstance(site, dict),
                 "evaluation site identity mismatch")
        # The routing resolver verifies full topology against the actual backbone.
        from .models.split_model import ResolvedSite
        try:
            resolved = ResolvedSite(**site)
        except (TypeError, ValueError) as error:
            raise ValueError("invalid resolved sharing site") from error
        _require(resolved.site_id == name and resolved.schema == "encoder-site-v2",
                 "sharing site identity/schema mismatch")
        _require(type(resolved.layer_count) is int and resolved.layer_count > 19
                 and type(resolved.hidden_dim) is int and resolved.hidden_dim > 0
                 and isinstance(resolved.model_revision, str) and bool(resolved.model_revision),
                 "invalid sharing topology")
        _require((name == "enc_fn" and resolved.where == "after_final_norm" and resolved.index is None)
                 or (name != "enc_fn" and resolved.where == "after_layer"
                     and type(resolved.index) is int and resolved.index == int(name[5:])),
                 "invalid sharing boundary")
        expected_path = "encoder.norm" if name == "enc_fn" else f"encoder.layers.{resolved.index}"
        _require(resolved.module_path == expected_path, "sharing module path mismatch")
        topologies.add((resolved.model_revision, resolved.layer_count, resolved.hidden_dim))
    _require(len(topologies) == 1, "sharing sites have inconsistent backbone topology")


def validate_sharing_state(payload):
    """Reject drift in sharing kind, phase, schedule, views and exposure."""
    _require(isinstance(payload, dict) and payload.get("schema") == "experiment-state-v2",
             "invalid sharing state envelope")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("sharing"), dict):
        raise ValueError("missing sharing metadata")
    sharing = metadata["sharing"]
    fields = ("synthetic", "learner", "ordered_view_identities", "source_valid_per_view",
              "target_valid_per_view", "padded_per_view", "batch_size")
    _require(all(field in sharing for field in fields), "incomplete sharing metadata")
    count = metadata.get("completed_updates")
    expected, stream = build_sharing_state(completed_updates=count,
        protocol_id=metadata.get("protocol_identity"), **{k: sharing[k] for k in fields})
    _require(canonical_bytes(sharing) == canonical_bytes(expected), "sharing counters/order/horizon mismatch")
    kind = "shared_encoder_v2" if sharing["learner"] == "shared" else "single_site_v2"
    _require(payload.get("kind") == kind, "sharing learner/kind mismatch")
    _require(metadata.get("phase") == "both", "sharing requires simultaneous objective phase")
    _require(canonical_bytes(payload.get("stream")) == canonical_bytes(stream), "sharing stream/view offsets mismatch")
    _require(isinstance(payload.get("scheduler"), dict) and type(payload["scheduler"].get("last_epoch")) is int
             and payload["scheduler"]["last_epoch"] == count,
             "sharing scheduler/global update mismatch")
    _site_policy(metadata.get("evaluation_sites"), learner=sharing["learner"])
    if kind == "single_site_v2":
        site = sharing["learner"].removeprefix("specialist_")
        _require(metadata.get("site") == metadata["evaluation_sites"][site]["site"], "specialist saved site mismatch")


def _reference(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_digest(path)}


def _read_reference(reference):
    _require(isinstance(reference, dict) and set(reference) == {"path", "sha256"}, "invalid freeze file reference")
    path = Path(reference["path"])
    _require(path.is_file() and file_digest(path) == reference["sha256"], "freeze evidence bytes changed")
    return path


def _verified_runs(trained_runs, *, synthetic, protocol_identity, site_policy, recipe_identity, architecture_decision):
    """Recompute completion from retained events and inspect checkpoint bytes."""
    import torch
    from .experiment_state import _validate

    _require(isinstance(trained_runs, dict) and set(trained_runs) == set(LEARNERS), "freeze requires all four learners")
    checkpoint_hashes = []
    common = None
    for learner, run in trained_runs.items():
        _require(isinstance(run, dict) and set(run) == {"root", "manifest", "inventory", "events"}, "invalid trained run references")
        root = Path(run["root"]).resolve()
        paths = {key: _read_reference(run[key]) for key in ("manifest", "inventory", "events")}
        _require(all(path.resolve().is_relative_to(root) for path in paths.values()), "run references escape root")
        manifest = json.loads(paths["manifest"].read_text())
        inventory = json.loads(paths["inventory"].read_text())
        verify_inventory(root, inventory)
        _require(inventory["mode"] == "tensor-complete", "freeze requires retained checkpoint bytes")
        phases = manifest.get("phases")
        _require(isinstance(phases, list) and len(phases) == 1, "freeze requires one simultaneous phase")
        horizon = phases[0].get("updates")
        _count(horizon, "freeze horizon")
        factor = 3 if learner == "shared" else 1
        _require(horizon > 0 and horizon % (4 * factor) == 0
                 and (synthetic or horizon == (400 if protocol_identity == DEPTH_PROTOCOL else 200) * factor), "freeze horizon mismatch")
        _require(manifest["phases"] == [{"phase_id": "combined", "updates": horizon,
                    "start_step": 0, "start_valid_tokens": 0}], "freeze trained horizon mismatch")
        events = read_events(paths["events"])
        status = completion_status(manifest, events, inventory, root)
        _require(status["status"] == "complete" and status["tensor_bytes_verified"], "trained run incomplete")
        trained_sites = SITES if learner == "shared" else (learner.removeprefix("specialist_"),)
        task_steps = [horizon // 2, horizon]
        objective_steps = [horizon * i // 4 for i in range(1, 5)]
        if learner == "shared":
            task_steps.insert(0, 0)
            objective_steps.insert(0, 0)
        expected_panels = {(purpose, site, step) for purpose, points in
                           (("task", task_steps), ("objective", objective_steps))
                           for site in trained_sites for step in points}
        observed_panels = []
        observed_ids = []
        for event in events:
            if event["event_type"] != "observation":
                continue
            observation = event["payload"]
            request = observation["request"]
            _require(observation["status"] == "complete" and request["role"] == "trained"
                     and request["task"] == "hellaswag" and request["conditions"] == ["no_noise", -6, 0, 6, 12, 18],
                     "freeze requires complete trained-site observations")
            observed_panels.append((request["purpose"], request["site"], request["step"]))
            observed_ids.append(observation["identity"])
        _require(set(observed_panels) == expected_panels and len(observed_panels) == len(expected_panels)
                 and set(manifest["expected_observations"]) == set(observed_ids),
                 "freeze trained observation cadence mismatch")
        steps = []
        checkpoints_by_step = {}
        checkpoint_sharing = None
        for artifact in inventory["artifacts"]:
            if artifact["kind"] != "tensor":
                continue
            payload = torch.load(root / artifact["path"], map_location="cpu", weights_only=False)
            _validate(payload)
            validate_sharing_state(payload)
            metadata = payload["metadata"]
            sharing = metadata["sharing"]
            checkpoint_sharing = sharing
            _require(sharing["learner"] == learner and sharing["synthetic"] is synthetic and sharing["horizon"] == horizon,
                     "freeze checkpoint learner mismatch")
            _require(metadata["protocol_identity"] == protocol_identity, "freeze checkpoint protocol mismatch")
            _require(metadata.get("recipe_identity") == recipe_identity
                     and metadata.get("architecture_decision") == architecture_decision,
                     "freeze checkpoint recipe/architecture decision mismatch")
            expected_policy = site_policy if learner == "shared" else {
                learner.removeprefix("specialist_"): site_policy[learner.removeprefix("specialist_")]}
            _require(metadata["evaluation_sites"] == expected_policy, "freeze checkpoint site policy mismatch")
            paired = {key: sharing[key] for key in ("q", "ordered_view_identities", "source_valid_per_view",
                      "target_valid_per_view", "padded_per_view", "batch_size")}
            paired["initialization_identity"] = metadata["initialization_identity"]
            paired["source_identity"] = metadata["source_identity"]
            if common is None:
                common = paired
            _require(paired == common, "freeze learners lack matched initialization/stream/source")
            steps.append(metadata["completed_updates"])
            checkpoints_by_step[metadata["completed_updates"]] = artifact["sha256"]
            checkpoint_hashes.append(artifact["sha256"])
        expected_steps = [horizon * i // 4 for i in range(1, 5)]
        if learner == "shared":
            expected_steps.insert(0, 0)
        _require(sorted(steps) == expected_steps, "missing/duplicate report checkpoint cadence")
        if checkpoint_sharing is None:
            raise ValueError("freeze requires sharing checkpoints")
        local_counts: dict[str, int] = dict.fromkeys(SITES, 0)
        expected_order = _updates(learner, checkpoint_sharing["q"])
        updates = [event["payload"] for event in events if event["event_type"] == "update"]
        _require(len(updates) == len(expected_order), "freeze update count mismatch")
        for row, site in zip(updates, expected_order):
            local = local_counts[site]
            _require(row["site_id"] == site and row["site_step"] == local + 1
                     and row["sweep"] == local + 1 and row["objective"]["kind"] == "combined",
                     "freeze update site/order/objective mismatch")
            exposure = row["exposure"]
            _require(exposure["sequences"] == checkpoint_sharing["batch_size"]
                     and exposure["source_tokens"] == checkpoint_sharing["source_valid_per_view"][local]
                     and exposure["target_tokens"] == checkpoint_sharing["target_valid_per_view"][local]
                     and exposure["padded_tokens"] == checkpoint_sharing["padded_per_view"][local],
                     "freeze update exposure differs from saved stream")
            local_counts[site] += 1
        for event in events:
            if event["event_type"] == "observation":
                request = event["payload"]["request"]
                _require(request["checkpoint"] == checkpoints_by_step.get(request["step"])
                         and request["protocol"] == protocol_identity,
                         "freeze observation checkpoint/protocol mismatch")
                _require(request["purpose"] != "objective" or request["objective_kind"] == "combined",
                         "freeze requires simultaneous objective observations")
    _require(len(checkpoint_hashes) == len(set(checkpoint_hashes)), "duplicate checkpoint bytes across learners")
    return sorted(checkpoint_hashes)


def create_study_freeze(path, *, protocol_identity, recipe_identity, architecture_decision,
                        site_policy, trained_runs, obligation_plan, synthetic=False):
    """Publish an immutable freeze only after trained-only manifests complete.

Each run supplies a root and hash-bound manifest/inventory/events references.
The separate obligation plan includes the still-pending held-out task, objective
and geometry work; these pending obligations cannot make trained runs circular.
"""
    for value in (protocol_identity, recipe_identity, architecture_decision):
        _require(isinstance(value, str) and bool(value.strip()), "freeze identities must be explicit")
    _require(type(synthetic) is bool, "synthetic freeze must be explicit")
    _site_policy(site_policy)
    _require(isinstance(obligation_plan, dict) and set(obligation_plan) == {"task", "objective", "geometry"},
             "freeze must bind task/objective/geometry obligations")
    panels = []
    for values in obligation_plan.values():
        _require(isinstance(values, list) and bool(values) and all(isinstance(v, str) and v for v in values),
                 "empty freeze obligation plan")
        panels.extend(values)
    _require(len(panels) == len(set(panels)), "duplicate freeze obligations")
    checkpoints = _verified_runs(trained_runs, synthetic=synthetic,
                                protocol_identity=protocol_identity, site_policy=site_policy,
                                recipe_identity=recipe_identity, architecture_decision=architecture_decision)
    receipt = {"schema": "study-freeze-v1", "status": "complete", "synthetic": synthetic,
               "protocol_identity": protocol_identity, "recipe_identity": recipe_identity,
               "architecture_decision": architecture_decision, "site_policy": site_policy,
               "trained_run_inventory": trained_runs, "checkpoint_hashes": checkpoints,
               "observation_plan": panels, "obligation_plan": obligation_plan}
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".freeze-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(canonical_bytes(receipt) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return _reference(path)


def verify_study_freeze(reference, *, checkpoint_sha256=None, protocol_identity=None,
                        site_policy=None, obligation=None):
    """Reverify retained evidence before any real held-out work, including capture."""
    receipt = json.loads(_read_reference(reference).read_text())
    _require(set(receipt) == {"schema", "status", "synthetic", "protocol_identity", "recipe_identity",
             "architecture_decision", "site_policy", "trained_run_inventory", "checkpoint_hashes",
             "observation_plan", "obligation_plan"}, "invalid freeze receipt fields")
    _require(receipt["schema"] == "study-freeze-v1" and receipt["status"] == "complete", "incomplete freeze")
    _site_policy(receipt["site_policy"])
    for key in ("protocol_identity", "recipe_identity", "architecture_decision"):
        _require(isinstance(receipt[key], str) and bool(receipt[key].strip()), "missing freeze identity")
    _require(type(receipt["synthetic"]) is bool, "invalid freeze synthetic flag")
    plan = receipt["obligation_plan"]
    _require(isinstance(plan, dict) and set(plan) == {"task", "objective", "geometry"}
             and all(isinstance(v, list) and v and all(isinstance(p, str) and p for p in v) for v in plan.values()),
             "invalid freeze obligations")
    panels = [panel for values in plan.values() for panel in values]
    _require(len(set(panels)) == len(panels) and set(panels) == set(receipt["observation_plan"]), "freeze obligation mismatch")
    checkpoints = _verified_runs(receipt["trained_run_inventory"], synthetic=receipt["synthetic"],
                    protocol_identity=receipt["protocol_identity"], site_policy=receipt["site_policy"],
                    recipe_identity=receipt["recipe_identity"], architecture_decision=receipt["architecture_decision"])
    _require(checkpoints == receipt["checkpoint_hashes"], "freeze checkpoint inventory mismatch")
    if checkpoint_sha256 is not None:
        _require(checkpoint_sha256 in checkpoints, "checkpoint absent from freeze")
    if protocol_identity is not None:
        _require(protocol_identity == receipt["protocol_identity"], "freeze protocol mismatch")
    if site_policy is not None:
        _require(site_policy == receipt["site_policy"], "freeze site policy mismatch")
    if obligation is not None:
        _require(obligation in receipt["observation_plan"], "undeclared post-freeze obligation")
    return receipt
