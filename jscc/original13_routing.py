"""Routing-owned M1 API. Stubs deliberately reject original13 execution.

The consumer supplies the actual SplitModel routes, role objectives and weight
aliases. Training/state orchestration stays in encoder_stages.py. The learner
must reuse established bound-batch, numerical update and exposure mechanics.
"""


def create_original13_learner(model, *, plan, metadata, event_sink):
    """Return a compatible learner with one canonical parameter list.

    Required fields: model, parameters, optimizer, scheduler, scaler, exposure,
    completed, attempted, offset, valid_tokens, phase_start_step, failed.
    update([batch], batch_identities=[binding]) consumes plan.update_at(completed),
    uses plan.objective_at(completed), advances scheduler once and returns the
    persisted update event with actual lr_used. It must increment completed and
    exposure exactly once, and stop on nonfinite/skipped updates. Reset and
    scheduler construction belong to the integration owner.
    """
    raise NotImplementedError('original13 routing learner not implemented')


def canonical_codec(model):
    """Return the single nn.Module whose names match canonical weight keys."""
    raise NotImplementedError('original13 canonical codec not implemented')


def export_codec(model):
    """Return flat codec-only state_dict; no role duplicates or backbone tensors."""
    raise NotImplementedError('original13 canonical export not implemented')


def load_codec(model, weights):
    """Strictly load export_codec's body and verify all role aliases."""
    raise NotImplementedError('original13 canonical loader not implemented')


def observation_state(model):
    """Return comparable detached route/noise/mode/counter state for restoration.

    Include active split, hook ownership, bypass/training flags, role masks,
    noise namespaces/generator state and channel/payload counters. No backbone
    tensor copy is required. The observation consumer restores these exactly.
    """
    raise NotImplementedError('original13 observation state not implemented')


def restore_runtime_state(model, state):
    """Restore exported idle route/noise/counter state, without learned tensors.

    observation_state must be serializable data and contain no Python hook
    handles/callables or duplicate codec/backbone tensors. Reject active routes.
    """
    raise NotImplementedError('original13 runtime-state restore not implemented')
