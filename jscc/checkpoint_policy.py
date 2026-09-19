"""Optional exact optimizer-step guard for fixed-budget evaluation."""


def validate_checkpoint_step(state, expected_step):
    if expected_step is None:
        return
    if type(expected_step) is not int or expected_step < 1:
        raise ValueError("expected step must be a positive integer")
    actual_step = state.get("step")
    if type(actual_step) is not int or actual_step != expected_step:
        raise ValueError(f"Checkpoint step must be {expected_step}; found {actual_step!r}")
