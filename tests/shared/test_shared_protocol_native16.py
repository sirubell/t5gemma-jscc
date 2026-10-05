"""Current primitive assertions extracted without resurrecting legacy execution."""
from jscc.presentation import PresentationSampler


def test_complete_pool_crosses_epochs_without_subset_cycle():
    pool = list(range(1000))
    stream = PresentationSampler(pool, 102400, seed=20260920).actual_ids.tolist()
    assert len(set(stream[:1000])) == 1000
    assert len(set(stream[1000:2000])) == 1000
    assert stream[:3200] != stream[3200:6400]
