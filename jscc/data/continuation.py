"""Checkpoint the consumed ordinary epoch without replaying tokenization."""
from collections.abc import Sized
from typing import Any, cast

from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler


class OrdinaryCursor:
    policy = "ordinary-epoch-batches-v1"

    def __init__(self, loader: DataLoader, state=None):
        if (loader.num_workers != 0 or loader.generator is not None
                or not isinstance(loader.batch_sampler, BatchSampler)
                or type(loader.sampler) not in (RandomSampler, SequentialSampler)
                or loader.drop_last or loader.persistent_workers):
            raise ValueError("exact continuation requires an ordinary zero-worker, non-dropping loader")
        if isinstance(loader.sampler, RandomSampler) and (
                loader.sampler.replacement or loader.sampler.num_samples != len(cast(Sized, loader.dataset))
                or loader.sampler.generator is not None):
            raise ValueError("exact continuation requires the ordinary epoch sampler")
        self.original = loader.batch_sampler
        self.rows = len(cast(Sized, loader.dataset))
        self.batch_size = cast(int, loader.batch_size)
        if not self.rows:
            raise ValueError("exact continuation requires nonempty training data")
        self.cycle = 0
        self.batches: list[list[int]] = []
        self.position = 0
        self.restoring = state is not None
        if state is not None:
            self._restore(state)
        self.loader = DataLoader(loader.dataset, batch_sampler=self, collate_fn=loader.collate_fn,
                                 num_workers=0, pin_memory=loader.pin_memory)

    def _restore(self, state):
        if (not isinstance(state, dict) or state.get("policy") != self.policy
                or state.get("rows") != self.rows or state.get("batch_size") != self.batch_size):
            raise ValueError("incompatible ordinary data cursor")
        batches = state.get("batches")
        position, cycle = state.get("next_batch"), state.get("cycle")
        if (not isinstance(batches, list) or not batches or type(position) is not int
                or not 0 <= position <= len(batches) or type(cycle) is not int or cycle < 0
                or any(not isinstance(batch, list) or not batch
                       or len(batch) != min(self.batch_size, self.rows - index * self.batch_size)
                       or any(type(row) is not int for row in batch)
                       for index, batch in enumerate(batches))
                or sorted(row for batch in batches for row in batch) != list(range(self.rows))):
            raise ValueError("invalid ordinary data cursor position/order")
        self.batches, self.position, self.cycle = batches, position, cycle

    def __iter__(self):
        if self.restoring:
            self.restoring = False
        else:
            if self.batches:
                self.cycle += 1
            # DataLoader draws its base seed before this call. The original
            # sampler then draws its ordinary shuffle seed, preserving fresh order.
            self.batches = list(self.original)
            self.position = 0
        yield from self.batches[self.position:]

    def __len__(self):
        return len(self.original)

    def consumed(self):
        # Advance after next(loader), never in the sampler's yield path.
        self.position += 1

    def state_dict(self) -> dict[str, Any]:
        return {"policy": self.policy, "rows": self.rows, "batch_size": self.batch_size,
                "cycle": self.cycle, "batches": self.batches, "next_batch": self.position}


def cursor_support(loader):
    if not isinstance(loader, DataLoader):
        return "exact continuation requires an ordinary map-style DataLoader"
    try:
        OrdinaryCursor(loader)
    except ValueError as error:
        return str(error)
    return None
