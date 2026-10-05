"""Native-policy row identity and gradient guards.

The historical native-v2 preflight training executor is retired. The native
policy trainer retains these helpers at their existing import path.
"""

import torch
from torch.utils.data import Dataset


class IndexedRows(Dataset):
    def __init__(self, dataset, row_ids):
        self.dataset, self.row_ids = dataset, row_ids

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index], self.row_ids[index]


class IndexedCollator:
    def __init__(self, collator):
        self.collator = collator

    def __call__(self, rows):
        batch = self.collator([row for row, _ in rows])
        batch["preflight_row_ids"] = [row_id for _, row_id in rows]
        return batch


def checked_gradient_names(model):
    names = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise RuntimeError(f"missing trainable gradient: {name}")
        if not torch.isfinite(parameter.grad).all().item():
            raise FloatingPointError(f"nonfinite trainable gradient: {name}")
        names.append(name)
    if not names:
        raise RuntimeError("no trainable gradients")
    return names


if __name__ == "__main__":
    raise SystemExit("Native preflight training is retired; use train.py with an explicit --output.")
