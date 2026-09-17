# Reserve HellaSwag validation for final evaluation

New HellaSwag experiments reserve a deterministic subset of the official training split for checkpoint selection and exclude those rows from optimization. This gives up some optimization examples in exchange for keeping the full official validation split separate from checkpoint selection, with identical data IDs across split-location variants. Historical runs that selected on official validation retain their original protocol labels; changing new defaults does not reinterpret their results.
