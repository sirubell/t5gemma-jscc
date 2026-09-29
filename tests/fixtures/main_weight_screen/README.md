# Main-weight screen regression fixtures

These repository-local inputs make the main-weight screen's CPU regression tests portable. They are test inputs, not complete historical experiment receipts or a recipe to start another run.

| File | Preserved test contract |
|---|---|
| `base-dec_l20.yaml` | Frozen base recipe; only the machine-specific `run.output_dir` was changed to `./runs/fixture-only`. |
| `fixture.json` | E3 selection seed and the exact ordered 128 `row_id` values. Prompt and token payloads are omitted because the selection-order regression does not read them. |
| `saved-data-ids.json` | Complete saved ID membership: 39,393 training rows and 512 validation rows, plus selection metadata. The ordering and disjointness checks need the full lists. |

`tests/test_main_weight_screen.py` uses these files to check the frozen selection order and screen configuration without private local paths. Byte-identical copies of the three original inputs are retained in the owner's MTK archive outside this repository; the local cleanup receipt records original and test-fixture hashes. The compact `fixture.json` and sanitized YAML deliberately have different hashes from their originals.
