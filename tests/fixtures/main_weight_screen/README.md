# Main-weight screen regression fixtures

These repository-local inputs make the main-weight screen's CPU regression tests portable. They are test inputs, not complete historical experiment receipts or a recipe to start another run.

| File | Preserved test contract |
|---|---|
| `base-dec_l20.yaml` | Frozen base recipe; only the machine-specific `run.output_dir` was changed to `./runs/fixture-only`. |
| `fixture.json` | E3 selection seed and the exact ordered 128 `row_id` values. Prompt and token payloads are omitted because the selection-order regression does not read them. |
| `saved-data-ids.json` | Complete saved ID membership: 39,393 training rows and 512 validation rows, plus selection metadata. The ordering and disjointness checks need the full lists. |

`tests/test_main_weight_screen.py` uses these files to check the frozen selection order and screen configuration without private local paths. Byte-identical copies of the three original inputs are retained in the owner's MTK archive outside this repository; the local cleanup receipt records original and test-fixture hashes. The compact `fixture.json` and sanitized YAML deliberately have different hashes from their originals.

The historical script helper directory has retired. Regression imports now follow the actual body owners:

| Preserved helpers | Current provider |
|---|---|
| Score/intervention/replay | `jscc/diagnostic_scoring.py` |
| Selection IDs/order, features, VJPs and pooled gradient algebra | `jscc/gradient_diagnostics.py` |
| Codec family constructors and role-seeded initialization | `jscc/models/codec.py` |
| Frozen screen/pilot configuration and finite-update instrumentation | `tests/model_helpers.py` |
| Strict saved checkpoint evaluation | `scripts/evaluate_historical_weight_screen.py` |

This is a source migration, not a reproduction receipt. Use the matching historical source/runtime for original scientific artifacts; fixture portability does not revise their hashes or authorize execution.
