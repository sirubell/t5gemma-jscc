# Historical experiment modules

These nine modules reproduce the evening diagnostics and main-weight screen. They are retained with their tests as source for interpreting historical evidence. Importing this package does not schedule or start an experiment. The drivers can start subprocesses when explicitly invoked; do not treat an old allocation or approval record as permission to rerun one.

From the repository root, invoke a moved module as `uv run --locked python -m scripts.experiments.<module> ...`. The former `scripts.<module>` import and `python -m` paths have changed as follows:

| Former module | Current module |
|---|---|
| `scripts.evening_e5_driver` | `scripts.experiments.evening_e5_driver` |
| `scripts.evening_eval` | `scripts.experiments.evening_eval` |
| `scripts.evening_gradients` | `scripts.experiments.evening_gradients` |
| `scripts.evening_pilot` | `scripts.experiments.evening_pilot` |
| `scripts.evening_suite_driver` | `scripts.experiments.evening_suite_driver` |
| `scripts.main_weight_screen_driver` | `scripts.experiments.main_weight_screen_driver` |
| `scripts.main_weight_screen_eval` | `scripts.experiments.main_weight_screen_eval` |
| `scripts.main_weight_screen_gradients` | `scripts.experiments.main_weight_screen_gradients` |
| `scripts.main_weight_screen_train` | `scripts.experiments.main_weight_screen_train` |

These modules use repository-root-relative paths after relocation. The portable regression inputs for the main-weight selection and recipe are in [`tests/fixtures/main_weight_screen/`](../../tests/fixtures/main_weight_screen/README.md). Original experiment inputs and receipts remain with the MTK archive outside this Git project; the test fixtures are intentionally smaller or path-sanitized copies.
