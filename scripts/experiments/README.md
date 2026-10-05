# Historical experiment modules

The retained evening helpers and main-weight screen modules support interpretation of historical evidence. Importing this package does not schedule or start an experiment. The historical main-weight launch graph is retired; old allocations and approvals do not authorize another run.

The evening E5 and suite drivers are retired from current source. The retained `evening_eval`, `evening_gradients` and `evening_pilot` modules preserve scoring/replay/intervention, feature/VJP/selection and codec-family initialization/configuration helpers. Their historical executors are unsupported: module invocation exits with an explicit retirement error before imports, model/data loading or output creation. Import the helpers through the retained module paths below.

| Former module | Retained module or disposition |
|---|---|
| `scripts.evening_e5_driver` | Retired historical driver; use its recorded source package to interpret old commands. |
| `scripts.evening_eval` | `scripts.experiments.evening_eval` (imported helpers only) |
| `scripts.evening_gradients` | `scripts.experiments.evening_gradients` (imported helpers only) |
| `scripts.evening_pilot` | `scripts.experiments.evening_pilot` (imported helpers only) |
| `scripts.evening_suite_driver` | Retired historical driver; use its recorded source package to interpret old commands. |
| `scripts.main_weight_screen_driver` | Retired historical driver; use its recorded source package to interpret old commands. |
| `scripts.main_weight_screen_eval` | `scripts.experiments.main_weight_screen_eval` |
| `scripts.main_weight_screen_gradients` | `scripts.experiments.main_weight_screen_gradients` (imported pooled probe/algebra helpers only) |
| `scripts.main_weight_screen_train` | `scripts.experiments.main_weight_screen_train` (imported configuration/order/instrumented training test helpers only) |

The main-weight gradient and training executors are unsupported: direct script or module invocation exits with an explicit retirement error before dependency imports, model/data loading or output creation. Their retained providers preserve pooled probe/algebra, configuration/order and the discarded two-update finite-gradient/nonzero-update/FP32Adam training test seam. The separately retained evaluator accepts only `--route dec_l20` and `--step 500` or `--step 1000`; authorized evaluation from the repository root uses `uv run --locked python -m scripts.experiments.main_weight_screen_eval ...`. Scoring, replay, payload and budget policies are unchanged.

Retained main-weight modules use repository-root-relative paths. The portable regression inputs for selection and recipe helpers are in [`tests/fixtures/main_weight_screen/`](../../tests/fixtures/main_weight_screen/README.md). Original experiment inputs, commands, source hashes and receipts remain with the MTK archive outside this Git project; test fixtures are intentionally smaller or path-sanitized copies. Current source does not promise historical reproduction. Future execution needs fresh source bindings; saved hashes and held scientific sources retain their original identities.
