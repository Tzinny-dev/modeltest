# Core concepts

Everything in `modeltest` revolves around four pieces:

```text
ModelSuite ──run(model, X_val, y_val)──▶ TestContext ──▶ [ModelTest, ...] ──▶ SuiteResult
```

1. A **`ModelSuite`** holds a list of tests and knows how to run them against
   a model plus validation data.
2. Running a suite builds a **`TestContext`** — everything a test may need:
   the model, `X_val` / `y_val`, optional train data, metadata, and a shared
   prediction cache.
3. Each **`ModelTest`** receives the context and either passes (returns
   normally after its `assert`s) or fails (raises `AssertionError`). An
   unexpected exception marks the test as **ERROR**, not FAILED.
4. The suite collects every test's `TestResult` into a **`SuiteResult`**,
   which aggregates pass/fail counts and renders reports.

## Test statuses

| Status | Meaning |
| ------ | ------- |
| `PASSED` | The test returned normally — every `assert` held. |
| `FAILED` | An `AssertionError` was raised — the contract is violated. |
| `ERROR`  | Something else went wrong (bad config, missing column...). |
| `SKIPPED`| The test decided not to run (returned a `TestResult` with this status). |

Only `PASSED` counts towards the suite passing: `SuiteResult.passed` is
`True` when every result passed.

## Prediction caching

Within a single `suite.run(...)`, predictions are computed **once and
reused** across every test. `TestContext.predict()` caches by a content hash
of the input, so tests predicting on the *same* data (accuracy, group
performance, the fairness tests...) each reuse the result instead of
re-running the model.

Perturbed inputs get their own cache entry, so the robustness test's noisy
copy never collides with the clean data — caching never compromises
correctness.

Disable it if you need a fresh prediction on every call:

```python
from modeltest import TestContext

ctx = TestContext(model=model, X_val=X_val, y_val=y_val, cache_predictions=False)
```

## Feature-name filtering

If the model was fit with named features (`feature_names_in_`, true for
sklearn estimators and `Pipeline`s), `TestContext.predict()` drops any extra
columns from `X_val` that the model does not expect, in the model's own
order. This lets your validation frame carry helper columns (a group column,
timestamps...) without breaking the model call.

## Reports

`SuiteResult.report()` (or `result.report(style=...)`) renders three ways:

```python
result.report(style="table")   # console table (default)
result.report(style="json")    # machine-readable JSON
result.report(style="junit")   # JUnit XML for CI test reporters
```

The JUnit output wraps everything in a standard `<testsuites>` element, which
is what GitHub Actions, GitLab and Jenkins reporters expect. Exit codes:
`modeltest validate` returns `0` when the suite passes and `1` when any test
fails, so plain `if` semantics work in shell scripts and CI.

## Metadata

Any extra keyword arguments to `suite.run(...)` land in
`ctx.metadata` — `model_name="fraud_rf"` shows up in reports and MLflow.

```python
result = suite.run(model, X_val, y_val, model_name="fraud_rf", version="2.1")
```
