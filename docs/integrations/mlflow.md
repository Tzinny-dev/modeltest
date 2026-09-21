# MLflow

Log a finished validation run into [MLflow](https://mlflow.org/) as an
experiment run — one param per test, one metric per numeric value, and the
full JSON report saved as an artifact.

```bash
pip install modeltest[mlflow]
```

## Basic usage

```python
import mlflow
from modeltest import ModelSuite
from modeltest.integrations.mlflow import log_suite_result
from modeltest.scenarios import MinimumAccuracyTest

suite = ModelSuite(name="fraud-v2")
suite.add_test(MinimumAccuracyTest(threshold=0.85))
result = suite.run(model, X_val, y_val)

with mlflow.start_run():
    log_suite_result(result)
```

## What gets logged

| What | Key pattern | Example |
| ---- | ----------- | ------- |
| Test status (param) | `<test>.status` | `MinimumAccuracyTest.status = PASSED` |
| Test duration (metric) | `<test>.duration_ms` | `12.4` |
| Numeric test metrics | `<test>.<metric>` | whatever your tests record in `metrics` |
| Aggregates (metrics) | `num_passed`, `num_failed`, `passed` | `3`, `1`, `0.0` |
| Full report (artifact) | `modeltest-report.json` | rendered JSON of the whole run |

## Logging into a specific run

`log_suite_result` supports logging into an already-open (or finished) run
by `run_id` — it uses an `MlflowClient` under the hood:

```python
log_suite_result(result, run_id="abc123")
```

## Namespacing

When several suites log into the same run, namespace the keys to avoid
collisions:

```python
log_suite_result(result, param_prefix="fraud.", metric_prefix="fraud.")
```

## Options

| Param | Default | Description |
| ----- | ------- | ----------- |
| `run_id` | `None` | Log into this run instead of the active one. |
| `param_prefix` | `""` | Prefix for logged param names. |
| `metric_prefix` | `""` | Prefix for logged metric names. |
| `log_artifacts` | `True` | Also save the JSON report artifact. |
| `flush` | `True` | Flush MLflow's async logging before returning. |

If MLflow is not installed, the integration raises
`MlflowNotInstalledError` with install instructions — the core library
never imports MLflow unless you call it.
