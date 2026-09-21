# Custom tests

Any class subclassing `ModelTest` works as a test. Override `test(self, ctx)`
and use plain `assert` statements — a failing `assert` marks the test as
FAILED with the assert message as the detail.

```python
from modeltest import ModelTest, TestContext


class ZeroPredictionShareTest(ModelTest):
    """Fail if too many predictions fall into a single class."""

    name = "zero_prediction_share"

    def __init__(self, max_share: float = 0.5):
        self.max_share = max_share

    def test(self, ctx: TestContext) -> None:
        import numpy as np

        y_pred = np.asarray(ctx.predict())
        values, counts = np.unique(y_pred, return_counts=True)
        share = counts.max() / counts.sum()
        assert share <= self.max_share, (
            f"{values[counts.argmax()]} covers {share:.2f} of predictions "
            f"(> max_share {self.max_share})"
        )
```

Notes:

- `ctx.predict()` returns the model's predictions (cached, shared with the
  other tests in the suite).
- Set `name` to control how the test appears in reports and JUnit XML.
- Raise `AssertionError` → **FAILED**. Any other exception → **ERROR**.

## Referencing custom tests from YAML

No registration required — reference them by dotted import path, both
`module.path:Class` and `module.path.Class` notations work. The module is
looked up on the import path; the current working directory is added
automatically, so a plain file next to your suite works from the CLI:

```yaml
suite:
  name: "Income Model"
  tests:
    - type: minimum_accuracy
      params: {threshold: 0.85}
    - type: myproject.custom_tests:ZeroPredictionShareTest
      params: {max_share: 0.5}
```

## Programmatic registration

For short, friendly names (or to replace a built-in), register the class:

```python
from modeltest import register, unregister
from my_tests import ZeroPredictionShareTest

register("zero_share", ZeroPredictionShareTest)
# ...now `type: zero_share` works in YAML:
```

```yaml
- type: zero_share
  params: {max_share: 0.5}
```

`unregister("zero_share")` removes it again (works for built-ins too).

## Returning a TestResult

For warning-only or non-blocking checks, return a `TestResult` instead of
raising — it short-circuits the outcome and lets you set the status and
metrics explicitly:

```python
from modeltest import TestResult, TestStatus

def test(self, ctx):
    ok, value = expensive_check(ctx)
    metrics = {"checks_value": value}
    if not ok:
        return TestResult(
            name=self.name, status=TestStatus.FAILED,
            detail="expensive check failed", metrics=metrics,
        )
    return TestResult(name=self.name, status=TestStatus.PASSED, metrics=metrics)
```

Whatever a test records into `metrics` (dict of numeric values) shows up in
the JSON report and as MLflow metrics via the
[integration](../integrations/mlflow.md).
