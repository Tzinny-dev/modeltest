# Performance

Performance tests answer: *is the model good enough?* — globally, per
subgroup, and with statistical confidence.

Supported `metric` values: `accuracy`, `precision`, `recall`, `f1`,
`roc_auc`.

## MinimumAccuracyTest

Assert the overall metric is at least a threshold.

```python
from modeltest.scenarios import MinimumAccuracyTest

MinimumAccuracyTest(threshold=0.85, metric="accuracy")
```

| Param | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `threshold` | float | `0.85` | Minimum acceptable value. |
| `metric` | str | `"accuracy"` | Any [supported metric](#performance). |

Fails with detail like `accuracy = 0.7810 < threshold 0.85`.

## GroupPerformanceTest

Assert the metric stays above a threshold **for every subgroup** of a
categorical column. Useful to catch models that are fine on average but weak
for a segment.

```python
from modeltest.scenarios import GroupPerformanceTest

GroupPerformanceTest(metric="accuracy", threshold=0.8, group_col="gender")
```

| Param | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `metric` | str | `"accuracy"` | Metric evaluated per group. |
| `threshold` | float | `0.8` | Minimum acceptable value for *every* group. |
| `group_col` | str | `"gender"` | Categorical column in `X_val`. |

The failure message names the offending group and its score, e.g.
`group 'male': accuracy = 0.7810 < threshold 0.8`.

## ConfidenceThresholdTest

Assert a metric exceeds a threshold **with a confidence interval**. Instead
of comparing a point estimate, it bootstrap-resamples the validation
predictions and only passes when even the *lower* bound of the interval
clears the threshold — the right way to gate on small validation samples,
where one lucky number can look fine.

```python
from modeltest.scenarios import ConfidenceThresholdTest

ConfidenceThresholdTest(metric="accuracy", threshold=0.75, n_boot=1000, alpha=0.05)
```

| Param | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `metric` | str | `"accuracy"` | Metric to bound. |
| `threshold` | float | `0.85` | Value the selected bound must clear. |
| `n_boot` | int | `1000` | Bootstrap resamples. |
| `alpha` | float | `0.05` | Interval level (`1 - alpha` confidence). |
| `bound` | str | `"lower"` | `"lower"`: pass when `lower >= threshold` (floors). `"upper"`: pass when `upper <= threshold` (caps, e.g. error-rate ceilings). |
| `random_state` | int | `0` | Seed for reproducible intervals. |

The detail line records the estimate, the interval and the bound compared:
`accuracy: est=0.8120 CI=[0.7601, 0.8620] (95%); lower CI bound >= threshold 0.75`.
