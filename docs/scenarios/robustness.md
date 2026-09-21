# Robustness

## RobustnessTest

Assert that adding Gaussian noise to the numeric features does not degrade
the metric by more than `max_drop` relative to the clean data. This is a
stability check: a production model should not collapse when inputs wobble
by sensor-level amounts.

```python
from modeltest.scenarios import RobustnessTest

RobustnessTest(noise_std=0.01, max_drop=0.03, metric="accuracy")
```

| Param | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `noise_std` | float | `0.05` | Standard deviation of the Gaussian noise added to numeric columns. |
| `max_drop` | float | `0.05` | Maximum allowed absolute drop (`clean - noisy`). |
| `metric` | str | `"accuracy"` | Metric compared clean vs. noisy. |
| `seed` | int | `42` | RNG seed — the perturbation is reproducible. |

How it works:

1. Copies `X_val` and adds `N(0, noise_std)` noise to **numeric columns
   only** (categorical columns are untouched).
2. Predicts on both the clean and the noisy frames (each gets its own
   [prediction-cache](../usage/core-concepts.md#prediction-caching) entry).
3. Fails if `score_clean - score_noisy > max_drop`.

```yaml
- type: robustness
  params: {noise_std: 0.01, max_drop: 0.03}
```

!!! note "Scale matters"

    `noise_std` is in the units of your features. A `noise_std` of `0.01`
    is negligible for a `0..1` normalized feature and invisible for an
    income column measured in thousands — tune it per dataset (the
    [examples suite](https://github.com/Tzinny-dev/modeltest/blob/main/examples/suite.yaml)
    uses `noise_std: 100` on raw features).
