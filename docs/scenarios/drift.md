# Drift

Drift tests detect distribution shift between the training data and the data
the model is being validated on. They compare **data distributions, not
predictions** — that is why they need `X_train` in the context and never
call the model.

On the CLI, pass `--train-data train.csv` to enable them. From Python:

```python
result = suite.run(model, X_val, y_val, X_train=X_train)
```

Without training data in the context they raise `ValueError` (reported as an
ERROR, not a FAILED test).

!!! tip "Rule of thumb for PSI"

    PSI `< 0.1`: no significant shift · `0.1–0.25`: moderate shift ·
    `> 0.25`: major shift. The default `max_psi=0.25` only fails on major
    shift; tighten it (e.g. `0.15`) for sensitive models.

## DataDriftTest

Assert the Population Stability Index (PSI) stays below a threshold for
every checked column.

```python
from modeltest.scenarios import DataDriftTest

DataDriftTest(feature_cols=["age", "income"], max_psi=0.15, n_bins=10)
```

| Param | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `feature_cols` | list[str] \| None | `None` | Columns to check; `None` checks **all numeric columns** of `X_train`. |
| `max_psi` | float | `0.25` | Maximum tolerated PSI per column. |
| `n_bins` | int | `10` | Quantile bins used to discretize each column. |

```yaml
- type: data_drift
  params: {features: [age, income], max_psi: 0.15}
```

Fails with detail like `PSI for income = 0.4102 exceeds max_psi 0.15`. A
column missing from `X_val` also fails the test.

## KSTest

Assert the two-sample Kolmogorov–Smirnov p-value stays **above** a threshold
per column: a low p-value means the train and validation distributions of
that column differ significantly.

```python
from modeltest.scenarios import KSTest

KSTest(feature_cols=None, min_p_value=0.05)
```

| Param | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `feature_cols` | list[str] \| None | `None` | Columns to check; `None` checks all numeric columns of `X_train`. |
| `min_p_value` | float | `0.05` | Minimum tolerated KS p-value per column. |

```yaml
- type: ks
  params: {columns: [age, income], min_p_value: 0.05}
```

!!! note "PSI vs KS"

    PSI is bucketed and symmetric-friendly (works well on binned score
    distributions); KS is a formal hypothesis test sensitive anywhere the
    ECDFs diverge. Running both gives a cheap, complementary drift tripwire.
