# Data invariants

Structural guarantees on the validation data itself. These tests never call
the model — they validate the *data contract* upstream of any modeling.

## DataInvariantTest

Assert expected columns are present and that no column has too many missing
values.

```python
from modeltest.scenarios import DataInvariantTest

DataInvariantTest(expected_columns=["age", "income"], max_null_ratio=0.02)
```

| Param | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `expected_columns` | list[str] \| None | `None` | Columns that must exist in `X_val`; `None` skips the column check. |
| `max_null_ratio` | float | `0.02` | Maximum tolerated fraction of nulls per column. Set **negative to disable** the null check. |

```yaml
- type: data_invariant
  params: {expected_columns: [age, income], max_null_ratio: 0.02}
```

Failure details: `Missing columns: ['income']` or
`Columns exceed null ratio 0.02: {'phone': 0.31}` (it names every offending
column and its ratio).

## NoNullTest

Assert **no null values** exist in the validation data (or in the selected
columns).

```python
from modeltest.scenarios import NoNullTest

NoNullTest()                       # whole frame
NoNullTest(columns=["age", "income"])  # subset
```

| Param | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `columns` | list[str] \| None | `None` | Columns to check; `None` checks the entire `X_val`. |

```yaml
- type: no_null
```

!!! tip "Invariants vs drift"

    Invariants are *hard* schema/quality rules (schema changed, feed broke);
    drift tests are *statistical* comparisons (distribution moved). Use
    invariants to catch pipeline breakage fast, drift to catch slow decay —
    see [Drift](drift.md).
