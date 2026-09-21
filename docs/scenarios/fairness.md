# Fairness

Fairness tests detect unintended bias across protected groups. Both take a
`protected_col` — a categorical column in `X_val` (e.g. `"gender"`) — and
compare outcome rates across its groups. YAML accepts the alias
`protected` for `protected_col`.

## EqualOpportunityTest

Assert that **true-positive rates** are balanced across protected groups —
equal opportunity means the model is equally good at catching actual
positives for every group.

```python
from modeltest.scenarios import EqualOpportunityTest

EqualOpportunityTest(protected_col="gender", max_diff=0.1)
```

| Param | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `protected_col` | str | *required* | Categorical column in `X_val` defining the groups. |
| `max_diff` | float | `0.1` | Maximum tolerated gap between the best and worst group TPR. |
| `pos_label` | int | `1` | Which class counts as *positive*. |

```yaml
- type: equal_opportunity
  params: {protected: "gender", max_diff: 0.1}
```

Failure detail names the gap and every group's TPR, e.g.
`Equal opportunity TPR gap = 0.1800 > max_diff 0.1; TPRs: {'male': 0.92, 'female': 0.74}`.

## StatisticalParityTest

Assert **selection (acceptance) rates** are balanced across groups. It
checks both classic disparity criteria at once:

- the absolute **demographic parity difference** (`max_diff`), and
- the **disparate-impact ratio** (`min_ratio` — `0.8` implements the
  4/5ths rule).

```python
from modeltest.scenarios import StatisticalParityTest

StatisticalParityTest(protected_col="gender", max_diff=0.1, min_ratio=0.8)
```

| Param | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `protected_col` | str | *required* | Categorical column defining the groups. |
| `max_diff` | float | `0.1` | Maximum tolerated difference in selection rates. |
| `min_ratio` | float | `0.8` | Minimum tolerated ratio between the least and most selected group. |
| `pos_label` | int | `1` | Which prediction value counts as *selected*. |

```yaml
- type: statistical_parity
  params: {protected: "gender", max_diff: 0.1, min_ratio: 0.8}
```

!!! note "Choosing a criterion"

    Equal opportunity conditions on *true* positives (it needs ground-truth
    labels per group); statistical parity only looks at predicted positives.
    Regulated domains often require the 4/5ths ratio — hence the dual check.
