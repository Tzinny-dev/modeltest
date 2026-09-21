# YAML suites

Define your whole model contract declaratively — no Python, reviewable in a
PR like any other config file:

```yaml
suite:
  name: "Credit Scoring Model"
  tests:
    - type: minimum_accuracy
      params: {threshold: 0.85}
    - type: group_performance
      params: {metric: accuracy, threshold: 0.8, group_col: "gender"}
    - type: robustness
      params: {noise_std: 0.01, max_drop: 0.03}
    - type: data_drift
      params: {features: [age, income], max_psi: 0.15}
    - type: equal_opportunity
      params: {protected: "gender", max_diff: 0.1}
    - type: statistical_parity
      params: {protected: "gender", max_diff: 0.1, min_ratio: 0.8}
    - type: data_invariant
      params: {expected_columns: [age, income], max_null_ratio: 0.02}
```

Run it with `modeltest validate --suite suite.yaml ...` or load it from
Python:

```python
from modeltest.config import load_suite_yaml

suite = load_suite_yaml("suite.yaml")
result = suite.run(model, X_val, y_val, X_train=X_train)
```

## Structure

- `suite.name` — a label for reports (default `"suite"`).
- `suite.tests` — the list of tests. Each entry has:
  - `type` — the test name (see the
    [registry table](#built-in-type-names)), or a
    [custom test path](custom-tests.md#referencing-custom-tests-from-yaml).
  - `params` — a mapping of constructor arguments for that test.

## Parameter aliases

A few constructor names have YAML-friendly aliases accepted by `params`:

| Alias | Resolves to |
| ----- | ----------- |
| `features` | `feature_cols` (drift tests) |
| `columns` | `feature_cols` (drift tests) / `columns` (no_null) |
| `group` | `group_col` |
| `protected` | `protected_col` |

## Validation behavior

- **Unknown params raise** a `ValueError` listing the valid ones — typos
  fail fast instead of being silently ignored.
- **Unknown types raise** a `ValueError` listing all known type names.
- A malformed structure (missing top-level `suite:`, non-mapping test
  entries) raises `ValueError` with a descriptive message.

## Built-in type names

| YAML `type` | Class | Checks |
| ----------- | ----- | ------ |
| `minimum_accuracy` | `MinimumAccuracyTest` | Global metric above threshold |
| `group_performance` | `GroupPerformanceTest` | Metric above threshold per subgroup |
| `confidence_threshold` | `ConfidenceThresholdTest` | Metric floor via bootstrap CI |
| `robustness` | `RobustnessTest` | Performance under feature noise |
| `data_invariant` | `DataInvariantTest` | Expected columns / null ratios |
| `no_null` | `NoNullTest` | No missing values |
| `data_drift` | `DataDriftTest` | PSI between train & validation |
| `ks` | `KSTest` | KS p-value per column |
| `equal_opportunity` | `EqualOpportunityTest` | Balanced TPR across protected groups |
| `statistical_parity` | `StatisticalParityTest` | Balanced selection rate (+ 4/5ths rule) |
| `feature_dominance` | `FeatureDominanceTest` | No single feature dominates attribution |
| `top_features` | `TopFeaturesTest` | Top-K attributed features are expected |

Every test's parameters and defaults are documented in the
[Scenarios section](../scenarios/index.md).

!!! tip "Round-trip"

    `dump_suite_yaml(suite, "suite.yaml")` serialises a Python-built suite
    back to YAML (best effort, built-in tests only) — handy to bootstrap a
    declarative contract from code.
