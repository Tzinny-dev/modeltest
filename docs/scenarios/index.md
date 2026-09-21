# Scenarios

`modeltest` ships **12 built-in test scenarios** grouped in six families.
Each page documents every parameter, its default, and when to reach for the
test. They are also the exact `type` names usable in
[YAML suites](../usage/yaml-suites.md).

| Family | Tests | Needs | Answers |
| ------ | ----- | ----- | ------- |
| [Performance](performance.md) | `minimum_accuracy`, `group_performance`, `confidence_threshold` | predictions | Is the model good enough, globally and per group — *confidently*? |
| [Robustness](robustness.md) | `robustness` | predictions | Does quality survive small input noise? |
| [Drift](drift.md) | `data_drift`, `ks` | **train data** | Is the incoming data still the data we trained on? |
| [Fairness](fairness.md) | `equal_opportunity`, `statistical_parity` | predictions + protected col | Are outcomes balanced across protected groups? |
| [Data invariants](data.md) | `data_invariant`, `no_null` | validation data only | Does the data still have the shape/schema we expect? |
| [Explainability](explainability.md) | `feature_dominance`, `top_features` | SHAP (`modeltest[explain]`) | Does the model rely on the features we think it does? |

All tests share the same contract: subclass `ModelTest`, run against a
`TestContext`, `PASSED` on clean return / `FAILED` on `AssertionError`.

## Quick reference

```yaml
suite:
  name: "Full contract example"
  tests:
    - type: minimum_accuracy
      params: {threshold: 0.85, metric: accuracy}
    - type: group_performance
      params: {metric: accuracy, threshold: 0.8, group_col: "gender"}
    - type: confidence_threshold
      params: {metric: accuracy, threshold: 0.75, n_boot: 1000, alpha: 0.05}
    - type: robustness
      params: {noise_std: 0.01, max_drop: 0.03}
    - type: data_drift
      params: {features: [age, income], max_psi: 0.15}
    - type: ks
      params: {min_p_value: 0.05}
    - type: equal_opportunity
      params: {protected: "gender", max_diff: 0.1}
    - type: statistical_parity
      params: {protected: "gender", max_diff: 0.1, min_ratio: 0.8}
    - type: data_invariant
      params: {expected_columns: [age, income], max_null_ratio: 0.02}
    - type: no_null
    - type: feature_dominance
      params: {max_top_share: 0.9}
    - type: top_features
      params: {expected_features: [age, income, score], k: 3}
```

Notes:

- Drift tests require passing `X_train` (CLI: `--train-data`) — they compare
  distributions, not predictions.
- Explainability tests require the `explain` extra (`pip install
  modeltest[explain]`) or a custom explainer callable.
