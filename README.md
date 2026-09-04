# modeltest

> If you test your code, why not your model?

`modeltest` is a unit-testing framework for machine learning models. It lets you
define contracts for model quality, robustness, fairness, and data invariants,
and run them automatically in your CI/CD pipeline — just like `pytest` for code.

## Quick start

```python
from modeltest import ModelSuite
from modeltest.scenarios import MinimumAccuracyTest, GroupPerformanceTest

suite = ModelSuite(name="Fraud Detection")
suite.add_test(MinimumAccuracyTest(threshold=0.85))
suite.add_test(GroupPerformanceTest(metric="accuracy", threshold=0.8, group_col="gender"))

result = suite.run(model, X_val, y_val, model_name="fraud_rf")
print(result.report(style="table"))
```

## CLI

After training, validate a model contract from the command line:

```bash
# Declarative suite (recommended)
modeltest validate \
  --suite suite.yaml \
  --model model.pkl \
  --data validation.csv \
  --target target \
  --train-data train.csv \
  --output report.xml

# Python suite (suite.py exposing `suite`)
modeltest validate --suite suite.py --model model.pkl --data validation.csv --target target
```

Extras:
- `--output report.xml` writes JUnit XML for CI reporters.
- `--train-data` enables the drift tests (they compare train vs. validation distributions).
- Exit code is `1` if any test fails, `0` otherwise.

## Prediction caching

Within a single `suite.run(...)`, predictions are computed once and reused
across every test. `TestContext.predict()` caches by a content hash of the
input, so tests predicting on the *same* data (`MinimumAccuracyTest`,
`GroupPerformanceTest`, the fairness tests, ...) each reuse the result instead
of re-running the model. Perturbed inputs (e.g. the robustness test's noisy
copy) get their own cache entry, so caching never compromises correctness.

Disable it if you need a fresh prediction every call:

```python
ctx = TestContext(model=model, X_val=X_val, y_val=y_val, cache_predictions=False)
```

## YAML suites

Define your contract declaratively — no code needed:

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

## Multi-framework support

`TestContext` talks to models through a small adapter interface
(`modeltest.wrappers`). Out of the box it normalizes:

- **scikit-learn** estimators(`predict`, and `predict_proba` when available)
- **PyTorch** `nn.Module` (`predict` = argmax over logits, `predict_proba` = softmax)
- **Keras / TensorFlow** models (binary threshold or multiclass argmax)

Pass any of these straight to `suite.run(model, ...)`; the right adapter is
picked automatically. Custom framework? Implement a
`ModelWrapper` subclass and pass an instance as the model.

## Built-in test types

| `type` (YAML) | Class | Checks |
|---------------|-------|--------|
| `minimum_accuracy` | `MinimumAccuracyTest` | Global metric above threshold |
| `group_performance` | `GroupPerformanceTest` | Metric above threshold per subgroup |
| `confidence_threshold` | `ConfidenceThresholdTest` | Metric floor via bootstrap CI (lower/upper bound) |
| `robustness` | `RobustnessTest` | Performance under feature noise |
| `data_invariant` | `DataInvariantTest` | Expected columns / null ratios |
| `no_null` | `NoNullTest` | No missing values |
| `data_drift` | `DataDriftTest` | PSI between train & validation |
| `ks` | `KSTest` | KS p-value per column |
| `equal_opportunity` | `EqualOpportunityTest` | Balanced TPR across protected groups |
| `statistical_parity` | `StatisticalParityTest` | Balanced selection rate (+ 4/5ths rule) |
| `feature_dominance` | `FeatureDominanceTest` | No single feature dominates attribution |
| `top_features` | `TopFeaturesTest` | Top-K attributed features are expected |

> Explainability tests use SHAP. Install with `pip install modeltest[explain]`
> (or `modeltest`'s `explain` extra). You can also pass your own explainer
> callable to any explainability test.

## Custom tests

Any class subclassing `ModelTest` can be referenced from a YAML suite
directly, by dotted import path — no registration required:

```yaml
suite:
  name: "Income Model"
  tests:
    - type: minimum_accuracy
      params: {threshold: 0.85}
    - type: myproject.custom_tests:ZeroPredictionShareTest
      params: {min_positive_share: 0.01}
```

Both `module.path:Class` and `module.path.Class` work. The module is looked
up on the import path (your current working directory is added automatically).
Programmatic registration is also available for short, friendlier names:

```python
from modeltest import register
register("zero_share", ZeroPredictionShareTest)
# ...now use `type: zero_share` in YAML
```

## Development

Install dev tools and run the quality gates:

```bash
make install          # pip install -e ".[dev]"
make lint             # ruff check
make format           # ruff format
make test             # pytest (with 80% coverage gate)
make precommit        # install git pre-commit hooks (lint+format)
```

Continuous integration mirrors these gates: `lint`, `test-library` (with
coverage) and `validate-model` all run on every push / PR
([.github/workflows/validate.yml](.github/workflows/validate.yml)).

## CI/CD

A ready-to-use [GitHub Actions workflow](.github/workflows/validate.yml) runs the
library's own tests and validates your model contract on every push / PR. It
publishes both reports (as JUnit) and fails the pipeline if the model doesn't
meet its contract.

The model job trains a sample model and validates it:

```bash
modeltest validate --suite examples/suite.yaml --model examples/model.pkl \
  --data examples/validation.csv --target target --train-data examples/train.csv
```

To point it at your real artifacts, update the `validate-model` job's `run` step paths.

