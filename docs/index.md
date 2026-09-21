# modeltest

> If you test your code, why not your model?

<div class="grid" markdown>

[Get started: core concepts](usage/core-concepts.md){ .md-button .md-button--primary }
[Run from the CLI](usage/cli.md){ .md-button }
[Browse the 12 built-in tests](scenarios/index.md){ .md-button }

</div>

`modeltest` is a unit-testing framework for machine learning models. It lets
you define **contracts** for model quality, robustness, fairness, and data
invariants — and run them automatically in your CI/CD pipeline, just like
`pytest` does for code.

```bash
pip install modeltest
```

## Why?

Model quality quietly degrades: data drifts, upstream pipelines change
schemas, a retrain produces a subtly worse model. `modeltest` turns those
concerns into executable checks that **fail your build** before your users
find out.

```python
from modeltest import ModelSuite
from modeltest.scenarios import MinimumAccuracyTest, GroupPerformanceTest

suite = ModelSuite(name="Fraud Detection")
suite.add_test(MinimumAccuracyTest(threshold=0.85))
suite.add_test(
    GroupPerformanceTest(metric="accuracy", threshold=0.8, group_col="gender")
)

result = suite.run(model, X_val, y_val, model_name="fraud_rf")
print(result.report(style="table"))
```

```text
Suite: Fraud Detection

STATUS   TEST                                TIME (ms)    DETAIL
--------------------------------------------------------------------------------
PASS     MinimumAccuracyTest                 12.4
FAIL     GroupPerformanceTest                13.0         group 'male': accuracy = 0.7810 < threshold 0.8
--------------------------------------------------------------------------------
1 passed, 1 failed
```

## Highlights

- :material-flask: **12 built-in scenarios** — accuracy floors, bootstrap
  confidence intervals, robustness to noise, PSI/KS drift, fairness gaps,
  data invariants, SHAP-based explainability.
- :material-cog: **Multi-framework** — scikit-learn, PyTorch, Keras/TensorFlow,
  sklearn `Pipeline`s, or your own adapter.
- :material-file-code: **Declarative YAML suites** — no code, reviewable in PRs.
  Plug in custom tests by dotted import path.
- :material-robot: **CI-native** — `modeltest validate` exits non-zero on
  failure and writes JUnit XML your CI can render.
- :material-chart-line: **MLflow tracking** — log every validation run as an
  experiment run with `pip install modeltest[mlflow]`.

## Where to next?

| I want to... | Go to |
| ------------ | ----- |
| Understand the run lifecycle, caching and reports | [Core concepts](usage/core-concepts.md) |
| Validate a trained model from the command line | [CLI](usage/cli.md) |
| Define contracts without writing Python | [YAML suites](usage/yaml-suites.md) |
| Test my own model class | [Custom tests](usage/custom-tests.md) |
| See every test type and its parameters | [Scenarios](scenarios/index.md) |
| Wire it into GitHub Actions | [GitHub Actions](integrations/github-actions.md) |
| Browse the full API | [API reference](api/core.md) |
