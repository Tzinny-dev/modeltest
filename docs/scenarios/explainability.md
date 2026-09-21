# Explainability

Explainability tests inspect **how the model uses its input features**.
Attributions are computed with SHAP by default — install the extra:

```bash
pip install modeltest[explain]
```

Heavy dependencies are imported lazily: the core stays light, and SHAP is
only required when these tests actually run. You can also pass **your own
explainer callable** to skip SHAP entirely.

## The explainer interface

```python
Explainer = Callable[[model, X], attribution_or_tuple]
```

Given the (wrapped) model and the feature frame, return an array of SHAP
values shaped `(n_samples, n_features)` — or a tuple
`(values, feature_names)` when your explainer knows the exact feature
columns the attribution refers to (useful after pipeline feature
engineering).

The default backend picks a SHAP explainer for you: `TreeExplainer` for
tree ensembles / logistic regression, falling back to `KernelExplainer`
(sampled) otherwise. sklearn `Pipeline`s are unwrapped — the final estimator
is explained over the *engineered* features, so names always line up.

## FeatureDominanceTest

Assert the model does not rely on a **single feature** to the exclusion of
all others. Catches degenerate models that effectively use one column (a
leaky feature, an ID, a target proxy).

```python
from modeltest.scenarios import FeatureDominanceTest

FeatureDominanceTest(max_top_share=0.9)
```

| Param | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `explainer` | callable \| None | `None` | Custom explainer; `None` uses the default SHAP backend. |
| `max_top_share` | float | `0.9` | Maximum share of total absolute attribution allowed for the single top feature. |

```yaml
- type: feature_dominance
  params: {max_top_share: 0.9}
```

If every attribution is zero (model outputs constant), the test fails with
`All SHAP attributions are zero; model is degenerate`.

## TopFeaturesTest

Assert the **top-K features by attribution** are within the expected set —
the features your team believes the model should be using.

```python
from modeltest.scenarios import TopFeaturesTest

TopFeaturesTest(expected_features=["age", "income", "score"], k=3)
```

| Param | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `expected_features` | list[str] | *required* | The plausible/desired feature set. |
| `k` | int | `3` | How many top features to check. |
| `explainer` | callable \| None | `None` | Custom explainer; `None` uses the default SHAP backend. |

```yaml
- type: top_features
  params: {expected_features: [age, income, score], k: 3}
```

Fails if any of the top-`k` ranked features falls outside
`expected_features`: `Unexpected top-3 features: ['zip_code']; attributions: {...}`.

!!! note "Performance"

    SHAP on large frames can be slow — run these tests on a sample of the
    validation data, or supply a fast custom explainer (e.g. a pre-built
    `TreeExplainer` wrapped in a callable).
