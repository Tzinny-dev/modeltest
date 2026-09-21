# Model wrappers

`TestContext` talks to models through a small adapter interface
(`modeltest.wrappers`):

```text
predict(X)       -> class labels
predict_proba(X) -> probability estimates (optional)
```

`wrap(model)` returns a normalized object implementing that interface,
dispatching on the framework. Out of the box it handles:

- **scikit-learn** estimators — any `BaseEstimator` with `predict`;
  `predict_proba` is used when available (label probabilities via
  `SklearnClassifier`).
- **sklearn `Pipeline`s** — the same adapter works; the
  [explainability tests](../scenarios/explainability.md) additionally unwrap
  the pipeline so SHAP sees the final estimator over engineered features.
- **PyTorch** `nn.Module` — `predict` returns argmax class indices,
  `predict_proba` returns softmax probabilities. The model is switched to
  `eval()` and gradients are disabled automatically.
- **Keras / TensorFlow** — `predict` thresholds single-output models at 0.5,
  or takes argmax for multiclass (`multiclass=True` forces it).
- **Anything else** — the fallback assumes the model already exposes
  `predict` / `predict_proba` (e.g. your own adapter), so custom model
  classes work without registration.

```python
from modeltest import ModelSuite
from modeltest.scenarios import MinimumAccuracyTest
from modeltest.wrappers import wrap

model = wrap(my_custom_model)     # optional — suites wrap automatically
suite = ModelSuite(name="suite", tests=[MinimumAccuracyTest()])
result = suite.run(model, X_val, y_val)
```

## Tuning adapters

`wrap` forwards keyword arguments to the adapter:

| Adapter | kwarg | Purpose |
| ------- | ----- | ------- |
| `TorchModel` | `input_key` | Key when the model expects a dict input. |
| `TorchModel` | `device` | Device to move inputs to before forward. |
| `KerasModel` | `multiclass` | Force argmax decoding even for 2 outputs. |

```python
model = wrap(torch_model, device="cuda")
```

## Feature-name filtering

Wrappers expose the model's `feature_names_in_` when available. The context
uses it to filter `X_val` down to the exact features the model expects —
in the right order — before calling it. Extra helper columns (group columns,
IDs, timestamps) therefore coexist peacefully in your validation frame.

## Writing your own adapter

Subclass `ModelWrapper` and implement `predict` (and optionally
`predict_proba`). Instances of `ModelWrapper` are passed through `wrap`
untouched, so you can pre-configure them and hand them to `suite.run`:

```python
from modeltest.wrappers import ModelWrapper

class MyModelWrapper(ModelWrapper):
    def predict(self, X):
        return self.model.classify(X)   # your API

    def predict_proba(self, X):
        return self.model.probabilities(X)
```

See the [wrappers API reference](../api/wrappers.md) for the full class
documentation.
