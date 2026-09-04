"""Tests for the prediction cache in TestContext."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from modeltest import ModelSuite, TestContext
from modeltest.scenarios import MinimumAccuracyTest


class _CountingModel(RandomForestClassifier):
    """Wraps RandomForestClassifier and counts predict() calls."""

    def __init__(self):
        super().__init__(n_estimators=20, random_state=0)
        self.call_count = 0

    def predict(self, X):
        self.call_count += 1
        return super().predict(X)


def _dataset():
    rng = np.random.default_rng(0)
    n = 800
    X = pd.DataFrame(
        {"age": rng.normal(45, 12, n), "income": rng.normal(60000, 20000, n)}
    )
    y = (X["income"] > X["age"] * 800 + rng.normal(0, 16000, n)).astype(int)
    return X, y


def _build():
    X, y = _dataset()
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.4, random_state=0
    )
    model = _CountingModel()
    model.fit(X_train, y_train)
    return model, X_val, y_val


class TestPredictionCache:
    def test_predict_called_once_across_suite(self):
        model, X_val, y_val = _build()
        suite = ModelSuite(name="cache")
        suite.add_tests(
            MinimumAccuracyTest(threshold=0.0),
            MinimumAccuracyTest(threshold=0.0),
        )
        suite.run(model, X_val, y_val)
        # Two tests predicting on the same data -> only one model.predict
        assert model.call_count == 1

    def test_extra_column_does_not_break_cache(self):
        model, X_val, y_val = _build()
        X_val["_id"] = np.arange(len(X_val))  # non-feature column
        ctx = TestContext(model=model, X_val=X_val, y_val=y_val)
        ctx.predict()
        ctx.predict()
        assert model.call_count == 1

    def test_numpy_array_input_cached(self):
        model, X_val, y_val = _build()
        arr = X_val[["age", "income"]].to_numpy()
        ctx = TestContext(model=model, X_val=arr, y_val=y_val.to_numpy())
        p1 = ctx.predict(arr)
        p2 = ctx.predict(arr)
        assert model.call_count == 1
        assert np.array_equal(p1, p2)

    def test_disabling_cache_forces_repredict(self):
        model, X_val, y_val = _build()
        ctx = TestContext(
            model=model, X_val=X_val, y_val=y_val, cache_predictions=False
        )
        ctx.predict()
        ctx.predict()
        assert model.call_count == 2

    def test_perturbed_input_gets_own_cache_entry(self):
        """A different (noisy) input must not reuse the clean prediction."""
        model, X_val, y_val = _build()
        ctx = TestContext(model=model, X_val=X_val, y_val=y_val)
        clean = ctx.predict(X_val)
        noisy = X_val.copy()
        noisy["age"] = noisy["age"] + 99999.0  # clearly different
        noisy_pred = ctx.predict(noisy)
        assert model.call_count == 2  # clean + noisy each computed once
        assert not np.array_equal(clean, noisy_pred)
